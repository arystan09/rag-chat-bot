"""FastAPI router for RAG chat endpoints."""
import time
import uuid
from typing import List
from fastapi import APIRouter, HTTPException, status
from loguru import logger
from sqlalchemy.orm import Session

from app.api.schemas import QueryRequest, QueryResponse, RetrievedChunk, Citation, ErrorResponse
from app.vector.elasticsearch.search import hybrid_search
from app.db.session import SessionLocal
from app.db.models import QueryLog
from app.core.settings import settings


router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def get_conversation_history(db: Session, conversation_id: str, user_id: int, limit: int = 5) -> List[dict]:
    """
    Get conversation history for context.
    
    Args:
        db: Database session
        conversation_id: Conversation ID
        user_id: User ID
        limit: Number of recent exchanges to retrieve
        
    Returns:
        List of conversation exchanges
    """
    try:
        # Get recent queries in this conversation
        queries = db.query(QueryLog).filter(
            QueryLog.conversation_id == conversation_id,
            QueryLog.user_id == user_id
        ).order_by(QueryLog.created_at.desc()).limit(limit).all()
        
        history = []
        for query in reversed(queries):  # Reverse to get chronological order
            history.append({
                "question": query.question,
                "answer": query.answer or f"Previous answer from {query.created_at.strftime('%H:%M')}",
                "timestamp": query.created_at.isoformat()
            })
        
        return history
        
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        return []


def call_llm(context: str, question: str, conversation_history: List[dict] = None) -> tuple[str, int, int]:
    """
    Call LLM with context and conversation history.
    
    Args:
        context: Retrieved document context
        question: User question
        conversation_history: Previous conversation exchanges
        
    Returns:
        Tuple of (answer, token_in, token_out)
    """
    try:
        from openai import OpenAI
        
        if not settings.ai.openai_api_key:
            logger.warning("OpenAI API key not configured")
            return "⚠️ OpenAI временно недоступен. Попробуйте позже.", 0, 0
        
        logger.info(f"Calling LLM with model: {settings.ai.llm_model}")
        logger.debug(f"Context length: {len(context)} characters")
        logger.debug(f"Question: {question}")
        logger.debug(f"Conversation history entries: {len(conversation_history or [])}")
        
        client = OpenAI(api_key=settings.ai.openai_api_key)
        
        # Build messages with conversation history
        messages = [
            {
                "role": "system", 
                "content": "Ты - помощник по документам. Отвечай на русском языке на основе предоставленного контекста."
            }
        ]
        
        # Add conversation history
        if conversation_history:
            messages.append({"role": "system", "content": "ПРЕДЫДУЩИЙ ДИАЛОГ:"})
            for exchange in conversation_history[-3:]:  # Last 3 exchanges
                messages.append({"role": "user", "content": exchange["question"]})
                messages.append({"role": "assistant", "content": exchange["answer"]})
        
        # Add current context and question
        prompt = f"""КОНТЕКСТ ДОКУМЕНТОВ:
{context}

ТЕКУЩИЙ ВОПРОС: {question}

ИНСТРУКЦИИ:
1. Отвечай ТОЛЬКО на русском языке
2. Используй информацию ТОЛЬКО из предоставленного контекста
3. Если информации нет в контексте - скажи об этом честно
4. Структурируй ответ логично и понятно
5. Не выдумывай информацию, которой нет в контексте
6. Отвечай подробно и структурированно
7. Если есть несколько аспектов вопроса - рассмотри их все
8. Используй конкретные детали из документов (цифры, даты, названия)
9. ВАЖНО: Если вопрос короткий и неясный (например, "Что за университет?"), попробуй понять его в контексте предыдущих вопросов в диалоге. Если в предыдущем диалоге обсуждалась тема визы или поступления, то "университет" скорее всего относится к университету, в который планируется поступление.

ОТВЕТ:"""
        
        messages.append({"role": "user", "content": prompt})
        
        logger.debug(f"Sending request to OpenAI with {len(messages)} messages")
        
        response = client.chat.completions.create(
            model=settings.ai.llm_model,
            messages=messages,
            max_tokens=settings.ai.max_tokens,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        token_in = response.usage.prompt_tokens
        token_out = response.usage.completion_tokens
        
        logger.info(f"LLM response received: {len(answer)} characters, tokens: {token_in} in, {token_out} out")
        
        return answer, token_in, token_out
        
    except ImportError:
        logger.warning("OpenAI library not installed")
        return "⚠️ OpenAI временно недоступен. Попробуйте позже.", 0, 0
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return "⚠️ Ошибка при обращении к ИИ. Попробуйте позже.", 0, 0


@router.post("/query", response_model=QueryResponse)
async def chat_query(request: QueryRequest):
    """
    Process RAG chat query with optional conversation history.
    
    Args:
        request: Query request with user_id, question, top_k, and optional conversation_id
        
    Returns:
        AI-generated answer with retrieved chunks and metadata
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing query for user {request.user_id}: {request.question[:50]}... (conversation_id: {request.conversation_id})")
        
        # Generate or use conversation ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Get conversation history if conversation_id is provided
        conversation_history = []
        if request.conversation_id:
            db = SessionLocal()
            try:
                conversation_history = get_conversation_history(db, conversation_id, request.user_id)
                logger.info(f"Retrieved {len(conversation_history)} conversation history entries")
            finally:
                db.close()
        
        # Perform hybrid search with more candidates
        logger.info(f"Performing hybrid search for: {request.question}")
        search_results = hybrid_search(request.question, min(request.top_k * 2, 15))  # Get more candidates
        
        if not search_results:
            logger.warning("No search results found")
            return QueryResponse(
                answer="К сожалению, я не смог найти релевантную информацию в ваших документах для ответа на этот вопрос.",
                retrieved_chunks=[],
                citations=[],
                model=settings.ai.llm_model,
                token_in=0,
                token_out=0,
                latency_ms=int((time.time() - start_time) * 1000),
                conversation_id=conversation_id
            )
        
        logger.info(f"Found {len(search_results)} search results")
        
        # Build context from retrieved chunks with better filtering
        context_parts = []
        retrieved_chunks = []
        
        # Filter and rank chunks by relevance with improved logic
        filtered_results = []
        for result in search_results:
            # Calculate relevance score based on multiple factors
            text = result['text'].lower()
            question_words = request.question.lower().split()
            
            # Factor 1: Keyword matches (weight: 0.3)
            keyword_matches = sum(1 for word in question_words if word in text)
            keyword_score = keyword_matches / len(question_words) if question_words else 0
            
            # Factor 2: Original search score (weight: 0.7)
            search_score = result['score']
            
            # Factor 3: Text length penalty (shorter texts are often more relevant)
            length_penalty = min(1.0, 500 / len(text)) if len(text) > 0 else 0
            
            # Combined relevance score
            relevance_score = (0.3 * keyword_score + 0.7 * search_score) * length_penalty
            
            # Only include results with minimum relevance threshold
            if relevance_score >= 0.3:  # Higher threshold for better relevance
                result['relevance_score'] = relevance_score
                filtered_results.append(result)
        
        # Sort by relevance score and take only the best result
        filtered_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        top_results = filtered_results[:1]  # Only return the most relevant result
        
        logger.info(f"Selected top {len(top_results)} results for context")
        logger.info(f"Top results scores: {[r['score'] for r in top_results]}")
        logger.info(f"Top results filenames: {[r['filename'] for r in top_results]}")
        
        # If no relevant results found, return appropriate response
        if not top_results:
            logger.warning("No relevant results found for query")
            return QueryResponse(
                answer="К сожалению, в предоставленных документах нет информации, которая позволила бы ответить на ваш вопрос.",
                retrieved_chunks=[],
                citations=[],
                model=settings.ai.llm_model,
                token_in=0,
                token_out=0,
                latency_ms=int((time.time() - start_time) * 1000),
                conversation_id=conversation_id
            )
        
        for result in top_results:
            context_parts.append(f"Document: {result['filename']}\nContent: {result['text']}")
            
            retrieved_chunks.append(RetrievedChunk(
                chunk_id=result['chunk_id'],
                doc_id=result['doc_id'],
                filename=result['filename'],
                url=result.get('url', ''),
                text=result['text'],
                score=result['score'],
                has_image=result['has_image'],
                image_urls=result['image_urls']
            ))
        
        # Ensure we only have one chunk for citations
        if len(retrieved_chunks) > 1:
            # Keep only the chunk with highest score
            retrieved_chunks = [max(retrieved_chunks, key=lambda x: x.score)]
        
        context = "\n\n".join(context_parts)
        
        # Truncate context if too long
        max_context_length = 8000  # Adjust based on model limits
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            logger.warning(f"Context truncated to {max_context_length} characters")
        
        # Call LLM with conversation history
        logger.info("Calling LLM for answer generation")
        answer, token_in, token_out = call_llm(context, request.question, conversation_history)
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Log query to database
        db = SessionLocal()
        try:
            # Find user by telegram_id
            from app.db.models import User
            user = db.query(User).filter(User.telegram_id == str(request.user_id)).first()
            user_id = user.id if user else None
            
            query_log = QueryLog(
                user_id=user_id,
                question=request.question,
                answer=answer,  # Save LLM response for conversation history
                top_k=request.top_k,
                retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved_chunks],
                used_document_ids=list(set([chunk.doc_id for chunk in retrieved_chunks if chunk.doc_id and chunk.doc_id.strip()])),
                latency_ms=latency_ms,
                model=settings.ai.llm_model,
                token_in=token_in,
                token_out=token_out,
                conversation_id=conversation_id
            )
            
            db.add(query_log)
            db.commit()
            
            logger.info(f"Query logged with ID {query_log.id}")
            
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
            db.rollback()
        finally:
            db.close()
        
        # Generate citations from retrieved chunks (only the most relevant)
        citations = []
        seen_docs = set()
        
        # Only include citations for chunks that meet relevance threshold
        relevant_chunks = [chunk for chunk in retrieved_chunks if chunk.score >= 0.3]
        
        # Limit to only the most relevant chunk
        if relevant_chunks:
            relevant_chunks = [max(relevant_chunks, key=lambda x: x.score)]
        
        for chunk in relevant_chunks:
            # Skip chunks with empty doc_id
            if not chunk.doc_id or chunk.doc_id.strip() == '':
                continue
                
            if chunk.doc_id not in seen_docs:
                seen_docs.add(chunk.doc_id)
                # Get document public_url from database
                db_citation = SessionLocal()
                try:
                    from app.db.models import Document
                    doc = db_citation.query(Document).filter(Document.id == chunk.doc_id).first()
                    if doc:
                        citations.append(Citation(
                            filename=chunk.filename,
                            public_url=doc.public_url or f"doc_id:{chunk.doc_id}",
                            url=chunk.url or doc.url or f"{settings.app_base_url}/api/v1/docs/{chunk.doc_id}/download",
                            doc_id=chunk.doc_id,
                            snippet=chunk.text[:300] + '...' if len(chunk.text) > 300 else chunk.text,
                            score=chunk.score
                        ))
                finally:
                    db_citation.close()
        
        logger.info(f"Query processed successfully in {latency_ms}ms")
        
        return QueryResponse(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            citations=citations,
            model=settings.ai.llm_model,
            token_in=token_in,
            token_out=token_out,
            latency_ms=latency_ms,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        logger.error(f"Chat query failed: {e}")
        
        # Return safe error response instead of raising exception
        latency_ms = int((time.time() - start_time) * 1000)
        
        return QueryResponse(
            answer="⚠️ Ошибка при обработке вопроса. Попробуйте позже.",
            retrieved_chunks=[],
            citations=[],
            model=settings.ai.llm_model,
            token_in=0,
            token_out=0,
            latency_ms=latency_ms,
            conversation_id=request.conversation_id or str(uuid.uuid4())
        )


@router.get("/health/openai")
async def health_check_openai():
    """
    Health check for OpenAI API.
    
    Returns:
        Status of OpenAI API connection
    """
    try:
        from app.ingestion.embeddings import test_openai_connection
        result = test_openai_connection()
        return result
    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")
        return {
            "status": "error",
            "detail": f"Health check failed: {str(e)}"
        }