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


def call_llm(context_gists: List[str], full_context: str, question: str, conversation_history: List[dict] = None) -> tuple[str, int, int]:
    """
    Call LLM with dual context (gists + full chunks) and conversation history.
    
    Args:
        context_gists: Short gists extracted from top chunks
        full_context: Concatenated best chunks text
        question: User question
        conversation_history: Previous conversation exchanges
        
    Returns:
        Tuple of (answer_text, token_in, token_out)
    """
    try:
        from openai import OpenAI
        import json as _json
        
        if not settings.ai.openai_api_key:
            logger.warning("OpenAI API key not configured")
            return "⚠️ OpenAI временно недоступен. Попробуйте позже.", 0, 0
        
        logger.info(f"Calling LLM with model: {settings.ai.llm_model}")
        logger.info(f"LLM input: gists={len(context_gists)}, full_chunks_chars={len(full_context)}")
        logger.debug(f"Question: {question}")
        logger.debug(f"Conversation history entries: {len(conversation_history or [])}")
        
        client = OpenAI(api_key=settings.ai.openai_api_key)
        
        # Build messages with conversation history and strict JSON instruction
        messages = [
            {
                "role": "system",
                "content": (
                    "Вы — умный, вежливый и лаконичный ассистент-помощник для внутренних документов компании. "
                    "Отвечайте по-русски. Всегда: "
                    "- Кратко: максимум 3–5 предложений в основном ответе (если не просят развернуть). "
                    "- Сначала давайте однострочную подсказку о темах: \"Можно: … | Нельзя: …\" (макс 1 предложение). "
                    "- Если для корректного ответа не хватает информации — задайте 1 короткий уточняющий вопрос (макс 12 слов). "
                    "- Форматируйте источники так: `📄 {filename} — {url}` (plain URL, без HTML). "
                    "- Не используйте лишние символы/звёздочки/повторяющиеся переносы строк. "
                    "- Возвращайте строго JSON-объект по схеме. "
                    "Правила ответов: Если вопрос про документы (виза/поступление и т.п.) — перечисли ВСЕ найденные пункты нумерованно (1., 2., 3., …), не сокращай. "
                    "Если вопрос про медстраховку — если в тексте прямо не сказано, ответь: \"В документах это не указано\" и предложи уточнить у консульства. "
                    "Не придумывай фактов. Если данных нет — скажи \"Нет данных\"."
                ),
            }
        ]
        
        # Add conversation history (last 3), before full context
        if conversation_history:
            messages.append({"role": "system", "content": "ПРЕДЫДУЩИЙ ДИАЛОГ:"})
            for exchange in conversation_history[-3:]:  # Last 3 exchanges
                messages.append({"role": "user", "content": exchange["question"]})
                messages.append({"role": "assistant", "content": exchange["answer"]})
        
        # Prepare dual context
        gists_block = "\n".join([f"- {g}" for g in context_gists]) if context_gists else "- Нет данных"
        full_block = full_context if full_context else ""
        
        # Build user prompt with strict output schema
        prompt = (
            "[GISTS]\n"
            f"{gists_block}\n\n"
            "[FULL CONTEXT]\n"
            f"{full_block}\n\n"
            "[QUESTION]\n"
            f"{question}\n\n"
            "[OUTPUT SCHEMA]\n"
            "Верни строго JSON без пояснений:\n"
            "{\n"
            "  \"one_line_topics\": \"Можно: ... | Нельзя: ...\",\n"
            "  \"answer\": \"...\",\n"
            "  \"steps\": [\"1. ...\", \"2. ...\"],\n"
            "  \"citations\": [{\"filename\":\"...\", \"url\":\"...\", \"snippet\":\"...\"}],\n"
            "  \"clarifying_question\": null,\n"
            "  \"confidence\": \"high|medium|low\"\n"
            "}"
        )
        messages.append({"role": "user", "content": prompt})

        logger.info(f"Prompt length (chars): {len(prompt)}")
        
        # Send request
        response = client.chat.completions.create(
            model=settings.ai.llm_model,
            messages=messages,
            max_tokens=settings.ai.max_tokens,
            temperature=0.9,
            top_p=0.9,
            frequency_penalty=0.2,
        )
        
        raw = response.choices[0].message.content or ""
        token_in = response.usage.prompt_tokens
        token_out = response.usage.completion_tokens
        
        logger.info(f"LLM response received: {len(raw)} characters, tokens: {token_in} in, {token_out} out")

        # Robust JSON parsing with cleanup and fallback
        fallback_json = {
            "one_line_topics": "Нет данных",
            "answer": "Нет данных",
            "steps": [],
            "citations": [],
            "clarifying_question": None,
            "confidence": "low"
        }
        parsed = None
        try:
            parsed = _json.loads(raw)
        except Exception:
            try:
                start = raw.find('{')
                end = raw.rfind('}')
                if start != -1 and end != -1 and end > start:
                    cleaned = raw[start:end+1]
                    parsed = _json.loads(cleaned)
            except Exception:
                parsed = None
        if not isinstance(parsed, dict):
            parsed = fallback_json
        # Ensure minimal fields
        for key in fallback_json.keys():
            if key not in parsed:
                parsed[key] = fallback_json[key]
        final_answer = parsed.get("answer") or fallback_json["answer"]
        if not final_answer or not str(final_answer).strip():
            final_answer = fallback_json["answer"]
        
        return final_answer, token_in, token_out
        
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
        # Retrieve more candidates to improve recall for list-style questions
        search_results = hybrid_search(request.question, max(12, min(request.top_k * 4, 30)))
        
        if not search_results:
            logger.warning("No search results found")
            return QueryResponse(
                answer="В документах этого нет.",
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
        
        # Sort by combined score from search (already normalized)
        search_results_sorted = sorted(search_results, key=lambda x: x['score'], reverse=True)
        
        # Dynamically select top chunks
        ql = request.question.lower()
        want_full_list = any(kw in ql for kw in ["список", "перечень", "весь список", "все пункты", "полный список", "все документы"])
        top_keep = 12 if want_full_list else 6
        
        # Apply a gentle minimum score threshold but fallback to top_keep if none pass
        MIN_SCORE = 0.1
        candidates = [r for r in search_results_sorted if r.get('score', 0.0) >= MIN_SCORE]
        if not candidates:
            candidates = search_results_sorted
        top_results = candidates[:top_keep]
        
        # Log top-3 chunk ids/files/scores
        top3_preview = [
            {
                'doc_id': r.get('doc_id', ''),
                'filename': r.get('filename', ''),
                'score': round(r.get('score', 0.0), 3)
            } for r in top_results[:3]
        ]
        logger.info(f"Selected top {len(top_results)} results; preview: {top3_preview}")
        
        # If no relevant results found, return appropriate response
        if not top_results:
            logger.warning("No relevant results found for query")
            return QueryResponse(
                answer="В документах этого нет.",
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
        
        # Create gists (<=25 words, first sentence heuristic) and keep 3–5
        def _make_gist(text: str) -> str:
            sentence = text.split('. ')[0].strip()
            words = sentence.split()
            if len(words) > 25:
                return " ".join(words[:25]) + "..."
            return sentence
        max_gists = 5 if want_full_list else 3
        context_gists = [_make_gist(r['text']) for r in top_results][:max_gists]
        
        # Assemble full context with budget at chunk granularity (avoid cutting mid-chunk)
        context_parts_joined = []
        total_len = 0
        max_context_length = 15000 if want_full_list else 7000
        for part in context_parts:
            if total_len + len(part) <= max_context_length:
                context_parts_joined.append(part)
                total_len += len(part)
            else:
                break
        full_context = "\n\n".join(context_parts_joined)
        
        logger.info(f"Prepared context for LLM: gists={len(context_gists)}, full_chunks={len(context_parts_joined)}, full_len={len(full_context)}")
        
        # Call LLM with conversation history
        logger.info("Calling LLM for answer generation")
        answer, token_in, token_out = call_llm(context_gists, full_context, request.question, conversation_history)
        if not answer or not answer.strip():
            answer = "Нет данных"
        
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
        
        # Generate citations from retrieved chunks (single highest-scoring source only)
        citations = []
        
        # Use only the single highest-scoring chunk for citation
        if retrieved_chunks:
            best_chunk = max(retrieved_chunks, key=lambda x: x.score)
            # Skip chunks with empty doc_id
            if best_chunk.doc_id and best_chunk.doc_id.strip():
                # Get document public_url from database
                db_citation = SessionLocal()
                try:
                    from app.db.models import Document
                    doc = db_citation.query(Document).filter(Document.id == best_chunk.doc_id).first()
                    if doc:
                        citations.append(Citation(
                            filename=best_chunk.filename,
                            public_url=doc.public_url or f"doc_id:{best_chunk.doc_id}",
                            url=best_chunk.url or doc.url or f"{settings.app_base_url}/api/v1/docs/{best_chunk.doc_id}/download",
                            doc_id=best_chunk.doc_id,
                            snippet=best_chunk.text[:300] + '...' if len(best_chunk.text) > 300 else best_chunk.text,
                            score=best_chunk.score
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