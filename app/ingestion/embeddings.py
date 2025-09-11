"""Embedding service for text vectorization."""
import random
from typing import List
from loguru import logger

from app.core.settings import settings


def embed_text(text: str) -> List[float]:
    """
    Generate embedding for text.
    
    Args:
        text: Input text to embed
        
    Returns:
        List of float values representing the embedding vector
    """
    try:
        logger.debug(f"Generating embedding for text: '{text[:100]}...' (length: {len(text)})")
        embeddings = embed_texts([text])
        result = embeddings[0] if embeddings else []
        logger.debug(f"Generated embedding with {len(result)} dimensions")
        return result
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return []


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.
    
    Args:
        texts: List of input texts to embed
        
    Returns:
        List of embedding vectors
    """
    try:
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Check if OpenAI model is configured
        if settings.ai.embedding_model.startswith("openai") or settings.ai.embedding_model.startswith("text-embedding"):
            return _embed_with_openai(texts)
        else:
            return _embed_with_huggingface(texts)
            
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return []


def _embed_with_openai(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using OpenAI API.
    
    Args:
        texts: List of input texts to embed
        
    Returns:
        List of embedding vectors
    """
    try:
        from openai import OpenAI
        
        if not settings.ai.openai_api_key:
            logger.warning("OpenAI API key not configured, falling back to stub embeddings")
            return _embed_with_stub(texts)
        
        logger.info(f"Using OpenAI API with model: {settings.ai.embedding_model}")
        logger.debug(f"API key configured: {bool(settings.ai.openai_api_key)}")
        
        client = OpenAI(api_key=settings.ai.openai_api_key)
        
        # Prepare texts for embedding
        # OpenAI has limits on batch size, so we process in chunks
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1} with {len(batch_texts)} texts")
            
            try:
                response = client.embeddings.create(
                    model=settings.ai.embedding_model,
                    input=batch_texts
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Successfully generated {len(batch_embeddings)} embeddings in batch {i//batch_size + 1}")
                
            except Exception as batch_error:
                logger.error(f"OpenAI API batch error: {batch_error}")
                logger.warning("Falling back to stub embeddings for this batch")
                # Generate stub embeddings for this batch
                batch_stubs = _embed_with_stub(batch_texts)
                all_embeddings.extend(batch_stubs)
        
        logger.info(f"Generated {len(all_embeddings)} total OpenAI embeddings")
        return all_embeddings
        
    except ImportError:
        logger.warning("OpenAI library not installed, falling back to stub embeddings")
        return _embed_with_stub(texts)
    except Exception as e:
        logger.error(f"OpenAI embedding failed: {e}, falling back to stub embeddings")
        return _embed_with_stub(texts)


def _embed_with_huggingface(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using HuggingFace SentenceTransformer.
    
    Args:
        texts: List of input texts to embed
        
    Returns:
        List of embedding vectors
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Using HuggingFace model: {settings.ai.embedding_model}")
        
        # Load model
        model = SentenceTransformer(settings.ai.embedding_model)
        
        # Generate embeddings
        embeddings = model.encode(texts, convert_to_tensor=False)
        
        # Convert to list of lists
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        
        logger.info(f"Generated {len(embeddings_list)} HuggingFace embeddings")
        return embeddings_list
        
    except ImportError:
        logger.warning("sentence-transformers library not installed, falling back to stub embeddings")
        return _embed_with_stub(texts)
    except Exception as e:
        logger.error(f"HuggingFace embedding failed: {e}, falling back to stub embeddings")
        return _embed_with_stub(texts)


def _embed_with_stub(texts: List[str]) -> List[List[float]]:
    """
    Generate stub embeddings for testing.
    
    Args:
        texts: List of input texts to embed
        
    Returns:
        List of stub embedding vectors
    """
    embeddings = []
    for text in texts:
        # Deterministic random vector based on text content
        random.seed(hash(text) % 2**32)
        embedding = [random.uniform(-1.0, 1.0) for _ in range(settings.ai.embedding_dimensions)]
        embeddings.append(embedding)
    
    logger.debug(f"Generated {len(embeddings)} stub embeddings")
    return embeddings


def get_embedding_dimension() -> int:
    """
    Get the dimension of embeddings.
    
    Returns:
        Embedding dimension
    """
    return settings.ai.embedding_dimensions


def normalize_embedding(embedding: List[float]) -> List[float]:
    """
    Normalize embedding vector to unit length.
    
    Args:
        embedding: Input embedding vector
        
    Returns:
        Normalized embedding vector
    """
    if not embedding:
        return []
    
    # Calculate L2 norm
    norm = sum(x * x for x in embedding) ** 0.5
    
    if norm == 0:
        return embedding
    
    # Normalize
    return [x / norm for x in embedding]


def test_openai_connection() -> dict:
    """
    Test OpenAI API connection.
    
    Returns:
        dict: Status and details of the test
    """
    try:
        from openai import OpenAI
        
        if not settings.ai.openai_api_key:
            return {
                "status": "error",
                "detail": "OpenAI API key not configured"
            }
        
        logger.info("Testing OpenAI API connection...")
        client = OpenAI(api_key=settings.ai.openai_api_key)
        
        # Test with a simple embedding
        test_text = "ping"
        response = client.embeddings.create(
            model=settings.ai.embedding_model,
            input=[test_text]
        )
        
        if response.data and len(response.data) > 0:
            logger.info("OpenAI API test successful")
            return {
                "status": "ok",
                "detail": f"Successfully generated embedding with {len(response.data[0].embedding)} dimensions"
            }
        else:
            return {
                "status": "error",
                "detail": "OpenAI API returned empty response"
            }
            
    except Exception as e:
        logger.error(f"OpenAI API test failed: {e}")
        return {
            "status": "error",
            "detail": str(e)
        }

