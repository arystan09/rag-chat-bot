"""Hybrid search combining dense vector (kNN) and BM25 search."""
from typing import List, Dict, Any, Tuple
from loguru import logger

from app.core.settings import settings
from app.vector.elasticsearch.client import get_es


def hybrid_search(question: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining kNN (semantic) and BM25 (keyword) search.
    
    Args:
        question: Search query
        k: Number of results to return
        
    Returns:
        List of documents with metadata including doc_id, filename, snippet, score
    """
    logger.info(f"Starting hybrid search for query: '{question}' (k={k})")
    
    try:
        # Generate embedding for the question
        from app.ingestion.embeddings import embed_text
        question_embedding = embed_text(question)
        
        if not question_embedding:
            logger.error("Failed to generate embedding for question")
            return bm25_search_fallback(question, k)
        
        logger.info(f"Generated embedding with {len(question_embedding)} dimensions")
        
        # Perform both searches
        knn_results = perform_knn_search(question_embedding, k * 2)  # Get more candidates
        bm25_results = perform_bm25_search(question, k * 2)
        
        logger.info(f"kNN found {len(knn_results)} results, BM25 found {len(bm25_results)} results")
        
        # Combine and score results
        combined_results = combine_search_results(knn_results, bm25_results, k)
        
        logger.info(f"Hybrid search returning {len(combined_results)} results")
        
        # Format results for API
        formatted_results = format_search_results(combined_results)
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return bm25_search_fallback(question, k)


def perform_knn_search(query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
    """
    Perform kNN search using dense vectors.
    
    Args:
        query_embedding: Query vector
        k: Number of results to return
        
    Returns:
        List of search results with kNN scores
    """
    try:
        query = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": k,
                "num_candidates": k * 5  # Search more candidates for better results
            },
            "size": k,
            "_source": ["text", "doc_id", "filename", "url", "chunk_id", "image_urls", "has_image"]
        }
        
        es = get_es()
        response = es.search(
            index=settings.elasticsearch.index_name,
            body=query
        )
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            result = {
                'chunk_id': source.get('chunk_id', 0),
                'doc_id': source.get('doc_id', ''),
                'filename': source.get('filename', ''),
                'url': source.get('url', ''),
                'text': source['text'],
                'knn_score': hit['_score'],
                'bm25_score': 0.0,  # Will be updated during combination
                'has_image': source.get('has_image', False),
                'image_urls': source.get('image_urls', [])
            }
            results.append(result)
        
        logger.info(f"kNN search completed: {len(results)} results")
        return results
        
    except Exception as e:
        logger.warning(f"kNN search failed: {e}")
        return []


def perform_bm25_search(question: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Perform BM25 keyword search.
    
    Args:
        question: Search query
        k: Number of results to return
        
    Returns:
        List of search results with BM25 scores
    """
    try:
        query = {
            "query": {
                "match": {
                    "text": {
                        "query": question,
                        "boost": 1.0
                    }
                }
            },
            "size": k,
            "_source": ["text", "doc_id", "filename", "url", "chunk_id", "image_urls", "has_image"]
        }
        
        es = get_es()
        response = es.search(
            index=settings.elasticsearch.index_name,
            body=query
        )
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            result = {
                'chunk_id': source.get('chunk_id', 0),
                'doc_id': source.get('doc_id', ''),
                'filename': source.get('filename', ''),
                'url': source.get('url', ''),
                'text': source['text'],
                'knn_score': 0.0,  # Will be updated during combination
                'bm25_score': hit['_score'],
                'has_image': source.get('has_image', False),
                'image_urls': source.get('image_urls', [])
            }
            results.append(result)
        
        logger.info(f"BM25 search completed: {len(results)} results")
        return results
        
    except Exception as e:
        logger.warning(f"BM25 search failed: {e}")
        return []


def combine_search_results(knn_results: List[Dict[str, Any]], 
                          bm25_results: List[Dict[str, Any]], 
                          k: int) -> List[Dict[str, Any]]:
    """
    Combine kNN and BM25 results with weighted scoring.
    
    Args:
        knn_results: Results from kNN search
        bm25_results: Results from BM25 search
        k: Number of final results to return
        
    Returns:
        Combined and scored results
    """
    # Normalize scores to [0, 1] range
    knn_results = normalize_scores(knn_results, 'knn_score')
    bm25_results = normalize_scores(bm25_results, 'bm25_score')
    
    # Create a dictionary to store combined results
    combined_dict = {}
    
    # Add kNN results
    for result in knn_results:
        key = f"{result['doc_id']}_{result['chunk_id']}"
        combined_dict[key] = result.copy()
    
    # Add BM25 results and combine scores
    for result in bm25_results:
        key = f"{result['doc_id']}_{result['chunk_id']}"
        if key in combined_dict:
            # Update BM25 score for existing result
            combined_dict[key]['bm25_score'] = result['bm25_score']
        else:
            # Add new result
            combined_dict[key] = result.copy()
    
    # Calculate final scores with weights
    KNN_WEIGHT = 0.7
    BM25_WEIGHT = 0.3
    
    for result in combined_dict.values():
        final_score = (KNN_WEIGHT * result['knn_score'] + 
                      BM25_WEIGHT * result['bm25_score'])
        result['final_score'] = final_score
        
        logger.debug(f"Doc {result['doc_id']}: kNN={result['knn_score']:.3f}, "
                    f"BM25={result['bm25_score']:.3f}, Final={final_score:.3f}")
    
    # Sort by final score and return top k
    sorted_results = sorted(combined_dict.values(), 
                           key=lambda x: x['final_score'], 
                           reverse=True)
    
    return sorted_results[:k]


def normalize_scores(results: List[Dict[str, Any]], score_field: str) -> List[Dict[str, Any]]:
    """
    Normalize scores to [0, 1] range using min-max normalization.
    
    Args:
        results: List of search results
        score_field: Name of the score field to normalize
        
    Returns:
        Results with normalized scores
    """
    if not results:
        return results
    
    scores = [result[score_field] for result in results]
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        # All scores are the same, set to 1.0
        for result in results:
            result[score_field] = 1.0
    else:
        # Normalize to [0, 1]
        for result in results:
            result[score_field] = (result[score_field] - min_score) / (max_score - min_score)
    
    return results


def format_search_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format search results for API response.
    
    Args:
        results: Combined search results
        
    Returns:
        Formatted results with required fields
    """
    formatted = []
    for result in results:
        formatted_result = {
            'doc_id': result['doc_id'],
            'filename': result['filename'],
            'url': result.get('url', ''),  # Add URL field
            'snippet': result['text'][:300] + '...' if len(result['text']) > 300 else result['text'],
            'score': result['final_score'],
            'chunk_id': result['chunk_id'],
            'text': result['text'],  # Keep full text for context building
            'has_image': result['has_image'],
            'image_urls': result['image_urls'],
            'knn_score': result['knn_score'],
            'bm25_score': result['bm25_score']
        }
        formatted.append(formatted_result)
    
    return formatted


def bm25_search_fallback(question: str, k: int) -> List[Dict[str, Any]]:
    """
    Fallback to BM25 search when kNN is unavailable.
    
    Args:
        question: Search query
        k: Number of results to return
        
    Returns:
        BM25 search results formatted for API
    """
    logger.info(f"Using BM25 fallback for query: '{question}'")
    
    try:
        results = perform_bm25_search(question, k)
        
        # Format results for API
        formatted_results = []
        for result in results:
            formatted_result = {
                'doc_id': result['doc_id'],
                'filename': result['filename'],
                'snippet': result['text'][:300] + '...' if len(result['text']) > 300 else result['text'],
                'score': result['bm25_score'],
                'chunk_id': result['chunk_id'],
                'text': result['text'],
                'has_image': result['has_image'],
                'image_urls': result['image_urls'],
                'knn_score': 0.0,
                'bm25_score': result['bm25_score']
            }
            formatted_results.append(formatted_result)
        
        logger.info(f"BM25 fallback returning {len(formatted_results)} results")
        return formatted_results
        
    except Exception as e:
        logger.error(f"BM25 fallback failed: {e}")
        return []


def simple_search(question: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Legacy function for backward compatibility.
    Now delegates to hybrid_search.
    
    Args:
        question: Search query
        k: Number of results to return
        
    Returns:
        List of search results
    """
    logger.info(f"simple_search called, delegating to hybrid_search")
    return hybrid_search(question, k)