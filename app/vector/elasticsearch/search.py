"""Hybrid search combining dense vector (kNN) and BM25 search."""
from typing import List, Dict, Any, Tuple
from loguru import logger

from app.core.settings import settings
from app.vector.elasticsearch.client import get_es

# -----------------------------
# Configurable constants
# -----------------------------
KNN_WEIGHT_DEFAULT: float = 0.7
BM25_WEIGHT_DEFAULT: float = 0.3
BM25_WEIGHT_ADAPTIVE: float = 0.5
BM25_STRONG_SCORE_THRESHOLD: float = 0.7  # normalized
BM25_STRONG_COUNT_THRESHOLD: int = 5      # number of distinct documents
MIN_SCORE_THRESHOLD: float = 0.15         # filtered out if below (after final score)
BOOST_FOR_DOMAIN_TERMS: float = 2.0
BOOST_FOR_OTHER_TERMS: float = 1.0


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


def _expand_query_terms(question: str) -> List[Tuple[str, float]]:
    """
    Generate simple query expansion terms with per-term boost.
    Heuristics only (no LLM): synonyms, morphological variants, key domain terms.
    Returns a list of (term, boost) pairs.
    """
    q = (question or "").lower()
    terms: Dict[str, float] = {}

    def add_term(term: str, boost: float) -> None:
        t = term.strip()
        if not t:
            return
        # Keep the max boost if duplicate
        current = terms.get(t, 0.0)
        if boost > current:
            terms[t] = boost

    # base tokens
    for token in q.replace(',', ' ').replace('.', ' ').split():
        token = token.strip()
        if not token:
            continue
        add_term(token, BOOST_FOR_OTHER_TERMS)
        # naive singular/plural
        if token.endswith('ы') or token.endswith('и'):
            add_term(token[:-1], BOOST_FOR_OTHER_TERMS)
        if token.endswith('а'):
            add_term(token[:-1], BOOST_FOR_OTHER_TERMS)

    # domain expansions (boosted)
    if 'список' in q or 'перечень' in q:
        for t in ['перечень', 'список', 'пункты', 'требования', 'документы', 'список документов', 'перечень документов']:
            add_term(t, BOOST_FOR_DOMAIN_TERMS)
    if 'документ' in q or 'документы' in q:
        for t in ['документ', 'документы', 'список документов', 'перечень документов']:
            add_term(t, BOOST_FOR_DOMAIN_TERMS)
    if 'страхов' in q or 'мед' in q:
        for t in ['медстраховка', 'медицинская страховка', 'страхование', 'страховой полис']:
            add_term(t, BOOST_FOR_DOMAIN_TERMS)
    if 'виза' in q:
        for t in ['виза', 'национальная виза', 'учебная виза']:
            add_term(t, BOOST_FOR_DOMAIN_TERMS)

    return list(terms.items())


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
        expanded = _expand_query_terms(question)
        if expanded:
            should_clauses = [
                {
                    "match": {
                        "text": {
                            "query": term,
                            "boost": boost
                        }
                    }
                } for term, boost in expanded
            ]
        else:
            should_clauses = [
                {
                    "match": {
                        "text": {
                            "query": question,
                            "boost": BOOST_FOR_OTHER_TERMS
                        }
                    }
                }
            ]

        query = {
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
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
    
    # Determine adaptive weights based on BM25 strength across distinct documents
    strong_bm25_doc_ids = set()
    for r in bm25_results:
        if r['bm25_score'] >= BM25_STRONG_SCORE_THRESHOLD and r.get('doc_id'):
            strong_bm25_doc_ids.add(r['doc_id'])
    if len(strong_bm25_doc_ids) >= BM25_STRONG_COUNT_THRESHOLD:
        knn_weight = max(0.0, 1.0 - BM25_WEIGHT_ADAPTIVE)
        bm25_weight = BM25_WEIGHT_ADAPTIVE
    else:
        knn_weight = KNN_WEIGHT_DEFAULT
        bm25_weight = BM25_WEIGHT_DEFAULT

    # Create a dictionary to store combined results
    combined_dict: Dict[str, Dict[str, Any]] = {}
    
    # Add kNN results
    for result in knn_results:
        key = f"{result['doc_id']}_{result['chunk_id']}"
        combined_dict[key] = result.copy()
    
    # Add BM25 results and preserve both scores
    for result in bm25_results:
        key = f"{result['doc_id']}_{result['chunk_id']}"
        if key in combined_dict:
            combined_dict[key]['bm25_score'] = result['bm25_score']
        else:
            combined_dict[key] = result.copy()
    
    # Calculate final scores with weights
    for result in combined_dict.values():
        final_score = (knn_weight * result['knn_score'] + bm25_weight * result['bm25_score'])
        result['final_score'] = final_score
        logger.debug(
            f"Doc {result['doc_id']}: kNN={result['knn_score']:.3f}, BM25={result['bm25_score']:.3f}, Final={final_score:.3f}")
    
    # Filter out weak results
    filtered = [r for r in combined_dict.values() if r['final_score'] >= MIN_SCORE_THRESHOLD]
    
    # Sort by final score and return top k
    sorted_results = sorted(filtered, key=lambda x: x['final_score'], reverse=True)

    # Log top-3 merged results at INFO
    log_preview = []
    for r in sorted_results[:3]:
        log_preview.append(
            {
                'doc_id': r.get('doc_id', ''),
                'filename': r.get('filename', ''),
                'final_score': round(r.get('final_score', 0.0), 3)
            }
        )
    if log_preview:
        logger.info(f"Top merged results: {log_preview}")
    
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