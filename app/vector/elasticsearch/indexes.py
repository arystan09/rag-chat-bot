"""Elasticsearch index management."""
from elasticsearch.exceptions import NotFoundError, RequestError
from loguru import logger

from app.core.settings import settings
from app.vector.elasticsearch.client import get_es


def ensure_index(name: str = "docs_chunks") -> bool:
    """
    Ensure Elasticsearch index exists with proper mapping.
    
    Args:
        name: Index name (default: "docs_chunks")
        
    Returns:
        bool: True if index exists or was created successfully, False otherwise
    """
    try:
        client = get_es()
        
        # Check if index exists
        if client.indices.exists(index=name):
            logger.info(f"Index '{name}' already exists")
            return True
        
        # Create index with mapping
        mapping = {
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "bm25_text": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "doc_id": {
                        "type": "keyword"
                    },
                    "filename": {
                        "type": "keyword"
                    },
                    "url": {
                        "type": "keyword"
                    },
                    "chunk_id": {
                        "type": "integer"
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": settings.ai.embedding_dimensions,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "has_image": {
                        "type": "boolean"
                    },
                    "image_urls": {
                        "type": "keyword"
                    },
                    "created_at": {
                        "type": "date"
                    },
                    "metadata": {
                        "type": "object",
                        "dynamic": True
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }
        
        # Create index
        client.indices.create(index=name, body=mapping)
        logger.info(f"Created index '{name}' with vector search enabled")
        
        # Verify index was created
        if client.indices.exists(index=name):
            logger.info(f"Successfully verified index '{name}' exists")
            return True
        else:
            logger.error(f"Failed to verify index '{name}' creation")
            return False
            
    except RequestError as e:
        logger.error(f"Failed to create index '{name}': {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating index '{name}': {e}")
        return False


def delete_index(name: str) -> bool:
    """
    Delete Elasticsearch index.
    
    Args:
        name: Index name
        
    Returns:
        bool: True if index was deleted successfully, False otherwise
    """
    try:
        client = get_es()
        
        if not client.indices.exists(index=name):
            logger.info(f"Index '{name}' does not exist")
            return True
        
        client.indices.delete(index=name)
        logger.info(f"Deleted index '{name}'")
        return True
        
    except NotFoundError:
        logger.info(f"Index '{name}' not found")
        return True
    except Exception as e:
        logger.error(f"Failed to delete index '{name}': {e}")
        return False


def get_index_info(name: str) -> dict:
    """
    Get information about an index.
    
    Args:
        name: Index name
        
    Returns:
        dict: Index information or empty dict if error
    """
    try:
        client = get_es()
        
        if not client.indices.exists(index=name):
            logger.warning(f"Index '{name}' does not exist")
            return {}
        
        info = client.indices.get(index=name)
        return info.get(name, {})
        
    except Exception as e:
        logger.error(f"Failed to get info for index '{name}': {e}")
        return {}


def list_indices() -> list:
    """
    List all indices.
    
    Returns:
        list: List of index names
    """
    try:
        client = get_es()
        indices = client.cat.indices(format="json")
        return [idx["index"] for idx in indices]
    except Exception as e:
        logger.error(f"Failed to list indices: {e}")
        return []


def index_document(index_name: str, doc_id: str, document: dict) -> bool:
    """
    Index a document in Elasticsearch.
    
    Args:
        index_name: Index name
        doc_id: Document ID
        document: Document data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        client = get_es()
        
        response = client.index(
            index=index_name,
            id=doc_id,
            body=document
        )
        
        if response["result"] in ["created", "updated"]:
            logger.debug(f"Document {doc_id} indexed successfully")
            return True
        else:
            logger.warning(f"Unexpected response for document {doc_id}: {response['result']}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to index document {doc_id}: {e}")
        return False


def search_documents(
    index_name: str,
    query: dict,
    size: int = 10,
    from_: int = 0
) -> dict:
    """
    Search documents in Elasticsearch.
    
    Args:
        index_name: Index name
        query: Search query
        size: Number of results to return
        from_: Starting offset
        
    Returns:
        dict: Search results
    """
    try:
        client = get_es()
        
        response = client.search(
            index=index_name,
            body=query,
            size=size,
            from_=from_
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"hits": {"hits": [], "total": {"value": 0}}}
