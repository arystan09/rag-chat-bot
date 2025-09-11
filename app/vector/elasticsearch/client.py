"""Elasticsearch client configuration."""
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, AuthenticationException
from loguru import logger

from app.core.settings import settings


def get_es() -> Elasticsearch:
    """
    Get Elasticsearch client instance.
    
    Returns:
        Elasticsearch: Configured Elasticsearch client
        
    Raises:
        ConnectionError: If unable to connect to Elasticsearch
        AuthenticationException: If authentication fails
    """
    try:
        # Prepare connection parameters
        es_config = {
            "hosts": [settings.elasticsearch.url],
            "request_timeout": 30,
            "max_retries": 3,
            "retry_on_timeout": True,
            "verify_certs": False,
            "ssl_show_warn": False,
            "headers": {
                "Accept": "application/vnd.elasticsearch+json; compatible-with=8"
            }
        }
        
        # Add authentication if provided
        if settings.elasticsearch.username and settings.elasticsearch.password:
            es_config["basic_auth"] = (
                settings.elasticsearch.username,
                settings.elasticsearch.password
            )
        
        # Create client
        client = Elasticsearch(**es_config)
        
        # Test connection
        if client.ping():
            logger.info(f"Connected to Elasticsearch at {settings.elasticsearch.url}")
            return client
        else:
            raise ConnectionError("Failed to ping Elasticsearch server")
            
    except AuthenticationException as e:
        logger.error(f"Elasticsearch authentication failed: {e}")
        raise
    except ConnectionError as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error connecting to Elasticsearch: {e}")
        raise


def check_es_connection() -> bool:
    """
    Check if Elasticsearch connection is working.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        client = get_es()
        info = client.info()
        logger.info(f"Elasticsearch version: {info['version']['number']}")
        return True
    except Exception as e:
        logger.error(f"Elasticsearch connection check failed: {e}")
        return False


def get_es_client() -> Elasticsearch:
    """
    Get Elasticsearch client (alias for get_es for backward compatibility).
    
    Returns:
        Elasticsearch: Configured Elasticsearch client
    """
    return get_es()



