"""Document indexing service."""
import os
import tempfile
from typing import Tuple, Optional
from loguru import logger
from sqlalchemy.orm import Session

from app.core.settings import settings
from app.db.session import SessionLocal
from app.db.models import User, Document, Chunk
from app.ingestion.ingest import ingest_document
from app.ingestion.embeddings import embed_texts
from app.storage.service import save_original_file, save_image, get_file_info
from app.vector.elasticsearch.indexes import index_document as es_index_document


def index_document(file_content: bytes, filename: str, user_id: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Index document with full ingestion pipeline.
    
    Args:
        file_content: File content as bytes
        filename: Original filename
        user_id: ID of the user uploading the file
        
    Returns:
        Tuple of (doc_id, filename) or (None, None) if failed
    """
    db = SessionLocal()
    try:
        # Save original file
        storage_path, public_url = save_original_file(file_content, filename)
        if not storage_path:
            logger.error("Failed to save original file")
            return None, None
        
        # Get file information
        sha256, size_bytes, mime_type = get_file_info(storage_path)
        
        # Check if document already exists (by SHA256)
        existing_doc = db.query(Document).filter(Document.sha256 == sha256).first()
        if existing_doc:
            logger.warning(f"Document with SHA256 {sha256} already exists")
            return str(existing_doc.id), existing_doc.filename
        
        # Create document record in database
        document = Document(
            owner_user_id=user_id,
            filename=filename,
            mime=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            storage_path=storage_path,
            public_url=public_url
        )
        
        db.add(document)
        db.flush()  # Get the ID without committing
        
        doc_id = str(document.id)
        
        # Generate and update download URL
        download_url = f"{settings.app_base_url}/api/v1/docs/{doc_id}/download"
        document.url = download_url
        logger.info(f"Created document record: {doc_id}")
        logger.info(f"Generated download URL: {download_url}")
        
        # Ingest document
        chunks = ingest_document(storage_path, doc_id)
        if not chunks:
            logger.error("Failed to ingest document")
            db.rollback()
            return None, None
        
        # Process chunks
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings for all chunks
        embeddings = embed_texts(chunk_texts)
        if len(embeddings) != len(chunks):
            logger.error(f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)}")
            db.rollback()
            return None, None
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = chunk["chunk_id"]
            text = chunk["text"]
            embedding = embeddings[i]
            is_ocr = chunk.get("is_ocr", False)
            
            # Get image URLs (already processed in ingest for OCR chunks)
            image_urls = chunk.get("image_urls", [])
            
            # For non-OCR chunks, process images if any
            if not is_ocr and chunk.get("has_image") and "_image_data" in chunk:
                image_data = chunk["_image_data"]
                image_url = save_image(image_data, doc_id, chunk_id, 0)
                if image_url:
                    image_urls.append(image_url)
            
            # Create chunk record in database
            chunk_record = Chunk(
                document_id=document.id,
                chunk_id=chunk_id,
                char_start=0,  # Could be calculated more precisely
                char_end=len(text),
                has_image=len(image_urls) > 0,
                image_urls=image_urls if image_urls else None
            )
            
            db.add(chunk_record)
            
            # Index in Elasticsearch
            es_doc = {
                "text": text,
                "bm25_text": text,  # Same as text for now
                "doc_id": doc_id,
                "filename": filename,
                "url": download_url,  # Use the new download URL
                "chunk_id": chunk_id,
                "embedding": embedding,
                "has_image": len(image_urls) > 0,
                "image_urls": image_urls,
                "is_ocr": is_ocr,  # Flag for OCR chunks
                "created_at": document.created_at.isoformat()
            }
            
            es_doc_id = f"{doc_id}_{chunk_id}"
            success = es_index_document("docs_chunks", es_doc_id, es_doc)
            if not success:
                logger.warning(f"Failed to index chunk {es_doc_id} in Elasticsearch")
        
        # Commit all changes
        db.commit()
        
        logger.info(f"Successfully indexed document {doc_id} with {len(chunks)} chunks")
        return doc_id, filename
        
    except Exception as e:
        logger.error(f"Failed to index document: {e}")
        db.rollback()
        return None, None
    finally:
        db.close()


def delete_document(doc_id: str, user_id: int) -> bool:
    """
    Delete document and all its chunks.
    
    Args:
        doc_id: Document ID
        user_id: User ID (for authorization)
        
    Returns:
        True if successful, False otherwise
    """
    db = SessionLocal()
    try:
        # Find document
        document = db.query(Document).filter(
            Document.id == doc_id,
            Document.owner_user_id == user_id
        ).first()
        
        if not document:
            logger.warning(f"Document {doc_id} not found or not owned by user {user_id}")
            return False
        
        # Delete from Elasticsearch
        from app.vector.elasticsearch.client import get_es
        es_client = get_es()
        
        # Delete all chunks for this document
        query = {
            "query": {
                "term": {
                    "doc_id": doc_id
                }
            }
        }
        
        es_client.delete_by_query(index="docs_chunks", body=query)
        
        # Delete chunks from database (cascade will handle this)
        # Delete document from database
        db.delete(document)
        
        # Delete original file
        if document.storage_path and os.path.exists(document.storage_path):
            os.remove(document.storage_path)
        
        db.commit()
        
        logger.info(f"Successfully deleted document {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def get_user_documents(user_id: int) -> list:
    """
    Get list of documents for a user.
    
    Args:
        user_id: User ID
        
    Returns:
        List of document dictionaries
    """
    db = SessionLocal()
    try:
        documents = db.query(Document).filter(
            Document.owner_user_id == user_id
        ).order_by(Document.created_at.desc()).all()
        
        result = []
        for doc in documents:
            result.append({
                "id": doc.id,
                "filename": doc.filename,
                "size_bytes": doc.size_bytes,
                "mime": doc.mime,
                "created_at": doc.created_at.isoformat(),
                "public_url": doc.public_url,
                "url": doc.url
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get documents for user {user_id}: {e}")
        return []
    finally:
        db.close()


def get_document_chunks(doc_id: str) -> list:
    """
    Get all chunks for a document.
    
    Args:
        doc_id: Document ID
        
    Returns:
        List of chunk dictionaries
    """
    db = SessionLocal()
    try:
        chunks = db.query(Chunk).filter(
            Chunk.document_id == doc_id
        ).order_by(Chunk.chunk_id).all()
        
        result = []
        for chunk in chunks:
            result.append({
                "id": chunk.id,
                "chunk_id": chunk.chunk_id,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "has_image": chunk.has_image,
                "image_urls": chunk.image_urls or [],
                "created_at": chunk.created_at.isoformat()
            })
        
        return result

    except Exception as e:
        logger.error(f"Failed to get chunks for document {doc_id}: {e}")
        return []
    finally:
        db.close()


