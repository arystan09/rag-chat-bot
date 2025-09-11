"""FastAPI router for file management endpoints."""
import time
from typing import List
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, status
from fastapi.responses import FileResponse
from loguru import logger
import os

from app.api.schemas import (
    UploadResponse, DocumentListResponse, DeleteResponse, 
    DocumentOut, ErrorResponse
)
from app.ingestion.indexer import index_document, delete_document, get_user_documents
from app.db.session import SessionLocal
from app.db.models import Document
from app.core.settings import settings


router = APIRouter(prefix="/api/v1/files", tags=["files"])


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user_id: int = Query(..., description="User ID")
):
    """
    Upload and index a document.
    
    Args:
        file: Uploaded file
        user_id: User ID
        
    Returns:
        Upload response with document ID and metadata
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )
        
        # Read file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        logger.info(f"Uploading file {file.filename} for user {user_id}")
        
        # Index document
        doc_id, filename = index_document(file_content, file.filename, user_id)
        
        if not doc_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to index document"
            )
        
        # Get chunks count (simplified - could be improved)
        # For now, we'll estimate based on file size
        estimated_chunks = max(1, len(file_content) // 1000)
        
        logger.info(f"Successfully uploaded document {doc_id} with {estimated_chunks} chunks")
        
        return UploadResponse(
            doc_id=doc_id,
            filename=filename,
            chunks_count=estimated_chunks,
            message="Document uploaded and indexed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    user_id: int = Query(..., description="User ID")
):
    """
    Get list of user's documents.
    
    Args:
        user_id: User ID
        
    Returns:
        List of user documents
    """
    try:
        logger.info(f"Listing documents for user {user_id}")
        
        documents = get_user_documents(user_id)
        
        # Convert to response format
        document_list = []
        for doc in documents:
            document_list.append(DocumentOut(
                id=doc["id"],
                filename=doc["filename"],
                size_bytes=doc["size_bytes"],
                mime=doc["mime"],
                created_at=doc["created_at"],
                public_url=doc["public_url"]
            ))
        
        logger.info(f"Found {len(document_list)} documents for user {user_id}")
        
        return DocumentListResponse(
            documents=document_list,
            total=len(document_list)
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document_endpoint(
    doc_id: str,
    user_id: int = Query(..., description="User ID")
):
    """
    Delete a document and all its chunks.
    
    Args:
        doc_id: Document ID
        user_id: User ID
        
    Returns:
        Deletion confirmation
    """
    try:
        logger.info(f"Deleting document {doc_id} for user {user_id}")
        
        success = delete_document(doc_id, user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or not owned by user"
            )
        
        logger.info(f"Successfully deleted document {doc_id}")
        
        return DeleteResponse(
            doc_id=doc_id,
            message="Document and all chunks deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


# New router for document downloads
docs_router = APIRouter(prefix="/api/v1/docs", tags=["docs"])


@docs_router.get("/{doc_id}/download")
async def download_document(doc_id: str):
    """
    Download a document by ID.
    
    Args:
        doc_id: Document ID
        
    Returns:
        File response with the document
    """
    try:
        logger.info(f"Download request for document {doc_id}")
        
        db = SessionLocal()
        try:
            # Get document from database
            document = db.query(Document).filter(Document.id == doc_id).first()
            
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )
            
            # Check if file exists
            file_path = document.storage_path
            if not os.path.exists(file_path):
                logger.error(f"File not found at path: {file_path}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File not found on server"
                )
            
            logger.info(f"Serving document {doc_id}: {document.filename}")
            
            # Return file response
            return FileResponse(
                path=file_path,
                filename=document.filename,
                media_type=document.mime
            )
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download document {doc_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Download failed: {str(e)}"
        )