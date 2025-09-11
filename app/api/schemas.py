"""Pydantic schemas for API requests and responses."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    """Document information for API responses."""
    
    id: str = Field(description="Document ID")
    filename: str = Field(description="Original filename")
    size_bytes: int = Field(description="File size in bytes")
    mime: str = Field(description="MIME type")
    created_at: datetime = Field(description="Creation timestamp")
    public_url: str = Field(description="Public URL for file access")
    
    class Config:
        from_attributes = True


class ChunkOut(BaseModel):
    """Chunk information for API responses."""
    
    chunk_id: int = Field(description="Chunk number")
    char_start: int = Field(description="Start character position")
    char_end: int = Field(description="End character position")
    has_image: bool = Field(description="Whether chunk contains images")
    image_urls: List[str] = Field(default=[], description="Image URLs")
    
    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    """Response for file upload."""
    
    doc_id: str = Field(description="Document ID")
    filename: str = Field(description="Original filename")
    chunks_count: int = Field(description="Number of chunks created")
    message: str = Field(default="Document uploaded successfully")


class DocumentListResponse(BaseModel):
    """Response for document list."""
    
    documents: List[DocumentOut] = Field(description="List of user documents")
    total: int = Field(description="Total number of documents")


class DeleteResponse(BaseModel):
    """Response for document deletion."""
    
    status: str = Field(default="deleted", description="Deletion status")
    doc_id: str = Field(description="Deleted document ID")
    message: str = Field(default="Document deleted successfully")


class QueryRequest(BaseModel):
    """Request for RAG chat query."""
    
    user_id: int = Field(description="User ID")
    question: str = Field(description="User question")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of chunks to retrieve")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")


class RetrievedChunk(BaseModel):
    """Information about retrieved chunk."""
    
    chunk_id: int = Field(description="Chunk ID")
    doc_id: str = Field(description="Document ID")
    filename: str = Field(description="Document filename")
    url: str = Field(default="", description="Download URL for document")
    text: str = Field(description="Chunk text")
    score: float = Field(description="Relevance score")
    has_image: bool = Field(description="Whether chunk has images")
    image_urls: List[str] = Field(default=[], description="Image URLs")


class Citation(BaseModel):
    """Citation information for sources."""
    
    filename: str = Field(description="Document filename")
    public_url: str = Field(description="Public URL for document access")
    url: str = Field(description="Download URL for document")
    doc_id: str = Field(description="Document ID")
    snippet: Optional[str] = Field(default=None, description="Text snippet from document")
    score: Optional[float] = Field(default=None, description="Relevance score")


class QueryResponse(BaseModel):
    """Response for RAG chat query."""
    
    answer: str = Field(description="AI-generated answer")
    retrieved_chunks: List[RetrievedChunk] = Field(description="Retrieved chunks used for answer")
    citations: List[Citation] = Field(description="Source citations with clickable links")
    model: str = Field(description="LLM model used")
    token_in: int = Field(description="Input tokens used")
    token_out: int = Field(description="Output tokens generated")
    latency_ms: int = Field(description="Response latency in milliseconds")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID")


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Additional error details")
    status_code: int = Field(description="HTTP status code")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(default="ok", description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    version: str = Field(default="1.0.0", description="API version")


