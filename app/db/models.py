"""SQLAlchemy database models."""
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.session import Base


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_id = Column(String(50), unique=True, nullable=False, index=True)
    role = Column(String(32), nullable=False, default='user')
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    query_logs = relationship("QueryLog", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, telegram_id='{self.telegram_id}', role='{self.role}')>"


class Document(Base):
    """Document model for storing file metadata."""
    
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    owner_user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    filename = Column(String(255), nullable=False)
    mime = Column(String(100), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    sha256 = Column(String(64), nullable=False, unique=True, index=True)
    storage_path = Column(String(500), nullable=False)
    public_url = Column(String(500), nullable=True)
    url = Column(String(500), nullable=True)  # New API download URL
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    owner = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', owner_id={self.owner_user_id})>"


class Chunk(Base):
    """Chunk model for storing document fragments."""
    
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(
        Integer, 
        ForeignKey("documents.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    chunk_id = Column(Integer, nullable=False)  # Sequential chunk number within document
    char_start = Column(Integer, nullable=False)
    char_end = Column(Integer, nullable=False)
    has_image = Column(Boolean, default=False, nullable=False)
    image_urls = Column(JSON, nullable=True)  # List of image URLs in this chunk
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('document_id', 'chunk_id', name='uq_document_chunk'),
    )
    
    def __repr__(self):
        return f"<Chunk(id={self.id}, document_id={self.document_id}, chunk_id={self.chunk_id})>"


class QueryLog(Base):
    """Query log model for tracking user interactions."""
    
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True, 
        index=True
    )
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)  # LLM response for conversation history
    top_k = Column(Integer, nullable=False)
    retrieved_chunk_ids = Column(JSON, nullable=True)  # List of chunk IDs used
    used_document_ids = Column(JSON, nullable=True)    # List of document IDs used
    latency_ms = Column(Integer, nullable=False)
    model = Column(String(100), nullable=False)
    token_in = Column(Integer, nullable=True)
    token_out = Column(Integer, nullable=True)
    conversation_id = Column(String(36), nullable=True, index=True)  # UUID for conversation grouping
    parent_id = Column(Integer, ForeignKey("query_logs.id", ondelete="SET NULL"), nullable=True)  # For threading
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="query_logs")
    
    def __repr__(self):
        return f"<QueryLog(id={self.id}, user_id={self.user_id}, question='{self.question[:50]}...')>"



