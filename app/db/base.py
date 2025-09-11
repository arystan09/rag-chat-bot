"""Database base and models import for Alembic."""
from app.db.session import Base
from app.db.models import User, Document, Chunk, QueryLog

# Import all models to ensure they are registered with Base
__all__ = ["Base", "User", "Document", "Chunk", "QueryLog"]


