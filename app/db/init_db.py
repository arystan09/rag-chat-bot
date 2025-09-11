"""Database initialization helper."""
from loguru import logger
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.settings import settings
from app.db.session import engine, Base
from app.db.models import User, Document, Chunk, QueryLog


def init_db() -> bool:
    """
    Initialize database by creating all tables.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Initializing database...")
        
        # Test database connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Verify tables were created
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]
            logger.info(f"Created tables: {', '.join(tables)}")
        
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"Database initialization failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during database initialization: {e}")
        return False


def check_db_connection() -> bool:
    """
    Check if database connection is working.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection check successful")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Database connection check failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during database connection check: {e}")
        return False


def drop_all_tables() -> bool:
    """
    Drop all tables (use with caution!).
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(bind=engine)
        logger.info("All tables dropped successfully")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Failed to drop tables: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while dropping tables: {e}")
        return False


