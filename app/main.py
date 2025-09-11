"""Main FastAPI application entrypoint."""
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.core.settings import settings
from app.api import health, files, chat
from app.db.init_db import check_db_connection
from app.vector.elasticsearch.client import check_es_connection
from app.vector.elasticsearch.indexes import ensure_index
from app.bot.bot import start_bot, stop_bot


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title=settings.app_name,
        description="RAG system with Telegram bot integration",
        version="1.0.0",
        debug=settings.debug
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(files.router, tags=["files"])
    app.include_router(files.docs_router, tags=["docs"])
    app.include_router(chat.router, tags=["chat"])
    
    # Mount static files for public access
    app.mount("/files", StaticFiles(directory=settings.storage.upload_dir), name="files")
    
    @app.on_event("startup")
    async def startup_event():
        """Application startup event."""
        logger.info(f"Starting {settings.app_name}")
        logger.info(f"Debug mode: {settings.debug}")
        logger.info(f"Database URL: {settings.database.url}")
        logger.info(f"Elasticsearch URL: {settings.elasticsearch.url}")
        
        # Check database connection
        if check_db_connection():
            logger.info("Database connection verified")
        else:
            logger.warning("Database connection failed - some features may not work")
        
        # Check Elasticsearch connection and create index
        if check_es_connection():
            logger.info("Elasticsearch connection verified")
            
            # Ensure docs_chunks index exists
            if ensure_index("docs_chunks"):
                logger.info("Elasticsearch index 'docs_chunks' is ready")
            else:
                logger.warning("Failed to create Elasticsearch index - vector search may not work")
        else:
            logger.warning("Elasticsearch connection failed - vector search may not work")
        
        # Start Telegram bot as background task
        if settings.telegram.bot_token:
            logger.info("Starting Telegram bot...")
            asyncio.create_task(start_bot())
        else:
            logger.warning("Telegram bot token not configured, skipping bot startup")
        
        logger.info("Application startup completed")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown event."""
        logger.info("Stopping Telegram bot...")
        await stop_bot()
        logger.info("Application shutdown")
    
    return app


# Create app instance
app = create_app()