"""Application settings and configuration."""
import json
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="rag_db", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="password", description="Database password")
    
    @property
    def url(self) -> str:
        """Get database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    class Config:
        env_prefix = "DATABASE_"
        extra = "ignore"


class ElasticsearchSettings(BaseSettings):
    """Elasticsearch configuration."""
    
    host: str = Field(default="localhost", description="Elasticsearch host")
    port: int = Field(default=9200, description="Elasticsearch port")
    username: Optional[str] = Field(default=None, description="Elasticsearch username")
    password: Optional[str] = Field(default=None, description="Elasticsearch password")
    index_name: str = Field(default="docs_chunks", description="Elasticsearch index name")
    
    @property
    def url(self) -> str:
        """Get Elasticsearch URL."""
        return f"http://{self.host}:{self.port}"
    
    class Config:
        env_prefix = "ELASTICSEARCH_"
        extra = "ignore"


class AISettings(BaseSettings):
    """AI and LLM configuration."""
    
    openai_api_key: str = Field(default="", description="OpenAI API key")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model name")
    llm_model: str = Field(default="gpt-3.5-turbo", description="LLM model name")
    embedding_dimensions: int = Field(default=1536, description="Embedding dimensions")
    max_tokens: int = Field(default=2000, description="Maximum tokens for LLM response")
    search_top_k: int = Field(default=10, description="Number of candidates for hybrid search")
    
    class Config:
        env_prefix = "AI_"
        extra = "ignore"


class TelegramSettings(BaseSettings):
    """Telegram bot configuration."""
    
    bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN", description="Telegram bot token")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for bot")
    webhook_secret: Optional[str] = Field(default=None, description="Webhook secret")
    allowed_user_ids: List[int] = Field(default_factory=list, description="Allowed Telegram user IDs")
    admin_telegram_ids: List[str] = Field(default_factory=list, description="Admin Telegram IDs (comma-separated)")
    max_images_per_response: int = Field(default=4, description="Maximum images per bot response")
    max_images_per_chunk: int = Field(default=2, description="Maximum images per chunk")
    max_message_length: int = Field(default=4000, description="Max Telegram message length")
    max_citations: int = Field(default=3, description="Maximum citations to show")
    dynamic_roles: bool = Field(default=True, description="Auto-update roles from admin_telegram_ids")
    
    @field_validator('admin_telegram_ids', mode='before')
    @classmethod
    def parse_admin_ids(cls, v):
        """Parse admin IDs from JSON string or list."""
        if isinstance(v, str):
            try:
                # Try to parse as JSON
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
                else:
                    return [str(parsed)]
            except (json.JSONDecodeError, TypeError):
                # If not JSON, treat as comma-separated string
                if v.strip():
                    return [item.strip() for item in v.split(',')]
                else:
                    return []
        elif isinstance(v, list):
            return [str(item) for item in v]
        return []
    
    class Config:
        env_prefix = "TELEGRAM_"
        extra = "ignore"


class StorageSettings(BaseSettings):
    """File storage configuration."""
    
    upload_dir: str = Field(default="./uploads", description="Upload directory")
    max_file_size: int = Field(default=250 * 1024 * 1024, description="Max file size in bytes (250MB)")
    base_url: str = Field(default="http://localhost:8000", description="Base URL for public file access")
    allowed_extensions: List[str] = Field(
        default=["pdf", "txt", "doc", "docx", "md", "png", "jpg", "jpeg"],
        description="Allowed file extensions"
    )
    
    class Config:
        env_prefix = "STORAGE_"
        extra = "ignore"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application
    app_name: str = Field(default="RAG Telegram Bot", description="Application name")
    app_base_url: str = Field(default="http://localhost:8000", description="Base URL for the application")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")
    
    # OpenAI API key (root level for compatibility)
    openai_api_key: str = Field(default="", description="OpenAI API key")
    
    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    elasticsearch: ElasticsearchSettings = Field(default_factory=ElasticsearchSettings)
    ai: AISettings = Field(default_factory=AISettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()