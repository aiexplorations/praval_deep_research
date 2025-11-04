"""
Configuration management for the agentic research system.

This module provides centralized configuration management with
environment variable support and validation.
"""

import os
import re
from typing import Dict, Any, Optional
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
import structlog
from urllib.parse import urlparse


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Core Application
    APP_NAME: str = Field(default="praval_deep_research", env="APP_NAME")
    APP_VERSION: str = Field(default="1.0.0", env="APP_VERSION")
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # Praval Framework Configuration
    PRAVAL_DEFAULT_PROVIDER: str = Field(default="openai", env="PRAVAL_DEFAULT_PROVIDER")
    PRAVAL_DEFAULT_MODEL: str = Field(default="gpt-4o-mini", env="PRAVAL_DEFAULT_MODEL")
    PRAVAL_MAX_THREADS: int = Field(default=10, env="PRAVAL_MAX_THREADS")
    PRAVAL_MEMORY_ENABLED: bool = Field(default=True, env="PRAVAL_MEMORY_ENABLED")
    PRAVAL_REEF_CAPACITY: int = Field(default=1000, env="PRAVAL_REEF_CAPACITY")
    
    # External APIs
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Infrastructure Services
    RABBITMQ_URL: str = Field(
        default="amqp://research_user:research_pass@localhost:5672/research_vhost",
        env="RABBITMQ_URL"
    )
    RABBITMQ_MANAGEMENT_URL: str = Field(
        default="http://localhost:15672",
        env="RABBITMQ_MANAGEMENT_URL"
    )
    RABBITMQ_PASSWORD: str = Field(default="research_pass", env="RABBITMQ_PASSWORD")
    QDRANT_URL: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = Field(default="research_vectors", env="QDRANT_COLLECTION_NAME")
    
    MINIO_ENDPOINT: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    MINIO_EXTERNAL_ENDPOINT: str = Field(default="localhost:9000", env="MINIO_EXTERNAL_ENDPOINT")
    MINIO_ACCESS_KEY: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    MINIO_BUCKET_NAME: str = Field(default="research-papers", env="MINIO_BUCKET_NAME")
    MINIO_SECURE: bool = Field(default=False, env="MINIO_SECURE")
    
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # ArXiv Configuration
    ARXIV_BASE_URL: str = Field(
        default="http://export.arxiv.org/api/query",
        env="ARXIV_BASE_URL"
    )
    ARXIV_MAX_RESULTS: int = Field(default=50, env="ARXIV_MAX_RESULTS")
    ARXIV_RATE_LIMIT: int = Field(default=3, env="ARXIV_RATE_LIMIT")
    ARXIV_DELAY_SECONDS: float = Field(default=1.0, env="ARXIV_DELAY_SECONDS")

    # Embedding Configuration
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    EMBEDDING_DIMENSIONS: int = Field(default=1536, env="EMBEDDING_DIMENSIONS")
    EMBEDDING_BATCH_SIZE: int = Field(default=100, env="EMBEDDING_BATCH_SIZE")

    # PDF Processing Configuration
    CHUNK_SIZE: int = Field(default=1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=200, env="CHUNK_OVERLAP")
    MAX_CHUNKS_PER_PAPER: int = Field(default=50, env="MAX_CHUNKS_PER_PAPER")
    PDF_EXTRACTION_METHOD: str = Field(default="pdfplumber", env="PDF_EXTRACTION_METHOD")
    PDF_DOWNLOAD_TIMEOUT: int = Field(default=60, env="PDF_DOWNLOAD_TIMEOUT")
    PDF_MAX_RETRIES: int = Field(default=3, env="PDF_MAX_RETRIES")

    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        env="API_CORS_ORIGINS"
    )
    API_ALLOWED_HOSTS: str = Field(
        default="localhost,127.0.0.1,0.0.0.0",
        env="API_ALLOWED_HOSTS"
    )
    
    # Monitoring
    PROMETHEUS_PORT: int = Field(default=9090, env="PROMETHEUS_PORT")
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    
    model_config = {"env_file": ".env", "case_sensitive": True}
    
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is supported."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {', '.join(valid_levels)}")
        return v.upper()
    
    @field_validator("PRAVAL_DEFAULT_PROVIDER")
    @classmethod
    def validate_praval_provider(cls, v: str) -> str:
        """Validate Praval provider is supported."""
        valid_providers = ["openai", "anthropic"]
        if v.lower() not in valid_providers:
            raise ValueError(f"PRAVAL_DEFAULT_PROVIDER must be one of: {', '.join(valid_providers)}")
        return v.lower()
    
    def validate_api_keys(self) -> None:
        """Validate that required API keys are present."""
        if self.PRAVAL_DEFAULT_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
        
        if self.PRAVAL_DEFAULT_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required when using Anthropic provider")
    
    def get_praval_config(self) -> Dict[str, Any]:
        """Get Praval framework configuration."""
        return {
            "default_provider": self.PRAVAL_DEFAULT_PROVIDER,
            "default_model": self.PRAVAL_DEFAULT_MODEL,
            "max_threads": self.PRAVAL_MAX_THREADS,
            "temperature": 0.3,
            "max_tokens": 2000,
            "reef_config": {
                "channel_capacity": self.PRAVAL_REEF_CAPACITY,
                "message_ttl": 3600,
                "enable_persistence": True
            },
            "memory_config": {
                "collection_name": "research_memory",
                "embedding_model": "text-embedding-ada-002",
                "chunk_size": 1000,
                "overlap": 200,
                "similarity_threshold": 0.7
            }
        }
    
    def get_cors_origins(self) -> list[str]:
        """Parse CORS origins into a list."""
        return [origin.strip() for origin in self.API_CORS_ORIGINS.split(",")]
    
    @property
    def CORS_ORIGINS(self) -> list[str]:
        """Get CORS origins as a list."""
        return self.get_cors_origins()
    
    @property
    def ALLOWED_HOSTS(self) -> list[str]:
        """Get allowed hosts as a list."""
        return [host.strip() for host in self.API_ALLOWED_HOSTS.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def setup_logging(log_level: str = "INFO", structured: bool = True) -> structlog.BoundLogger:
    """
    Setup structured logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        structured: Whether to use structured (JSON) logging
        
    Returns:
        Configured logger instance
    """
    import logging
    
    # Map string levels to logging constants
    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    
    if structured:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            level_mapping.get(log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()


def validate_rabbitmq_url(url: str) -> bool:
    """
    Validate RabbitMQ URL format.
    
    Args:
        url: RabbitMQ URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("amqp", "amqps") and bool(parsed.hostname)
    except Exception:
        return False


def validate_api_endpoints(config: Dict[str, str]) -> bool:
    """
    Validate external API endpoint configurations.
    
    Args:
        config: Dictionary of endpoint configurations
        
    Returns:
        True if all endpoints are valid, False otherwise
    """
    try:
        for endpoint_name, endpoint_url in config.items():
            if not endpoint_url:
                return False
                
            if endpoint_name == "MINIO_ENDPOINT":
                # MinIO endpoint can be just host:port
                if not re.match(r"^[\w\.-]+:\d+$", endpoint_url):
                    return False
            else:
                # Other endpoints should be proper URLs
                parsed = urlparse(endpoint_url)
                if not parsed.scheme or not parsed.hostname:
                    return False
        
        return True
    except Exception:
        return False


# Global settings instance
settings = get_settings()