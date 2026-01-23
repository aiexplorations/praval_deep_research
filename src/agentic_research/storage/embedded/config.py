"""
Embedded Configuration - Unified configuration for desktop/standalone deployment.

This module provides configuration and initialization for all embedded services,
replacing Docker-based infrastructure with local alternatives.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedConfig:
    """
    Configuration for embedded/desktop deployment.

    All paths are relative to the data_dir, which defaults to ./data
    but can be customized (e.g., ~/Library/Application Support/Praval on macOS).
    """

    # Base data directory
    data_dir: Path = field(default_factory=lambda: Path("./data"))

    # Storage configuration
    storage_bucket: str = "research-papers"

    # Vector database configuration
    vector_size: int = 1536
    vector_backend: str = "auto"  # "auto", "lancedb", "faiss", "numpy"

    # Cache configuration
    cache_backend: str = "auto"  # "auto", "diskcache", "dict"
    cache_ttl: Optional[int] = None  # Default TTL in seconds

    # Database configuration
    database_url: Optional[str] = None  # If None, uses SQLite

    # Embedding configuration
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    openai_api_key: Optional[str] = None

    # LLM configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    # Optional external services (for hybrid mode)
    use_external_qdrant: bool = False
    qdrant_url: Optional[str] = None

    use_external_postgres: bool = False
    postgres_url: Optional[str] = None

    # Feature flags
    enable_knowledge_graph: bool = False
    enable_bm25_search: bool = True

    # Logging
    log_level: str = "INFO"

    def __post_init__(self):
        """Ensure data directory exists and set derived paths."""
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Set database URL if not provided
        if not self.database_url:
            self.database_url = f"sqlite:///{self.data_dir}/praval.db"

        # Load API key from environment if not provided
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")

    @property
    def storage_path(self) -> Path:
        return self.data_dir / "storage"

    @property
    def vectors_path(self) -> Path:
        return self.data_dir / "vectors"

    @property
    def cache_path(self) -> Path:
        return self.data_dir / "cache"

    @property
    def indexes_path(self) -> Path:
        return self.data_dir / "vajra_indexes"

    @classmethod
    def from_env(cls) -> "EmbeddedConfig":
        """Create configuration from environment variables."""
        return cls(
            data_dir=Path(os.environ.get("PRAVAL_DATA_DIR", "./data")),
            storage_bucket=os.environ.get("PRAVAL_STORAGE_BUCKET", "research-papers"),
            vector_size=int(os.environ.get("EMBEDDING_DIMENSIONS", "1536")),
            embedding_model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            llm_provider=os.environ.get("PRAVAL_DEFAULT_PROVIDER", "openai"),
            llm_model=os.environ.get("PRAVAL_DEFAULT_MODEL", "gpt-4o-mini"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )

    @classmethod
    def for_desktop(cls, app_name: str = "Praval") -> "EmbeddedConfig":
        """
        Create configuration suitable for desktop app.

        Uses platform-specific data directories:
        - macOS: ~/Library/Application Support/Praval
        - Windows: %APPDATA%/Praval
        - Linux: ~/.local/share/praval
        """
        import platform

        system = platform.system()

        if system == "Darwin":  # macOS
            base = Path.home() / "Library" / "Application Support" / app_name
        elif system == "Windows":
            base = Path(os.environ.get("APPDATA", "")) / app_name
        else:  # Linux
            base = Path.home() / ".local" / "share" / app_name.lower()

        return cls(data_dir=base)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data_dir": str(self.data_dir),
            "storage_bucket": self.storage_bucket,
            "vector_size": self.vector_size,
            "vector_backend": self.vector_backend,
            "cache_backend": self.cache_backend,
            "database_url": self.database_url,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "enable_knowledge_graph": self.enable_knowledge_graph,
            "enable_bm25_search": self.enable_bm25_search,
        }


# Global config instance
_config: Optional[EmbeddedConfig] = None


def get_embedded_config() -> EmbeddedConfig:
    """Get or create the global embedded configuration."""
    global _config
    if _config is None:
        _config = EmbeddedConfig.from_env()
    return _config


def set_embedded_config(config: EmbeddedConfig):
    """Set the global embedded configuration."""
    global _config
    _config = config


# =========================================================================
# Service Initialization
# =========================================================================

class EmbeddedServices:
    """
    Centralized initialization and management of all embedded services.

    Usage:
        services = EmbeddedServices(config)
        await services.initialize()

        # Access services
        services.storage.upload_pdf(...)
        services.vectors.add_vectors(...)
        services.cache.set(...)
        services.queue.publish(...)

        # Cleanup
        await services.shutdown()
    """

    def __init__(self, config: Optional[EmbeddedConfig] = None):
        self.config = config or get_embedded_config()

        # Service instances (initialized lazily)
        self._storage = None
        self._vectors = None
        self._cache = None
        self._queue = None
        self._db_engine = None

        self._initialized = False

    async def initialize(self):
        """Initialize all embedded services."""
        if self._initialized:
            return

        logger.info("Initializing embedded services...")

        # Initialize storage
        from .storage_client import EmbeddedStorageClient
        self._storage = EmbeddedStorageClient(
            base_path=str(self.config.storage_path),
            bucket_name=self.config.storage_bucket
        )

        # Initialize vector database
        from .vector_db import EmbeddedVectorDB
        self._vectors = EmbeddedVectorDB(
            base_path=str(self.config.vectors_path),
            vector_size=self.config.vector_size,
            backend=self.config.vector_backend
        )

        # Initialize cache
        from .cache_store import EmbeddedCacheStore
        self._cache = EmbeddedCacheStore(
            base_path=str(self.config.cache_path),
            use_disk=(self.config.cache_backend != "dict"),
            default_ttl=self.config.cache_ttl
        )

        # Initialize message queue
        from .message_queue import LocalMessageQueue
        self._queue = LocalMessageQueue()
        await self._queue.start()

        # Initialize database (SQLite)
        await self._init_database()

        self._initialized = True
        logger.info("Embedded services initialized successfully")

    async def _init_database(self):
        """Initialize SQLite database with SQLAlchemy."""
        try:
            from sqlalchemy.ext.asyncio import create_async_engine

            # Convert sqlite:// to sqlite+aiosqlite://
            db_url = self.config.database_url
            if db_url.startswith("sqlite://"):
                db_url = db_url.replace("sqlite://", "sqlite+aiosqlite://")

            self._db_engine = create_async_engine(
                db_url,
                echo=False,
                future=True
            )

            # Import and create tables
            # Note: This assumes the models are defined elsewhere
            # In full implementation, you'd run migrations here

            logger.info(f"Database initialized: {self.config.database_url}")

        except ImportError:
            logger.warning("SQLAlchemy/aiosqlite not installed, database features disabled")
            self._db_engine = None

    async def shutdown(self):
        """Shutdown all services gracefully."""
        logger.info("Shutting down embedded services...")

        if self._queue:
            await self._queue.stop()

        if self._vectors:
            self._vectors.close()

        if self._cache:
            self._cache.close()

        if self._db_engine:
            await self._db_engine.dispose()

        self._initialized = False
        logger.info("Embedded services shut down")

    # =========================================================================
    # Service Accessors
    # =========================================================================

    @property
    def storage(self):
        """Get storage client."""
        if not self._storage:
            raise RuntimeError("Services not initialized. Call initialize() first.")
        return self._storage

    @property
    def vectors(self):
        """Get vector database."""
        if not self._vectors:
            raise RuntimeError("Services not initialized. Call initialize() first.")
        return self._vectors

    @property
    def cache(self):
        """Get cache store."""
        if not self._cache:
            raise RuntimeError("Services not initialized. Call initialize() first.")
        return self._cache

    @property
    def queue(self):
        """Get message queue."""
        if not self._queue:
            raise RuntimeError("Services not initialized. Call initialize() first.")
        return self._queue

    @property
    def db_engine(self):
        """Get database engine."""
        return self._db_engine

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """Check health of all services."""
        return {
            "initialized": self._initialized,
            "storage": {
                "status": "ok" if self._storage else "not_initialized",
                "stats": self._storage.get_storage_stats() if self._storage else None
            },
            "vectors": {
                "status": "ok" if self._vectors else "not_initialized",
                "stats": self._vectors.get_stats() if self._vectors else None
            },
            "cache": {
                "status": "ok" if self._cache else "not_initialized",
                "stats": self._cache.get_stats() if self._cache else None
            },
            "queue": {
                "status": "ok" if self._queue and self._queue._running else "not_running",
                "stats": self._queue.get_stats() if self._queue else None
            },
            "database": {
                "status": "ok" if self._db_engine else "not_initialized"
            }
        }


# Global services instance
_services: Optional[EmbeddedServices] = None


async def get_embedded_services() -> EmbeddedServices:
    """Get or create the global embedded services instance."""
    global _services
    if _services is None:
        _services = EmbeddedServices()
        await _services.initialize()
    return _services


def get_embedded_services_sync() -> Optional[EmbeddedServices]:
    """Get services without initialization (returns None if not initialized)."""
    return _services
