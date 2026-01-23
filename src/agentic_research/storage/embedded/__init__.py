"""
Embedded storage implementations for desktop/standalone deployment.

These implementations replace Docker-based services (MinIO, Qdrant, Redis, RabbitMQ)
with embedded alternatives that work without external dependencies.

Usage:
    from agentic_research.storage.embedded import get_embedded_storage

    storage = get_embedded_storage()
    storage.upload_pdf(paper_id, pdf_bytes)
"""

from .storage_client import EmbeddedStorageClient
from .vector_db import EmbeddedVectorDB
from .cache_store import EmbeddedCacheStore
from .message_queue import LocalMessageQueue
from .config import EmbeddedConfig, get_embedded_config

__all__ = [
    "EmbeddedStorageClient",
    "EmbeddedVectorDB",
    "EmbeddedCacheStore",
    "LocalMessageQueue",
    "EmbeddedConfig",
    "get_embedded_config",
]
