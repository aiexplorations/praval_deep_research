"""
Storage module for agentic deep research.

This module provides clients for various storage backends:
- MinIO: Object storage for PDFs
- Qdrant: Vector database for embeddings
- Embeddings: OpenAI embedding generation
"""

from agentic_research.storage.minio_client import MinIOClient
from agentic_research.storage.qdrant_client import QdrantClientWrapper
from agentic_research.storage.embeddings import EmbeddingsGenerator

__all__ = ["MinIOClient", "QdrantClientWrapper", "EmbeddingsGenerator"]
