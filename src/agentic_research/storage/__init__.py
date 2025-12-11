"""
Storage module for agentic deep research.

This module provides clients for various storage backends:
- MinIO: Object storage for PDFs
- Qdrant: Vector database for embeddings (multi-collection support)
- Embeddings: OpenAI embedding generation

Collection types for context engineering:
- RESEARCH_PAPERS: Main KB paper chunks
- PAPER_SUMMARIES: One summary embedding per paper for fast retrieval
- LINKED_PAPERS: Fully indexed cited papers (1-hop)
"""

from agentic_research.storage.minio_client import MinIOClient
from agentic_research.storage.qdrant_client import QdrantClientWrapper, CollectionType
from agentic_research.storage.embeddings import EmbeddingsGenerator

__all__ = ["MinIOClient", "QdrantClientWrapper", "CollectionType", "EmbeddingsGenerator"]
