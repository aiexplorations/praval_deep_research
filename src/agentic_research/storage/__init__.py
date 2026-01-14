"""
Storage module for agentic deep research.

This module provides clients for various storage backends:
- MinIO: Object storage for PDFs
- Qdrant: Vector database for embeddings (multi-collection support)
- Embeddings: OpenAI embedding generation
- BM25: Full-text search using Vajra BM25 engine

Collection types for context engineering:
- RESEARCH_PAPERS: Main KB paper chunks
- PAPER_SUMMARIES: One summary embedding per paper for fast retrieval
- LINKED_PAPERS: Fully indexed cited papers (1-hop)
"""

from agentic_research.storage.minio_client import MinIOClient
from agentic_research.storage.qdrant_client import QdrantClientWrapper, CollectionType
from agentic_research.storage.embeddings import EmbeddingsGenerator
from agentic_research.storage.bm25_index_manager import (
    BM25IndexManager,
    IndexedDocument,
    SearchHit,
    get_index_manager,
    register_index_manager,
)
from agentic_research.storage.conversation_index import (
    ConversationIndex,
    get_conversation_index,
)
from agentic_research.storage.paper_index import (
    PaperIndex,
    get_paper_index,
)
from agentic_research.storage.hybrid_search import (
    HybridPaperSearch,
    HybridSearchResult,
    get_hybrid_search,
)

__all__ = [
    "MinIOClient",
    "QdrantClientWrapper",
    "CollectionType",
    "EmbeddingsGenerator",
    "BM25IndexManager",
    "IndexedDocument",
    "SearchHit",
    "get_index_manager",
    "register_index_manager",
    "ConversationIndex",
    "get_conversation_index",
    "PaperIndex",
    "get_paper_index",
    "HybridPaperSearch",
    "HybridSearchResult",
    "get_hybrid_search",
]
