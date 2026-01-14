"""
BM25 Index Manager for Vajra-based full-text search.

This module provides base index management using Vajra BM25 search engine.
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import structlog

from vajra_bm25 import (
    Document,
    DocumentCorpus,
    VajraSearchOptimized,
    SearchResult,
)

from agentic_research.core.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class IndexedDocument:
    """Represents a document to be indexed."""

    id: str
    title: str
    content: str
    metadata: Dict[str, Any]


@dataclass
class SearchHit:
    """Represents a search result with metadata."""

    document_id: str
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any]


class BM25IndexManager(ABC):
    """
    Base class for BM25 index management using Vajra.

    Provides common functionality for creating, searching, and persisting
    BM25 indexes. Subclasses implement index-specific logic.
    """

    def __init__(
        self,
        index_name: str,
        index_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize the BM25 index manager.

        Args:
            index_name: Name identifier for this index
            index_path: Base path for storing indexes (defaults to config)
        """
        self.settings = get_settings()
        self.index_name = index_name

        # Set up index path
        base_path = index_path or Path(self.settings.BM25_INDEX_PATH)
        self.index_path = base_path / index_name
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Index file paths
        self._index_file = self.index_path / "index.joblib"
        self._corpus_file = self.index_path / "corpus.jsonl"
        self._metadata_file = self.index_path / "metadata.json"

        # Engine and corpus
        self._engine: Optional[VajraSearchOptimized] = None
        self._corpus: Optional[DocumentCorpus] = None
        self._documents: Dict[str, IndexedDocument] = {}

        # Track if index needs rebuild
        self._dirty = False
        self._doc_count_at_last_build = 0

        logger.info(
            "BM25IndexManager initialized",
            index_name=index_name,
            index_path=str(self.index_path),
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the index is currently loaded."""
        return self._engine is not None

    @property
    def document_count(self) -> int:
        """Get the number of documents in the index."""
        return len(self._documents)

    def load_index(self) -> bool:
        """
        Load an existing index from disk.

        Returns:
            True if index was loaded successfully, False otherwise
        """
        if not self._index_file.exists():
            logger.info("No existing index found", index_name=self.index_name)
            return False

        try:
            # Load corpus first
            if self._corpus_file.exists():
                self._corpus = DocumentCorpus.load_jsonl(self._corpus_file)
                # Rebuild documents dict from corpus
                for doc in self._corpus.documents:
                    self._documents[doc.id] = IndexedDocument(
                        id=doc.id,
                        title=doc.title,
                        content=doc.content,
                        metadata=doc.metadata or {},
                    )

            # Load index
            self._engine = VajraSearchOptimized.load_index(
                self._index_file, self._corpus
            )
            self._doc_count_at_last_build = len(self._documents)
            self._dirty = False

            logger.info(
                "Index loaded successfully",
                index_name=self.index_name,
                document_count=self.document_count,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to load index",
                index_name=self.index_name,
                error=str(e),
            )
            return False

    def save_index(self) -> bool:
        """
        Save the current index to disk.

        Returns:
            True if index was saved successfully, False otherwise
        """
        if self._engine is None or self._corpus is None:
            logger.warning(
                "Cannot save: no index built",
                index_name=self.index_name,
            )
            return False

        try:
            # Save corpus
            self._corpus.save_jsonl(self._corpus_file)

            # Save index
            self._engine.save_index(self._index_file)

            # Save metadata
            import json

            metadata = {
                "index_name": self.index_name,
                "document_count": self.document_count,
                "last_updated": datetime.utcnow().isoformat(),
                "bm25_k1": self.settings.BM25_K1,
                "bm25_b": self.settings.BM25_B,
            }
            with open(self._metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                "Index saved successfully",
                index_name=self.index_name,
                document_count=self.document_count,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to save index",
                index_name=self.index_name,
                error=str(e),
            )
            return False

    def add_document(self, doc: IndexedDocument) -> None:
        """
        Add a document to the index (marks index as dirty).

        Args:
            doc: Document to add
        """
        self._documents[doc.id] = doc
        self._dirty = True
        logger.debug(
            "Document added to index",
            index_name=self.index_name,
            doc_id=doc.id,
        )

    def add_documents(self, docs: List[IndexedDocument]) -> None:
        """
        Add multiple documents to the index.

        Args:
            docs: List of documents to add
        """
        for doc in docs:
            self._documents[doc.id] = doc
        self._dirty = True
        logger.debug(
            "Documents added to index",
            index_name=self.index_name,
            count=len(docs),
        )

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the index.

        Args:
            doc_id: ID of document to remove

        Returns:
            True if document was found and removed
        """
        if doc_id in self._documents:
            del self._documents[doc_id]
            self._dirty = True
            return True
        return False

    def rebuild_index(self) -> bool:
        """
        Rebuild the index from current documents.

        Returns:
            True if rebuild was successful
        """
        if not self._documents:
            logger.warning(
                "Cannot rebuild: no documents",
                index_name=self.index_name,
            )
            return False

        try:
            # Convert IndexedDocuments to Vajra Documents
            vajra_docs = [
                Document(
                    id=doc.id,
                    title=doc.title,
                    content=doc.content,
                    metadata=doc.metadata,
                )
                for doc in self._documents.values()
            ]

            # Create corpus
            self._corpus = DocumentCorpus(vajra_docs)

            # Build optimized search engine
            self._engine = VajraSearchOptimized(
                self._corpus,
                k1=self.settings.BM25_K1,
                b=self.settings.BM25_B,
                cache_size=self.settings.BM25_CACHE_SIZE,
                use_sparse=self.settings.BM25_USE_SPARSE,
                use_eager=self.settings.BM25_USE_EAGER,
            )

            self._dirty = False
            self._doc_count_at_last_build = len(self._documents)

            logger.info(
                "Index rebuilt successfully",
                index_name=self.index_name,
                document_count=self.document_count,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to rebuild index",
                index_name=self.index_name,
                error=str(e),
            )
            return False

    async def rebuild_index_async(self) -> bool:
        """
        Rebuild the index asynchronously.

        Returns:
            True if rebuild was successful
        """
        return await asyncio.to_thread(self.rebuild_index)

    def ensure_index_current(self, rebuild_threshold: int = 10) -> None:
        """
        Ensure index is current, rebuilding if necessary.

        Args:
            rebuild_threshold: Number of new docs before auto-rebuild
        """
        new_docs = len(self._documents) - self._doc_count_at_last_build

        if self._engine is None:
            # Try to load existing index
            if not self.load_index():
                # No existing index, rebuild if we have documents
                if self._documents:
                    self.rebuild_index()
        elif self._dirty and new_docs >= rebuild_threshold:
            # Auto-rebuild if we've added enough documents
            self.rebuild_index()

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        """
        Search the index.

        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of SearchHit results
        """
        self.ensure_index_current()

        if self._engine is None:
            logger.warning(
                "Search called with no index",
                index_name=self.index_name,
            )
            return []

        try:
            results: List[SearchResult] = self._engine.search(query, top_k=top_k)

            hits = []
            for result in results:
                # Get the original IndexedDocument for metadata
                doc_id = result.document.id
                original_doc = self._documents.get(doc_id)
                metadata = original_doc.metadata if original_doc else {}

                # Apply filters if provided
                if filters and not self._matches_filters(metadata, filters):
                    continue

                hits.append(
                    SearchHit(
                        document_id=doc_id,
                        content=result.document.content,
                        score=result.score,
                        rank=result.rank,
                        metadata=metadata,
                    )
                )

            return hits

        except Exception as e:
            logger.error(
                "Search failed",
                index_name=self.index_name,
                query=query,
                error=str(e),
            )
            return []

    def _matches_filters(
        self, metadata: Dict[str, Any], filters: Dict[str, Any]
    ) -> bool:
        """
        Check if metadata matches all filters.

        Args:
            metadata: Document metadata
            filters: Filter criteria

        Returns:
            True if all filters match
        """
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    @abstractmethod
    def get_index_type(self) -> str:
        """Return the type of this index (for logging/identification)."""
        pass


# Singleton registry for index managers
_index_managers: Dict[str, BM25IndexManager] = {}


def get_index_manager(index_name: str) -> Optional[BM25IndexManager]:
    """
    Get a registered index manager by name.

    Args:
        index_name: Name of the index manager

    Returns:
        The index manager if found, None otherwise
    """
    return _index_managers.get(index_name)


def register_index_manager(manager: BM25IndexManager) -> None:
    """
    Register an index manager globally.

    Args:
        manager: The index manager to register
    """
    _index_managers[manager.index_name] = manager
    logger.info(
        "Index manager registered",
        index_name=manager.index_name,
        index_type=manager.get_index_type(),
    )
