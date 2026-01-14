"""
Paper Index for BM25 search over research paper content.

This module provides full-text search over indexed paper chunks
and summaries using Vajra BM25 search engine.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import structlog

from agentic_research.storage.bm25_index_manager import (
    BM25IndexManager,
    IndexedDocument,
    SearchHit,
    register_index_manager,
)

logger = structlog.get_logger(__name__)


class PaperIndex(BM25IndexManager):
    """
    BM25 index for research paper content.

    Provides full-text search over paper chunks with metadata
    like paper_id, title, authors, and categories.
    """

    INDEX_NAME = "paper_chunks"

    def __init__(self, index_path: Optional[Path] = None) -> None:
        """
        Initialize the paper index.

        Args:
            index_path: Base path for storing indexes (defaults to config)
        """
        super().__init__(index_name=self.INDEX_NAME, index_path=index_path)

    def get_index_type(self) -> str:
        """Return the type of this index."""
        return "paper_chunks"

    def index_paper_chunk(
        self,
        chunk_id: str,
        paper_id: str,
        title: str,
        content: str,
        chunk_index: int,
        total_chunks: int,
        authors: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        abstract: Optional[str] = None,
        published_date: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Index a paper chunk.

        Args:
            chunk_id: Unique identifier for this chunk
            paper_id: ArXiv ID or other paper identifier
            title: Paper title
            content: Chunk text content
            chunk_index: Position of this chunk in the paper
            total_chunks: Total number of chunks in the paper
            authors: List of author names
            categories: List of ArXiv categories
            abstract: Paper abstract
            published_date: Publication date
            metadata: Additional metadata
        """
        if not content or not content.strip():
            logger.debug("Skipping empty chunk", chunk_id=chunk_id)
            return

        doc_metadata = {
            "paper_id": paper_id,
            "title": title,  # Include title in metadata for search results
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "authors": authors or [],
            "categories": categories or [],
            "abstract": abstract or "",
            "published_date": published_date or "",
            "indexed_at": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }

        doc = IndexedDocument(
            id=chunk_id,
            title=title,
            content=content,
            metadata=doc_metadata,
        )

        self.add_document(doc)

        logger.debug(
            "Paper chunk indexed",
            chunk_id=chunk_id,
            paper_id=paper_id,
            chunk_index=chunk_index,
        )

    def index_paper(
        self,
        paper_id: str,
        title: str,
        chunks: List[str],
        authors: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        abstract: Optional[str] = None,
        published_date: Optional[str] = None,
    ) -> int:
        """
        Index all chunks of a paper at once.

        Args:
            paper_id: ArXiv ID or other paper identifier
            title: Paper title
            chunks: List of chunk texts
            authors: List of author names
            categories: List of ArXiv categories
            abstract: Paper abstract
            published_date: Publication date

        Returns:
            Number of chunks indexed
        """
        indexed_count = 0
        total_chunks = len(chunks)

        for i, chunk_text in enumerate(chunks):
            if not chunk_text or not chunk_text.strip():
                continue

            chunk_id = f"{paper_id}_chunk_{i}"
            self.index_paper_chunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                title=title,
                content=chunk_text,
                chunk_index=i,
                total_chunks=total_chunks,
                authors=authors,
                categories=categories,
                abstract=abstract,
                published_date=published_date,
            )
            indexed_count += 1

        if indexed_count > 0:
            logger.info(
                "Paper indexed",
                paper_id=paper_id,
                chunks_indexed=indexed_count,
                total_chunks=total_chunks,
            )

        return indexed_count

    def search_papers(
        self,
        query: str,
        top_k: int = 10,
        paper_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
    ) -> List[SearchHit]:
        """
        Search paper content.

        Args:
            query: Search query string
            top_k: Number of results to return
            paper_id: Filter by specific paper ID
            categories: Filter by ArXiv categories

        Returns:
            List of SearchHit results
        """
        # Build filters
        filters: Dict[str, Any] = {}
        if paper_id:
            filters["paper_id"] = paper_id

        # Get more results for filtering
        search_limit = top_k * 3 if filters or categories else top_k
        results = self.search(query, top_k=search_limit, filters=filters)

        # Filter by categories if specified
        if categories:
            filtered_results = []
            for hit in results:
                hit_categories = hit.metadata.get("categories", [])
                if any(cat in hit_categories for cat in categories):
                    filtered_results.append(hit)
            results = filtered_results

        return results[:top_k]

    def get_paper_chunks(self, paper_id: str) -> List[SearchHit]:
        """
        Get all chunks for a specific paper.

        Args:
            paper_id: Paper ID to get chunks for

        Returns:
            List of chunks sorted by chunk_index
        """
        chunks = []
        for doc in self._documents.values():
            if doc.metadata.get("paper_id") == paper_id:
                chunks.append(
                    SearchHit(
                        document_id=doc.id,
                        content=doc.content,
                        score=1.0,
                        rank=0,
                        metadata=doc.metadata,
                    )
                )

        # Sort by chunk index
        chunks.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        return chunks

    def get_indexed_papers(self) -> List[Dict[str, Any]]:
        """
        Get list of all indexed papers with metadata.

        Returns:
            List of paper metadata dicts
        """
        papers: Dict[str, Dict[str, Any]] = {}

        for doc in self._documents.values():
            paper_id = doc.metadata.get("paper_id")
            if paper_id and paper_id not in papers:
                papers[paper_id] = {
                    "paper_id": paper_id,
                    "title": doc.title,
                    "authors": doc.metadata.get("authors", []),
                    "categories": doc.metadata.get("categories", []),
                    "abstract": doc.metadata.get("abstract", ""),
                    "published_date": doc.metadata.get("published_date", ""),
                    "chunk_count": doc.metadata.get("total_chunks", 0),
                }

        return list(papers.values())

    def get_paper_count(self) -> int:
        """Get the number of unique papers indexed."""
        paper_ids = set()
        for doc in self._documents.values():
            paper_id = doc.metadata.get("paper_id")
            if paper_id:
                paper_ids.add(paper_id)
        return len(paper_ids)


# Global singleton instance
_paper_index: Optional[PaperIndex] = None


def get_paper_index() -> PaperIndex:
    """
    Get the global paper index singleton.

    Returns:
        The PaperIndex instance
    """
    global _paper_index

    if _paper_index is None:
        _paper_index = PaperIndex()
        register_index_manager(_paper_index)

        # Try to load existing index
        _paper_index.load_index()

    return _paper_index
