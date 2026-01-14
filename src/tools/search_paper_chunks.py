"""
Search Paper Chunks Tool.

This module provides a Praval tool for searching research paper content
using Vajra BM25 full-text search.

Single Responsibility: Search paper content by keywords.
Used by: qa_specialist, document_processor, research_advisor agents.
"""

from typing import Dict, List, Any, Optional
import structlog

from praval import tool

from agentic_research.storage.paper_index import get_paper_index

logger = structlog.get_logger(__name__)


@tool(
    tool_name="search_paper_chunks",
    description="Search research paper content using BM25 full-text search. "
    "Find relevant paper sections by keywords, technical terms, or concepts. "
    "Returns matching chunks with paper metadata.",
    category="search",
    shared=True,
    version="1.0.0",
    author="Research Team",
    tags=["search", "bm25", "papers", "research", "chunks"],
)
def search_paper_chunks(
    query: str,
    top_k: int = 10,
    paper_id: Optional[str] = None,
    categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search research paper content using BM25 full-text search.

    This tool searches across all indexed paper chunks to find relevant
    content. Useful for:
    - Finding papers discussing specific topics or methods
    - Locating technical details in research papers
    - Discovering related work on a subject

    Args:
        query: Search query - keywords, technical terms, or concepts (required)
        top_k: Number of results to return (default: 10)
        paper_id: Filter results to a specific paper by its ID (optional)
        categories: Filter by ArXiv categories like ["cs.AI", "cs.LG"] (optional)

    Returns:
        List of matching chunks with:
        - chunk_id: Unique identifier
        - content: Chunk text
        - score: BM25 relevance score
        - paper_id: Source paper ID
        - title: Paper title
        - authors: List of authors
        - categories: ArXiv categories
        - chunk_index: Position in paper
    """
    if not query or not query.strip():
        logger.warning("search_paper_chunks called with empty query")
        return []

    logger.info(
        "Searching paper chunks",
        query=query[:100],
        top_k=top_k,
        paper_id=paper_id,
        categories=categories,
    )

    try:
        index = get_paper_index()

        results = index.search_papers(
            query=query,
            top_k=top_k,
            paper_id=paper_id,
            categories=categories,
        )

        output = []
        for hit in results:
            output.append({
                "chunk_id": hit.document_id,
                "content": hit.content,
                "score": round(hit.score, 4),
                "paper_id": hit.metadata.get("paper_id"),
                "title": hit.metadata.get("title", ""),
                "authors": hit.metadata.get("authors", []),
                "categories": hit.metadata.get("categories", []),
                "chunk_index": hit.metadata.get("chunk_index"),
                "total_chunks": hit.metadata.get("total_chunks"),
            })

        logger.info(
            "Paper chunk search completed",
            query=query[:50],
            results_count=len(output),
        )

        return output

    except Exception as e:
        logger.error(
            "search_paper_chunks failed",
            query=query[:50],
            error=str(e),
        )
        return []
