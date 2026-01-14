"""
List Knowledge Base Tool.

This module provides a Praval tool for listing and exploring
the contents of the research knowledge base.

Single Responsibility: Show what papers are in the knowledge base.
Used by: All agents to understand available research context.
"""

from typing import Dict, List, Any, Optional
import structlog

from praval import tool

from agentic_research.storage.paper_index import get_paper_index

logger = structlog.get_logger(__name__)


@tool(
    tool_name="list_knowledge_base",
    description="List papers in the research knowledge base. "
    "Shows what research papers have been indexed and are available for search. "
    "Use this to understand what context is available before answering questions.",
    category="knowledge",
    shared=True,
    version="1.0.0",
    author="Research Team",
    tags=["knowledge", "papers", "index", "list", "overview"],
)
def list_knowledge_base(
    limit: int = 20,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List papers in the research knowledge base.

    This tool shows what research papers have been indexed and are
    available for searching. Use this to:
    - Understand what research context is available
    - Check if specific topics have been indexed
    - Get an overview of the knowledge base contents

    Args:
        limit: Maximum number of papers to return (default: 20)
        category: Filter by ArXiv category like "cs.AI" (optional)

    Returns:
        Dictionary with:
        - total_papers: Number of unique papers indexed
        - total_chunks: Total document chunks in the index
        - papers: List of paper metadata (id, title, authors, categories)
    """
    logger.info(
        "Listing knowledge base",
        limit=limit,
        category=category,
    )

    try:
        index = get_paper_index()

        # Get all indexed papers
        papers = index.get_indexed_papers()

        # Filter by category if specified
        if category:
            papers = [
                p for p in papers
                if category in p.get("categories", [])
            ]

        # Sort by title
        papers.sort(key=lambda p: p.get("title", ""))

        # Limit results
        papers = papers[:limit]

        result = {
            "total_papers": index.get_paper_count(),
            "total_chunks": index.document_count,
            "papers": papers,
        }

        logger.info(
            "Knowledge base listed",
            total_papers=result["total_papers"],
            returned=len(papers),
        )

        return result

    except Exception as e:
        logger.error(
            "list_knowledge_base failed",
            error=str(e),
        )
        return {
            "total_papers": 0,
            "total_chunks": 0,
            "papers": [],
            "error": str(e),
        }


@tool(
    tool_name="get_paper_details",
    description="Get detailed information about a specific paper in the knowledge base. "
    "Returns full metadata and chunk overview for a paper by its ID.",
    category="knowledge",
    shared=True,
    version="1.0.0",
    author="Research Team",
    tags=["knowledge", "papers", "details", "metadata"],
)
def get_paper_details(
    paper_id: str,
) -> Dict[str, Any]:
    """
    Get detailed information about a specific paper.

    Args:
        paper_id: The paper ID (e.g., ArXiv ID like "2301.12345")

    Returns:
        Dictionary with:
        - paper_id: Paper identifier
        - title: Paper title
        - authors: List of authors
        - categories: ArXiv categories
        - abstract: Paper abstract
        - chunk_count: Number of indexed chunks
        - chunks_preview: First few chunks for context
    """
    if not paper_id or not paper_id.strip():
        logger.warning("get_paper_details called with empty paper_id")
        return {"error": "paper_id is required"}

    logger.info("Getting paper details", paper_id=paper_id)

    try:
        index = get_paper_index()

        # Get all chunks for this paper
        chunks = index.get_paper_chunks(paper_id)

        if not chunks:
            return {
                "paper_id": paper_id,
                "error": "Paper not found in knowledge base",
            }

        # Get metadata from first chunk
        first_chunk = chunks[0]
        metadata = first_chunk.metadata

        result = {
            "paper_id": paper_id,
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", []),
            "categories": metadata.get("categories", []),
            "abstract": metadata.get("abstract", ""),
            "published_date": metadata.get("published_date", ""),
            "chunk_count": len(chunks),
            "chunks_preview": [
                {
                    "chunk_index": c.metadata.get("chunk_index"),
                    "content_preview": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                }
                for c in chunks[:3]  # First 3 chunks as preview
            ],
        }

        logger.info(
            "Paper details retrieved",
            paper_id=paper_id,
            chunk_count=len(chunks),
        )

        return result

    except Exception as e:
        logger.error(
            "get_paper_details failed",
            paper_id=paper_id,
            error=str(e),
        )
        return {
            "paper_id": paper_id,
            "error": str(e),
        }
