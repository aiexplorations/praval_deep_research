"""
Knowledge Base Search API endpoints.

Provides hybrid search (BM25 + Vector) over indexed research papers
using Vajra BM25 and Qdrant vector search with RRF fusion.
"""

import time
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from agentic_research.storage.hybrid_search import get_hybrid_search, HybridSearchResult
from agentic_research.storage.paper_index import get_paper_index
from agentic_research.storage.conversation_store import get_conversation_store

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/kb-search", tags=["Knowledge Base Search"])


# Request/Response Models

class KBSearchRequest(BaseModel):
    """Request model for knowledge base search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query",
        example="transformer attention mechanism"
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of results to return"
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="BM25 weight (1.0=keyword, 0.5=hybrid, 0.0=semantic)"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Filter by ArXiv categories (e.g., ['cs.AI', 'cs.LG'])"
    )
    paper_ids: Optional[List[str]] = Field(
        default=None,
        description="Filter to specific paper IDs"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "transformer attention mechanism",
                "top_k": 20,
                "alpha": 0.5,
                "categories": ["cs.AI", "cs.LG"]
            }
        }


class PaperSearchResult(BaseModel):
    """Individual paper search result."""

    paper_id: str
    title: str
    authors: List[str]
    categories: List[str]
    abstract: str
    combined_score: float = Field(description="Fused hybrid score")
    bm25_score: Optional[float] = Field(None, description="BM25 keyword score")
    bm25_rank: Optional[int] = Field(None, description="Rank in BM25 results")
    vector_score: Optional[float] = Field(None, description="Vector similarity score")
    vector_rank: Optional[int] = Field(None, description="Rank in vector results")
    matching_chunks: int = Field(default=0, description="Number of matching chunks")


class KBSearchResponse(BaseModel):
    """Response model for knowledge base search."""

    query: str
    search_mode: str = Field(description="'keyword', 'hybrid', or 'semantic'")
    alpha: float
    results: List[PaperSearchResult]
    total_found: int
    search_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StartChatRequest(BaseModel):
    """Request to start a chat with selected papers."""

    paper_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="IDs of papers to chat about"
    )
    initial_question: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional first question to ask"
    )
    title: Optional[str] = Field(
        None,
        max_length=200,
        description="Optional conversation title"
    )


class StartChatResponse(BaseModel):
    """Response after creating a paper chat."""

    conversation_id: str
    paper_count: int
    paper_titles: List[str]
    redirect_url: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PaperChunksResponse(BaseModel):
    """Response with paper chunks."""

    paper_id: str
    title: str
    chunk_count: int
    chunks: List[Dict[str, Any]]


# Endpoints

@router.post(
    "",
    response_model=KBSearchResponse,
    summary="Hybrid search over knowledge base"
)
async def search_knowledge_base(request: KBSearchRequest) -> KBSearchResponse:
    """
    Search indexed papers using hybrid search (BM25 + Vector).

    Combines keyword matching (BM25) with semantic similarity (vector)
    using Reciprocal Rank Fusion. Adjust the `alpha` parameter to control
    the balance:
    - alpha=1.0: Pure keyword/BM25 search (exact term matching)
    - alpha=0.5: Balanced hybrid search (default, recommended)
    - alpha=0.0: Pure semantic/vector search (conceptual similarity)

    Returns papers ranked by combined relevance score with both
    individual BM25 and vector scores for transparency.
    """
    start_time = time.time()

    try:
        hybrid = get_hybrid_search()

        # Perform hybrid search
        results = hybrid.search(
            query=request.query,
            top_k=request.top_k,
            alpha=request.alpha,
            categories=request.categories,
            paper_ids=request.paper_ids,
        )

        # Convert to response model
        paper_results = [
            PaperSearchResult(
                paper_id=r.paper_id,
                title=r.title,
                authors=r.authors,
                categories=r.categories,
                abstract=r.abstract,
                combined_score=round(r.combined_score, 4),
                bm25_score=round(r.bm25_score, 2) if r.bm25_score else None,
                bm25_rank=r.bm25_rank,
                vector_score=round(r.vector_score, 4) if r.vector_score else None,
                vector_rank=r.vector_rank,
                matching_chunks=r.matching_chunks,
            )
            for r in results
        ]

        search_time_ms = int((time.time() - start_time) * 1000)

        return KBSearchResponse(
            query=request.query,
            search_mode=hybrid.get_search_mode(request.alpha),
            alpha=request.alpha,
            results=paper_results,
            total_found=len(paper_results),
            search_time_ms=search_time_ms,
        )

    except Exception as e:
        logger.error("KB search failed", error=str(e), query=request.query[:50])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get(
    "/papers/{paper_id}/chunks",
    response_model=PaperChunksResponse,
    summary="Get all chunks for a paper"
)
async def get_paper_chunks(paper_id: str) -> PaperChunksResponse:
    """
    Get all indexed chunks for a specific paper.

    Returns the paper title and all text chunks that have been
    indexed for full-text search.
    """
    try:
        index = get_paper_index()
        chunks = index.get_paper_chunks(paper_id)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Paper {paper_id} not found in knowledge base"
            )

        # Get title from first chunk
        title = chunks[0].metadata.get("title", "") if chunks else ""

        return PaperChunksResponse(
            paper_id=paper_id,
            title=title,
            chunk_count=len(chunks),
            chunks=[
                {
                    "chunk_index": c.metadata.get("chunk_index", i),
                    "content": c.content,
                }
                for i, c in enumerate(chunks)
            ]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get paper chunks failed", paper_id=paper_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get paper chunks: {str(e)}"
        )


@router.post(
    "/start-chat",
    response_model=StartChatResponse,
    summary="Start a chat with selected papers"
)
async def start_paper_chat(request: StartChatRequest) -> StartChatResponse:
    """
    Create a new conversation focused on specific papers.

    The conversation will be created with metadata indicating which
    papers to use as context. When asking questions in this conversation,
    the Q&A system will prioritize content from these papers.
    """
    try:
        # Get paper titles for display
        index = get_paper_index()
        paper_titles = []

        for paper_id in request.paper_ids:
            chunks = index.get_paper_chunks(paper_id)
            if chunks:
                title = chunks[0].metadata.get("title", paper_id)
                paper_titles.append(title)
            else:
                paper_titles.append(paper_id)

        # Generate conversation title
        if request.title:
            conv_title = request.title
        elif len(paper_titles) == 1:
            conv_title = f"Chat: {paper_titles[0][:50]}"
        else:
            conv_title = f"Chat: {len(paper_titles)} papers"

        # Create conversation with paper context metadata
        store = get_conversation_store()
        conversation = await store.create_conversation(title=conv_title)

        # Store paper_ids in conversation metadata
        # Note: This requires the context_data field to be added to the Conversation model
        conversation_id = conversation.id

        # Update conversation with metadata (paper context)
        # Include paper_titles so the chat UI can display them
        await store.update_conversation_metadata(
            conversation_id,
            metadata={
                "paper_ids": request.paper_ids,
                "paper_titles": paper_titles,
                "source": "kb_search",
                "scope": "primary_plus_related",
            }
        )

        # If initial question provided, add it as first message
        if request.initial_question:
            await store.add_message(
                conversation_id=conversation_id,
                role="user",
                content=request.initial_question,
            )

        logger.info(
            "Paper chat created",
            conversation_id=conversation_id,
            paper_count=len(request.paper_ids),
        )

        return StartChatResponse(
            conversation_id=conversation_id,
            paper_count=len(request.paper_ids),
            paper_titles=paper_titles,
            redirect_url=f"/chat?conversation_id={conversation_id}",
        )

    except Exception as e:
        logger.error("Start paper chat failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}"
        )


@router.get(
    "/stats",
    summary="Get knowledge base statistics"
)
async def get_kb_stats() -> Dict[str, Any]:
    """
    Get statistics about the knowledge base.

    Returns counts of indexed papers, chunks, and categories.
    """
    try:
        index = get_paper_index()

        papers = index.get_indexed_papers()
        total_papers = len(papers)
        total_chunks = index.document_count

        # Count categories
        categories: Dict[str, int] = {}
        for paper in papers:
            for cat in paper.get("categories", []):
                categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_papers": total_papers,
            "total_chunks": total_chunks,
            "categories": categories,
            "avg_chunks_per_paper": round(total_chunks / total_papers, 1) if total_papers > 0 else 0,
        }

    except Exception as e:
        logger.error("Get KB stats failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )
