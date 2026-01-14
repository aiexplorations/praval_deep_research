"""
Knowledge Base Search Agent - Research Domain.

I am a knowledge base search specialist who uses hybrid search (BM25 + Vector)
to find relevant papers from the indexed research collection.
"""

import logging
from typing import Dict, Any, List, Optional
from praval import agent, broadcast, Spore

# Import with proper path handling
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', '..')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from agentic_research.storage.hybrid_search import get_hybrid_search
from agentic_research.storage.paper_index import get_paper_index

logger = logging.getLogger(__name__)


@agent(
    "kb_search",
    channel="broadcast",
    responds_to=["kb_search_request", "start_paper_chat_request"],
    memory=True
)
def kb_search_agent(spore: Spore) -> None:
    """
    I am a knowledge base search specialist. I use Vajra's hybrid search
    (BM25 + Vector with RRF fusion) to find relevant papers from the
    indexed research collection.

    My expertise:
    - Hybrid search combining keyword and semantic matching
    - Adjustable search balance (keyword-heavy to semantic-heavy)
    - Paper aggregation and relevance scoring
    - Context-aware paper selection for chat
    """
    logger.info("=" * 80)
    logger.info("KB SEARCH AGENT TRIGGERED!")
    logger.info("=" * 80)
    logger.info(f"Received spore type: {spore.knowledge.get('type')}")

    spore_type = spore.knowledge.get("type")

    if spore_type == "kb_search_request":
        _handle_search_request(spore)
    elif spore_type == "start_paper_chat_request":
        _handle_start_chat_request(spore)
    else:
        logger.warning(f"Unknown spore type: {spore_type}")


def _handle_search_request(spore: Spore) -> None:
    """Handle knowledge base search request."""
    query = spore.knowledge.get("query", "")
    top_k = spore.knowledge.get("top_k", 20)
    alpha = spore.knowledge.get("alpha", 0.5)
    categories = spore.knowledge.get("categories")
    paper_ids = spore.knowledge.get("paper_ids")
    session_id = spore.knowledge.get("session_id")

    if not query:
        logger.warning("KB search received empty query")
        broadcast({
            "type": "kb_search_error",
            "session_id": session_id,
            "error": "Empty query provided",
        })
        return

    logger.info(f"KB Search: query='{query[:50]}', alpha={alpha}, top_k={top_k}")

    try:
        hybrid = get_hybrid_search()
        results = hybrid.search(
            query=query,
            top_k=top_k,
            alpha=alpha,
            categories=categories,
            paper_ids=paper_ids,
        )

        # Convert results to serializable format
        paper_results = []
        for r in results:
            paper_results.append({
                "paper_id": r.paper_id,
                "title": r.title,
                "authors": r.authors,
                "categories": r.categories,
                "abstract": r.abstract,
                "combined_score": round(r.combined_score, 4),
                "bm25_score": round(r.bm25_score, 2) if r.bm25_score else None,
                "bm25_rank": r.bm25_rank,
                "vector_score": round(r.vector_score, 4) if r.vector_score else None,
                "vector_rank": r.vector_rank,
                "matching_chunks": r.matching_chunks,
            })

        # Remember successful searches
        if paper_results:
            kb_search_agent.remember(
                f"successful_kb_search: '{query[:50]}' -> {len(paper_results)} papers",
                importance=0.6
            )

        logger.info(f"KB Search completed: {len(paper_results)} papers found")

        # Broadcast results
        broadcast({
            "type": "kb_search_results",
            "session_id": session_id,
            "query": query,
            "search_mode": hybrid.get_search_mode(alpha),
            "alpha": alpha,
            "results": paper_results,
            "total_found": len(paper_results),
        })

    except Exception as e:
        logger.error(f"KB search failed: {e}")
        broadcast({
            "type": "kb_search_error",
            "session_id": session_id,
            "error": str(e),
        })


def _handle_start_chat_request(spore: Spore) -> None:
    """Handle request to start a chat with selected papers."""
    paper_ids = spore.knowledge.get("paper_ids", [])
    initial_question = spore.knowledge.get("initial_question")
    session_id = spore.knowledge.get("session_id")
    conversation_id = spore.knowledge.get("conversation_id")

    if not paper_ids:
        logger.warning("Start chat request with no papers")
        broadcast({
            "type": "paper_chat_error",
            "session_id": session_id,
            "error": "No papers selected",
        })
        return

    logger.info(f"Starting paper chat with {len(paper_ids)} papers")

    try:
        # Get paper titles
        index = get_paper_index()
        paper_titles = []
        for paper_id in paper_ids:
            chunks = index.get_paper_chunks(paper_id)
            if chunks:
                title = chunks[0].metadata.get("title", paper_id)
                paper_titles.append(title)
            else:
                paper_titles.append(paper_id)

        # Remember this chat context
        kb_search_agent.remember(
            f"paper_chat_started: {len(paper_ids)} papers - {', '.join(paper_titles[:3])}",
            importance=0.7
        )

        # Broadcast chat initiated
        broadcast({
            "type": "paper_chat_initiated",
            "session_id": session_id,
            "conversation_id": conversation_id,
            "paper_ids": paper_ids,
            "paper_titles": paper_titles,
            "paper_count": len(paper_ids),
            "initial_question": initial_question,
        })

        logger.info(f"Paper chat initiated: conversation={conversation_id}")

    except Exception as e:
        logger.error(f"Start chat failed: {e}")
        broadcast({
            "type": "paper_chat_error",
            "session_id": session_id,
            "error": str(e),
        })


# Agent metadata for introspection
AGENT_METADATA = {
    "identity": "knowledge base search specialist",
    "domain": "research",
    "capabilities": [
        "hybrid search (BM25 + Vector)",
        "adjustable search balance",
        "paper aggregation",
        "context-aware selection",
    ],
    "responds_to": ["kb_search_request", "start_paper_chat_request"],
    "broadcasts": ["kb_search_results", "kb_search_error", "paper_chat_initiated", "paper_chat_error"],
    "memory_enabled": True,
    "learning_focus": "successful search patterns and user preferences",
}
