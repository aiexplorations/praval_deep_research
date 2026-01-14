"""
Search Conversations Tool.

This module provides a Praval tool for searching conversation history
using Vajra BM25 full-text search.

Single Responsibility: Search conversation history.
Used by: qa_specialist, research_advisor agents.
"""

from typing import Dict, List, Any, Optional
import structlog

from praval import tool

from agentic_research.storage.conversation_index import get_conversation_index

logger = structlog.get_logger(__name__)


@tool(
    tool_name="search_conversations",
    description="Search conversation history using BM25 full-text search. "
    "Returns relevant messages from past conversations based on query similarity.",
    category="search",
    shared=True,
    version="1.0.0",
    author="Research Team",
    tags=["search", "bm25", "conversations", "history", "context"],
)
def search_conversations(
    query: str,
    top_k: int = 10,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    role: Optional[str] = None,
    recency_boost: bool = True,
) -> List[Dict[str, Any]]:
    """
    Search conversation history using BM25 full-text search.

    This tool searches across all indexed conversation messages to find
    relevant prior discussions. Useful for:
    - Finding related past Q&A interactions
    - Building context from conversation history
    - Retrieving similar questions asked before

    Args:
        query: Search query string (required)
        top_k: Number of results to return (default: 10)
        user_id: Filter by specific user (optional)
        conversation_id: Filter by specific conversation (optional)
        role: Filter by message role: "user", "assistant", "system" (optional)
        recency_boost: Apply recency weighting - newer messages score higher (default: True)

    Returns:
        List of matching messages with:
        - message_id: Unique identifier
        - content: Message text
        - score: BM25 relevance score
        - conversation_id: Source conversation
        - user_id: Owner of the conversation
        - role: Message role (user/assistant/system)
        - timestamp: When the message was created
    """
    if not query or not query.strip():
        logger.warning("search_conversations called with empty query")
        return []

    logger.info(
        "Searching conversations",
        query=query[:100],
        top_k=top_k,
        user_id=user_id,
        recency_boost=recency_boost,
    )

    try:
        # Get the conversation index
        index = get_conversation_index()

        # Perform search
        results = index.search_conversations(
            query=query,
            top_k=top_k,
            user_id=user_id,
            conversation_id=conversation_id,
            role=role,
            recency_boost=recency_boost,
        )

        # Convert to plain dicts for tool return
        output = []
        for hit in results:
            output.append({
                "message_id": hit.document_id,
                "content": hit.content,
                "score": round(hit.score, 4),
                "conversation_id": hit.metadata.get("conversation_id"),
                "user_id": hit.metadata.get("user_id"),
                "role": hit.metadata.get("role"),
                "timestamp": hit.metadata.get("timestamp"),
            })

        logger.info(
            "Conversation search completed",
            query=query[:50],
            results_count=len(output),
        )

        return output

    except Exception as e:
        logger.error(
            "search_conversations failed",
            query=query[:50],
            error=str(e),
        )
        return []
