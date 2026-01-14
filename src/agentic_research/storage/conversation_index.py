"""
Conversation Index for BM25 search over conversation history.

This module provides full-text search over conversation messages
using Vajra BM25 search engine.
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


class ConversationIndex(BM25IndexManager):
    """
    BM25 index for conversation messages.

    Provides global search across all conversations with optional
    filtering by user_id and conversation_id. Supports recency-weighted
    scoring for relevant context retrieval.
    """

    INDEX_NAME = "conversations"

    def __init__(self, index_path: Optional[Path] = None) -> None:
        """
        Initialize the conversation index.

        Args:
            index_path: Base path for storing indexes (defaults to config)
        """
        super().__init__(index_name=self.INDEX_NAME, index_path=index_path)

    def get_index_type(self) -> str:
        """Return the type of this index."""
        return "conversation"

    def index_message(
        self,
        message_id: str,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Index a conversation message.

        Args:
            message_id: Unique identifier for the message
            conversation_id: ID of the conversation this message belongs to
            user_id: ID of the user who owns this conversation
            role: Message role (user, assistant, system)
            content: Message content
            timestamp: When the message was created
            metadata: Additional metadata to store
        """
        if not content or not content.strip():
            logger.debug(
                "Skipping empty message",
                message_id=message_id,
            )
            return

        doc_metadata = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "role": role,
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            **(metadata or {}),
        }

        doc = IndexedDocument(
            id=message_id,
            title=f"{role}:{conversation_id[:8]}",
            content=content,
            metadata=doc_metadata,
        )

        self.add_document(doc)

        logger.debug(
            "Message indexed",
            message_id=message_id,
            conversation_id=conversation_id,
            role=role,
        )

    def index_messages_batch(
        self,
        messages: List[Dict[str, Any]],
    ) -> int:
        """
        Index multiple messages at once.

        Args:
            messages: List of message dicts with keys:
                - message_id, conversation_id, user_id, role, content
                - Optional: timestamp, metadata

        Returns:
            Number of messages indexed
        """
        indexed_count = 0

        for msg in messages:
            content = msg.get("content", "")
            if not content or not content.strip():
                continue

            doc_metadata = {
                "conversation_id": msg["conversation_id"],
                "user_id": msg["user_id"],
                "role": msg["role"],
                "timestamp": msg.get(
                    "timestamp", datetime.utcnow().isoformat()
                ),
                **(msg.get("metadata") or {}),
            }

            doc = IndexedDocument(
                id=msg["message_id"],
                title=f"{msg['role']}:{msg['conversation_id'][:8]}",
                content=content,
                metadata=doc_metadata,
            )

            self._documents[doc.id] = doc
            indexed_count += 1

        if indexed_count > 0:
            self._dirty = True
            logger.info(
                "Batch indexed messages",
                count=indexed_count,
                total_documents=self.document_count,
            )

        return indexed_count

    def search_conversations(
        self,
        query: str,
        top_k: int = 10,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        role: Optional[str] = None,
        recency_boost: bool = False,
    ) -> List[SearchHit]:
        """
        Search conversation history.

        Args:
            query: Search query string
            top_k: Number of results to return
            user_id: Filter by user ID (optional)
            conversation_id: Filter by conversation ID (optional)
            role: Filter by message role (user, assistant, system)
            recency_boost: Apply recency weighting to scores

        Returns:
            List of SearchHit results
        """
        # Build filters
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if conversation_id:
            filters["conversation_id"] = conversation_id
        if role:
            filters["role"] = role

        # Get more results for filtering
        search_limit = top_k * 3 if filters else top_k
        results = self.search(query, top_k=search_limit, filters=filters)

        # Apply recency boost if requested
        if recency_boost and results:
            results = self._apply_recency_boost(results)
            # Re-sort by boosted score
            results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def _apply_recency_boost(
        self,
        results: List[SearchHit],
    ) -> List[SearchHit]:
        """
        Apply recency weighting to search results.

        More recent messages get a score boost based on config weight.

        Args:
            results: Original search results

        Returns:
            Results with recency-boosted scores
        """
        if not results:
            return results

        recency_weight = self.settings.CONTEXT_RECENCY_WEIGHT
        now = datetime.utcnow()

        boosted_results = []
        for hit in results:
            # Parse timestamp from metadata
            timestamp_str = hit.metadata.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    # Calculate age in hours
                    age_hours = (now - timestamp).total_seconds() / 3600
                    # Decay factor: newer = higher boost
                    decay = 1.0 / (1.0 + age_hours / 24)  # Half-life of ~1 day
                    boost = 1.0 + (recency_weight * decay)
                    boosted_score = hit.score * boost
                except (ValueError, TypeError):
                    boosted_score = hit.score
            else:
                boosted_score = hit.score

            # Create new hit with boosted score
            boosted_results.append(
                SearchHit(
                    document_id=hit.document_id,
                    content=hit.content,
                    score=boosted_score,
                    rank=hit.rank,
                    metadata=hit.metadata,
                )
            )

        return boosted_results

    def get_conversation_messages(
        self,
        conversation_id: str,
    ) -> List[SearchHit]:
        """
        Get all messages from a specific conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            List of messages sorted by timestamp
        """
        messages = []
        for doc in self._documents.values():
            if doc.metadata.get("conversation_id") == conversation_id:
                messages.append(
                    SearchHit(
                        document_id=doc.id,
                        content=doc.content,
                        score=1.0,
                        rank=0,
                        metadata=doc.metadata,
                    )
                )

        # Sort by timestamp
        messages.sort(
            key=lambda x: x.metadata.get("timestamp", ""),
        )

        return messages

    def get_user_conversations(
        self,
        user_id: str,
    ) -> List[str]:
        """
        Get list of conversation IDs for a user.

        Args:
            user_id: ID of the user

        Returns:
            List of unique conversation IDs
        """
        conversation_ids = set()
        for doc in self._documents.values():
            if doc.metadata.get("user_id") == user_id:
                conv_id = doc.metadata.get("conversation_id")
                if conv_id:
                    conversation_ids.add(conv_id)

        return list(conversation_ids)


# Global singleton instance
_conversation_index: Optional[ConversationIndex] = None


def get_conversation_index() -> ConversationIndex:
    """
    Get the global conversation index singleton.

    Returns:
        The ConversationIndex instance
    """
    global _conversation_index

    if _conversation_index is None:
        _conversation_index = ConversationIndex()
        register_index_manager(_conversation_index)

        # Try to load existing index
        _conversation_index.load_index()

    return _conversation_index
