"""
Unit tests for search_conversations Praval tool.

Tests the search_conversations tool that provides BM25 full-text
search over conversation history.
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))


class TestSearchConversationsTool:
    """Tests for the search_conversations tool."""

    @pytest.fixture
    def mock_conversation_index(self):
        """Create a mock ConversationIndex."""
        mock_index = MagicMock()
        return mock_index

    @pytest.fixture
    def mock_search_hit(self):
        """Create a mock SearchHit."""
        from agentic_research.storage.bm25_index_manager import SearchHit

        return SearchHit(
            document_id="msg-001",
            content="What is machine learning?",
            score=5.5,
            rank=1,
            metadata={
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "role": "user",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def test_search_conversations_returns_results(
        self, mock_conversation_index, mock_search_hit
    ):
        """Test that search_conversations returns formatted results."""
        mock_conversation_index.search_conversations.return_value = [mock_search_hit]

        with patch(
            "tools.search_conversations.get_conversation_index",
            return_value=mock_conversation_index,
        ):
            from tools.search_conversations import search_conversations

            results = search_conversations(
                query="machine learning",
                top_k=10,
            )

            assert len(results) == 1
            assert results[0]["message_id"] == "msg-001"
            assert results[0]["content"] == "What is machine learning?"
            assert results[0]["score"] == 5.5
            assert results[0]["conversation_id"] == "conv-001"
            assert results[0]["user_id"] == "user-001"
            assert results[0]["role"] == "user"

    def test_search_conversations_empty_query(self, mock_conversation_index):
        """Test that empty query returns empty results."""
        with patch(
            "tools.search_conversations.get_conversation_index",
            return_value=mock_conversation_index,
        ):
            from tools.search_conversations import search_conversations

            results = search_conversations(query="")

            assert results == []
            mock_conversation_index.search_conversations.assert_not_called()

    def test_search_conversations_whitespace_query(self, mock_conversation_index):
        """Test that whitespace-only query returns empty results."""
        with patch(
            "tools.search_conversations.get_conversation_index",
            return_value=mock_conversation_index,
        ):
            from tools.search_conversations import search_conversations

            results = search_conversations(query="   ")

            assert results == []
            mock_conversation_index.search_conversations.assert_not_called()

    def test_search_conversations_with_filters(
        self, mock_conversation_index, mock_search_hit
    ):
        """Test that filters are passed to the index."""
        mock_conversation_index.search_conversations.return_value = [mock_search_hit]

        with patch(
            "tools.search_conversations.get_conversation_index",
            return_value=mock_conversation_index,
        ):
            from tools.search_conversations import search_conversations

            results = search_conversations(
                query="test query",
                top_k=5,
                user_id="user-001",
                conversation_id="conv-001",
                role="user",
                recency_boost=False,
            )

            mock_conversation_index.search_conversations.assert_called_once_with(
                query="test query",
                top_k=5,
                user_id="user-001",
                conversation_id="conv-001",
                role="user",
                recency_boost=False,
            )

    def test_search_conversations_handles_exception(self, mock_conversation_index):
        """Test that exceptions are handled gracefully."""
        mock_conversation_index.search_conversations.side_effect = Exception(
            "Search failed"
        )

        with patch(
            "tools.search_conversations.get_conversation_index",
            return_value=mock_conversation_index,
        ):
            from tools.search_conversations import search_conversations

            results = search_conversations(query="test query")

            assert results == []

    def test_search_conversations_default_params(
        self, mock_conversation_index, mock_search_hit
    ):
        """Test default parameter values."""
        mock_conversation_index.search_conversations.return_value = [mock_search_hit]

        with patch(
            "tools.search_conversations.get_conversation_index",
            return_value=mock_conversation_index,
        ):
            from tools.search_conversations import search_conversations

            results = search_conversations(query="test")

            mock_conversation_index.search_conversations.assert_called_once_with(
                query="test",
                top_k=10,
                user_id=None,
                conversation_id=None,
                role=None,
                recency_boost=True,  # Default is True
            )


class TestSearchConversationsToolMetadata:
    """Tests for the tool decorator metadata."""

    def test_tool_has_correct_metadata(self):
        """Test that the tool has correct Praval metadata."""
        from tools.search_conversations import search_conversations
        from praval import get_tool_info, is_tool

        # Check it's decorated as a tool
        assert is_tool(search_conversations)

        # Check metadata
        info = get_tool_info(search_conversations)
        assert info is not None
        assert info.get("tool_name") == "search_conversations"
        assert info.get("category") == "search"
        assert info.get("shared") is True
        assert "bm25" in info.get("tags", [])
        assert "conversations" in info.get("tags", [])
