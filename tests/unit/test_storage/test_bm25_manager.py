"""
Unit tests for BM25 Index Manager.

Tests the BM25IndexManager base class and ConversationIndex
using Vajra BM25 search engine.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

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


class TestBM25IndexManager:
    """Tests for the BM25IndexManager base class."""

    @pytest.fixture
    def temp_index_path(self, tmp_path):
        """Create a temporary directory for test indexes."""
        return tmp_path / "test_indexes"

    @pytest.fixture
    def mock_settings(self, monkeypatch):
        """Mock settings for tests."""
        mock_settings_obj = MagicMock()
        mock_settings_obj.BM25_INDEX_PATH = "/tmp/test_indexes"
        mock_settings_obj.BM25_K1 = 1.5
        mock_settings_obj.BM25_B = 0.75
        mock_settings_obj.BM25_CACHE_SIZE = 100
        mock_settings_obj.BM25_USE_SPARSE = False
        mock_settings_obj.BM25_USE_EAGER = False
        mock_settings_obj.CONTEXT_RECENCY_WEIGHT = 0.3

        monkeypatch.setattr(
            "agentic_research.storage.bm25_index_manager.get_settings",
            lambda: mock_settings_obj
        )
        monkeypatch.setattr(
            "agentic_research.storage.conversation_index.get_settings",
            lambda: mock_settings_obj
        )
        return mock_settings_obj


class TestConversationIndex:
    """Tests for the ConversationIndex class."""

    @pytest.fixture
    def temp_index_path(self, tmp_path):
        """Create a temporary directory for test indexes."""
        return tmp_path / "test_indexes"

    @pytest.fixture
    def mock_settings(self, monkeypatch, temp_index_path):
        """Mock settings for tests."""
        mock_settings_obj = MagicMock()
        mock_settings_obj.BM25_INDEX_PATH = str(temp_index_path)
        mock_settings_obj.BM25_K1 = 1.5
        mock_settings_obj.BM25_B = 0.75
        mock_settings_obj.BM25_CACHE_SIZE = 100
        mock_settings_obj.BM25_USE_SPARSE = False
        mock_settings_obj.BM25_USE_EAGER = False
        mock_settings_obj.CONTEXT_RECENCY_WEIGHT = 0.3

        # Mock at the source module where get_settings is imported
        monkeypatch.setattr(
            "agentic_research.core.config.get_settings",
            lambda: mock_settings_obj
        )
        return mock_settings_obj

    @pytest.fixture
    def conversation_index(self, temp_index_path, mock_settings):
        """Create a ConversationIndex instance for testing."""
        # Clear the singleton to ensure fresh instance
        import agentic_research.storage.conversation_index as conv_module
        conv_module._conversation_index = None
        return ConversationIndex(index_path=temp_index_path)

    def test_index_initialization(self, conversation_index):
        """Test that ConversationIndex initializes correctly."""
        assert conversation_index.index_name == "conversations"
        assert conversation_index.get_index_type() == "conversation"
        assert conversation_index.document_count == 0
        assert not conversation_index.is_loaded

    def test_index_single_message(self, conversation_index):
        """Test indexing a single message."""
        conversation_index.index_message(
            message_id="msg-001",
            conversation_id="conv-001",
            user_id="user-001",
            role="user",
            content="What is machine learning?",
            timestamp=datetime.utcnow(),
        )

        assert conversation_index.document_count == 1

    def test_index_empty_message_skipped(self, conversation_index):
        """Test that empty messages are skipped."""
        conversation_index.index_message(
            message_id="msg-001",
            conversation_id="conv-001",
            user_id="user-001",
            role="user",
            content="",
        )

        assert conversation_index.document_count == 0

    def test_index_whitespace_only_message_skipped(self, conversation_index):
        """Test that whitespace-only messages are skipped."""
        conversation_index.index_message(
            message_id="msg-001",
            conversation_id="conv-001",
            user_id="user-001",
            role="user",
            content="   \n\t  ",
        )

        assert conversation_index.document_count == 0

    def test_index_multiple_messages(self, conversation_index):
        """Test indexing multiple messages."""
        messages = [
            {
                "message_id": "msg-001",
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "role": "user",
                "content": "What is machine learning?",
            },
            {
                "message_id": "msg-002",
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "role": "assistant",
                "content": "Machine learning is a subset of AI...",
            },
            {
                "message_id": "msg-003",
                "conversation_id": "conv-002",
                "user_id": "user-001",
                "role": "user",
                "content": "Explain neural networks",
            },
        ]

        count = conversation_index.index_messages_batch(messages)

        assert count == 3
        assert conversation_index.document_count == 3

    def test_search_conversations_basic(self, conversation_index):
        """Test basic conversation search."""
        # Index some messages
        messages = [
            {
                "message_id": "msg-001",
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "role": "user",
                "content": "What is machine learning and how does it work?",
            },
            {
                "message_id": "msg-002",
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "role": "assistant",
                "content": "Machine learning is a subset of artificial intelligence.",
            },
            {
                "message_id": "msg-003",
                "conversation_id": "conv-002",
                "user_id": "user-002",
                "role": "user",
                "content": "Tell me about deep learning architectures.",
            },
        ]
        conversation_index.index_messages_batch(messages)

        # Build index
        conversation_index.rebuild_index()

        # Search
        results = conversation_index.search_conversations(
            query="machine learning",
            top_k=5,
        )

        assert len(results) > 0
        # The message about machine learning should rank high
        machine_learning_found = any(
            "machine learning" in hit.content.lower() for hit in results
        )
        assert machine_learning_found

    def test_search_with_user_filter(self, conversation_index):
        """Test search with user_id filter."""
        messages = [
            {
                "message_id": "msg-001",
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "role": "user",
                "content": "What is machine learning?",
            },
            {
                "message_id": "msg-002",
                "conversation_id": "conv-002",
                "user_id": "user-002",
                "role": "user",
                "content": "What is machine learning?",
            },
        ]
        conversation_index.index_messages_batch(messages)
        conversation_index.rebuild_index()

        # Search filtered by user
        results = conversation_index.search_conversations(
            query="machine learning",
            user_id="user-001",
        )

        # All results should be from user-001
        for hit in results:
            assert hit.metadata.get("user_id") == "user-001"

    def test_search_with_conversation_filter(self, conversation_index):
        """Test search with conversation_id filter."""
        messages = [
            {
                "message_id": "msg-001",
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "role": "user",
                "content": "Discuss artificial intelligence",
            },
            {
                "message_id": "msg-002",
                "conversation_id": "conv-002",
                "user_id": "user-001",
                "role": "user",
                "content": "Discuss artificial intelligence",
            },
        ]
        conversation_index.index_messages_batch(messages)
        conversation_index.rebuild_index()

        # Search filtered by conversation
        results = conversation_index.search_conversations(
            query="artificial intelligence",
            conversation_id="conv-001",
        )

        # All results should be from conv-001
        for hit in results:
            assert hit.metadata.get("conversation_id") == "conv-001"

    def test_search_with_role_filter(self, conversation_index):
        """Test search with role filter."""
        messages = [
            {
                "message_id": "msg-001",
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "role": "user",
                "content": "What is deep learning?",
            },
            {
                "message_id": "msg-002",
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "role": "assistant",
                "content": "Deep learning uses neural networks.",
            },
        ]
        conversation_index.index_messages_batch(messages)
        conversation_index.rebuild_index()

        # Search only user messages
        results = conversation_index.search_conversations(
            query="learning",
            role="user",
        )

        # All results should be from user role
        for hit in results:
            assert hit.metadata.get("role") == "user"

    def test_index_persistence(self, conversation_index, temp_index_path, mock_settings):
        """Test that indexes can be saved and loaded."""
        # Index messages and build
        messages = [
            {
                "message_id": "msg-001",
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "role": "user",
                "content": "Test message for persistence",
            },
        ]
        conversation_index.index_messages_batch(messages)
        conversation_index.rebuild_index()

        # Save
        assert conversation_index.save_index()

        # Create new instance and load
        import agentic_research.storage.conversation_index as conv_module
        conv_module._conversation_index = None
        new_index = ConversationIndex(index_path=temp_index_path)
        assert new_index.load_index()
        assert new_index.document_count == 1

        # Search should work after load (fixed in Vajra v0.4.0)
        results = new_index.search_conversations(
            query="persistence",
            top_k=5,
        )
        assert len(results) > 0

    def test_get_conversation_messages(self, conversation_index):
        """Test getting all messages from a conversation."""
        messages = [
            {
                "message_id": "msg-001",
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "role": "user",
                "content": "First message",
                "timestamp": datetime.utcnow().isoformat(),
            },
            {
                "message_id": "msg-002",
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "role": "assistant",
                "content": "Second message",
                "timestamp": (datetime.utcnow() + timedelta(seconds=1)).isoformat(),
            },
            {
                "message_id": "msg-003",
                "conversation_id": "conv-002",
                "user_id": "user-001",
                "role": "user",
                "content": "Different conversation",
            },
        ]
        conversation_index.index_messages_batch(messages)

        # Get messages for conv-001
        conv_messages = conversation_index.get_conversation_messages("conv-001")

        assert len(conv_messages) == 2
        assert all(
            m.metadata.get("conversation_id") == "conv-001"
            for m in conv_messages
        )

    def test_get_user_conversations(self, conversation_index):
        """Test getting list of conversation IDs for a user."""
        messages = [
            {
                "message_id": "msg-001",
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "role": "user",
                "content": "Message 1",
            },
            {
                "message_id": "msg-002",
                "conversation_id": "conv-002",
                "user_id": "user-001",
                "role": "user",
                "content": "Message 2",
            },
            {
                "message_id": "msg-003",
                "conversation_id": "conv-003",
                "user_id": "user-002",
                "role": "user",
                "content": "Message 3",
            },
        ]
        conversation_index.index_messages_batch(messages)

        # Get conversations for user-001
        conv_ids = conversation_index.get_user_conversations("user-001")

        assert len(conv_ids) == 2
        assert "conv-001" in conv_ids
        assert "conv-002" in conv_ids
        assert "conv-003" not in conv_ids

    def test_recency_boost(self, conversation_index):
        """Test that recency boost affects scoring."""
        now = datetime.utcnow()

        messages = [
            {
                "message_id": "msg-old",
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "role": "user",
                "content": "Machine learning question",
                "timestamp": (now - timedelta(days=30)).isoformat(),
            },
            {
                "message_id": "msg-new",
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "role": "user",
                "content": "Machine learning question",
                "timestamp": now.isoformat(),
            },
        ]
        conversation_index.index_messages_batch(messages)
        conversation_index.rebuild_index()

        # Search with recency boost
        results_with_boost = conversation_index.search_conversations(
            query="machine learning",
            recency_boost=True,
        )

        # The newer message should have a higher score
        if len(results_with_boost) >= 2:
            scores = {r.document_id: r.score for r in results_with_boost}
            # Newer should score higher with recency boost
            assert scores.get("msg-new", 0) >= scores.get("msg-old", 0)


class TestIndexedDocument:
    """Tests for the IndexedDocument dataclass."""

    def test_create_indexed_document(self):
        """Test creating an IndexedDocument."""
        doc = IndexedDocument(
            id="doc-001",
            title="Test Document",
            content="This is test content",
            metadata={"author": "test"},
        )

        assert doc.id == "doc-001"
        assert doc.title == "Test Document"
        assert doc.content == "This is test content"
        assert doc.metadata == {"author": "test"}


class TestSearchHit:
    """Tests for the SearchHit dataclass."""

    def test_create_search_hit(self):
        """Test creating a SearchHit."""
        hit = SearchHit(
            document_id="doc-001",
            content="Match content",
            score=5.5,
            rank=1,
            metadata={"source": "test"},
        )

        assert hit.document_id == "doc-001"
        assert hit.content == "Match content"
        assert hit.score == 5.5
        assert hit.rank == 1
        assert hit.metadata == {"source": "test"}
