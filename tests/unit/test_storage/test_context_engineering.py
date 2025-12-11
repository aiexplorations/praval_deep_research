"""
Unit tests for Context Engineering features.

Tests the multi-collection support in QdrantClientWrapper:
- paper_summaries collection
- linked_papers collection
- Two-tier retrieval functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from agentic_research.storage.qdrant_client import (
    QdrantClientWrapper,
    CollectionType
)


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock()
    settings.QDRANT_URL = "http://localhost:6333"
    settings.QDRANT_API_KEY = None
    settings.QDRANT_COLLECTION_NAME = "research_vectors"
    settings.QDRANT_SUMMARIES_COLLECTION = "paper_summaries"
    settings.QDRANT_LINKED_PAPERS_COLLECTION = "linked_papers"
    settings.EMBEDDING_DIMENSIONS = 1536
    return settings


@pytest.fixture
def qdrant_client(mock_settings):
    """Create QdrantClientWrapper with mocked dependencies."""
    with patch('agentic_research.storage.qdrant_client.QdrantClient') as mock_qdrant:
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant.return_value = mock_qdrant_instance

        with patch('agentic_research.storage.qdrant_client.get_settings', return_value=mock_settings):
            client = QdrantClientWrapper(settings=mock_settings)
            client.client = mock_qdrant_instance
            yield client


class TestCollectionType:
    """Tests for CollectionType enum."""

    def test_collection_type_values(self):
        """Test CollectionType enum has correct values."""
        assert CollectionType.RESEARCH_PAPERS.value == "research_papers"
        assert CollectionType.PAPER_SUMMARIES.value == "paper_summaries"
        assert CollectionType.LINKED_PAPERS.value == "linked_papers"

    def test_collection_type_is_string_enum(self):
        """Test CollectionType can be used as string."""
        assert str(CollectionType.RESEARCH_PAPERS) == "CollectionType.RESEARCH_PAPERS"
        assert CollectionType.RESEARCH_PAPERS == "research_papers"


class TestQdrantClientMultiCollection:
    """Tests for multi-collection support in QdrantClientWrapper."""

    def test_initialization_with_collection_type(self, mock_settings):
        """Test client can be initialized with specific collection type."""
        with patch('agentic_research.storage.qdrant_client.QdrantClient') as mock_qdrant:
            mock_qdrant.return_value = Mock()

            with patch('agentic_research.storage.qdrant_client.get_settings', return_value=mock_settings):
                # Initialize with PAPER_SUMMARIES collection
                client = QdrantClientWrapper(
                    settings=mock_settings,
                    collection_type=CollectionType.PAPER_SUMMARIES
                )

                assert client.collection_name == "paper_summaries"

    def test_initialization_default_collection(self, mock_settings):
        """Test client uses default collection when no type specified."""
        with patch('agentic_research.storage.qdrant_client.QdrantClient') as mock_qdrant:
            mock_qdrant.return_value = Mock()

            with patch('agentic_research.storage.qdrant_client.get_settings', return_value=mock_settings):
                client = QdrantClientWrapper(settings=mock_settings)
                assert client.collection_name == "research_vectors"

    def test_get_collection_name(self, qdrant_client):
        """Test get_collection_name returns correct names."""
        assert qdrant_client.get_collection_name(CollectionType.RESEARCH_PAPERS) == "research_vectors"
        assert qdrant_client.get_collection_name(CollectionType.PAPER_SUMMARIES) == "paper_summaries"
        assert qdrant_client.get_collection_name(CollectionType.LINKED_PAPERS) == "linked_papers"

    def test_switch_collection(self, qdrant_client):
        """Test switching between collections."""
        assert qdrant_client.collection_name == "research_vectors"

        qdrant_client.switch_collection(CollectionType.PAPER_SUMMARIES)
        assert qdrant_client.collection_name == "paper_summaries"

        qdrant_client.switch_collection(CollectionType.LINKED_PAPERS)
        assert qdrant_client.collection_name == "linked_papers"


class TestContextEngineeringCollections:
    """Tests for context engineering collection initialization."""

    def test_initialize_context_engineering_collections_creates_new(self, qdrant_client):
        """Test that collections are created when they don't exist."""
        # Mock empty collections list
        qdrant_client.client.get_collections.return_value.collections = []

        result = qdrant_client.initialize_context_engineering_collections()

        # Should attempt to create both collections
        assert qdrant_client.client.create_collection.call_count == 2
        assert "paper_summaries" in result
        assert "linked_papers" in result

    def test_initialize_context_engineering_collections_skips_existing(self, qdrant_client):
        """Test that existing collections are not recreated."""
        # Mock existing collections
        mock_summaries = Mock()
        mock_summaries.name = "paper_summaries"
        mock_linked = Mock()
        mock_linked.name = "linked_papers"

        qdrant_client.client.get_collections.return_value.collections = [mock_summaries, mock_linked]

        result = qdrant_client.initialize_context_engineering_collections()

        # Should not create any collections
        qdrant_client.client.create_collection.assert_not_called()
        assert result["paper_summaries"] == False
        assert result["linked_papers"] == False


class TestPaperSummaries:
    """Tests for paper summary operations."""

    def test_add_paper_summary(self, qdrant_client):
        """Test adding a paper summary."""
        qdrant_client.client.upload_points.return_value = None

        summary_data = {
            "title": "Test Paper",
            "one_line": "A test paper about testing",
            "abstract_summary": "This paper tests things",
            "key_contributions": ["Testing", "More testing"],
            "methodology": "Unit tests",
            "domains": ["testing", "software"]
        }
        embedding = [0.1] * 1536

        result = qdrant_client.add_paper_summary(
            paper_id="test123",
            summary_data=summary_data,
            embedding=embedding
        )

        assert result == True
        qdrant_client.client.upload_points.assert_called_once()

        # Verify upload was to paper_summaries collection
        call_args = qdrant_client.client.upload_points.call_args
        assert call_args.kwargs['collection_name'] == "paper_summaries"

    def test_add_paper_summary_handles_error(self, qdrant_client):
        """Test that add_paper_summary handles errors gracefully."""
        qdrant_client.client.upload_points.side_effect = Exception("Upload failed")

        result = qdrant_client.add_paper_summary(
            paper_id="test123",
            summary_data={"title": "Test"},
            embedding=[0.1] * 1536
        )

        assert result == False

    def test_search_summaries(self, qdrant_client):
        """Test searching paper summaries."""
        # Mock search results
        mock_result = Mock()
        mock_result.id = 123
        mock_result.score = 0.85
        mock_result.payload = {
            "paper_id": "test123",
            "title": "Test Paper",
            "one_line": "A test paper",
            "domains": ["testing"]
        }

        qdrant_client.client.search.return_value = [mock_result]

        results = qdrant_client.search_summaries(
            query_vector=[0.1] * 1536,
            limit=10,
            score_threshold=0.6
        )

        assert len(results) == 1
        assert results[0]["paper_id"] == "test123"
        assert results[0]["title"] == "Test Paper"
        assert results[0]["one_line"] == "A test paper"
        assert results[0]["score"] == 0.85

        # Verify search was on paper_summaries collection
        call_args = qdrant_client.client.search.call_args
        assert call_args.kwargs['collection_name'] == "paper_summaries"

    def test_search_summaries_returns_empty_on_error(self, qdrant_client):
        """Test that search_summaries returns empty list on error."""
        qdrant_client.client.search.side_effect = Exception("Search failed")

        results = qdrant_client.search_summaries(
            query_vector=[0.1] * 1536,
            limit=10
        )

        assert results == []


class TestLinkedPapers:
    """Tests for linked paper operations."""

    def test_add_linked_paper_vectors(self, qdrant_client):
        """Test adding vectors for a linked paper."""
        qdrant_client.client.upload_points.return_value = None

        chunks = [
            {
                "chunk_text": "First chunk",
                "chunk_index": 0,
                "embedding": [0.1] * 1536,
                "title": "Linked Paper",
                "authors": ["Author 1"]
            },
            {
                "chunk_text": "Second chunk",
                "chunk_index": 1,
                "embedding": [0.2] * 1536,
                "title": "Linked Paper",
                "authors": ["Author 1"]
            }
        ]

        result = qdrant_client.add_linked_paper_vectors(
            paper_id="linked123",
            chunks=chunks,
            source_paper_id="source456"
        )

        assert result == 2
        qdrant_client.client.upload_points.assert_called_once()

        # Verify upload was to linked_papers collection
        call_args = qdrant_client.client.upload_points.call_args
        assert call_args.kwargs['collection_name'] == "linked_papers"

        # Verify points have source_paper_id
        points = call_args.kwargs['points']
        assert all(p.payload.get("source_paper_id") == "source456" for p in points)
        assert all(p.payload.get("is_linked_paper") == True for p in points)

    def test_search_linked_papers(self, qdrant_client):
        """Test searching linked papers."""
        mock_result = Mock()
        mock_result.id = 456
        mock_result.score = 0.75
        mock_result.payload = {
            "paper_id": "linked123",
            "source_paper_id": "source456",
            "chunk_text": "Linked paper content",
            "is_linked_paper": True
        }

        qdrant_client.client.search.return_value = [mock_result]

        results = qdrant_client.search_linked_papers(
            query_vector=[0.1] * 1536,
            limit=5
        )

        assert len(results) == 1
        assert results[0]["payload"]["paper_id"] == "linked123"
        assert results[0]["payload"]["source_paper_id"] == "source456"

        # Verify search was on linked_papers collection
        call_args = qdrant_client.client.search.call_args
        assert call_args.kwargs['collection_name'] == "linked_papers"

    def test_search_linked_papers_with_source_filter(self, qdrant_client):
        """Test searching linked papers filtered by source paper."""
        qdrant_client.client.search.return_value = []

        qdrant_client.search_linked_papers(
            query_vector=[0.1] * 1536,
            limit=5,
            source_paper_id="source456"
        )

        # Verify filter was applied
        call_args = qdrant_client.client.search.call_args
        assert call_args.kwargs['query_filter'] is not None


class TestGetAllCollectionsInfo:
    """Tests for getting info about all collections."""

    def test_get_all_collections_info(self, qdrant_client):
        """Test getting info about all context engineering collections."""
        # Mock collection info
        mock_info = Mock()
        mock_info.name = "research_vectors"
        mock_info.vectors_count = 100
        mock_info.points_count = 100
        mock_info.status.value = "green"

        qdrant_client.client.get_collection.return_value = mock_info

        info = qdrant_client.get_all_collections_info()

        # Should have info for all three collection types
        assert "research_papers" in info
        assert "paper_summaries" in info
        assert "linked_papers" in info

    def test_get_all_collections_info_handles_missing(self, qdrant_client):
        """Test that missing collections are handled gracefully."""
        qdrant_client.client.get_collection.side_effect = Exception("Collection not found")

        info = qdrant_client.get_all_collections_info()

        # Should still return info dict with error markers
        assert "research_papers" in info
        assert info["research_papers"].get("exists") == False
