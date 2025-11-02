"""
Unit tests for VectorSearchClient.

Tests the vector search functionality including embedding generation
and similarity search with proper error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from agentic_research.storage.vector_search import VectorSearchClient


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock()
    settings.QDRANT_URL = "http://localhost:6333"
    settings.QDRANT_COLLECTION_NAME = "test_collection"
    settings.OPENAI_API_KEY = "test_key"
    settings.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
    settings.EMBEDDING_DIMENSIONS = 1536
    return settings


@pytest.fixture
def vector_client(mock_settings):
    """Create VectorSearchClient with mocked dependencies."""
    with patch('agentic_research.storage.vector_search.openai') as mock_openai, \
         patch('agentic_research.storage.vector_search.QdrantClient') as mock_qdrant:

        # Mock Qdrant client
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant.return_value = mock_qdrant_instance

        # Mock OpenAI
        mock_openai.api_key = "test_key"

        client = VectorSearchClient()
        client.settings = mock_settings
        client.client = mock_qdrant_instance

        yield client


def test_search_returns_empty_list_when_no_results(vector_client):
    """Test that search returns empty list when no results found."""
    # Mock empty search results
    vector_client.client.search.return_value = []

    # Mock embedding generation
    with patch.object(vector_client, 'generate_embedding') as mock_embed:
        mock_embed.return_value = [0.1] * 1536

        # Execute search
        results = vector_client.search("test query", top_k=5, score_threshold=0.5)

        # Verify
        assert results == []
        assert isinstance(results, list)


def test_search_returns_formatted_chunks(vector_client):
    """Test that search properly formats Qdrant results."""
    # Mock search result
    mock_result = Mock()
    mock_result.score = 0.85
    mock_result.payload = {
        "title": "Test Paper",
        "paper_id": "test123",
        "chunk_index": 0,
        "text": "Short text",
        "full_text": "This is the full text of the chunk",
        "extra_metadata": "value"
    }

    vector_client.client.search.return_value = [mock_result]

    # Mock embedding generation
    with patch.object(vector_client, 'generate_embedding') as mock_embed:
        mock_embed.return_value = [0.1] * 1536

        # Execute search
        results = vector_client.search("test query", top_k=5, score_threshold=0.5)

        # Verify structure
        assert len(results) == 1
        assert isinstance(results, list)

        chunk = results[0]
        assert chunk["title"] == "Test Paper"
        assert chunk["paper_id"] == "test123"
        assert chunk["chunk_index"] == 0
        assert chunk["text"] == "This is the full text of the chunk"
        assert chunk["excerpt"] == "Short text"
        assert chunk["relevance_score"] == 0.85
        assert "extra_metadata" in chunk["metadata"]


def test_search_handles_exception_gracefully(vector_client):
    """Test that search returns empty list on exception."""
    # Mock search to raise exception
    vector_client.client.search.side_effect = Exception("Qdrant connection error")

    # Mock embedding generation
    with patch.object(vector_client, 'generate_embedding') as mock_embed:
        mock_embed.return_value = [0.1] * 1536

        # Execute search
        results = vector_client.search("test query", top_k=5, score_threshold=0.5)

        # Verify it returns empty list, not None
        assert results == []
        assert isinstance(results, list)


def test_search_with_various_score_thresholds(vector_client):
    """Test search with different score thresholds."""
    # Create multiple mock results with different scores
    results_data = [
        (0.9, "High relevance"),
        (0.6, "Medium relevance"),
        (0.3, "Low relevance")
    ]

    mock_results = []
    for score, title in results_data:
        mock_result = Mock()
        mock_result.score = score
        mock_result.payload = {
            "title": title,
            "paper_id": f"paper_{score}",
            "chunk_index": 0,
            "text": f"Text with score {score}",
            "full_text": f"Full text with score {score}"
        }
        mock_results.append(mock_result)

    vector_client.client.search.return_value = mock_results

    # Mock embedding
    with patch.object(vector_client, 'generate_embedding') as mock_embed:
        mock_embed.return_value = [0.1] * 1536

        # Test with low threshold
        results = vector_client.search("test", top_k=10, score_threshold=0.2)
        assert len(results) == 3

        # Verify all scores are included
        scores = [r["relevance_score"] for r in results]
        assert scores == [0.9, 0.6, 0.3]


def test_generate_embedding_called_with_query(vector_client):
    """Test that generate_embedding is called with the search query."""
    vector_client.client.search.return_value = []

    with patch.object(vector_client, 'generate_embedding') as mock_embed:
        mock_embed.return_value = [0.1] * 1536

        query = "What are transformers?"
        vector_client.search(query, top_k=5, score_threshold=0.5)

        # Verify embedding was generated for the query
        mock_embed.assert_called_once_with(query)


@pytest.mark.parametrize("query,top_k,threshold", [
    ("simple query", 5, 0.5),
    ("complex query about machine learning", 10, 0.3),
    ("short", 3, 0.7),
])
def test_search_parameters(vector_client, query, top_k, threshold):
    """Test that search parameters are passed correctly to Qdrant."""
    vector_client.client.search.return_value = []

    with patch.object(vector_client, 'generate_embedding') as mock_embed:
        mock_embed.return_value = [0.1] * 1536

        vector_client.search(query, top_k=top_k, score_threshold=threshold)

        # Verify Qdrant search was called with correct parameters
        call_args = vector_client.client.search.call_args
        assert call_args.kwargs['limit'] == top_k
        assert call_args.kwargs['score_threshold'] == threshold
        assert call_args.kwargs['collection_name'] == "test_collection"
