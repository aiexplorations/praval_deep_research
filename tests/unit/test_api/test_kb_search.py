"""
Unit tests for the Knowledge Base Hybrid Search API endpoints.

This module tests the KB search functionality including:
- Hybrid search (BM25 + Vector with RRF fusion)
- Search mode detection based on alpha parameter
- Paper chunk retrieval
- Start paper chat endpoint
- Error handling
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from agentic_research.api.main import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_hybrid_search():
    """Mock the hybrid search instance."""
    with patch('agentic_research.api.routes.kb_search.get_hybrid_search') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_paper_index():
    """Mock the paper index instance."""
    with patch('agentic_research.api.routes.kb_search.get_paper_index') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_conversation_store():
    """Mock the conversation store."""
    with patch('agentic_research.api.routes.kb_search.get_conversation_store') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


class TestKBSearchEndpoint:
    """Test suite for /kb-search endpoint."""

    def test_kb_search_hybrid_mode(self, client, mock_hybrid_search):
        """Test KB search with default hybrid alpha (0.5)."""
        # Setup mock results
        mock_result = MagicMock()
        mock_result.paper_id = "paper123"
        mock_result.title = "Test Paper on Neural Networks"
        mock_result.authors = ["Author A", "Author B"]
        mock_result.categories = ["cs.AI", "cs.LG"]
        mock_result.abstract = "This paper explores neural network architectures."
        mock_result.combined_score = 0.85
        mock_result.bm25_score = 8.5
        mock_result.bm25_rank = 1
        mock_result.vector_score = 0.92
        mock_result.vector_rank = 2
        mock_result.matching_chunks = 5

        mock_hybrid_search.search.return_value = [mock_result]
        mock_hybrid_search.get_search_mode.return_value = "hybrid"

        response = client.post("/kb-search", json={
            "query": "neural network architectures",
            "top_k": 10,
            "alpha": 0.5
        })

        assert response.status_code == 200
        data = response.json()

        assert data["query"] == "neural network architectures"
        assert data["search_mode"] == "hybrid"
        assert data["alpha"] == 0.5
        assert len(data["results"]) == 1
        assert data["total_found"] == 1

        result = data["results"][0]
        assert result["paper_id"] == "paper123"
        assert result["title"] == "Test Paper on Neural Networks"
        assert result["combined_score"] == 0.85
        assert result["bm25_score"] == 8.5
        assert result["vector_score"] == 0.92

    def test_kb_search_keyword_mode(self, client, mock_hybrid_search):
        """Test KB search with keyword-heavy alpha (1.0)."""
        mock_result = MagicMock()
        mock_result.paper_id = "paper456"
        mock_result.title = "BM25 Keyword Paper"
        mock_result.authors = ["Author C"]
        mock_result.categories = ["cs.IR"]
        mock_result.abstract = "A paper about information retrieval."
        mock_result.combined_score = 0.95
        mock_result.bm25_score = 12.3
        mock_result.bm25_rank = 1
        mock_result.vector_score = None
        mock_result.vector_rank = None
        mock_result.matching_chunks = 3

        mock_hybrid_search.search.return_value = [mock_result]
        mock_hybrid_search.get_search_mode.return_value = "keyword"

        response = client.post("/kb-search", json={
            "query": "information retrieval algorithms",
            "alpha": 1.0
        })

        assert response.status_code == 200
        data = response.json()

        assert data["search_mode"] == "keyword"
        assert data["alpha"] == 1.0
        assert data["results"][0]["bm25_score"] == 12.3
        assert data["results"][0]["vector_score"] is None

    def test_kb_search_semantic_mode(self, client, mock_hybrid_search):
        """Test KB search with semantic-heavy alpha (0.0)."""
        mock_result = MagicMock()
        mock_result.paper_id = "paper789"
        mock_result.title = "Semantic Vector Paper"
        mock_result.authors = ["Author D"]
        mock_result.categories = ["cs.CL"]
        mock_result.abstract = "A paper about semantic embeddings."
        mock_result.combined_score = 0.88
        mock_result.bm25_score = None
        mock_result.bm25_rank = None
        mock_result.vector_score = 0.95
        mock_result.vector_rank = 1
        mock_result.matching_chunks = 2

        mock_hybrid_search.search.return_value = [mock_result]
        mock_hybrid_search.get_search_mode.return_value = "semantic"

        response = client.post("/kb-search", json={
            "query": "semantic similarity models",
            "alpha": 0.0
        })

        assert response.status_code == 200
        data = response.json()

        assert data["search_mode"] == "semantic"
        assert data["alpha"] == 0.0
        assert data["results"][0]["bm25_score"] is None
        assert data["results"][0]["vector_score"] == 0.95

    def test_kb_search_empty_query(self, client):
        """Test KB search with empty query returns error."""
        response = client.post("/kb-search", json={
            "query": "",
            "top_k": 10
        })

        assert response.status_code == 422

    def test_kb_search_no_results(self, client, mock_hybrid_search):
        """Test KB search with no matching results."""
        mock_hybrid_search.search.return_value = []
        mock_hybrid_search.get_search_mode.return_value = "hybrid"

        response = client.post("/kb-search", json={
            "query": "nonexistent topic xyz123"
        })

        assert response.status_code == 200
        data = response.json()

        assert data["total_found"] == 0
        assert len(data["results"]) == 0

    def test_kb_search_with_category_filter(self, client, mock_hybrid_search):
        """Test KB search with category filtering."""
        mock_result = MagicMock()
        mock_result.paper_id = "paper_filtered"
        mock_result.title = "Filtered Paper"
        mock_result.authors = ["Author E"]
        mock_result.categories = ["cs.AI"]
        mock_result.abstract = "An AI paper."
        mock_result.combined_score = 0.75
        mock_result.bm25_score = 6.0
        mock_result.bm25_rank = 1
        mock_result.vector_score = 0.80
        mock_result.vector_rank = 1
        mock_result.matching_chunks = 4

        mock_hybrid_search.search.return_value = [mock_result]
        mock_hybrid_search.get_search_mode.return_value = "hybrid"

        response = client.post("/kb-search", json={
            "query": "machine learning",
            "categories": ["cs.AI", "cs.LG"]
        })

        assert response.status_code == 200
        mock_hybrid_search.search.assert_called_once()
        call_kwargs = mock_hybrid_search.search.call_args[1]
        assert call_kwargs.get("categories") == ["cs.AI", "cs.LG"]

    def test_kb_search_with_paper_ids_filter(self, client, mock_hybrid_search):
        """Test KB search filtering by specific paper IDs."""
        mock_result = MagicMock()
        mock_result.paper_id = "specific_paper"
        mock_result.title = "Specific Paper"
        mock_result.authors = ["Author F"]
        mock_result.categories = ["cs.CV"]
        mock_result.abstract = "A specific paper."
        mock_result.combined_score = 0.90
        mock_result.bm25_score = 9.0
        mock_result.bm25_rank = 1
        mock_result.vector_score = 0.88
        mock_result.vector_rank = 1
        mock_result.matching_chunks = 6

        mock_hybrid_search.search.return_value = [mock_result]
        mock_hybrid_search.get_search_mode.return_value = "hybrid"

        response = client.post("/kb-search", json={
            "query": "image recognition",
            "paper_ids": ["specific_paper", "other_paper"]
        })

        assert response.status_code == 200
        mock_hybrid_search.search.assert_called_once()
        call_kwargs = mock_hybrid_search.search.call_args[1]
        assert call_kwargs.get("paper_ids") == ["specific_paper", "other_paper"]


class TestKBSearchStatsEndpoint:
    """Test suite for /kb-search/stats endpoint."""

    def test_get_stats_success(self, client, mock_hybrid_search):
        """Test getting KB search statistics."""
        mock_hybrid_search.get_stats.return_value = {
            "total_papers": 150,
            "total_chunks": 2500,
            "bm25_ready": True,
            "vector_ready": True,
            "last_indexed": "2025-01-14T10:30:00Z"
        }

        response = client.get("/kb-search/stats")

        assert response.status_code == 200
        data = response.json()

        assert data["total_papers"] == 150
        assert data["total_chunks"] == 2500
        assert data["bm25_ready"] is True
        assert data["vector_ready"] is True


class TestStartPaperChatEndpoint:
    """Test suite for /kb-search/start-chat endpoint."""

    def test_start_chat_success(self, client, mock_paper_index, mock_conversation_store):
        """Test starting a chat with selected papers."""
        # Mock paper chunks
        mock_chunk = MagicMock()
        mock_chunk.metadata = {"title": "Test Paper Title"}
        mock_paper_index.get_paper_chunks.return_value = [mock_chunk]

        # Mock conversation creation
        mock_conversation_store.create_conversation = AsyncMock(return_value={
            "id": "conv123",
            "title": "Chat with Test Paper Title",
            "created_at": datetime.now().isoformat()
        })
        mock_conversation_store.update_conversation_metadata = AsyncMock(return_value=True)

        response = client.post("/kb-search/start-chat", json={
            "paper_ids": ["paper1", "paper2"],
            "initial_question": "What are the main findings?"
        })

        assert response.status_code == 200
        data = response.json()

        assert "conversation_id" in data
        assert data["paper_count"] == 2
        assert "redirect_url" in data
        assert "chat" in data["redirect_url"]

    def test_start_chat_no_papers(self, client):
        """Test starting chat with no papers returns error."""
        response = client.post("/kb-search/start-chat", json={
            "paper_ids": []
        })

        assert response.status_code == 422

    def test_start_chat_single_paper(self, client, mock_paper_index, mock_conversation_store):
        """Test starting chat with a single paper."""
        mock_chunk = MagicMock()
        mock_chunk.metadata = {"title": "Single Paper"}
        mock_paper_index.get_paper_chunks.return_value = [mock_chunk]

        mock_conversation_store.create_conversation = AsyncMock(return_value={
            "id": "conv456",
            "title": "Chat with Single Paper",
            "created_at": datetime.now().isoformat()
        })
        mock_conversation_store.update_conversation_metadata = AsyncMock(return_value=True)

        response = client.post("/kb-search/start-chat", json={
            "paper_ids": ["single_paper"]
        })

        assert response.status_code == 200
        data = response.json()
        assert data["paper_count"] == 1


class TestKBSearchErrorHandling:
    """Test error handling in KB search endpoints."""

    def test_search_internal_error(self, client, mock_hybrid_search):
        """Test handling of internal search errors."""
        mock_hybrid_search.search.side_effect = Exception("Database connection failed")

        response = client.post("/kb-search", json={
            "query": "test query"
        })

        assert response.status_code == 500

    def test_invalid_alpha_value(self, client):
        """Test validation of alpha parameter bounds."""
        # Alpha > 1.0 should be invalid
        response = client.post("/kb-search", json={
            "query": "test",
            "alpha": 1.5
        })
        assert response.status_code == 422

        # Alpha < 0.0 should be invalid
        response = client.post("/kb-search", json={
            "query": "test",
            "alpha": -0.5
        })
        assert response.status_code == 422

    def test_invalid_top_k_value(self, client):
        """Test validation of top_k parameter."""
        response = client.post("/kb-search", json={
            "query": "test",
            "top_k": -5
        })
        assert response.status_code == 422

        response = client.post("/kb-search", json={
            "query": "test",
            "top_k": 0
        })
        assert response.status_code == 422
