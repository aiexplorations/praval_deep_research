"""
Unit tests for vector search module.

Tests the VectorSearchClient class for Qdrant vector storage and similarity search.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any

from qdrant_client.models import PointStruct, Distance, VectorParams
from agentic_research.storage.vector_search import VectorSearchClient, get_vector_search_client


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock()
    settings.QDRANT_URL = "http://localhost:6333"
    settings.QDRANT_COLLECTION_NAME = "test_collection"
    settings.OPENAI_API_KEY = "test-api-key"
    settings.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
    settings.EMBEDDING_DIMENSIONS = 1536
    return settings


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    client = Mock()

    # Mock get_collections response
    mock_collection = Mock()
    mock_collection.name = "test_collection"
    mock_collections = Mock()
    mock_collections.collections = [mock_collection]
    client.get_collections.return_value = mock_collections

    return client


@pytest.fixture
def vector_search_client(mock_settings, mock_qdrant_client):
    """Create VectorSearchClient with mocked dependencies."""
    with patch('agentic_research.storage.vector_search.QdrantClient') as mock_qc, \
         patch('agentic_research.storage.vector_search.get_settings') as mock_gs, \
         patch('agentic_research.storage.vector_search.openai'):

        mock_gs.return_value = mock_settings
        mock_qc.return_value = mock_qdrant_client

        client = VectorSearchClient()
        client.client = mock_qdrant_client

        return client


class TestVectorSearchClient:
    """Test suite for VectorSearchClient class."""

    def test_initialization(self, mock_settings, mock_qdrant_client):
        """Test that VectorSearchClient initializes correctly."""
        with patch('agentic_research.storage.vector_search.QdrantClient') as mock_qc, \
             patch('agentic_research.storage.vector_search.get_settings') as mock_gs, \
             patch('agentic_research.storage.vector_search.openai'):

            mock_gs.return_value = mock_settings
            mock_qc.return_value = mock_qdrant_client

            client = VectorSearchClient()

            assert client.collection_name == "test_collection"
            assert client.embedding_model == "text-embedding-3-small"
            assert client.embedding_dim == 1536
            mock_qdrant_client.get_collections.assert_called_once()

    def test_ensure_collection_creates_if_not_exists(self, mock_settings):
        """Test that collection is created if it doesn't exist."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []  # Empty - collection doesn't exist
        mock_client.get_collections.return_value = mock_collections

        with patch('agentic_research.storage.vector_search.QdrantClient') as mock_qc, \
             patch('agentic_research.storage.vector_search.get_settings') as mock_gs, \
             patch('agentic_research.storage.vector_search.openai'):

            mock_gs.return_value = mock_settings
            mock_qc.return_value = mock_client

            client = VectorSearchClient()

            # Verify collection creation was called
            mock_client.create_collection.assert_called_once()
            call_args = mock_client.create_collection.call_args
            assert call_args[1]['collection_name'] == "test_collection"
            assert call_args[1]['vectors_config'].size == 1536
            assert call_args[1]['vectors_config'].distance == Distance.COSINE

    def test_ensure_collection_skips_if_exists(self, mock_settings, mock_qdrant_client):
        """Test that collection creation is skipped if already exists."""
        with patch('agentic_research.storage.vector_search.QdrantClient') as mock_qc, \
             patch('agentic_research.storage.vector_search.get_settings') as mock_gs, \
             patch('agentic_research.storage.vector_search.openai'):

            mock_gs.return_value = mock_settings
            mock_qc.return_value = mock_qdrant_client

            client = VectorSearchClient()

            # Verify collection creation was NOT called
            mock_qdrant_client.create_collection.assert_not_called()

    def test_generate_embedding_success(self, vector_search_client):
        """Test successful embedding generation."""
        test_text = "Test document for embedding"
        mock_embedding = [0.1] * 1536

        with patch('agentic_research.storage.vector_search.openai.embeddings.create') as mock_create:
            mock_response = Mock()
            mock_response.data = [Mock(embedding=mock_embedding)]
            mock_create.return_value = mock_response

            result = vector_search_client.generate_embedding(test_text)

            assert result == mock_embedding
            assert len(result) == 1536
            mock_create.assert_called_once_with(
                model="text-embedding-3-small",
                input=test_text
            )

    def test_add_paper_chunks_success(self, vector_search_client):
        """Test successful addition of paper chunks."""
        paper_id = "arxiv:2301.12345"
        title = "Test Paper Title"
        chunks = [
            {"text": "First chunk of text", "metadata": {"page": 1}},
            {"text": "Second chunk of text", "metadata": {"page": 2}},
        ]

        mock_embeddings = [[0.1] * 1536, [0.2] * 1536]

        with patch.object(vector_search_client, 'generate_embedding', side_effect=mock_embeddings):
            vector_search_client.add_paper_chunks(paper_id, title, chunks)

            # Verify upsert was called
            vector_search_client.client.upsert.assert_called_once()
            call_args = vector_search_client.client.upsert.call_args

            assert call_args[1]['collection_name'] == "test_collection"

            points = call_args[1]['points']
            assert len(points) == 2

            # Verify point IDs are positive integers
            for point in points:
                assert isinstance(point.id, int)
                assert point.id > 0
                assert point.id < 2**63  # Within valid range

            # Verify point structure
            assert points[0].vector == mock_embeddings[0]
            assert points[0].payload['paper_id'] == paper_id
            assert points[0].payload['title'] == title
            assert points[0].payload['chunk_index'] == 0
            assert points[0].payload['text'] == "First chunk of text"
            assert points[0].payload['page'] == 1

            assert points[1].vector == mock_embeddings[1]
            assert points[1].payload['chunk_index'] == 1

    def test_add_paper_chunks_with_long_text(self, vector_search_client):
        """Test that long text is properly truncated in payload."""
        paper_id = "arxiv:2301.12345"
        title = "Test Paper"
        long_text = "x" * 2000  # 2000 characters
        chunks = [{"text": long_text}]

        mock_embedding = [0.1] * 1536

        with patch.object(vector_search_client, 'generate_embedding', return_value=mock_embedding):
            vector_search_client.add_paper_chunks(paper_id, title, chunks)

            call_args = vector_search_client.client.upsert.call_args
            points = call_args[1]['points']

            # Verify text field is truncated to 1000 chars
            assert len(points[0].payload['text']) == 1000
            # But full_text contains everything
            assert len(points[0].payload['full_text']) == 2000

    def test_add_paper_chunks_skips_empty_chunks(self, vector_search_client):
        """Test that empty chunks are skipped."""
        paper_id = "arxiv:2301.12345"
        title = "Test Paper"
        chunks = [
            {"text": "Valid chunk"},
            {"text": ""},  # Empty
            {"text": "Another valid chunk"},
            {},  # No text key
        ]

        mock_embedding = [0.1] * 1536

        with patch.object(vector_search_client, 'generate_embedding', return_value=mock_embedding):
            vector_search_client.add_paper_chunks(paper_id, title, chunks)

            # Only 2 valid chunks should be processed
            assert vector_search_client.generate_embedding.call_count == 2

            call_args = vector_search_client.client.upsert.call_args
            points = call_args[1]['points']
            assert len(points) == 2

    def test_add_paper_chunks_no_valid_chunks(self, vector_search_client):
        """Test that nothing is uploaded when no valid chunks exist."""
        paper_id = "arxiv:2301.12345"
        title = "Test Paper"
        chunks = [{"text": ""}, {}]  # All invalid

        vector_search_client.add_paper_chunks(paper_id, title, chunks)

        # Verify upsert was NOT called
        vector_search_client.client.upsert.assert_not_called()

    def test_add_paper_chunks_id_generation(self, vector_search_client):
        """Test that point IDs are generated correctly and consistently."""
        paper_id = "arxiv:2301.12345"
        title = "Test Paper"
        chunks = [{"text": f"Chunk {i}"} for i in range(5)]

        mock_embedding = [0.1] * 1536

        with patch.object(vector_search_client, 'generate_embedding', return_value=mock_embedding):
            vector_search_client.add_paper_chunks(paper_id, title, chunks)

            call_args = vector_search_client.client.upsert.call_args
            points = call_args[1]['points']

            # Verify all IDs are unique
            ids = [p.id for p in points]
            assert len(ids) == len(set(ids)), "IDs should be unique"

            # Verify all IDs are positive
            assert all(id > 0 for id in ids), "All IDs should be positive"

            # Verify IDs are within valid range
            assert all(id < 2**63 for id in ids), "IDs should be within 64-bit signed int range"

    def test_search_success(self, vector_search_client):
        """Test successful similarity search."""
        query = "What is machine learning?"
        mock_query_embedding = [0.5] * 1536

        # Mock search results
        mock_result_1 = Mock()
        mock_result_1.payload = {
            "title": "ML Paper 1",
            "paper_id": "arxiv:2301.001",
            "chunk_index": 0,
            "text": "Short excerpt",
            "full_text": "Full text of first result",
            "page": 1
        }
        mock_result_1.score = 0.95

        mock_result_2 = Mock()
        mock_result_2.payload = {
            "title": "ML Paper 2",
            "paper_id": "arxiv:2301.002",
            "chunk_index": 1,
            "text": "Another excerpt",
            "full_text": "Full text of second result",
            "page": 2
        }
        mock_result_2.score = 0.88

        mock_results = [mock_result_1, mock_result_2]

        with patch.object(vector_search_client, 'generate_embedding', return_value=mock_query_embedding):
            vector_search_client.client.search.return_value = mock_results

            results = vector_search_client.search(query, top_k=2, score_threshold=0.7)

            # Verify search was called correctly
            vector_search_client.client.search.assert_called_once_with(
                collection_name="test_collection",
                query_vector=mock_query_embedding,
                limit=2,
                score_threshold=0.7
            )

            # Verify results formatting
            assert len(results) == 2

            assert results[0]['title'] == "ML Paper 1"
            assert results[0]['paper_id'] == "arxiv:2301.001"
            assert results[0]['chunk_index'] == 0
            assert results[0]['text'] == "Full text of first result"
            assert results[0]['excerpt'] == "Short excerpt"
            assert results[0]['relevance_score'] == 0.95
            assert results[0]['metadata']['page'] == 1

            assert results[1]['relevance_score'] == 0.88

    def test_search_no_results(self, vector_search_client):
        """Test search with no results."""
        query = "Nonexistent topic"
        mock_query_embedding = [0.5] * 1536

        with patch.object(vector_search_client, 'generate_embedding', return_value=mock_query_embedding):
            vector_search_client.client.search.return_value = []

            results = vector_search_client.search(query)

            assert results == []

    def test_search_handles_missing_full_text(self, vector_search_client):
        """Test that search handles missing full_text field gracefully."""
        query = "Test query"
        mock_query_embedding = [0.5] * 1536

        mock_result = Mock()
        mock_result.payload = {
            "title": "Test Paper",
            "paper_id": "arxiv:2301.001",
            "chunk_index": 0,
            "text": "Short text only",
            # No full_text field
        }
        mock_result.score = 0.9

        with patch.object(vector_search_client, 'generate_embedding', return_value=mock_query_embedding):
            vector_search_client.client.search.return_value = [mock_result]

            results = vector_search_client.search(query)

            # Should fallback to 'text' field
            assert results[0]['text'] == "Short text only"
            assert results[0]['excerpt'] == "Short text only"

    def test_search_error_handling(self, vector_search_client):
        """Test that search errors are handled gracefully."""
        query = "Test query"
        mock_query_embedding = [0.5] * 1536

        with patch.object(vector_search_client, 'generate_embedding', return_value=mock_query_embedding):
            vector_search_client.client.search.side_effect = Exception("Qdrant error")

            results = vector_search_client.search(query)

            # Should return empty list on error
            assert results == []

    def test_add_paper_chunks_error_handling(self, vector_search_client):
        """Test that chunk addition errors are raised."""
        paper_id = "arxiv:2301.12345"
        title = "Test Paper"
        chunks = [{"text": "Test chunk"}]

        with patch.object(vector_search_client, 'generate_embedding', side_effect=Exception("API error")):
            with pytest.raises(Exception, match="API error"):
                vector_search_client.add_paper_chunks(paper_id, title, chunks)


class TestGetVectorSearchClient:
    """Test the global client singleton."""

    def test_get_vector_search_client_creates_singleton(self):
        """Test that get_vector_search_client returns singleton instance."""
        with patch('agentic_research.storage.vector_search.VectorSearchClient') as mock_vsc:
            mock_instance = Mock()
            mock_vsc.return_value = mock_instance

            # First call creates instance
            client1 = get_vector_search_client()
            assert client1 == mock_instance
            assert mock_vsc.call_count == 1

            # Second call returns same instance
            client2 = get_vector_search_client()
            assert client2 == mock_instance
            assert mock_vsc.call_count == 1  # Not called again

            # Verify same instance
            assert client1 is client2

    def test_get_vector_search_client_thread_safety(self):
        """Test concurrent access to singleton."""
        import threading

        with patch('agentic_research.storage.vector_search.VectorSearchClient') as mock_vsc:
            mock_instance = Mock()
            mock_vsc.return_value = mock_instance

            clients = []

            def get_client():
                client = get_vector_search_client()
                clients.append(client)

            threads = [threading.Thread(target=get_client) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All should be the same instance
            assert all(c == mock_instance for c in clients)
            # Should only be created once
            assert mock_vsc.call_count == 1


@pytest.mark.integration
class TestVectorSearchIntegration:
    """Integration tests requiring actual Qdrant instance."""

    @pytest.mark.skip(reason="Requires running Qdrant instance")
    def test_real_vector_storage_and_search(self):
        """Test actual vector storage and retrieval with Qdrant."""
        import os
        from agentic_research.core.config import get_settings

        if not os.getenv("QDRANT_URL") or not os.getenv("OPENAI_API_KEY"):
            pytest.skip("QDRANT_URL or OPENAI_API_KEY not set")

        settings = get_settings()
        client = VectorSearchClient()

        # Add test chunks
        paper_id = "test:integration:001"
        title = "Integration Test Paper"
        chunks = [
            {"text": "Machine learning is a subset of artificial intelligence.", "metadata": {"page": 1}},
            {"text": "Deep learning uses neural networks with multiple layers.", "metadata": {"page": 2}},
        ]

        client.add_paper_chunks(paper_id, title, chunks)

        # Search for relevant content
        results = client.search("What is machine learning?", top_k=2, score_threshold=0.3)

        assert len(results) > 0
        assert results[0]['paper_id'] == paper_id
        assert results[0]['title'] == title
        assert 'machine learning' in results[0]['text'].lower()
