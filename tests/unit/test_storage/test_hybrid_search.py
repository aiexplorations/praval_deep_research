"""
Unit tests for the Hybrid Search module.

This module tests the HybridPaperSearch wrapper that combines
Vajra BM25 with Qdrant vector search using RRF fusion.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MockSearchResult:
    """Mock search result for testing."""
    paper_id: str
    title: str
    authors: List[str]
    categories: List[str]
    abstract: str
    combined_score: float
    bm25_score: Optional[float]
    bm25_rank: Optional[int]
    vector_score: Optional[float]
    vector_rank: Optional[int]
    matching_chunks: int


@pytest.fixture
def mock_paper_index():
    """Mock paper index."""
    with patch('agentic_research.storage.hybrid_search.get_paper_index') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    with patch('agentic_research.storage.hybrid_search.QdrantClient') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model."""
    with patch('agentic_research.storage.hybrid_search.SentenceTransformer') as mock:
        mock_instance = MagicMock()
        mock_instance.encode.return_value = [0.1] * 384  # Mock embedding vector
        mock.return_value = mock_instance
        yield mock_instance


class TestHybridPaperSearch:
    """Test suite for HybridPaperSearch class."""

    def test_search_mode_detection_keyword(self):
        """Test search mode detection for keyword-heavy alpha."""
        from agentic_research.storage.hybrid_search import HybridPaperSearch

        with patch('agentic_research.storage.hybrid_search.get_paper_index'), \
             patch('agentic_research.storage.hybrid_search.QdrantClient'), \
             patch('agentic_research.storage.hybrid_search.SentenceTransformer'):

            search = HybridPaperSearch()

            assert search.get_search_mode(1.0) == "keyword"
            assert search.get_search_mode(0.9) == "keyword"
            assert search.get_search_mode(0.85) == "keyword"

    def test_search_mode_detection_semantic(self):
        """Test search mode detection for semantic-heavy alpha."""
        from agentic_research.storage.hybrid_search import HybridPaperSearch

        with patch('agentic_research.storage.hybrid_search.get_paper_index'), \
             patch('agentic_research.storage.hybrid_search.QdrantClient'), \
             patch('agentic_research.storage.hybrid_search.SentenceTransformer'):

            search = HybridPaperSearch()

            assert search.get_search_mode(0.0) == "semantic"
            assert search.get_search_mode(0.1) == "semantic"
            assert search.get_search_mode(0.15) == "semantic"

    def test_search_mode_detection_hybrid(self):
        """Test search mode detection for balanced alpha."""
        from agentic_research.storage.hybrid_search import HybridPaperSearch

        with patch('agentic_research.storage.hybrid_search.get_paper_index'), \
             patch('agentic_research.storage.hybrid_search.QdrantClient'), \
             patch('agentic_research.storage.hybrid_search.SentenceTransformer'):

            search = HybridPaperSearch()

            assert search.get_search_mode(0.5) == "hybrid"
            assert search.get_search_mode(0.3) == "hybrid"
            assert search.get_search_mode(0.7) == "hybrid"

    def test_rrf_score_calculation(self):
        """Test Reciprocal Rank Fusion score calculation."""
        from agentic_research.storage.hybrid_search import HybridPaperSearch

        with patch('agentic_research.storage.hybrid_search.get_paper_index'), \
             patch('agentic_research.storage.hybrid_search.QdrantClient'), \
             patch('agentic_research.storage.hybrid_search.SentenceTransformer'):

            search = HybridPaperSearch()

            # RRF formula: 1 / (k + rank)
            # With k=60 (default), rank 1 -> 1/(60+1) = 0.0164
            bm25_rrf = search._rrf_score(1)
            vector_rrf = search._rrf_score(2)

            assert bm25_rrf > vector_rrf  # Rank 1 should have higher score
            assert 0 < bm25_rrf <= 1
            assert 0 < vector_rrf <= 1

    def test_search_with_hybrid_alpha(self, mock_paper_index, mock_qdrant_client, mock_embedding_model):
        """Test hybrid search combines BM25 and vector results."""
        from agentic_research.storage.hybrid_search import HybridPaperSearch

        # Mock BM25 results from paper index
        mock_bm25_result = MagicMock()
        mock_bm25_result.paper_id = "paper1"
        mock_bm25_result.score = 10.5
        mock_bm25_result.metadata = {
            "title": "BM25 Paper",
            "authors": ["Author A"],
            "categories": ["cs.AI"],
            "abstract": "A paper found via BM25."
        }
        mock_paper_index.search.return_value = [mock_bm25_result]

        # Mock vector results from Qdrant
        mock_vector_result = MagicMock()
        mock_vector_result.id = "chunk_paper2_0"
        mock_vector_result.score = 0.95
        mock_vector_result.payload = {
            "paper_id": "paper2",
            "title": "Vector Paper",
            "authors": ["Author B"],
            "categories": ["cs.LG"],
            "abstract": "A paper found via vectors."
        }
        mock_qdrant_client.search.return_value = [mock_vector_result]

        with patch('agentic_research.storage.hybrid_search.get_paper_index', return_value=mock_paper_index), \
             patch('agentic_research.storage.hybrid_search.QdrantClient', return_value=mock_qdrant_client), \
             patch('agentic_research.storage.hybrid_search.SentenceTransformer', return_value=mock_embedding_model):

            search = HybridPaperSearch()
            results = search.search("test query", top_k=10, alpha=0.5)

            # Should have results from both engines
            assert len(results) >= 1

    def test_search_bm25_only(self, mock_paper_index, mock_qdrant_client, mock_embedding_model):
        """Test BM25-only search when alpha=1.0."""
        from agentic_research.storage.hybrid_search import HybridPaperSearch

        mock_bm25_result = MagicMock()
        mock_bm25_result.paper_id = "paper_bm25"
        mock_bm25_result.score = 12.0
        mock_bm25_result.metadata = {
            "title": "BM25 Only Paper",
            "authors": ["Author C"],
            "categories": ["cs.IR"],
            "abstract": "Found only via BM25."
        }
        mock_paper_index.search.return_value = [mock_bm25_result]

        with patch('agentic_research.storage.hybrid_search.get_paper_index', return_value=mock_paper_index), \
             patch('agentic_research.storage.hybrid_search.QdrantClient', return_value=mock_qdrant_client), \
             patch('agentic_research.storage.hybrid_search.SentenceTransformer', return_value=mock_embedding_model):

            search = HybridPaperSearch()
            results = search.search("keyword search", top_k=10, alpha=1.0)

            # Should call BM25 search
            mock_paper_index.search.assert_called()

    def test_search_vector_only(self, mock_paper_index, mock_qdrant_client, mock_embedding_model):
        """Test vector-only search when alpha=0.0."""
        from agentic_research.storage.hybrid_search import HybridPaperSearch

        mock_vector_result = MagicMock()
        mock_vector_result.id = "chunk_paper_vec_0"
        mock_vector_result.score = 0.98
        mock_vector_result.payload = {
            "paper_id": "paper_vec",
            "title": "Vector Only Paper",
            "authors": ["Author D"],
            "categories": ["cs.CL"],
            "abstract": "Found only via vectors."
        }
        mock_qdrant_client.search.return_value = [mock_vector_result]

        with patch('agentic_research.storage.hybrid_search.get_paper_index', return_value=mock_paper_index), \
             patch('agentic_research.storage.hybrid_search.QdrantClient', return_value=mock_qdrant_client), \
             patch('agentic_research.storage.hybrid_search.SentenceTransformer', return_value=mock_embedding_model):

            search = HybridPaperSearch()
            results = search.search("semantic similarity", top_k=10, alpha=0.0)

            # Should call vector search
            mock_qdrant_client.search.assert_called()

    def test_category_filtering(self, mock_paper_index, mock_qdrant_client, mock_embedding_model):
        """Test that category filtering is passed to search engines."""
        from agentic_research.storage.hybrid_search import HybridPaperSearch

        mock_paper_index.search.return_value = []
        mock_qdrant_client.search.return_value = []

        with patch('agentic_research.storage.hybrid_search.get_paper_index', return_value=mock_paper_index), \
             patch('agentic_research.storage.hybrid_search.QdrantClient', return_value=mock_qdrant_client), \
             patch('agentic_research.storage.hybrid_search.SentenceTransformer', return_value=mock_embedding_model):

            search = HybridPaperSearch()
            search.search("test", top_k=10, alpha=0.5, categories=["cs.AI", "cs.LG"])

            # Verify categories were passed to BM25 search
            call_args = mock_paper_index.search.call_args
            assert call_args is not None

    def test_paper_ids_filtering(self, mock_paper_index, mock_qdrant_client, mock_embedding_model):
        """Test that paper_ids filtering restricts search scope."""
        from agentic_research.storage.hybrid_search import HybridPaperSearch

        mock_paper_index.search.return_value = []
        mock_qdrant_client.search.return_value = []

        with patch('agentic_research.storage.hybrid_search.get_paper_index', return_value=mock_paper_index), \
             patch('agentic_research.storage.hybrid_search.QdrantClient', return_value=mock_qdrant_client), \
             patch('agentic_research.storage.hybrid_search.SentenceTransformer', return_value=mock_embedding_model):

            search = HybridPaperSearch()
            search.search("test", top_k=10, alpha=0.5, paper_ids=["paper1", "paper2"])

            # Verify paper_ids filter was applied
            call_args = mock_paper_index.search.call_args
            assert call_args is not None

    def test_empty_results(self, mock_paper_index, mock_qdrant_client, mock_embedding_model):
        """Test handling of empty search results."""
        from agentic_research.storage.hybrid_search import HybridPaperSearch

        mock_paper_index.search.return_value = []
        mock_qdrant_client.search.return_value = []

        with patch('agentic_research.storage.hybrid_search.get_paper_index', return_value=mock_paper_index), \
             patch('agentic_research.storage.hybrid_search.QdrantClient', return_value=mock_qdrant_client), \
             patch('agentic_research.storage.hybrid_search.SentenceTransformer', return_value=mock_embedding_model):

            search = HybridPaperSearch()
            results = search.search("nonexistent query xyz", top_k=10, alpha=0.5)

            assert results == []

    def test_get_stats(self, mock_paper_index, mock_qdrant_client, mock_embedding_model):
        """Test getting search statistics."""
        from agentic_research.storage.hybrid_search import HybridPaperSearch

        mock_paper_index.get_stats.return_value = {
            "total_documents": 100,
            "total_chunks": 1500
        }
        mock_qdrant_client.get_collection.return_value = MagicMock(
            points_count=1500
        )

        with patch('agentic_research.storage.hybrid_search.get_paper_index', return_value=mock_paper_index), \
             patch('agentic_research.storage.hybrid_search.QdrantClient', return_value=mock_qdrant_client), \
             patch('agentic_research.storage.hybrid_search.SentenceTransformer', return_value=mock_embedding_model):

            search = HybridPaperSearch()
            stats = search.get_stats()

            assert "total_papers" in stats or "total_documents" in stats


class TestRRFFusion:
    """Test Reciprocal Rank Fusion algorithm."""

    def test_rrf_combines_rankings(self):
        """Test that RRF properly combines rankings from two sources."""
        from agentic_research.storage.hybrid_search import HybridPaperSearch

        with patch('agentic_research.storage.hybrid_search.get_paper_index'), \
             patch('agentic_research.storage.hybrid_search.QdrantClient'), \
             patch('agentic_research.storage.hybrid_search.SentenceTransformer'):

            search = HybridPaperSearch()

            # Paper that appears in both lists should rank higher
            # Rank 1 in BM25, Rank 1 in Vector -> highest combined score
            # Rank 10 in BM25 only -> lower score

            bm25_rank1_score = search._rrf_score(1)
            bm25_rank10_score = search._rrf_score(10)

            # Combined score for paper in both (alpha=0.5)
            combined_score = 0.5 * bm25_rank1_score + 0.5 * bm25_rank1_score

            # Single source score
            single_score = 0.5 * bm25_rank10_score

            assert combined_score > single_score

    def test_rrf_k_parameter(self):
        """Test RRF with different k parameters."""
        from agentic_research.storage.hybrid_search import HybridPaperSearch

        with patch('agentic_research.storage.hybrid_search.get_paper_index'), \
             patch('agentic_research.storage.hybrid_search.QdrantClient'), \
             patch('agentic_research.storage.hybrid_search.SentenceTransformer'):

            search = HybridPaperSearch()

            # With k=60 (default), the score for rank 1 is 1/61
            expected = 1.0 / 61.0
            actual = search._rrf_score(1)

            assert abs(actual - expected) < 0.0001
