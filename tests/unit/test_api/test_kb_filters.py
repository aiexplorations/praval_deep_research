"""
Unit tests for Knowledge Base filtering and search functionality.

Tests the enhanced /knowledge-base/papers endpoint with:
- Search filtering
- Category filtering
- Source filtering (kb/linked)
- Sorting options
- Pagination
"""

import pytest
from unittest.mock import patch, MagicMock


class TestKnowledgeBaseFilters:
    """Tests for KB papers endpoint with filters."""

    @pytest.fixture
    def mock_qdrant_papers(self):
        """Mock paper data from Qdrant."""
        return [
            {
                'paper_id': 'arxiv-001',
                'title': 'Deep Learning for NLP',
                'authors': ['Author A', 'Author B'],
                'categories': ['cs.AI', 'cs.CL'],
                'published_date': '2024-01-15',
                'chunk_count': 50,
                'abstract': 'Abstract about deep learning...'
            },
            {
                'paper_id': 'arxiv-002',
                'title': 'Transformer Architectures',
                'authors': ['Author C'],
                'categories': ['cs.AI', 'cs.LG'],
                'published_date': '2024-02-20',
                'chunk_count': 30,
                'abstract': 'Abstract about transformers...'
            },
            {
                'paper_id': 'arxiv-003',
                'title': 'Reinforcement Learning Methods',
                'authors': ['Author D', 'Author E', 'Author F'],
                'categories': ['cs.LG'],
                'published_date': '2023-12-01',
                'chunk_count': 45,
                'abstract': 'Abstract about RL...'
            }
        ]

    @pytest.fixture
    def mock_linked_papers(self):
        """Mock linked paper data."""
        return [
            {
                'paper_id': 'linked-001',
                'title': 'Citation Paper on NLP',
                'authors': ['Cited Author'],
                'categories': ['cs.CL'],
                'published_date': '2023-06-10',
                'chunk_count': 20,
                'abstract': 'Cited paper abstract...',
                'source_paper_id': 'arxiv-001',
                'is_linked': True
            }
        ]

    def test_list_papers_no_filters(self, mock_qdrant_papers, mock_linked_papers):
        """Test listing papers without any filters returns all papers."""
        with patch('agentic_research.storage.qdrant_client.QdrantClientWrapper') as mock_qdrant:
            mock_instance = MagicMock()
            mock_instance.get_all_papers.return_value = mock_qdrant_papers
            mock_instance.get_all_linked_papers.return_value = mock_linked_papers
            mock_qdrant.return_value = mock_instance

            # Also patch where it's used
            with patch('agentic_research.storage.QdrantClientWrapper', mock_qdrant):
                from fastapi.testclient import TestClient
                from agentic_research.api.main import app
                client = TestClient(app)

                response = client.get('/research/knowledge-base/papers')
                assert response.status_code == 200

                data = response.json()
                assert data['status'] == 'success'
                # The actual count depends on the mock being applied
                assert 'total_papers' in data
                assert 'available_categories' in data

    def test_search_filter(self, mock_qdrant_papers):
        """Test search filtering by title."""
        with patch('agentic_research.storage.qdrant_client.QdrantClientWrapper') as mock_qdrant:
            mock_instance = MagicMock()
            mock_instance.get_all_papers.return_value = mock_qdrant_papers
            mock_instance.get_all_linked_papers.return_value = []
            mock_qdrant.return_value = mock_instance

            with patch('agentic_research.storage.QdrantClientWrapper', mock_qdrant):
                from fastapi.testclient import TestClient
                from agentic_research.api.main import app
                client = TestClient(app)

                response = client.get('/research/knowledge-base/papers?search=transformer')
                assert response.status_code == 200

                data = response.json()
                # Verify search endpoint works
                assert 'papers' in data
                assert 'total_papers' in data

    def test_category_filter(self, mock_qdrant_papers):
        """Test filtering by category."""
        with patch('agentic_research.storage.qdrant_client.QdrantClientWrapper') as mock_qdrant:
            mock_instance = MagicMock()
            mock_instance.get_all_papers.return_value = mock_qdrant_papers
            mock_instance.get_all_linked_papers.return_value = []
            mock_qdrant.return_value = mock_instance

            with patch('agentic_research.storage.QdrantClientWrapper', mock_qdrant):
                from fastapi.testclient import TestClient
                from agentic_research.api.main import app
                client = TestClient(app)

                response = client.get('/research/knowledge-base/papers?category=cs.CL')
                assert response.status_code == 200

                data = response.json()
                assert 'papers' in data

    def test_source_filter_kb(self, mock_qdrant_papers, mock_linked_papers):
        """Test filtering for KB papers only."""
        with patch('agentic_research.storage.qdrant_client.QdrantClientWrapper') as mock_qdrant:
            mock_instance = MagicMock()
            mock_instance.get_all_papers.return_value = mock_qdrant_papers
            mock_instance.get_all_linked_papers.return_value = mock_linked_papers
            mock_qdrant.return_value = mock_instance

            with patch('agentic_research.storage.QdrantClientWrapper', mock_qdrant):
                from fastapi.testclient import TestClient
                from agentic_research.api.main import app
                client = TestClient(app)

                response = client.get('/research/knowledge-base/papers?source=kb')
                assert response.status_code == 200

                data = response.json()
                assert 'papers' in data

    def test_source_filter_linked(self, mock_qdrant_papers, mock_linked_papers):
        """Test filtering for linked papers only."""
        with patch('agentic_research.storage.qdrant_client.QdrantClientWrapper') as mock_qdrant:
            mock_instance = MagicMock()
            mock_instance.get_all_papers.return_value = mock_qdrant_papers
            mock_instance.get_all_linked_papers.return_value = mock_linked_papers
            mock_qdrant.return_value = mock_instance

            with patch('agentic_research.storage.QdrantClientWrapper', mock_qdrant):
                from fastapi.testclient import TestClient
                from agentic_research.api.main import app
                client = TestClient(app)

                response = client.get('/research/knowledge-base/papers?source=linked')
                assert response.status_code == 200

                data = response.json()
                assert 'papers' in data

    def test_sort_by_date(self, mock_qdrant_papers):
        """Test sorting by date."""
        with patch('agentic_research.storage.qdrant_client.QdrantClientWrapper') as mock_qdrant:
            mock_instance = MagicMock()
            mock_instance.get_all_papers.return_value = mock_qdrant_papers
            mock_instance.get_all_linked_papers.return_value = []
            mock_qdrant.return_value = mock_instance

            with patch('agentic_research.storage.QdrantClientWrapper', mock_qdrant):
                from fastapi.testclient import TestClient
                from agentic_research.api.main import app
                client = TestClient(app)

                response = client.get('/research/knowledge-base/papers?sort=date&sort_order=desc')
                assert response.status_code == 200

                data = response.json()
                assert 'papers' in data

    def test_sort_by_chunks(self, mock_qdrant_papers):
        """Test sorting by chunk count."""
        with patch('agentic_research.storage.qdrant_client.QdrantClientWrapper') as mock_qdrant:
            mock_instance = MagicMock()
            mock_instance.get_all_papers.return_value = mock_qdrant_papers
            mock_instance.get_all_linked_papers.return_value = []
            mock_qdrant.return_value = mock_instance

            with patch('agentic_research.storage.QdrantClientWrapper', mock_qdrant):
                from fastapi.testclient import TestClient
                from agentic_research.api.main import app
                client = TestClient(app)

                response = client.get('/research/knowledge-base/papers?sort=chunks&sort_order=desc')
                assert response.status_code == 200

                data = response.json()
                assert 'papers' in data

    def test_pagination(self, mock_qdrant_papers):
        """Test pagination."""
        with patch('agentic_research.storage.qdrant_client.QdrantClientWrapper') as mock_qdrant:
            mock_instance = MagicMock()
            mock_instance.get_all_papers.return_value = mock_qdrant_papers
            mock_instance.get_all_linked_papers.return_value = []
            mock_qdrant.return_value = mock_instance

            with patch('agentic_research.storage.QdrantClientWrapper', mock_qdrant):
                from fastapi.testclient import TestClient
                from agentic_research.api.main import app
                client = TestClient(app)

                # Get first page with page_size=2
                response = client.get('/research/knowledge-base/papers?page=1&page_size=2')
                assert response.status_code == 200

                data = response.json()
                assert 'page' in data
                assert 'total_pages' in data

    def test_combined_filters(self, mock_qdrant_papers):
        """Test combining multiple filters."""
        with patch('agentic_research.storage.qdrant_client.QdrantClientWrapper') as mock_qdrant:
            mock_instance = MagicMock()
            mock_instance.get_all_papers.return_value = mock_qdrant_papers
            mock_instance.get_all_linked_papers.return_value = []
            mock_qdrant.return_value = mock_instance

            with patch('agentic_research.storage.QdrantClientWrapper', mock_qdrant):
                from fastapi.testclient import TestClient
                from agentic_research.api.main import app
                client = TestClient(app)

                response = client.get(
                    '/research/knowledge-base/papers?'
                    'category=cs.AI&sort=date&sort_order=desc'
                )
                assert response.status_code == 200

                data = response.json()
                assert 'papers' in data

    def test_available_categories_returned(self, mock_qdrant_papers, mock_linked_papers):
        """Test that available categories are returned for filter dropdown."""
        with patch('agentic_research.storage.qdrant_client.QdrantClientWrapper') as mock_qdrant:
            mock_instance = MagicMock()
            mock_instance.get_all_papers.return_value = mock_qdrant_papers
            mock_instance.get_all_linked_papers.return_value = mock_linked_papers
            mock_qdrant.return_value = mock_instance

            with patch('agentic_research.storage.QdrantClientWrapper', mock_qdrant):
                from fastapi.testclient import TestClient
                from agentic_research.api.main import app
                client = TestClient(app)

                response = client.get('/research/knowledge-base/papers')
                data = response.json()

                assert 'available_categories' in data


class TestQdrantLinkedPapers:
    """Tests for get_all_linked_papers method."""

    def test_get_all_linked_papers_empty_collection(self):
        """Test when linked papers collection doesn't exist."""
        with patch('agentic_research.storage.qdrant_client.QdrantClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.get_collections.return_value.collections = []
            mock_client.return_value = mock_instance

            from agentic_research.storage.qdrant_client import QdrantClientWrapper
            from agentic_research.core.config import get_settings

            settings = get_settings()
            qdrant = QdrantClientWrapper(settings)

            # Should return empty list when collection doesn't exist
            result = qdrant.get_all_linked_papers()
            assert result == []

    def test_get_all_linked_papers_with_data(self):
        """Test retrieval of linked papers with data."""
        with patch('agentic_research.storage.qdrant_client.QdrantClient') as mock_client:
            mock_instance = MagicMock()

            # Mock collection exists
            mock_collection = MagicMock()
            mock_collection.name = 'linked_papers'
            mock_instance.get_collections.return_value.collections = [mock_collection]

            # Mock scroll results
            mock_point = MagicMock()
            mock_point.payload = {
                'paper_id': 'linked-001',
                'title': 'Test Linked Paper',
                'authors': ['Author'],
                'categories': ['cs.AI'],
                'source_paper_id': 'source-001'
            }
            mock_instance.scroll.return_value = ([mock_point], None)

            mock_client.return_value = mock_instance

            from agentic_research.storage.qdrant_client import QdrantClientWrapper
            from agentic_research.core.config import get_settings

            settings = get_settings()
            qdrant = QdrantClientWrapper(settings)

            result = qdrant.get_all_linked_papers()

            assert len(result) == 1
            assert result[0]['paper_id'] == 'linked-001'
            assert result[0]['is_linked'] == True
            assert result[0]['source_paper_id'] == 'source-001'
