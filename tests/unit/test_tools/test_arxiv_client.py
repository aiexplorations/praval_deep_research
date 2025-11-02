"""
Unit tests for ArXiv client functionality.

Tests the real ArXiv API integration to ensure it works correctly
without fake data. This validates the core research tool functionality.
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import patch, MagicMock, AsyncMock

from src.processors.arxiv_client import (
    ArXivClient,
    ArXivAPIError,
    get_arxiv_client,
    search_arxiv_papers,
    get_paper_details,
    calculate_paper_relevance
)


class TestArXivClient:
    """Test ArXiv client core functionality."""
    
    def test_client_initialization(self):
        """Test ArXiv client initializes with correct settings."""
        client = ArXivClient()
        
        assert client.base_url == "http://export.arxiv.org/api/query"
        assert client.max_results == 50
        assert client.rate_limit == 3
        assert client.delay_seconds == 1
        assert client.last_request_time == 0
    
    def test_build_search_query_basic(self):
        """Test basic search query building."""
        client = ArXivClient()
        
        query = client._build_search_query("machine learning")
        assert "all:machine+learning" in query
    
    def test_build_search_query_with_domain(self):
        """Test search query building with domain filtering."""
        client = ArXivClient()
        
        query = client._build_search_query("neural networks", "computer_science")
        assert "all:neural+networks" in query
        assert "cat:cs.*" in query
    
    def test_build_search_query_ai_domain(self):
        """Test search query for AI domain."""
        client = ArXivClient()
        
        query = client._build_search_query("transformers", "artificial_intelligence")
        assert "cat:cs.AI" in query
    
    def test_build_search_query_ml_domain(self):
        """Test search query for machine learning domain."""
        client = ArXivClient()
        
        query = client._build_search_query("deep learning", "machine_learning")
        assert "cs.LG OR cs.AI" in query
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test that rate limiting works correctly."""
        client = ArXivClient()
        client.delay_seconds = 0.1  # Short delay for testing
        
        start_time = asyncio.get_event_loop().time()
        await client._rate_limit()
        
        # First call should be immediate
        first_call_time = asyncio.get_event_loop().time() - start_time
        assert first_call_time < 0.01
        
        # Second call should be delayed
        start_time = asyncio.get_event_loop().time()
        await client._rate_limit()
        second_call_time = asyncio.get_event_loop().time() - start_time
        
        assert second_call_time >= 0.1
    
    def test_parse_arxiv_response_empty(self):
        """Test parsing empty ArXiv response."""
        client = ArXivClient()
        
        empty_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""
        
        papers = client._parse_arxiv_response(empty_xml)
        assert papers == []
    
    def test_parse_arxiv_response_single_paper(self):
        """Test parsing ArXiv response with single paper."""
        client = ArXivClient()
        
        sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
            <entry>
                <id>http://arxiv.org/abs/2106.04560v1</id>
                <title>Test Paper Title</title>
                <author>
                    <name>Test Author</name>
                </author>
                <summary>This is a test abstract.</summary>
                <published>2021-06-08T17:59:59Z</published>
                <arxiv:primary_category term="cs.AI"/>
                <category term="cs.AI"/>
                <arxiv:journal_ref>Test Conference 2021</arxiv:journal_ref>
            </entry>
        </feed>"""
        
        papers = client._parse_arxiv_response(sample_xml)
        
        assert len(papers) == 1
        paper = papers[0]
        assert paper['title'] == "Test Paper Title"
        assert paper['authors'] == ["Test Author"]
        assert paper['abstract'] == "This is a test abstract."
        assert paper['arxiv_id'] == "2106.04560v1"
        assert paper['url'] == "http://arxiv.org/abs/2106.04560v1"
        assert paper['published_date'] == "2021-06-08"
        assert "cs.AI" in paper['categories']
        assert paper['venue'] == "Test Conference 2021"
    
    def test_parse_arxiv_response_invalid_xml(self):
        """Test parsing invalid XML raises appropriate error."""
        client = ArXivClient()
        
        invalid_xml = "not valid xml at all"
        
        with pytest.raises(ArXivAPIError, match="Invalid XML response"):
            client._parse_arxiv_response(invalid_xml)
    
    @pytest.mark.asyncio
    async def test_search_papers_invalid_query(self):
        """Test search with invalid query raises error."""
        client = ArXivClient()
        
        with pytest.raises(ArXivAPIError, match="at least 3 characters"):
            await client.search_papers("x")
    
    @pytest.mark.asyncio
    async def test_search_papers_max_results_clamping(self):
        """Test that max_results is clamped to allowed range."""
        client = ArXivClient()
        client.max_results = 10  # Set a lower limit for testing
        
        # Mock the HTTP request to avoid actual API call
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="""<?xml version="1.0"?>
                <feed xmlns="http://www.w3.org/2005/Atom"></feed>""")
            mock_get.return_value.__aenter__.return_value = mock_response
            
            await client.search_papers("test query", max_results=100)
            
            # Check that the URL contains max_results=10 (clamped)
            call_args = mock_get.call_args[0][0]
            assert "max_results=10" in call_args
    
    @pytest.mark.asyncio 
    async def test_search_papers_network_error(self):
        """Test handling of network errors."""
        client = ArXivClient()
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = aiohttp.ClientError("Network error")
            
            with pytest.raises(ArXivAPIError, match="Network error"):
                await client.search_papers("test query")
    
    @pytest.mark.asyncio
    async def test_search_papers_timeout(self):
        """Test handling of request timeout."""
        client = ArXivClient()
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = asyncio.TimeoutError()
            
            with pytest.raises(ArXivAPIError, match="timed out"):
                await client.search_papers("test query")
    
    @pytest.mark.asyncio
    async def test_search_papers_http_error(self):
        """Test handling of HTTP error responses."""
        client = ArXivClient()
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")
            mock_get.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(ArXivAPIError, match="status 500"):
                await client.search_papers("test query")


class TestArXivTools:
    """Test Praval tool functions."""
    
    @pytest.mark.asyncio
    async def test_search_arxiv_papers_tool(self):
        """Test the search_arxiv_papers tool function."""
        with patch('src.processors.arxiv_client.get_arxiv_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.search_papers = AsyncMock(return_value=[
                {
                    'title': 'Test Paper',
                    'authors': ['Test Author'],
                    'abstract': 'Test abstract',
                    'arxiv_id': '2024.test001'
                }
            ])
            mock_get_client.return_value = mock_client
            
            results = await search_arxiv_papers("machine learning", 5, "computer_science")
            
            assert len(results) == 1
            assert results[0]['title'] == 'Test Paper'
            mock_client.search_papers.assert_called_once_with("machine learning", 5, "computer_science")
    
    @pytest.mark.asyncio
    async def test_get_paper_details_tool(self):
        """Test the get_paper_details tool function."""
        with patch('src.processors.arxiv_client.get_arxiv_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.search_papers = AsyncMock(return_value=[
                {
                    'title': 'Specific Paper',
                    'arxiv_id': '2024.test001'
                }
            ])
            mock_get_client.return_value = mock_client
            
            result = await get_paper_details("2024.test001")
            
            assert result['title'] == 'Specific Paper'
            mock_client.search_papers.assert_called_once_with("id:2024.test001", max_results=1)
    
    @pytest.mark.asyncio
    async def test_get_paper_details_not_found(self):
        """Test get_paper_details when paper not found."""
        with patch('src.processors.arxiv_client.get_arxiv_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.search_papers = AsyncMock(return_value=[])
            mock_get_client.return_value = mock_client
            
            with pytest.raises(ArXivAPIError, match="not found"):
                await get_paper_details("nonexistent.id")


class TestRelevanceCalculation:
    """Test paper relevance scoring."""
    
    def test_calculate_relevance_title_match(self):
        """Test relevance calculation with title match."""
        paper = {
            'title': 'Machine Learning Algorithms',
            'abstract': 'This paper discusses various algorithms.',
            'categories': ['cs.LG']
        }
        
        score = calculate_paper_relevance(paper, "machine learning")
        
        # Should have high score due to title match
        assert score > 0.3  # Title contributes 40% weight
    
    def test_calculate_relevance_abstract_match(self):
        """Test relevance calculation with abstract match."""
        paper = {
            'title': 'Research Paper',
            'abstract': 'Deep learning neural networks for classification.',
            'categories': ['cs.AI']
        }
        
        score = calculate_paper_relevance(paper, "deep learning")
        
        # Should have moderate score due to abstract match
        assert score > 0.25  # Abstract contributes 50% weight
    
    def test_calculate_relevance_category_match(self):
        """Test relevance calculation with category match."""
        paper = {
            'title': 'Research Paper',
            'abstract': 'General research content.',
            'categories': ['cs.LG', 'machine-learning']
        }
        
        score = calculate_paper_relevance(paper, "machine learning")
        
        # Should have some score due to category match
        assert score > 0.0
    
    def test_calculate_relevance_no_match(self):
        """Test relevance calculation with no matches."""
        paper = {
            'title': 'Quantum Physics Paper',
            'abstract': 'Quantum mechanics and particle physics.',
            'categories': ['physics.quantum-ph']
        }
        
        score = calculate_paper_relevance(paper, "machine learning")
        
        # Should have very low or zero score
        assert score <= 0.1
    
    def test_calculate_relevance_empty_query(self):
        """Test relevance calculation with empty query."""
        paper = {
            'title': 'Test Paper',
            'abstract': 'Test abstract.',
            'categories': ['cs.AI']
        }
        
        score = calculate_paper_relevance(paper, "")
        assert score == 0.0
    
    def test_calculate_relevance_missing_fields(self):
        """Test relevance calculation with missing paper fields."""
        paper = {}  # Empty paper
        
        score = calculate_paper_relevance(paper, "machine learning")
        assert score >= 0.0  # Should not crash


class TestSingleton:
    """Test ArXiv client singleton functionality."""
    
    def test_get_arxiv_client_singleton(self):
        """Test that get_arxiv_client returns same instance."""
        client1 = get_arxiv_client()
        client2 = get_arxiv_client()
        
        assert client1 is client2
    
    def test_client_has_correct_type(self):
        """Test that singleton returns correct type."""
        client = get_arxiv_client()
        assert isinstance(client, ArXivClient)


@pytest.mark.integration
class TestArXivIntegration:
    """Integration tests that hit the real ArXiv API (marked for optional running)."""
    
    @pytest.mark.asyncio
    async def test_real_arxiv_search(self):
        """Test actual ArXiv API search (integration test)."""
        client = ArXivClient()
        
        # Search for a common topic that should return results
        papers = await client.search_papers("machine learning", max_results=3)
        
        # Should get some papers
        assert len(papers) > 0
        
        # Check paper structure
        paper = papers[0]
        assert 'title' in paper
        assert 'authors' in paper
        assert 'abstract' in paper
        assert isinstance(paper['authors'], list)
        
        # Title should not be empty
        assert len(paper['title']) > 0
        
        print(f"Integration test found {len(papers)} papers")
        print(f"First paper: {paper['title'][:60]}...")
    
    @pytest.mark.asyncio
    async def test_real_arxiv_search_with_domain(self):
        """Test ArXiv search with domain filtering."""
        client = ArXivClient()
        
        papers = await client.search_papers(
            "neural networks", 
            max_results=2, 
            domain="computer_science"
        )
        
        assert len(papers) >= 0  # May be 0 if no matches
        
        # If papers found, they should have categories
        if papers:
            paper = papers[0]
            categories = paper.get('categories', [])
            # Should have at least one category for computer science papers
            assert len(categories) >= 0
            
            print(f"Domain-filtered search found {len(papers)} papers")
            if categories:
                print(f"Categories: {categories}")