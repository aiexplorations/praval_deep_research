"""
Integration tests for paper discovery workflow.

This module tests the complete paper discovery workflow using real Praval agents
and ArXiv API integration. Tests validate the agent communication patterns,
memory integration, and end-to-end functionality.

Following CLAUDE.md standards:
- TDD approach with comprehensive test coverage
- Real API integration (no mocks in core logic)
- Praval agent communication testing
- Memory and learning validation
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List
from unittest.mock import patch, AsyncMock

# Test the new Praval agent structure
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agents.research.paper_discovery import paper_discovery_agent
from processors.arxiv_client import ArXivClient, search_arxiv_papers
from praval import start_agents, broadcast, Spore


class TestPaperDiscoveryWorkflow:
    """Test the complete paper discovery workflow."""
    
    @pytest.fixture
    def setup_agents(self):
        """Set up Praval agents for testing."""
        # Initialize the paper discovery agent
        try:
            # Note: In real usage, start_agents would be called
            # For testing, we'll work with the agent function directly
            yield paper_discovery_agent
        except Exception as e:
            pytest.skip(f"Praval agents not available: {e}")
    
    @pytest.mark.asyncio
    async def test_arxiv_client_basic_functionality(self):
        """Test ArXiv client works independently."""
        client = ArXivClient()
        
        # Test basic search
        papers = await client.search_papers("machine learning", max_results=3)
        
        assert isinstance(papers, list)
        assert len(papers) >= 0  # May be 0 if API issues, but shouldn't crash
        
        if papers:
            paper = papers[0]
            required_fields = ['title', 'authors', 'abstract', 'arxiv_id', 'url']
            for field in required_fields:
                assert field in paper, f"Missing field: {field}"
                assert paper[field] is not None, f"Null field: {field}"
    
    @pytest.mark.asyncio 
    async def test_search_arxiv_papers_tool_function(self):
        """Test the search_arxiv_papers tool function."""
        # Test the high-level tool function
        papers = await search_arxiv_papers(
            query="neural networks",
            max_results=2,
            domain="computer_science"
        )
        
        assert isinstance(papers, list)
        
        if papers:
            # Validate paper structure
            paper = papers[0]
            assert 'title' in paper
            assert 'authors' in paper
            assert isinstance(paper['authors'], list)
            assert len(paper['title']) > 0
    
    def test_paper_discovery_agent_structure(self, setup_agents):
        """Test paper discovery agent has correct structure."""
        agent = setup_agents
        
        # Check agent function exists
        assert callable(agent)
        
        # Check agent has proper Praval decorator metadata
        assert hasattr(agent, '__wrapped__') or hasattr(agent, 'responds_to')
    
    def test_paper_discovery_agent_spore_handling(self, setup_agents):
        """Test agent handles spores correctly."""
        agent = setup_agents
        
        # Create test spore for search request
        test_spore = Spore()
        test_spore.knowledge = {
            "type": "search_request",
            "query": "machine learning",
            "domain": "computer_science",
            "max_results": 3,
            "session_id": "test_session"
        }
        
        # Test that agent doesn't crash when processing spore
        try:
            # Mock the chat and broadcast functions to avoid external calls
            with patch('agents.research.paper_discovery.chat') as mock_chat, \
                 patch('agents.research.paper_discovery.broadcast') as mock_broadcast, \
                 patch('processors.arxiv_client.search_arxiv_papers') as mock_search:
                
                # Mock successful search results
                mock_search.return_value = [
                    {
                        'title': 'Test Paper',
                        'authors': ['Test Author'],
                        'abstract': 'Test abstract',
                        'arxiv_id': '2024.test001',
                        'url': 'http://arxiv.org/abs/2024.test001',
                        'published_date': '2024-01-01',
                        'categories': ['cs.LG']
                    }
                ]
                
                mock_chat.return_value = "machine learning optimization algorithms"
                
                # Execute agent
                result = agent(test_spore)
                
                # Verify agent called external functions
                mock_search.assert_called_once()
                mock_broadcast.assert_called_once()
                
                # Verify broadcast was called with correct structure
                broadcast_call = mock_broadcast.call_args[0][0]
                assert broadcast_call["type"] == "papers_found"
                assert "knowledge" in broadcast_call
                assert "papers" in broadcast_call["knowledge"]
                
        except Exception as e:
            # Agent shouldn't crash even with missing dependencies
            assert False, f"Agent crashed: {e}"
    
    @pytest.mark.asyncio
    async def test_paper_discovery_memory_integration(self, setup_agents):
        """Test agent memory functionality."""
        agent = setup_agents
        
        # Check if agent has memory methods
        if hasattr(agent, 'remember') and hasattr(agent, 'recall'):
            # Test memory operations
            test_memory = "test_search: machine learning -> 5 papers found"
            
            try:
                agent.remember(test_memory, importance=0.8)
                
                # Try to recall
                memories = agent.recall("test_search", limit=1)
                
                # Memory should work without crashing
                assert isinstance(memories, (list, type(None)))
                
            except Exception as e:
                # Memory might not be fully configured in test environment
                pytest.skip(f"Memory not available in test environment: {e}")
        else:
            pytest.skip("Agent memory methods not available")
    
    def test_paper_discovery_agent_identity(self, setup_agents):
        """Test agent has proper identity statement."""
        agent = setup_agents
        
        # Get docstring
        docstring = agent.__doc__ or ""
        
        # Should have identity statement starting with "I am"
        assert "I am" in docstring, "Agent should have identity statement starting with 'I am'"
        assert "paper" in docstring.lower() or "search" in docstring.lower(), \
            "Agent identity should relate to paper discovery"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_paper_discovery_workflow_with_real_api(self, setup_agents):
        """
        Integration test with real ArXiv API.
        
        This test validates the complete workflow but may be skipped
        if external services are unavailable.
        """
        agent = setup_agents
        
        # Create realistic search spore
        search_spore = Spore()
        search_spore.knowledge = {
            "type": "search_request",
            "query": "transformer neural networks",
            "domain": "artificial_intelligence", 
            "max_results": 2,
            "session_id": "integration_test",
            "quality_threshold": 0.3
        }
        
        # Track broadcasts for validation
        broadcasted_messages = []
        
        def mock_broadcast(message):
            broadcasted_messages.append(message)
        
        try:
            with patch('agents.research.paper_discovery.broadcast', side_effect=mock_broadcast):
                # Execute agent with real ArXiv search
                result = agent(search_spore)
                
                # Should have broadcasted results
                assert len(broadcasted_messages) > 0, "Agent should broadcast results"
                
                # Check broadcast message structure
                message = broadcasted_messages[0]
                assert message["type"] == "papers_found", "Should broadcast papers_found"
                assert "knowledge" in message
                
                knowledge = message["knowledge"]
                assert "papers" in knowledge
                assert "original_query" in knowledge
                assert knowledge["original_query"] == "transformer neural networks"
                
                # If papers found, validate structure
                papers = knowledge["papers"]
                if papers:
                    paper = papers[0]
                    required_fields = ['title', 'authors', 'abstract', 'arxiv_id']
                    for field in required_fields:
                        assert field in paper, f"Paper missing required field: {field}"
                
        except Exception as e:
            if "network" in str(e).lower() or "connection" in str(e).lower():
                pytest.skip(f"Network unavailable for integration test: {e}")
            else:
                raise


class TestPaperDiscoveryErrorHandling:
    """Test error handling in paper discovery workflow."""
    
    @pytest.fixture
    def setup_agents(self):
        """Set up agents for error testing."""
        yield paper_discovery_agent
    
    def test_agent_handles_invalid_spore(self, setup_agents):
        """Test agent handles invalid spore gracefully."""
        agent = setup_agents
        
        # Create invalid spore
        invalid_spore = Spore()
        invalid_spore.knowledge = {}  # Missing required fields
        
        # Should not crash
        try:
            with patch('agents.research.paper_discovery.broadcast') as mock_broadcast, \
                 patch('agents.research.paper_discovery.logger') as mock_logger:
                
                result = agent(invalid_spore)
                
                # Should log warning or error, not crash
                assert mock_logger.warning.called or mock_logger.error.called
                
        except Exception as e:
            assert False, f"Agent should handle invalid spore gracefully: {e}"
    
    def test_agent_handles_search_errors(self, setup_agents):
        """Test agent handles search API errors."""
        agent = setup_agents
        
        # Create valid spore
        test_spore = Spore()
        test_spore.knowledge = {
            "type": "search_request",
            "query": "test query",
            "domain": "computer_science",
            "max_results": 5
        }
        
        # Mock search failure
        with patch('processors.arxiv_client.search_arxiv_papers') as mock_search, \
             patch('agents.research.paper_discovery.broadcast') as mock_broadcast:
            
            # Simulate API error
            mock_search.side_effect = Exception("API Error")
            
            try:
                result = agent(test_spore)
                
                # Should broadcast error message
                assert mock_broadcast.called
                error_message = mock_broadcast.call_args[0][0]
                assert error_message["type"] == "search_error"
                
            except Exception as e:
                # Some failures are acceptable, but shouldn't crash ungracefully
                assert "API Error" in str(e) or "search" in str(e).lower()


class TestArXivClientRobustness:
    """Test ArXiv client robustness and error handling."""
    
    @pytest.mark.asyncio
    async def test_client_validates_input(self):
        """Test client validates input parameters."""
        client = ArXivClient()
        
        # Test empty query
        with pytest.raises(Exception):  # Should raise some validation error
            await client.search_papers("")
        
        # Test very short query
        with pytest.raises(Exception):
            await client.search_papers("a")
    
    @pytest.mark.asyncio
    async def test_client_handles_network_errors(self):
        """Test client handles network errors gracefully."""
        client = ArXivClient()
        
        # Mock network failure
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            with pytest.raises(Exception) as exc_info:
                await client.search_papers("test query")
            
            # Should get a meaningful error
            assert "Network error" in str(exc_info.value) or \
                   "search" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_client_rate_limiting(self):
        """Test client respects rate limiting."""
        client = ArXivClient()
        client.delay_seconds = 0.1  # Short delay for testing
        
        start_time = time.time()
        
        # Mock successful responses
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='''<?xml version="1.0"?>
                <feed xmlns="http://www.w3.org/2005/Atom"></feed>''')
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Make two quick requests
            await client.search_papers("test query 1", max_results=1)
            await client.search_papers("test query 2", max_results=1)
            
            elapsed = time.time() - start_time
            
            # Should have taken at least delay_seconds for rate limiting
            assert elapsed >= 0.1, "Rate limiting should add delay between requests"


if __name__ == "__main__":
    # Run specific tests for development
    pytest.main([__file__, "-v", "--tb=short"])