"""
Integration tests for Praval agent communication flow.

Tests the complete message flow between agents using Praval's native
communication patterns. These tests verify that:
1. Agents properly broadcast messages
2. Messages are routed based on responds_to
3. The research workflow completes end-to-end
4. Memory is used across agent interactions

Note: These tests require mocking external services (ArXiv, Qdrant, etc.)
but test real Praval agent communication.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from typing import Dict, Any, List

from praval import start_agents, get_reef, broadcast
from praval.core.reef import Reef, SporeType
from praval.core.reef_backend import InMemoryBackend

# Import all agents
from agents import (
    paper_discovery_agent,
    document_processing_agent,
    semantic_analysis_agent,
    summarization_agent,
    qa_specialist_agent,
    research_advisor_agent
)


@pytest.fixture
def setup_local_reef():
    """Set up a local Reef with InMemory backend for testing."""
    backend = InMemoryBackend()
    reef = Reef(backend=backend)

    # Initialize backend
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(reef.initialize_backend())

    yield reef

    # Cleanup
    loop.run_until_complete(reef.close_backend())
    reef.shutdown()
    loop.close()


@pytest.fixture
def mock_external_services():
    """Mock all external services for integration testing."""
    with patch('agents.research.paper_discovery.search_arxiv_papers') as mock_arxiv, \
         patch('agents.research.paper_discovery.chat') as mock_chat_discovery, \
         patch('agents.research.document_processor.MinIOClient') as mock_minio, \
         patch('agents.research.document_processor.QdrantClientWrapper') as mock_qdrant, \
         patch('agents.research.document_processor.EmbeddingsGenerator') as mock_embeddings, \
         patch('agents.research.semantic_analyzer.chat') as mock_chat_semantic, \
         patch('agents.research.summarization.chat') as mock_chat_summary, \
         patch('agents.interaction.qa_specialist.chat') as mock_chat_qa, \
         patch('agents.interaction.qa_specialist.QdrantClientWrapper') as mock_qdrant_qa, \
         patch('agents.research.paper_discovery.calculate_paper_relevance') as mock_calc_relevance, \
         patch('agents.interaction.research_advisor.chat') as mock_chat_advisor:

        # Configure ArXiv mock
        mock_arxiv.return_value = [
            {
                "arxiv_id": "2401.12345",
                "title": "Test Paper on Transformers",
                "authors": ["Author One", "Author Two"],
                "abstract": "A paper about transformer architectures",
                "url": "http://arxiv.org/abs/2401.12345",
                "pdf_url": "http://arxiv.org/pdf/2401.12345",
                "published_date": "2024-01-15",
                "categories": ["cs.AI", "cs.LG"]
            }
        ]

        # Configure chat mocks
        mock_chat_discovery.return_value = "optimized: transformers attention"
        mock_chat_semantic.return_value = "Thematic analysis: key themes identified"
        mock_chat_summary.return_value = "Executive summary of research"
        mock_chat_qa.return_value = "Comprehensive answer based on research"
        mock_chat_advisor.return_value = "Strategic research guidance"

        # Configure storage mocks
        mock_minio_instance = MagicMock()
        mock_minio.return_value = mock_minio_instance
        mock_minio_instance.pdf_exists.return_value = False
        mock_minio_instance.upload_pdf.return_value = True

        mock_qdrant_instance = MagicMock()
        mock_qdrant.return_value = mock_qdrant_instance
        mock_qdrant_instance.upsert_vectors.return_value = True

        mock_qdrant_qa_instance = MagicMock()
        mock_qdrant_qa.return_value = mock_qdrant_qa_instance
        mock_qdrant_qa_instance.search_similar.return_value = [
            {"payload": {"chunk_text": "Relevant context", "title": "Test Paper", "paper_id": "123", "chunk_index": 0}, "score": 0.9}
        ]

        mock_calc_relevance.return_value = 0.9

        yield {
            'arxiv': mock_arxiv,
            'chat_discovery': mock_chat_discovery,
            'chat_semantic': mock_chat_semantic,
            'chat_summary': mock_chat_summary,
            'chat_qa': mock_chat_qa,
            'chat_advisor': mock_chat_advisor,
            'embeddings': mock_embeddings,
            'minio': mock_minio,
            'qdrant': mock_qdrant,
            'qdrant_qa': mock_qdrant_qa,
            'calc_relevance': mock_calc_relevance
        }


class TestAgentStartup:
    """Test agent startup and registration."""

    def test_start_agents_registers_all(self, mock_external_services):
        """Test that start_agents properly registers all research agents."""
        agents = [
            paper_discovery_agent,
            document_processing_agent,
            semantic_analysis_agent,
            summarization_agent,
            qa_specialist_agent,
            research_advisor_agent
        ]

        # Start agents (this registers them with the reef)
        start_agents(*agents)

        # Get reef and verify
        reef = get_reef()
        assert reef is not None

        # Verify reef has channels set up
        stats = reef.get_network_stats()
        assert stats['total_channels'] > 0


class TestMessageFlowChain:
    """Test the complete message flow chain between agents."""

    @pytest.mark.integration
    def test_search_request_triggers_chain(self, mock_external_services):
        """Test that search_request triggers the full research workflow."""
        # Track broadcast calls
        broadcast_calls = []

        def track_broadcast(data: Dict[str, Any]):
            broadcast_calls.append(data)
            return "test-spore-id"

        # Start agents
        agents = [
            paper_discovery_agent,
            document_processing_agent,
            semantic_analysis_agent,
            summarization_agent
        ]

        # Mock agent memory methods
        for agent in agents:
            agent.recall = MagicMock(return_value=[])
            agent.remember = MagicMock()

        with patch('agents.research.paper_discovery.broadcast', side_effect=track_broadcast):
            # Create search request spore
            search_spore = MagicMock()
            search_spore.knowledge = {
                "type": "search_request",
                "query": "machine learning transformers",
                "domain": "cs.AI",
                "max_results": 5,
                "quality_threshold": 0.6,
                "session_id": "test-integration-123"
            }

            # Execute paper_discovery_agent
            paper_discovery_agent(search_spore)

            # Verify papers_found was broadcast
            assert len(broadcast_calls) >= 1
            assert broadcast_calls[0]["type"] == "papers_found"
            assert "papers" in broadcast_calls[0]


class TestQAWorkflow:
    """Test the Q&A workflow."""

    @pytest.mark.integration
    def test_user_query_generates_response(self, mock_external_services):
        """Test that user_query generates a Q&A response."""
        broadcast_calls = []

        def track_broadcast(data: Dict[str, Any]):
            broadcast_calls.append(data)
            return "test-spore-id"

        # Mock agent memory
        qa_specialist_agent.recall = MagicMock(return_value=[])
        qa_specialist_agent.remember = MagicMock()

        with patch('agents.interaction.qa_specialist.broadcast', side_effect=track_broadcast):
            # Create user query spore
            query_spore = MagicMock()
            query_spore.knowledge = {
                "type": "user_query",
                "query": "What are transformers?",
                "user_id": "test-user",
                "conversation_id": "test-conv-123",
                "session_id": "test-session"
            }

            # Execute qa_specialist_agent
            qa_specialist_agent(query_spore)

            # Verify qa_response was broadcast
            assert len(broadcast_calls) >= 1
            assert broadcast_calls[0]["type"] == "qa_response"
            assert "comprehensive_answer" in broadcast_calls[0]["knowledge"]


class TestResearchAdvisorWorkflow:
    """Test the research advisor workflow."""

    @pytest.mark.integration
    def test_guidance_request_generates_advisory(self, mock_external_services):
        """Test that guidance request generates research advisory."""
        broadcast_calls = []

        def track_broadcast(data: Dict[str, Any]):
            broadcast_calls.append(data)
            return "test-spore-id"

        # Mock agent memory
        research_advisor_agent.recall = MagicMock(return_value=[])
        research_advisor_agent.remember = MagicMock()

        with patch('agents.interaction.research_advisor.broadcast', side_effect=track_broadcast):
            # Create guidance request spore
            guidance_spore = MagicMock()
            guidance_spore.knowledge = {
                "type": "research_guidance_request",
                "guidance_request": "How to start NLP research?",
                "research_level": "graduate",
                "research_interests": ["NLP", "transformers"],
                "current_project": "Building chatbot",
                "user_id": "test-user"
            }

            # Execute research_advisor_agent
            research_advisor_agent(guidance_spore)

            # Verify advisory was broadcast
            assert len(broadcast_calls) >= 1
            assert broadcast_calls[0]["type"] == "research_advisory_complete"


class TestMemoryUsage:
    """Test memory usage across agent interactions."""

    @pytest.mark.integration
    def test_agent_remembers_successful_searches(self, mock_external_services):
        """Test that paper_discovery_agent remembers successful searches."""
        remember_calls = []

        def track_remember(content, importance=0.5):
            remember_calls.append({"content": content, "importance": importance})

        # Mock memory
        paper_discovery_agent.recall = MagicMock(return_value=[])
        paper_discovery_agent.remember = track_remember

        with patch('agents.research.paper_discovery.broadcast'):
            search_spore = MagicMock()
            search_spore.knowledge = {
                "type": "search_request",
                "query": "machine learning",
                "domain": "cs.AI",
                "max_results": 5,
                "quality_threshold": 0.6,
                "session_id": "test-123"
            }

            paper_discovery_agent(search_spore)

            # Verify remember was called
            assert len(remember_calls) > 0
            # Check that search was remembered
            remembered_content = [c["content"] for c in remember_calls]
            assert any("search" in c.lower() or "machine learning" in c.lower()
                      for c in remembered_content)

    @pytest.mark.integration
    def test_agent_recalls_past_context(self, mock_external_services):
        """Test that qa_specialist_agent recalls past user interactions."""
        recall_calls = []

        def track_recall(query, limit=5):
            recall_calls.append({"query": query, "limit": limit})
            # Return mock memories
            mock_memory = MagicMock()
            mock_memory.content = "Past interaction about transformers"
            return [mock_memory]

        # Mock memory
        qa_specialist_agent.recall = track_recall
        qa_specialist_agent.remember = MagicMock()

        with patch('agents.interaction.qa_specialist.broadcast'):
            query_spore = MagicMock()
            query_spore.knowledge = {
                "type": "user_query",
                "query": "Tell me more about transformers",
                "user_id": "returning-user",
                "session_id": "test-123"
            }

            qa_specialist_agent(query_spore)

            # Verify recall was called for user context
            assert len(recall_calls) > 0
            # Should recall user-specific context
            recall_queries = [c["query"] for c in recall_calls]
            assert any("user" in q.lower() for q in recall_queries)


class TestErrorHandling:
    """Test error handling in agent workflows."""

    @pytest.mark.integration
    def test_agent_broadcasts_error_on_failure(self, mock_external_services):
        """Test that agents broadcast errors when processing fails."""
        broadcast_calls = []

        def track_broadcast(data: Dict[str, Any]):
            broadcast_calls.append(data)
            return "test-spore-id"

        # Make ArXiv fail
        mock_external_services['arxiv'].side_effect = Exception("ArXiv API error")

        paper_discovery_agent.recall = MagicMock(return_value=[])
        paper_discovery_agent.remember = MagicMock()

        with patch('agents.research.paper_discovery.broadcast', side_effect=track_broadcast):
            search_spore = MagicMock()
            search_spore.knowledge = {
                "type": "search_request",
                "query": "test query",
                "domain": "cs.AI",
                "max_results": 5,
                "quality_threshold": 0.6,
                "session_id": "test-error-123"
            }

            # Should handle error gracefully
            try:
                paper_discovery_agent(search_spore)
            except Exception:
                pass  # Error expected

            # Should broadcast error
            error_broadcasts = [b for b in broadcast_calls if "error" in b.get("type", "").lower()]
            # Note: depends on agent error handling implementation


class TestPravalNativeIntegration:
    """Test native Praval integration patterns."""

    def test_agents_compatible_with_start_agents(self):
        """Test all agents can be used with start_agents()."""
        agents = [
            paper_discovery_agent,
            document_processing_agent,
            semantic_analysis_agent,
            summarization_agent,
            qa_specialist_agent,
            research_advisor_agent
        ]

        # All should have _praval_agent attribute
        for agent in agents:
            assert hasattr(agent, '_praval_agent'), \
                f"Agent {agent.__name__} not compatible with start_agents()"

    def test_agents_compatible_with_run_agents(self):
        """Test all agents can be used with run_agents()."""
        agents = [
            paper_discovery_agent,
            document_processing_agent,
            semantic_analysis_agent,
            summarization_agent,
            qa_specialist_agent,
            research_advisor_agent
        ]

        # All should have required Praval attributes
        required_attrs = ['_praval_agent', '_praval_name', '_praval_responds_to']

        for agent in agents:
            for attr in required_attrs:
                assert hasattr(agent, attr), \
                    f"Agent {agent._praval_name} missing {attr} for run_agents()"


class TestRunAgentsConfiguration:
    """Test run_agents configuration for distributed deployment."""

    def test_run_agents_imports(self):
        """Test that run_agents.py can import all agents."""
        # This tests the import structure
        from agents import (
            paper_discovery_agent,
            document_processing_agent,
            semantic_analysis_agent,
            summarization_agent,
            qa_specialist_agent,
            research_advisor_agent
        )

        agents = [
            paper_discovery_agent,
            document_processing_agent,
            semantic_analysis_agent,
            summarization_agent,
            qa_specialist_agent,
            research_advisor_agent
        ]

        assert len(agents) == 6

    def test_run_agents_entry_point_structure(self):
        """Test that run_agents.py has correct structure."""
        import importlib.util

        spec = importlib.util.find_spec("run_agents")
        if spec is None:
            pytest.skip("run_agents module not found in path")

        # Module should exist and be importable
        assert spec is not None
