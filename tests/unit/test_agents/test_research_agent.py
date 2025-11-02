"""
Unit tests for Praval-based Research Agents.

This module tests the autonomous research agents that communicate through
spores without any central coordinator.
"""

import pytest
from unittest.mock import patch, MagicMock

from agentic_research.agents.research_agent import (
    paper_search_agent,
    document_processing_agent,
    qa_specialist_agent,
    start_distributed_research,
    ask_distributed_question,
    research_and_ask_distributed,
    ResearchContext,
    ResearchReef,
    initialize_research_network,
    shutdown_research_network
)


class TestPravalResearchAgents:
    """Test suite for Praval research agents."""

    def test_research_context_creation(self):
        """Test research context creation and validation."""
        context = ResearchContext(
            query="machine learning",
            domain="computer science", 
            max_results=5,
            quality_threshold=0.8
        )
        
        assert context.query == "machine learning"
        assert context.domain == "computer science"
        assert context.max_results == 5
        assert context.quality_threshold == 0.8
        assert context.timestamp is not None

    def test_paper_search_agent_existence(self):
        """Test that paper search agent is properly decorated."""
        assert hasattr(paper_search_agent, '_praval_agent')
        assert hasattr(paper_search_agent, '_praval_name')
        assert hasattr(paper_search_agent, '_praval_memory_enabled')
        assert paper_search_agent._praval_name == "paper_searcher"
        assert paper_search_agent._praval_memory_enabled is True

    def test_document_processor_agent_existence(self):
        """Test that document processor agent is properly decorated."""
        assert hasattr(document_processing_agent, '_praval_agent')
        assert hasattr(document_processing_agent, '_praval_name')
        assert hasattr(document_processing_agent, '_praval_memory_enabled')
        assert document_processing_agent._praval_name == "document_processor"
        assert document_processing_agent._praval_memory_enabled is True

    def test_qa_specialist_agent_existence(self):
        """Test that Q&A specialist agent is properly decorated."""
        assert hasattr(qa_specialist_agent, '_praval_agent')
        assert hasattr(qa_specialist_agent, '_praval_name')
        assert hasattr(qa_specialist_agent, '_praval_memory_enabled')
        assert qa_specialist_agent._praval_name == "qa_specialist"
        assert qa_specialist_agent._praval_memory_enabled is True

    def test_agent_responds_to_configuration(self):
        """Test that agents respond to correct spore types."""
        assert "research_request" in paper_search_agent._praval_responds_to
        assert "papers_found" in document_processing_agent._praval_responds_to
        assert "papers_processed" in qa_specialist_agent._praval_responds_to
        assert "research_question" in qa_specialist_agent._praval_responds_to

    @patch('agentic_research.agents.research_agent.chat')
    @patch('agentic_research.agents.research_agent.broadcast')
    def test_paper_search_agent_workflow(self, mock_broadcast, mock_chat):
        """Test paper search agent processes research requests correctly."""
        mock_chat.return_value = "Focus on recent deep learning papers, prioritize top conferences"
        
        # Create mock spore
        mock_spore = MagicMock()
        mock_spore.knowledge = {
            "query": "deep learning",
            "domain": "computer science",
            "max_results": 3
        }
        
        # Simulate the agent function call
        result = paper_search_agent(mock_spore)
        
        # Verify LLM was called for search strategy
        mock_chat.assert_called_once()
        call_args = mock_chat.call_args[0][0]
        assert "deep learning" in call_args
        assert "computer science" in call_args
        
        # Verify papers were broadcast
        mock_broadcast.assert_called_once()
        broadcast_data = mock_broadcast.call_args[0][0]
        assert broadcast_data["type"] == "papers_found"
        assert broadcast_data["query"] == "deep learning"
        assert "papers" in broadcast_data
        
        # Verify return value
        assert "papers_found" in result
        assert result["papers_found"] > 0

    @patch('agentic_research.agents.research_agent.chat')  
    @patch('agentic_research.agents.research_agent.broadcast')
    def test_document_processor_workflow(self, mock_broadcast, mock_chat):
        """Test document processor handles papers_found spores correctly."""
        mock_chat.return_value = "Key insights: novel architecture, improved performance, extensive evaluation"
        
        # Create mock spore with papers
        mock_spore = MagicMock()
        mock_spore.knowledge = {
            "type": "papers_found",
            "query": "transformers",
            "papers": [
                {
                    "title": "Attention is All You Need",
                    "abstract": "The Transformer architecture...",
                    "authors": ["Vaswani et al."]
                },
                {
                    "title": "BERT: Pre-training Representations",
                    "abstract": "Bidirectional Encoder Representations...", 
                    "authors": ["Devlin et al."]
                }
            ]
        }
        
        result = document_processing_agent(mock_spore)
        
        # Verify LLM was called for each paper
        assert mock_chat.call_count == 2
        
        # Verify processed papers were broadcast
        mock_broadcast.assert_called_once()
        broadcast_data = mock_broadcast.call_args[0][0]
        assert broadcast_data["type"] == "papers_processed"
        assert len(broadcast_data["processed_papers"]) == 2
        
        # Verify return value
        assert result["papers_processed"] == 2

    @patch('agentic_research.agents.research_agent.chat')
    @patch('agentic_research.agents.research_agent.broadcast')
    def test_qa_specialist_papers_processed_workflow(self, mock_broadcast, mock_chat):
        """Test Q&A specialist stores processed papers."""
        # Create mock spore with processed papers
        mock_spore = MagicMock()
        mock_spore.knowledge = {
            "type": "papers_processed",
            "query": "neural networks",
            "processed_papers": [
                {"title": "Paper 1", "key_insights": "Insight 1"},
                {"title": "Paper 2", "key_insights": "Insight 2"}
            ]
        }
        
        result = qa_specialist_agent(mock_spore)
        
        # Verify papers were stored (indicated by broadcast)
        mock_broadcast.assert_called_once()
        broadcast_data = mock_broadcast.call_args[0][0]
        assert broadcast_data["type"] == "qa_ready"
        assert broadcast_data["papers_available"] == 2
        
        # Verify return value
        assert result["papers_stored"] == 2

    @patch('agentic_research.agents.research_agent.chat')
    @patch('agentic_research.agents.research_agent.broadcast')
    def test_qa_specialist_question_answering(self, mock_broadcast, mock_chat):
        """Test Q&A specialist answers questions when papers are available."""
        mock_chat.return_value = "Based on recent research, neural networks achieve state-of-the-art performance..."
        
        # Mock the recall method to return relevant papers
        with patch.object(qa_specialist_agent, 'recall') as mock_recall:
            mock_recall.return_value = [
                MagicMock(content="Neural networks show excellent performance on vision tasks"),
                MagicMock(content="Attention mechanisms improve model interpretability")
            ]
            
            # Create mock spore with question
            mock_spore = MagicMock()
            mock_spore.knowledge = {
                "type": "research_question",
                "question": "What are the advantages of neural networks?",
                "topic": "neural networks"
            }
            
            result = qa_specialist_agent(mock_spore)
            
            # Verify recall was called to find relevant papers
            mock_recall.assert_called_once()
            
            # Verify LLM was called to generate answer
            mock_chat.assert_called_once()
            
            # Verify answer was broadcast
            mock_broadcast.assert_called_once()
            broadcast_data = mock_broadcast.call_args[0][0]
            assert broadcast_data["type"] == "question_answered"
            assert "question" in broadcast_data
            assert "answer" in broadcast_data
            
            # Verify return value
            assert result["question_answered"] is True
            assert result["evidence_sources"] == 2

    def test_start_research_function(self):
        """Test start_distributed_research function initiates the workflow correctly."""
        with patch('agentic_research.agents.research_agent.start_agents') as mock_start:
            mock_start.return_value = {"papers_found": 3}
            
            result = start_distributed_research("machine learning", "AI", 5)
            
            # Verify start_agents was called with correct parameters
            mock_start.assert_called_once()
            call_args = mock_start.call_args
            
            # Check agents were passed
            agents = call_args[0]
            assert paper_search_agent in agents
            assert document_processing_agent in agents  
            assert qa_specialist_agent in agents
            
            # Check initial data
            initial_data = call_args[1]["initial_data"]
            assert initial_data["type"] == "research_request"
            assert initial_data["query"] == "machine learning"
            assert initial_data["domain"] == "AI"
            assert initial_data["max_results"] == 5

    def test_ask_question_function(self):
        """Test ask_distributed_question function works correctly."""
        with patch('agentic_research.agents.research_agent.start_agents') as mock_start:
            mock_start.return_value = {"question_answered": True}
            
            result = ask_distributed_question("How do neural networks work?", "AI")
            
            # Verify start_agents was called
            mock_start.assert_called_once()
            call_args = mock_start.call_args
            
            # Check Q&A agent was included
            agents = call_args[0]
            assert qa_specialist_agent in agents
            
            # Check initial data
            initial_data = call_args[1]["initial_data"]
            assert initial_data["type"] == "research_question"
            assert initial_data["question"] == "How do neural networks work?"
            assert initial_data["topic"] == "AI"

    def test_research_and_ask_workflow(self):
        """Test complete research and ask workflow."""
        with patch('agentic_research.agents.research_agent.start_distributed_research') as mock_research:
            with patch('agentic_research.agents.research_agent.ask_distributed_question') as mock_ask:
                mock_research.return_value = {"papers_found": 5}
                mock_ask.return_value = {"question_answered": True}
                
                result = research_and_ask_distributed(
                    "deep learning", 
                    "What are the latest trends in deep learning?",
                    "computer science"
                )
                
                # Verify both functions were called
                mock_research.assert_called_once_with("deep learning", "computer science")
                mock_ask.assert_called_once_with("What are the latest trends in deep learning?", "deep learning")
                
                # Verify return structure
                assert "research" in result
                assert "qa" in result
                assert result["research"]["papers_found"] == 5
                assert result["qa"]["question_answered"] is True

    def test_agent_memory_methods_available(self):
        """Test that agents have memory methods available from Praval."""
        # All agents should have memory methods since memory=True
        for agent in [paper_search_agent, document_processing_agent, qa_specialist_agent]:
            assert hasattr(agent, 'remember')
            assert hasattr(agent, 'recall')
            assert hasattr(agent, 'memory')

    def test_agent_communication_methods_available(self):
        """Test that agents have Praval communication methods available."""
        for agent in [paper_search_agent, document_processing_agent, qa_specialist_agent]:
            assert hasattr(agent, 'send_knowledge')
            assert hasattr(agent, 'broadcast_knowledge')
            assert hasattr(agent, 'request_knowledge')

    def test_no_coordinator_agent_exists(self):
        """Test that there is no coordinator agent - pure autonomous behavior."""
        # This test ensures we don't accidentally create a coordinator
        from agentic_research.agents import research_agent
        
        # Check that no coordinator-related functions or classes exist
        assert not hasattr(research_agent, 'coordinator_agent')
        assert not hasattr(research_agent, 'research_coordinator_agent')
        assert not hasattr(research_agent, 'CoordinatorAgent')
        
        # The only workflow functions should be simple triggers
        workflow_functions = [
            name for name in dir(research_agent) 
            if not name.startswith('_') and callable(getattr(research_agent, name))
        ]
        
        # Should not have orchestration-style functions
        orchestration_terms = ['orchestrate', 'coordinate', 'manage', 'control']
        for func_name in workflow_functions:
            for term in orchestration_terms:
                assert term not in func_name.lower(), f"Found orchestration function: {func_name}"

    def test_emergent_workflow_design(self):
        """Test that the system is designed for emergent behavior."""
        # Agents should respond to different spore types for natural flow
        search_responses = paper_search_agent._praval_responds_to
        processor_responses = document_processing_agent._praval_responds_to  
        qa_responses = qa_specialist_agent._praval_responds_to
        
        # Verify the spore cascade: research_request -> papers_found -> papers_processed
        assert "research_request" in search_responses
        assert "papers_found" in processor_responses
        assert "papers_processed" in qa_responses
        
        # Q&A should also handle direct questions
        assert "research_question" in qa_responses
        
        # No agent should respond to coordination/orchestration signals
        coordination_types = ["coordinate", "orchestrate", "manage", "control"]
        all_responses = search_responses + processor_responses + qa_responses
        
        for response_type in all_responses:
            for coord_type in coordination_types:
                assert coord_type not in response_type.lower()