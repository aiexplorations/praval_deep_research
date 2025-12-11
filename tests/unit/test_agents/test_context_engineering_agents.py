"""
Unit tests for Context Engineering Agents.

Tests for:
- paper_summarizer agent
- citation_extractor agent
- linked_paper_indexer agent
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any


class TestPaperSummarizerAgent:
    """Tests for the paper_summarizer agent."""

    @pytest.fixture
    def mock_spore(self):
        """Create a mock spore with processed papers."""
        spore = Mock()
        spore.knowledge = {
            "processed_papers": [
                {
                    "arxiv_id": "2106.04560",
                    "title": "Test Paper on Machine Learning",
                    "abstract": "This paper presents a novel approach...",
                    "authors": ["Author One", "Author Two"],
                    "categories": ["cs.LG", "cs.AI"],
                    "processing": {
                        "analysis": "Good paper",
                        "text_extracted": True
                    }
                }
            ],
            "original_query": "machine learning",
            "processing_stats": {
                "successful": 1,
                "total_papers": 1
            }
        }
        return spore

    def test_paper_summarizer_responds_to_documents_processed(self):
        """Test that paper_summarizer responds to documents_processed."""
        from agents.research.paper_summarizer import paper_summarizer_agent

        assert hasattr(paper_summarizer_agent, '_praval_responds_to')
        assert 'documents_processed' in paper_summarizer_agent._praval_responds_to

    def test_paper_summarizer_has_memory(self):
        """Test that paper_summarizer has memory enabled."""
        from agents.research.paper_summarizer import paper_summarizer_agent

        assert hasattr(paper_summarizer_agent, '_praval_memory_enabled')
        assert paper_summarizer_agent._praval_memory_enabled == True

    def test_paper_summarizer_metadata(self):
        """Test paper_summarizer agent metadata."""
        from agents.research.paper_summarizer import AGENT_METADATA

        assert AGENT_METADATA["identity"] == "paper summarization specialist"
        assert AGENT_METADATA["domain"] == "research"
        assert "Structured summary generation" in AGENT_METADATA["capabilities"]
        assert AGENT_METADATA["memory_enabled"] == True

    @patch('agents.research.paper_summarizer.chat')
    @patch('agents.research.paper_summarizer.broadcast')
    @patch('agents.research.paper_summarizer.QdrantClientWrapper')
    @patch('agents.research.paper_summarizer.EmbeddingsGenerator')
    def test_paper_summarizer_generates_summary(
        self,
        mock_embeddings,
        mock_qdrant,
        mock_broadcast,
        mock_chat,
        mock_spore
    ):
        """Test that paper_summarizer generates structured summaries."""
        from agents.research.paper_summarizer import paper_summarizer_agent

        # Setup mocks
        mock_chat.return_value = """
ONE_LINE: A novel ML approach
ABSTRACT_SUMMARY: This paper introduces a new method.
KEY_CONTRIBUTIONS:
- First contribution
- Second contribution
METHODOLOGY: Uses deep learning
DOMAINS: machine learning, neural networks
"""
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.generate_embedding.return_value = [0.1] * 1536
        mock_embeddings.return_value = mock_embeddings_instance

        mock_qdrant_instance = Mock()
        mock_qdrant_instance.initialize_context_engineering_collections.return_value = {}
        mock_qdrant_instance.add_paper_summary.return_value = True
        mock_qdrant.return_value = mock_qdrant_instance

        # Mock agent memory functions
        with patch.object(paper_summarizer_agent, 'remember', return_value=None):
            with patch.object(paper_summarizer_agent, 'recall', return_value=[]):
                # Execute agent
                paper_summarizer_agent(mock_spore)

        # Verify broadcast was called with papers_summarized
        mock_broadcast.assert_called_once()
        broadcast_call = mock_broadcast.call_args[0][0]
        assert broadcast_call["type"] == "papers_summarized"
        assert "summarized_papers" in broadcast_call["knowledge"]


class TestCitationExtractorAgent:
    """Tests for the citation_extractor agent."""

    @pytest.fixture
    def mock_spore(self):
        """Create a mock spore with summarized papers."""
        spore = Mock()
        spore.knowledge = {
            "summarized_papers": [
                {
                    "arxiv_id": "2106.04560",
                    "title": "Test Paper",
                    "abstract": "This paper cites arXiv:2105.12345 and other works...",
                    "summary": {
                        "title": "Test Paper",
                        "one_line": "A test paper",
                        "domains": ["machine learning"]
                    }
                }
            ],
            "original_query": "machine learning"
        }
        return spore

    def test_citation_extractor_responds_to_papers_summarized(self):
        """Test that citation_extractor responds to papers_summarized."""
        from agents.research.citation_extractor import citation_extractor_agent

        assert hasattr(citation_extractor_agent, '_praval_responds_to')
        assert 'papers_summarized' in citation_extractor_agent._praval_responds_to

    def test_citation_extractor_has_memory(self):
        """Test that citation_extractor has memory enabled."""
        from agents.research.citation_extractor import citation_extractor_agent

        assert hasattr(citation_extractor_agent, '_praval_memory_enabled')
        assert citation_extractor_agent._praval_memory_enabled == True

    def test_citation_extractor_metadata(self):
        """Test citation_extractor agent metadata."""
        from agents.research.citation_extractor import AGENT_METADATA

        assert AGENT_METADATA["identity"] == "citation extraction specialist"
        assert "Reference parsing from PDF text" in AGENT_METADATA["capabilities"]
        assert "citations_extracted" in AGENT_METADATA["broadcasts"]

    def test_extract_arxiv_ids_from_text(self):
        """Test regex-based arXiv ID extraction."""
        from agents.research.citation_extractor import _extract_arxiv_ids_from_text

        text = """
        This paper builds on arXiv:2106.04560 and the work presented at
        arxiv.org/abs/2105.12345v2. We also reference [2104.00001] in our
        methodology section.
        """

        ids = _extract_arxiv_ids_from_text(text)

        assert "2106.04560" in ids
        assert "2105.12345v2" in ids
        assert "2104.00001" in ids

    def test_extract_arxiv_ids_handles_various_formats(self):
        """Test arXiv ID extraction handles various citation formats."""
        from agents.research.citation_extractor import _extract_arxiv_ids_from_text

        # Test different formats
        test_cases = [
            ("arXiv:2106.04560", ["2106.04560"]),
            ("arxiv.org/abs/2105.12345", ["2105.12345"]),
            ("arxiv.org/pdf/2104.00001", ["2104.00001"]),
            ("[2103.12345]", ["2103.12345"]),
            ("arXiv: 2102.11111v3", ["2102.11111v3"]),
        ]

        for text, expected in test_cases:
            ids = _extract_arxiv_ids_from_text(text)
            for exp_id in expected:
                assert exp_id in ids, f"Failed to extract {exp_id} from '{text}'"


class TestLinkedPaperIndexerAgent:
    """Tests for the linked_paper_indexer agent."""

    @pytest.fixture
    def mock_spore(self):
        """Create a mock spore with citation candidates."""
        spore = Mock()
        spore.knowledge = {
            "citation_candidates": [
                {
                    "arxiv_id": "2105.12345",
                    "source_paper_id": "2106.04560",
                    "title": "Cited Paper",
                    "extraction_method": "regex",
                    "confidence": "high"
                }
            ],
            "source_papers": ["2106.04560"],
            "original_query": "machine learning"
        }
        return spore

    def test_linked_paper_indexer_responds_to_citations_extracted(self):
        """Test that linked_paper_indexer responds to citations_extracted."""
        from agents.research.linked_paper_indexer import linked_paper_indexer_agent

        assert hasattr(linked_paper_indexer_agent, '_praval_responds_to')
        assert 'citations_extracted' in linked_paper_indexer_agent._praval_responds_to

    def test_linked_paper_indexer_has_memory(self):
        """Test that linked_paper_indexer has memory enabled."""
        from agents.research.linked_paper_indexer import linked_paper_indexer_agent

        assert hasattr(linked_paper_indexer_agent, '_praval_memory_enabled')
        assert linked_paper_indexer_agent._praval_memory_enabled == True

    def test_linked_paper_indexer_metadata(self):
        """Test linked_paper_indexer agent metadata."""
        from agents.research.linked_paper_indexer import AGENT_METADATA

        assert AGENT_METADATA["identity"] == "linked paper indexing specialist"
        assert "Full PDF processing and indexing" in AGENT_METADATA["capabilities"]
        assert "linked_papers_indexed" in AGENT_METADATA["broadcasts"]
        assert "Qdrant (linked_papers collection)" in AGENT_METADATA["storage_integrations"]


class TestContextEngineeringPipeline:
    """Tests for the complete context engineering pipeline."""

    def test_pipeline_message_flow(self):
        """Test that agents form a proper broadcast chain."""
        from agents.research.paper_summarizer import paper_summarizer_agent
        from agents.research.citation_extractor import citation_extractor_agent
        from agents.research.linked_paper_indexer import linked_paper_indexer_agent

        # paper_summarizer responds to documents_processed
        assert 'documents_processed' in paper_summarizer_agent._praval_responds_to

        # citation_extractor responds to papers_summarized
        assert 'papers_summarized' in citation_extractor_agent._praval_responds_to

        # linked_paper_indexer responds to citations_extracted
        assert 'citations_extracted' in linked_paper_indexer_agent._praval_responds_to

    def test_all_agents_have_praval_name(self):
        """Test all context engineering agents have proper Praval names."""
        from agents.research.paper_summarizer import paper_summarizer_agent
        from agents.research.citation_extractor import citation_extractor_agent
        from agents.research.linked_paper_indexer import linked_paper_indexer_agent

        assert paper_summarizer_agent._praval_name == "paper_summarizer"
        assert citation_extractor_agent._praval_name == "citation_extractor"
        assert linked_paper_indexer_agent._praval_name == "linked_paper_indexer"

    def test_agents_exported_from_research_module(self):
        """Test that agents are properly exported from research module."""
        from agents.research import (
            paper_summarizer_agent,
            citation_extractor_agent,
            linked_paper_indexer_agent
        )

        assert paper_summarizer_agent is not None
        assert citation_extractor_agent is not None
        assert linked_paper_indexer_agent is not None

    def test_agents_included_in_main_agents_module(self):
        """Test that agents are included in main agents module."""
        from agents import (
            paper_summarizer_agent,
            citation_extractor_agent,
            linked_paper_indexer_agent
        )

        assert paper_summarizer_agent is not None
        assert citation_extractor_agent is not None
        assert linked_paper_indexer_agent is not None
