"""
Praval agents for the agentic research system.

This module contains specialized agents organized by domain:
- research/: Research domain agents (paper discovery, processing, analysis)
- interaction/: User interaction agents (Q&A, advisory)

Agents self-organize through spore communication - no central coordination needed.

Context Engineering Pipeline (DISABLED - requires user consent):
document_processor -> paper_summarizer -> citation_extractor -> linked_paper_indexer
"""

# Import all agents for easy access
from .research.paper_discovery import paper_discovery_agent
from .research.document_processor import document_processing_agent
from .research.semantic_analyzer import semantic_analysis_agent
from .research.summarization import summarization_agent

# Context Engineering agents - DISABLED (auto-indexes without user consent)
# These agents auto-index cited papers which causes "Indexing Papers" popups
# to appear without user explicitly selecting papers. Uncomment when user
# preference toggle is implemented.
# from .research.paper_summarizer import paper_summarizer_agent
# from .research.citation_extractor import citation_extractor_agent
# from .research.linked_paper_indexer import linked_paper_indexer_agent

from .interaction.qa_specialist import qa_specialist_agent
from .interaction.research_advisor import research_advisor_agent

__all__ = [
    # Research agents
    'paper_discovery_agent',
    'document_processing_agent',
    'semantic_analysis_agent',
    'summarization_agent',

    # Context Engineering - DISABLED (requires user consent)
    # 'paper_summarizer_agent',
    # 'citation_extractor_agent',
    # 'linked_paper_indexer_agent',

    # Interaction agents
    'qa_specialist_agent',
    'research_advisor_agent'
]