"""
Praval agents for the agentic research system.

This module contains specialized agents organized by domain:
- research/: Research domain agents (paper discovery, processing, analysis)
- interaction/: User interaction agents (Q&A, advisory)

Agents self-organize through spore communication - no central coordination needed.
"""

# Import all agents for easy access
from .research.paper_discovery import paper_discovery_agent
from .research.document_processor import document_processing_agent
from .research.semantic_analyzer import semantic_analysis_agent
from .research.summarization import summarization_agent
from .interaction.qa_specialist import qa_specialist_agent
from .interaction.research_advisor import research_advisor_agent

__all__ = [
    # Research agents
    'paper_discovery_agent',
    'document_processing_agent', 
    'semantic_analysis_agent',
    'summarization_agent',
    
    # Interaction agents
    'qa_specialist_agent',
    'research_advisor_agent'
]