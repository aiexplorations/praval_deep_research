"""
Praval-based agents for the Praval Deep Research system.

This package provides autonomous research agents built on Praval 0.6.0
that self-organize through spore communication for intelligent research workflows.
"""

from .research_agent import (
    ResearchContext,
    ResearchReef,
    paper_search_agent,
    document_processing_agent,
    qa_specialist_agent,
    start_distributed_research,
    ask_distributed_question,
    research_and_ask_distributed,
    initialize_research_network,
    shutdown_research_network
)

__all__ = [
    "ResearchContext",
    "ResearchReef", 
    "paper_search_agent",
    "document_processing_agent", 
    "qa_specialist_agent",
    "start_distributed_research",
    "ask_distributed_question",
    "research_and_ask_distributed",
    "initialize_research_network",
    "shutdown_research_network"
]