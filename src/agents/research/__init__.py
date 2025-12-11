"""
Research domain agents.

These agents specialize in academic research tasks:
- Paper discovery and retrieval
- Document processing and parsing
- Semantic analysis and clustering
- Summarization and synthesis

Context Engineering Pipeline:
document_processor -> paper_summarizer -> citation_extractor -> linked_paper_indexer
"""

from .paper_discovery import paper_discovery_agent
from .document_processor import document_processing_agent
from .semantic_analyzer import semantic_analysis_agent
from .summarization import summarization_agent
# Context Engineering agents
from .paper_summarizer import paper_summarizer_agent
from .citation_extractor import citation_extractor_agent
from .linked_paper_indexer import linked_paper_indexer_agent

__all__ = [
    'paper_discovery_agent',
    'document_processing_agent',
    'semantic_analysis_agent',
    'summarization_agent',
    # Context Engineering
    'paper_summarizer_agent',
    'citation_extractor_agent',
    'linked_paper_indexer_agent',
]