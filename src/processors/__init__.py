"""
Processing components for research operations.

This module provides document and data processing functionality
for the agentic research system.
"""

from .arxiv_client import (
    ArXivClient,
    ArXivAPIError,
    get_arxiv_client,
    search_arxiv_papers,
    get_paper_details,
    calculate_paper_relevance
)

__all__ = [
    'ArXivClient',
    'ArXivAPIError', 
    'get_arxiv_client',
    'search_arxiv_papers',
    'get_paper_details',
    'calculate_paper_relevance'
]