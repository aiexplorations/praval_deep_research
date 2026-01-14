"""
Praval Tools for Deep Research.

This module provides tools that can be used by agents for various operations.
Tools are decorated with @tool and registered in the Praval tool registry.

Available tools:
- search_conversations: BM25 search over conversation history
- search_paper_chunks: BM25 search over paper content
- list_knowledge_base: List papers in the knowledge base
- get_paper_details: Get detailed info about a specific paper
"""

from tools.search_conversations import search_conversations
from tools.search_paper_chunks import search_paper_chunks
from tools.list_knowledge_base import list_knowledge_base, get_paper_details

__all__ = [
    "search_conversations",
    "search_paper_chunks",
    "list_knowledge_base",
    "get_paper_details",
]
