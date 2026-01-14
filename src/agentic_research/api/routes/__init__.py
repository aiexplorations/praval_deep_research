"""
API routes for the Praval Deep Research system.
"""

from .health import router as health_router
from .research import router as research_router
from .kb_search import router as kb_search_router

__all__ = ["health_router", "research_router", "kb_search_router"]