"""
Extraction module for LangExtract-based structured information extraction.

This module provides schemas, examples, and processors for extracting
structured entities from research papers using LangExtract.
"""

from .schemas import (
    ExtractionType,
    SourceSpan,
    Extraction,
    PaperExtractions,
    METHOD_SCHEMA,
    DATASET_SCHEMA,
    FINDING_SCHEMA,
    CITATION_SCHEMA,
    METRIC_SCHEMA,
    LIMITATION_SCHEMA,
)

from .langextract_processor import LangExtractProcessor

__all__ = [
    "ExtractionType",
    "SourceSpan",
    "Extraction",
    "PaperExtractions",
    "METHOD_SCHEMA",
    "DATASET_SCHEMA",
    "FINDING_SCHEMA",
    "CITATION_SCHEMA",
    "METRIC_SCHEMA",
    "LIMITATION_SCHEMA",
    "LangExtractProcessor",
]
