"""
Extraction schemas for academic paper entity extraction using LangExtract.

Defines the entity types, data models, and schema specifications
for extracting structured information from research papers.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


class ExtractionType(str, Enum):
    """Types of entities that can be extracted from research papers."""

    METHOD = "method"
    DATASET = "dataset"
    FINDING = "finding"
    CITATION = "citation"
    METRIC = "metric"
    LIMITATION = "limitation"
    APPLICATION = "application"
    AUTHOR = "author"
    CONTRIBUTION = "contribution"


@dataclass
class SourceSpan:
    """Source location in the original text for grounding."""

    start: int
    end: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass
class Extraction:
    """A single extracted entity with source grounding."""

    extraction_type: ExtractionType
    name: str
    content: str
    source_span: SourceSpan
    confidence: float = 0.8
    attributes: Dict[str, Any] = field(default_factory=dict)
    paper_id: Optional[str] = None
    extracted_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.extraction_type.value,
            "name": self.name,
            "content": self.content,
            "source_span": self.source_span.to_dict(),
            "confidence": self.confidence,
            "attributes": self.attributes,
            "paper_id": self.paper_id,
            "extracted_at": self.extracted_at.isoformat() if self.extracted_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Extraction":
        """Create an Extraction from a dictionary."""
        source_span_data = data.get("source_span", {})
        source_span = SourceSpan(
            start=source_span_data.get("start", 0),
            end=source_span_data.get("end", 0),
            text=source_span_data.get("text", ""),
        )
        extracted_at = None
        if data.get("extracted_at"):
            extracted_at = datetime.fromisoformat(data["extracted_at"])

        return cls(
            extraction_type=ExtractionType(data.get("type", "method")),
            name=data.get("name", ""),
            content=data.get("content", ""),
            source_span=source_span,
            confidence=data.get("confidence", 0.8),
            attributes=data.get("attributes", {}),
            paper_id=data.get("paper_id"),
            extracted_at=extracted_at,
        )


@dataclass
class PaperExtractions:
    """Collection of all extractions from a single paper."""

    paper_id: str
    title: str
    extractions: List[Extraction]
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)

    def get_by_type(self, extraction_type: ExtractionType) -> List[Extraction]:
        """Get all extractions of a specific type."""
        return [e for e in self.extractions if e.extraction_type == extraction_type]

    def get_methods(self) -> List[Extraction]:
        return self.get_by_type(ExtractionType.METHOD)

    def get_datasets(self) -> List[Extraction]:
        return self.get_by_type(ExtractionType.DATASET)

    def get_findings(self) -> List[Extraction]:
        return self.get_by_type(ExtractionType.FINDING)

    def get_citations(self) -> List[Extraction]:
        return self.get_by_type(ExtractionType.CITATION)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "extractions": [e.to_dict() for e in self.extractions],
            "extraction_metadata": self.extraction_metadata,
            "summary": {
                "total": len(self.extractions),
                "methods": len(self.get_methods()),
                "datasets": len(self.get_datasets()),
                "findings": len(self.get_findings()),
                "citations": len(self.get_citations()),
            },
        }


# Schema specifications for LangExtract
# These define the structure expected for each extraction type

METHOD_SCHEMA = {
    "type": "method",
    "description": "A proposed technique, algorithm, or methodology described in the paper",
    "fields": {
        "name": {
            "type": "string",
            "description": "Name of the method or technique (e.g., 'BERT', 'Transformer', 'Adam optimizer')",
            "required": True,
        },
        "description": {
            "type": "string",
            "description": "Brief description of what the method does and how it works",
            "required": True,
        },
        "domain": {
            "type": "string",
            "description": "Research domain or field (e.g., 'NLP', 'Computer Vision', 'Optimization')",
            "required": False,
        },
        "novelty": {
            "type": "string",
            "description": "What makes this method novel or different from prior work",
            "required": False,
        },
        "components": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key components or building blocks of the method",
            "required": False,
        },
    },
    "examples": [
        {
            "name": "BERT",
            "description": "Bidirectional Encoder Representations from Transformers for language understanding",
            "domain": "NLP",
            "novelty": "Uses bidirectional pre-training unlike previous unidirectional models",
            "components": ["Transformer encoder", "Masked language modeling", "Next sentence prediction"],
        }
    ],
}

DATASET_SCHEMA = {
    "type": "dataset",
    "description": "A dataset used for training, evaluation, or benchmarking",
    "fields": {
        "name": {
            "type": "string",
            "description": "Name of the dataset (e.g., 'ImageNet', 'GLUE', 'SQuAD')",
            "required": True,
        },
        "description": {
            "type": "string",
            "description": "Brief description of the dataset contents and purpose",
            "required": True,
        },
        "size": {
            "type": "string",
            "description": "Size of the dataset (e.g., '1M images', '100K examples')",
            "required": False,
        },
        "domain": {
            "type": "string",
            "description": "Domain or task the dataset covers",
            "required": False,
        },
        "usage": {
            "type": "string",
            "description": "How the dataset is used in this paper (training, evaluation, etc.)",
            "required": False,
        },
    },
    "examples": [
        {
            "name": "SQuAD 2.0",
            "description": "Stanford Question Answering Dataset with unanswerable questions",
            "size": "150K question-answer pairs",
            "domain": "Reading comprehension",
            "usage": "Evaluation benchmark for question answering models",
        }
    ],
}

FINDING_SCHEMA = {
    "type": "finding",
    "description": "A key result, conclusion, or claim made in the paper",
    "fields": {
        "claim": {
            "type": "string",
            "description": "The main finding or claim being made",
            "required": True,
        },
        "evidence": {
            "type": "string",
            "description": "Evidence or experiments supporting this finding",
            "required": True,
        },
        "significance": {
            "type": "string",
            "description": "Why this finding is significant or impactful",
            "required": False,
        },
        "conditions": {
            "type": "string",
            "description": "Conditions or constraints under which this finding holds",
            "required": False,
        },
    },
    "examples": [
        {
            "claim": "BERT achieves state-of-the-art results on 11 NLP tasks",
            "evidence": "Experiments on GLUE, SQuAD, and SWAG benchmarks",
            "significance": "Demonstrates the effectiveness of bidirectional pre-training",
            "conditions": "With fine-tuning on task-specific data",
        }
    ],
}

CITATION_SCHEMA = {
    "type": "citation",
    "description": "A reference to another work cited in the paper",
    "fields": {
        "cited_work": {
            "type": "string",
            "description": "Title or name of the cited work",
            "required": True,
        },
        "context": {
            "type": "string",
            "description": "Context in which the citation appears",
            "required": True,
        },
        "relationship": {
            "type": "string",
            "description": "Relationship type (extends, compares, contrasts, builds_upon, etc.)",
            "required": False,
        },
        "cited_paper_id": {
            "type": "string",
            "description": "ArXiv ID or DOI of the cited paper if available",
            "required": False,
        },
    },
    "examples": [
        {
            "cited_work": "Attention Is All You Need",
            "context": "We build upon the Transformer architecture introduced in...",
            "relationship": "builds_upon",
            "cited_paper_id": "1706.03762",
        }
    ],
}

METRIC_SCHEMA = {
    "type": "metric",
    "description": "An evaluation metric or performance measure reported in the paper",
    "fields": {
        "name": {
            "type": "string",
            "description": "Name of the metric (e.g., 'Accuracy', 'F1 Score', 'BLEU')",
            "required": True,
        },
        "value": {
            "type": "string",
            "description": "Reported value of the metric",
            "required": True,
        },
        "dataset": {
            "type": "string",
            "description": "Dataset or benchmark on which this was measured",
            "required": False,
        },
        "comparison": {
            "type": "string",
            "description": "Comparison to baseline or prior work",
            "required": False,
        },
    },
    "examples": [
        {
            "name": "F1 Score",
            "value": "93.2%",
            "dataset": "SQuAD 2.0",
            "comparison": "+5.1 points over previous SOTA",
        }
    ],
}

LIMITATION_SCHEMA = {
    "type": "limitation",
    "description": "A limitation, weakness, or constraint acknowledged by the authors",
    "fields": {
        "description": {
            "type": "string",
            "description": "Description of the limitation",
            "required": True,
        },
        "impact": {
            "type": "string",
            "description": "Impact or consequences of this limitation",
            "required": False,
        },
        "mitigation": {
            "type": "string",
            "description": "Any proposed mitigations or future work to address it",
            "required": False,
        },
    },
    "examples": [
        {
            "description": "BERT requires significant computational resources for pre-training",
            "impact": "Limits accessibility for researchers with limited compute",
            "mitigation": "Future work on more efficient pre-training strategies",
        }
    ],
}

# Combined schema for full paper extraction
PAPER_EXTRACTION_SCHEMA = {
    "description": "Extract structured entities from an academic research paper",
    "entity_types": [
        METHOD_SCHEMA,
        DATASET_SCHEMA,
        FINDING_SCHEMA,
        CITATION_SCHEMA,
        METRIC_SCHEMA,
        LIMITATION_SCHEMA,
    ],
    "extraction_guidelines": """
    Extract the following types of entities from the research paper:

    1. METHODS: Proposed techniques, algorithms, or architectures
       - Focus on novel contributions and key technical approaches
       - Include the method's name, description, and components

    2. DATASETS: Data sources used for training or evaluation
       - Include dataset name, size, and how it's used
       - Note both standard benchmarks and custom datasets

    3. FINDINGS: Key results and conclusions
       - Focus on main claims with supporting evidence
       - Note significance and any conditions/limitations

    4. CITATIONS: Important references to prior work
       - Focus on citations that establish context or comparison
       - Identify the relationship (extends, contrasts, builds_upon)

    5. METRICS: Quantitative results and performance measures
       - Include metric name, value, and dataset
       - Note comparisons to baselines when available

    6. LIMITATIONS: Acknowledged weaknesses or constraints
       - Include impact and any proposed mitigations

    For each extraction, provide the exact source text location for grounding.
    """,
}
