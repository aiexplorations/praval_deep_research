"""
Knowledge Graph API routes.

Provides endpoints for viewing and interacting with the
knowledge graph built from LangExtract extractions.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import structlog

from agentic_research.storage.kg_client import KnowledgeGraphClient
from agentic_research.storage.qdrant_client import QdrantClientWrapper
from agentic_research.core.config import get_settings


logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph"])
settings = get_settings()


# Response models
class GraphNodeResponse(BaseModel):
    id: str
    label: str
    properties: dict = Field(default_factory=dict)


class GraphEdgeResponse(BaseModel):
    source: str
    target: str
    relationship: str
    properties: dict = Field(default_factory=dict)


class GraphDataResponse(BaseModel):
    nodes: List[GraphNodeResponse]
    edges: List[GraphEdgeResponse]
    node_count: int
    edge_count: int


class GraphStatsResponse(BaseModel):
    status: str
    paper_count: int = 0
    method_count: int = 0
    dataset_count: int = 0
    finding_count: int = 0
    author_count: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    kg_enabled: bool = True


class ExtractionResponse(BaseModel):
    paper_id: str
    extraction_type: str
    name: str
    content: str
    confidence: float
    attributes: dict = Field(default_factory=dict)
    source_span: dict = Field(default_factory=dict)


class PaperExtractionsResponse(BaseModel):
    paper_id: str
    extractions: List[ExtractionResponse]
    total: int


# Lazy-initialized clients
_kg_client: Optional[KnowledgeGraphClient] = None
_qdrant_client: Optional[QdrantClientWrapper] = None


def get_kg_client() -> KnowledgeGraphClient:
    global _kg_client
    if _kg_client is None:
        _kg_client = KnowledgeGraphClient()
    return _kg_client


def get_qdrant_client() -> QdrantClientWrapper:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClientWrapper(settings)
    return _qdrant_client


@router.get("/status")
async def get_kg_status() -> dict:
    """Get knowledge graph service status."""
    kg_client = get_kg_client()
    is_healthy = await kg_client.health_check()

    return {
        "enabled": settings.KG_ENABLED,
        "healthy": is_healthy,
        "kg_builder_url": settings.KG_BUILDER_URL,
        "neo4j_uri": settings.NEO4J_URI
    }


@router.get("/stats", response_model=GraphStatsResponse)
async def get_kg_stats() -> GraphStatsResponse:
    """Get knowledge graph statistics."""
    kg_client = get_kg_client()
    qdrant_client = get_qdrant_client()

    try:
        # Get stats from Kay-Gee-Go if available
        kg_stats = await kg_client.get_stats()

        # Also get extraction count from Qdrant
        extraction_count = 0
        try:
            # Get extractions collection info
            collections_info = qdrant_client.get_all_collections_info()
            extraction_info = collections_info.get("paper_extractions", {})
            if extraction_info.get("vectors_count"):
                extraction_count = extraction_info["vectors_count"]
        except Exception:
            pass

        return GraphStatsResponse(
            status=kg_stats.get("status", "unknown"),
            paper_count=kg_stats.get("paper_count", 0),
            method_count=kg_stats.get("method_count", 0),
            dataset_count=kg_stats.get("dataset_count", 0),
            finding_count=kg_stats.get("finding_count", 0),
            author_count=kg_stats.get("author_count", 0),
            total_nodes=kg_stats.get("total_nodes", extraction_count),
            total_edges=kg_stats.get("total_edges", 0),
            kg_enabled=settings.KG_ENABLED
        )

    except Exception as e:
        logger.error("Failed to get KG stats", error=str(e))
        return GraphStatsResponse(
            status="error",
            kg_enabled=settings.KG_ENABLED
        )


@router.get("/graph", response_model=GraphDataResponse)
async def get_graph(
    paper_ids: Optional[str] = Query(None, description="Comma-separated paper IDs"),
    categories: Optional[str] = Query(None, description="Comma-separated categories"),
    limit: int = Query(200, ge=1, le=1000, description="Maximum nodes to return")
) -> GraphDataResponse:
    """Get graph data for visualization."""
    kg_client = get_kg_client()

    paper_id_list = paper_ids.split(",") if paper_ids else None
    category_list = categories.split(",") if categories else None

    graph_data = await kg_client.get_graph(
        paper_ids=paper_id_list,
        categories=category_list,
        limit=limit
    )

    return GraphDataResponse(
        nodes=[
            GraphNodeResponse(
                id=n.id,
                label=n.label,
                properties=n.properties
            )
            for n in graph_data.nodes
        ],
        edges=[
            GraphEdgeResponse(
                source=e.source,
                target=e.target,
                relationship=e.relationship,
                properties=e.properties or {}
            )
            for e in graph_data.edges
        ],
        node_count=len(graph_data.nodes),
        edge_count=len(graph_data.edges)
    )


@router.get("/papers/{paper_id}/graph", response_model=GraphDataResponse)
async def get_paper_graph(paper_id: str) -> GraphDataResponse:
    """Get graph centered on a specific paper."""
    kg_client = get_kg_client()

    graph_data = await kg_client.get_paper_graph(paper_id)

    return GraphDataResponse(
        nodes=[
            GraphNodeResponse(
                id=n.id,
                label=n.label,
                properties=n.properties
            )
            for n in graph_data.nodes
        ],
        edges=[
            GraphEdgeResponse(
                source=e.source,
                target=e.target,
                relationship=e.relationship,
                properties=e.properties or {}
            )
            for e in graph_data.edges
        ],
        node_count=len(graph_data.nodes),
        edge_count=len(graph_data.edges)
    )


@router.get("/search")
async def search_graph(
    query: str = Query(..., min_length=1, description="Search query"),
    node_types: Optional[str] = Query(None, description="Comma-separated node types"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results")
) -> dict:
    """Search the knowledge graph."""
    kg_client = get_kg_client()

    node_type_list = node_types.split(",") if node_types else None

    nodes = await kg_client.search(
        query=query,
        node_types=node_type_list,
        limit=limit
    )

    return {
        "query": query,
        "results": [
            {
                "id": n.id,
                "label": n.label,
                "properties": n.properties
            }
            for n in nodes
        ],
        "total": len(nodes)
    }


@router.get("/papers/{paper_id}/extractions", response_model=PaperExtractionsResponse)
async def get_paper_extractions(paper_id: str) -> PaperExtractionsResponse:
    """Get all extractions for a specific paper from Qdrant."""
    qdrant_client = get_qdrant_client()

    try:
        extractions = qdrant_client.get_paper_extractions(paper_id)

        return PaperExtractionsResponse(
            paper_id=paper_id,
            extractions=[
                ExtractionResponse(
                    paper_id=e.get("paper_id", paper_id),
                    extraction_type=e.get("extraction_type", "unknown"),
                    name=e.get("name", ""),
                    content=e.get("content", ""),
                    confidence=e.get("confidence", 0.0),
                    attributes=e.get("attributes", {}),
                    source_span=e.get("source_span", {})
                )
                for e in extractions
            ],
            total=len(extractions)
        )

    except Exception as e:
        logger.error("Failed to get paper extractions", paper_id=paper_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/extractions")
async def list_all_extractions(
    extraction_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
) -> dict:
    """List all extractions across all papers."""
    qdrant_client = get_qdrant_client()

    try:
        # Get all papers and their extractions
        papers = qdrant_client.get_all_papers()
        all_extractions = []

        for paper in papers[:20]:  # Limit to 20 papers for performance
            paper_id = paper.get("paper_id", "")
            if paper_id:
                extractions = qdrant_client.get_paper_extractions(paper_id)
                for ext in extractions:
                    if extraction_type is None or ext.get("extraction_type") == extraction_type:
                        all_extractions.append({
                            **ext,
                            "paper_title": paper.get("title", "Unknown")
                        })

        # Apply pagination
        paginated = all_extractions[offset:offset + limit]

        return {
            "extractions": paginated,
            "total": len(all_extractions),
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error("Failed to list extractions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods")
async def list_methods(
    limit: int = Query(50, ge=1, le=100, description="Maximum results")
) -> dict:
    """Get all methods extracted from papers."""
    kg_client = get_kg_client()

    nodes = await kg_client.get_methods(limit=limit)

    return {
        "methods": [
            {
                "id": n.id,
                "name": n.label,
                "properties": n.properties
            }
            for n in nodes
        ],
        "total": len(nodes)
    }


@router.get("/datasets")
async def list_datasets(
    limit: int = Query(50, ge=1, le=100, description="Maximum results")
) -> dict:
    """Get all datasets extracted from papers."""
    kg_client = get_kg_client()

    nodes = await kg_client.get_datasets(limit=limit)

    return {
        "datasets": [
            {
                "id": n.id,
                "name": n.label,
                "properties": n.properties
            }
            for n in nodes
        ],
        "total": len(nodes)
    }


@router.get("/category/{category}")
async def get_category_graph(
    category: str,
    limit: int = Query(100, ge=1, le=500, description="Maximum nodes")
) -> GraphDataResponse:
    """Get graph for papers in a specific ArXiv category."""
    kg_client = get_kg_client()

    graph_data = await kg_client.get_graph(categories=[category], limit=limit)

    return GraphDataResponse(
        nodes=[
            GraphNodeResponse(
                id=n.id,
                label=n.label,
                properties=n.properties
            )
            for n in graph_data.nodes
        ],
        edges=[
            GraphEdgeResponse(
                source=e.source,
                target=e.target,
                relationship=e.relationship,
                properties=e.properties or {}
            )
            for e in graph_data.edges
        ],
        node_count=len(graph_data.nodes),
        edge_count=len(graph_data.edges)
    )
