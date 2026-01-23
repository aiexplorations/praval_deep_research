"""
Knowledge Graph Client for Kay-Gee-Go integration.

This module provides a client for sending LangExtract extractions
to the Kay-Gee-Go knowledge graph builder and querying the graph.
"""

import httpx
from typing import List, Dict, Any, Optional
import structlog
from dataclasses import dataclass

from agentic_research.core.config import get_settings


logger = structlog.get_logger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    label: str
    properties: Dict[str, Any]


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any] = None


@dataclass
class GraphData:
    """Represents a subgraph from the knowledge graph."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {"id": n.id, "label": n.label, "properties": n.properties}
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relationship": e.relationship,
                    "properties": e.properties or {}
                }
                for e in self.edges
            ]
        }


class KnowledgeGraphClient:
    """
    Client for sending extractions to Kay-Gee-Go and querying the graph.

    The Kay-Gee-Go service builds and maintains a Neo4j knowledge graph
    from structured paper extractions.
    """

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the Knowledge Graph client.

        Args:
            base_url: Kay-Gee-Go service URL. Uses config default if None.
        """
        self.settings = get_settings()
        self.base_url = base_url or self.settings.KG_BUILDER_URL
        self.enabled = self.settings.KG_ENABLED
        self.timeout = 30.0

        logger.info(
            "Knowledge Graph client initialized",
            base_url=self.base_url,
            enabled=self.enabled
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get async HTTP client."""
        return httpx.AsyncClient(timeout=self.timeout)

    def _get_sync_client(self) -> httpx.Client:
        """Get sync HTTP client."""
        return httpx.Client(timeout=self.timeout)

    async def ingest_paper(
        self,
        paper_id: str,
        title: str,
        authors: List[str],
        categories: List[str],
        extractions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Send paper extractions to Kay-Gee-Go for graph creation.

        Args:
            paper_id: Unique paper identifier (ArXiv ID)
            title: Paper title
            authors: List of author names
            categories: ArXiv categories (for color coding)
            extractions: List of extraction dictionaries from LangExtract

        Returns:
            Response from Kay-Gee-Go with graph creation results
        """
        if not self.enabled:
            logger.debug("Knowledge graph disabled, skipping ingest")
            return {"status": "disabled", "message": "KG integration disabled"}

        async with await self._get_client() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/ingest-paper",
                    json={
                        "paper_id": paper_id,
                        "title": title,
                        "authors": authors,
                        "categories": categories,
                        "extractions": extractions
                    }
                )
                response.raise_for_status()

                logger.info(
                    "Paper ingested to knowledge graph",
                    paper_id=paper_id,
                    extraction_count=len(extractions)
                )

                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(
                    "Failed to ingest paper to KG",
                    paper_id=paper_id,
                    status_code=e.response.status_code,
                    error=str(e)
                )
                return {
                    "status": "error",
                    "message": str(e),
                    "paper_id": paper_id
                }

            except httpx.RequestError as e:
                logger.error(
                    "KG request failed",
                    paper_id=paper_id,
                    error=str(e)
                )
                return {
                    "status": "error",
                    "message": f"Connection failed: {str(e)}",
                    "paper_id": paper_id
                }

    def ingest_paper_sync(
        self,
        paper_id: str,
        title: str,
        authors: List[str],
        categories: List[str],
        extractions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synchronous version of ingest_paper.

        Used by agents that run in sync context.
        """
        if not self.enabled:
            logger.debug("Knowledge graph disabled, skipping ingest")
            return {"status": "disabled", "message": "KG integration disabled"}

        with self._get_sync_client() as client:
            try:
                response = client.post(
                    f"{self.base_url}/api/ingest-paper",
                    json={
                        "paper_id": paper_id,
                        "title": title,
                        "authors": authors,
                        "categories": categories,
                        "extractions": extractions
                    }
                )
                response.raise_for_status()

                logger.info(
                    "Paper ingested to knowledge graph (sync)",
                    paper_id=paper_id,
                    extraction_count=len(extractions)
                )

                return response.json()

            except Exception as e:
                logger.error(
                    "KG ingest failed (sync)",
                    paper_id=paper_id,
                    error=str(e)
                )
                return {
                    "status": "error",
                    "message": str(e),
                    "paper_id": paper_id
                }

    async def get_graph(
        self,
        paper_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        limit: int = 200
    ) -> GraphData:
        """
        Fetch graph data for visualization.

        Args:
            paper_ids: Optional filter by paper IDs
            categories: Optional filter by ArXiv categories
            limit: Maximum number of nodes to return

        Returns:
            GraphData containing nodes and edges
        """
        if not self.enabled:
            return GraphData(nodes=[], edges=[])

        async with await self._get_client() as client:
            try:
                params = {"limit": limit}
                if paper_ids:
                    params["paper_ids"] = ",".join(paper_ids)
                if categories:
                    params["categories"] = ",".join(categories)

                response = await client.get(
                    f"{self.base_url}/api/graph",
                    params=params
                )
                response.raise_for_status()

                data = response.json()
                nodes = [
                    GraphNode(
                        id=n["id"],
                        label=n["label"],
                        properties=n.get("properties", {})
                    )
                    for n in data.get("nodes", [])
                ]
                edges = [
                    GraphEdge(
                        source=e["source"],
                        target=e["target"],
                        relationship=e["relationship"],
                        properties=e.get("properties", {})
                    )
                    for e in data.get("edges", [])
                ]

                logger.info(
                    "Retrieved graph data",
                    node_count=len(nodes),
                    edge_count=len(edges)
                )

                return GraphData(nodes=nodes, edges=edges)

            except Exception as e:
                logger.error("Failed to get graph data", error=str(e))
                return GraphData(nodes=[], edges=[])

    async def get_paper_graph(self, paper_id: str) -> GraphData:
        """
        Get graph centered on a specific paper.

        Args:
            paper_id: Paper identifier

        Returns:
            GraphData for the paper and its connected entities
        """
        if not self.enabled:
            return GraphData(nodes=[], edges=[])

        async with await self._get_client() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/api/graph/paper/{paper_id}"
                )
                response.raise_for_status()

                data = response.json()
                nodes = [
                    GraphNode(
                        id=n["id"],
                        label=n["label"],
                        properties=n.get("properties", {})
                    )
                    for n in data.get("nodes", [])
                ]
                edges = [
                    GraphEdge(
                        source=e["source"],
                        target=e["target"],
                        relationship=e["relationship"],
                        properties=e.get("properties", {})
                    )
                    for e in data.get("edges", [])
                ]

                return GraphData(nodes=nodes, edges=edges)

            except Exception as e:
                logger.error(
                    "Failed to get paper graph",
                    paper_id=paper_id,
                    error=str(e)
                )
                return GraphData(nodes=[], edges=[])

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge graph statistics.

        Returns:
            Dictionary with node counts, edge counts, etc.
        """
        if not self.enabled:
            return {
                "status": "disabled",
                "nodes": 0,
                "edges": 0
            }

        async with await self._get_client() as client:
            try:
                response = await client.get(f"{self.base_url}/api/stats")
                response.raise_for_status()

                return response.json()

            except Exception as e:
                logger.error("Failed to get KG stats", error=str(e))
                return {
                    "status": "error",
                    "error": str(e)
                }

    async def search(
        self,
        query: str,
        node_types: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[GraphNode]:
        """
        Search the knowledge graph for nodes matching a query.

        Args:
            query: Search query
            node_types: Optional filter by node types (Paper, Method, Dataset, etc.)
            limit: Maximum number of results

        Returns:
            List of matching nodes
        """
        if not self.enabled:
            return []

        async with await self._get_client() as client:
            try:
                params = {"query": query, "limit": limit}
                if node_types:
                    params["node_types"] = ",".join(node_types)

                response = await client.get(
                    f"{self.base_url}/api/search",
                    params=params
                )
                response.raise_for_status()

                data = response.json()
                nodes = [
                    GraphNode(
                        id=n["id"],
                        label=n["label"],
                        properties=n.get("properties", {})
                    )
                    for n in data.get("results", [])
                ]

                return nodes

            except Exception as e:
                logger.error("KG search failed", query=query, error=str(e))
                return []

    async def health_check(self) -> bool:
        """
        Check if Kay-Gee-Go service is healthy.

        Returns:
            True if service is healthy
        """
        if not self.enabled:
            return False

        async with await self._get_client() as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
            except Exception:
                return False

    def health_check_sync(self) -> bool:
        """Synchronous health check."""
        if not self.enabled:
            return False

        with self._get_sync_client() as client:
            try:
                response = client.get(f"{self.base_url}/health")
                return response.status_code == 200
            except Exception:
                return False

    async def get_methods(self, limit: int = 50) -> List[GraphNode]:
        """Get all Method nodes in the graph."""
        return await self.search("", node_types=["Method"], limit=limit)

    async def get_datasets(self, limit: int = 50) -> List[GraphNode]:
        """Get all Dataset nodes in the graph."""
        return await self.search("", node_types=["Dataset"], limit=limit)

    async def get_papers(self, limit: int = 50) -> List[GraphNode]:
        """Get all Paper nodes in the graph."""
        return await self.search("", node_types=["Paper"], limit=limit)

    async def get_related_entities(
        self,
        paper_id: str,
        entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get entities related to a paper.

        Args:
            paper_id: Paper identifier
            entity_type: Optional filter by entity type (Method, Dataset, etc.)

        Returns:
            List of related entities with relationship info
        """
        if not self.enabled:
            return []

        async with await self._get_client() as client:
            try:
                params = {}
                if entity_type:
                    params["entity_type"] = entity_type

                response = await client.get(
                    f"{self.base_url}/api/paper/{paper_id}/entities",
                    params=params
                )
                response.raise_for_status()

                return response.json().get("entities", [])

            except Exception as e:
                logger.error(
                    "Failed to get related entities",
                    paper_id=paper_id,
                    error=str(e)
                )
                return []
