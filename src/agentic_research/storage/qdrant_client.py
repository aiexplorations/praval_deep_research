"""
Qdrant vector database client for semantic search.

This module provides a client for interacting with Qdrant vector database
for storing and retrieving paper embeddings.

Supports multiple collections:
- research_vectors: Main paper chunks with embeddings
- paper_summaries: One embedding per paper for fast retrieval
- linked_papers: Fully indexed cited papers (1-hop from KB)
"""

from typing import List, Dict, Any, Optional
from uuid import uuid4
from enum import Enum
import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
    Batch
)
from qdrant_client.http.exceptions import UnexpectedResponse

from agentic_research.core.config import get_settings


logger = structlog.get_logger(__name__)


class CollectionType(str, Enum):
    """Qdrant collection types for context engineering."""
    RESEARCH_PAPERS = "research_papers"  # Main KB paper chunks
    PAPER_SUMMARIES = "paper_summaries"  # One summary per paper for fast retrieval
    LINKED_PAPERS = "linked_papers"      # Fully indexed cited papers
    PAPER_EXTRACTIONS = "paper_extractions"  # Structured entity extractions


class QdrantClientWrapper:
    """
    Wrapper for Qdrant vector database operations.

    I am a Qdrant vector database client who handles embedding storage,
    similarity search, and vector management for research papers.
    """

    def __init__(self, settings=None, collection_type: CollectionType = None):
        """
        Initialize Qdrant client.

        Args:
            settings: Optional settings object (uses get_settings() if None)
            collection_type: Optional collection type for multi-collection support.
                           If None, uses the default research_vectors collection.
        """
        self.settings = settings or get_settings()

        # Map collection types to configured collection names
        self._collection_names = {
            CollectionType.RESEARCH_PAPERS: self.settings.QDRANT_COLLECTION_NAME,
            CollectionType.PAPER_SUMMARIES: self.settings.QDRANT_SUMMARIES_COLLECTION,
            CollectionType.LINKED_PAPERS: self.settings.QDRANT_LINKED_PAPERS_COLLECTION,
            CollectionType.PAPER_EXTRACTIONS: self.settings.QDRANT_EXTRACTIONS_COLLECTION,
        }

        # Set current collection - defaults to research papers for backward compatibility
        self._collection_type = collection_type
        if collection_type:
            self.collection_name = self._collection_names.get(
                collection_type, self.settings.QDRANT_COLLECTION_NAME
            )
        else:
            self.collection_name = self.settings.QDRANT_COLLECTION_NAME

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.settings.QDRANT_URL,
            api_key=self.settings.QDRANT_API_KEY
        )

        logger.info(
            "Qdrant client initialized",
            url=self.settings.QDRANT_URL,
            collection=self.collection_name,
            collection_type=collection_type.value if collection_type else "default"
        )

    def get_collection_name(self, collection_type: CollectionType) -> str:
        """Get the configured collection name for a collection type."""
        return self._collection_names.get(collection_type, self.settings.QDRANT_COLLECTION_NAME)

    def switch_collection(self, collection_type: CollectionType) -> None:
        """Switch the active collection context."""
        self._collection_type = collection_type
        self.collection_name = self._collection_names.get(
            collection_type, self.settings.QDRANT_COLLECTION_NAME
        )
        logger.info("Switched collection", collection=self.collection_name, type=collection_type.value)

    def create_collection(
        self,
        vector_size: int = 1536,
        distance: Distance = Distance.COSINE,
        recreate: bool = False
    ) -> None:
        """
        Create collection for storing embeddings.

        Args:
            vector_size: Dimension of vectors (default: 1536 for OpenAI text-embedding-3-small)
            distance: Distance metric (default: COSINE)
            recreate: Whether to recreate collection if it exists

        Raises:
            UnexpectedResponse: If collection creation fails
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(
                col.name == self.collection_name for col in collections
            )

            if collection_exists:
                if recreate:
                    logger.info(
                        "Deleting existing collection",
                        collection=self.collection_name
                    )
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(
                        "Collection already exists",
                        collection=self.collection_name
                    )
                    return

            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )

            logger.info(
                "Collection created successfully",
                collection=self.collection_name,
                vector_size=vector_size,
                distance=distance.value
            )

        except UnexpectedResponse as e:
            logger.error(
                "Failed to create collection",
                collection=self.collection_name,
                error=str(e)
            )
            raise

    def add_vectors(
        self,
        paper_id: str,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Add vector embeddings for paper chunks.

        Args:
            paper_id: Unique identifier for the paper
            chunks: List of chunk dictionaries containing:
                - chunk_text: The text content
                - chunk_index: Index of chunk in paper
                - embedding: Vector embedding (list of floats)
                - Additional metadata (title, authors, etc.)
            batch_size: Number of vectors to upload per batch

        Returns:
            Number of vectors added

        Raises:
            UnexpectedResponse: If vector addition fails
        """
        try:
            points = []

            for chunk in chunks:
                # Generate unique integer ID (Qdrant requires int or UUID)
                # Use hash of paper_id and chunk_index, ensure positive
                point_id = abs(hash(f"{paper_id}_chunk_{chunk['chunk_index']}")) % (2**63 - 1)

                # Build payload with all metadata
                payload = {
                    "paper_id": paper_id,
                    "chunk_text": chunk["chunk_text"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk.get("total_chunks", len(chunks)),
                }

                # Add optional metadata
                for key in ["title", "authors", "abstract", "categories",
                           "published_date", "pdf_path", "search_context",
                           "processing_timestamp"]:
                    if key in chunk:
                        payload[key] = chunk[key]

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=chunk["embedding"],
                        payload=payload
                    )
                )

            # Upload points in batches
            # FIX: Use upload_points instead of upsert for better compatibility
            # The upsert method with PointStruct can cause JSON serialization issues
            total_uploaded = 0
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]

                # Debug: Log first point structure
                if i == 0 and batch:
                    first_point = batch[0]
                    logger.debug(
                        "First point structure",
                        point_id=first_point.id,
                        point_id_type=type(first_point.id).__name__,
                        vector_length=len(first_point.vector) if first_point.vector else 0,
                        payload_keys=list(first_point.payload.keys()) if first_point.payload else [],
                        has_vector=first_point.vector is not None,
                        vector_is_list=isinstance(first_point.vector, list) if first_point.vector else False
                    )

                try:
                    # Use upload_points for better reliability with Qdrant 1.7.3
                    # This method handles PointStruct serialization correctly
                    result = self.client.upload_points(
                        collection_name=self.collection_name,
                        points=batch,
                        wait=True
                    )
                    total_uploaded += len(batch)

                    logger.debug(
                        "Batch uploaded",
                        paper_id=paper_id,
                        batch_size=len(batch),
                        total_uploaded=total_uploaded
                    )
                except UnexpectedResponse as batch_error:
                    # Log the raw response for debugging
                    logger.error(
                        "Qdrant batch upload failed - UnexpectedResponse",
                        paper_id=paper_id,
                        batch_index=i // batch_size,
                        batch_size=len(batch),
                        error=str(batch_error),
                        first_point_id=batch[0].id if batch else None,
                        first_point_vector_len=len(batch[0].vector) if batch and batch[0].vector else 0,
                        first_point_payload_keys=list(batch[0].payload.keys()) if batch and batch[0].payload else []
                    )
                    raise
                except Exception as batch_error:
                    logger.error(
                        "Batch upload failed - unexpected error",
                        paper_id=paper_id,
                        batch_index=i // batch_size,
                        batch_size=len(batch),
                        error=str(batch_error),
                        error_type=type(batch_error).__name__
                    )
                    raise

            logger.info(
                "Vectors added successfully",
                paper_id=paper_id,
                num_chunks=len(chunks),
                num_vectors=total_uploaded
            )

            return total_uploaded

        except UnexpectedResponse as e:
            logger.error(
                "Failed to add vectors",
                paper_id=paper_id,
                error=str(e)
            )
            raise

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.7,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1)
            filter_dict: Optional filters (e.g., {"paper_id": "2106.04560"})

        Returns:
            List of search results with payload and score

        Raises:
            UnexpectedResponse: If search fails
        """
        try:
            # Build filter if provided
            query_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                query_filter = Filter(must=conditions)

            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False
            )

            results = []
            for hit in search_results:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                results.append(result)

            logger.info(
                "Search completed",
                num_results=len(results),
                score_threshold=score_threshold,
                limit=limit
            )

            return results

        except UnexpectedResponse as e:
            logger.error(
                "Search failed",
                error=str(e),
                score_threshold=score_threshold,
                limit=limit
            )
            raise

    def delete_vectors(self, paper_id: str) -> None:
        """
        Delete all vectors for a paper.

        Args:
            paper_id: Unique identifier for the paper

        Raises:
            UnexpectedResponse: If deletion fails
        """
        try:
            # Delete points with matching paper_id
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="paper_id",
                            match=MatchValue(value=paper_id)
                        )
                    ]
                )
            )

            logger.info(
                "Vectors deleted successfully",
                paper_id=paper_id
            )

        except UnexpectedResponse as e:
            logger.error(
                "Failed to delete vectors",
                paper_id=paper_id,
                error=str(e)
            )
            raise

    def get_paper_chunks(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific paper.

        Args:
            paper_id: Unique identifier for the paper

        Returns:
            List of chunk payloads sorted by chunk_index

        Raises:
            UnexpectedResponse: If retrieval fails
        """
        try:
            # Search with paper_id filter and high limit
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="paper_id",
                            match=MatchValue(value=paper_id)
                        )
                    ]
                ),
                limit=1000,  # Max chunks per paper
                with_payload=True,
                with_vectors=False
            )

            chunks = []
            for point in results[0]:  # results is tuple (points, next_page_offset)
                chunk = {
                    "id": point.id,
                    "payload": point.payload
                }
                chunks.append(chunk)

            # Sort by chunk_index
            chunks.sort(key=lambda x: x["payload"].get("chunk_index", 0))

            logger.info(
                "Retrieved paper chunks",
                paper_id=paper_id,
                num_chunks=len(chunks)
            )

            return chunks

        except UnexpectedResponse as e:
            logger.error(
                "Failed to retrieve paper chunks",
                paper_id=paper_id,
                error=str(e)
            )
            raise

    def collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection statistics

        Raises:
            UnexpectedResponse: If retrieval fails
        """
        try:
            info = self.client.get_collection(self.collection_name)

            return {
                "name": info.name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status.value,
                "optimizer_status": info.optimizer_status.value,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.value
            }

        except UnexpectedResponse as e:
            logger.error(
                "Failed to get collection info",
                collection=self.collection_name,
                error=str(e)
            )
            raise

    def health_check(self) -> bool:
        """
        Check if Qdrant service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get collections list
            self.client.get_collections()
            return True
        except Exception as e:
            logger.warning("Qdrant health check failed", error=str(e))
            return False

    def count_paper_chunks(self, paper_id: str) -> int:
        """
        Count number of chunks stored for a paper.

        Args:
            paper_id: Unique identifier for the paper

        Returns:
            Number of chunks stored

        Raises:
            UnexpectedResponse: If count fails
        """
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="paper_id",
                            match=MatchValue(value=paper_id)
                        )
                    ]
                )
            )

            return result.count

        except UnexpectedResponse as e:
            logger.error(
                "Failed to count paper chunks",
                paper_id=paper_id,
                error=str(e)
            )
            raise

    def get_all_papers(self) -> List[Dict[str, Any]]:
        """
        Get list of all unique papers in the collection with metadata.

        This method scrolls through all points in the collection, groups them
        by paper_id, and returns aggregated metadata for each unique paper.

        Returns:
            List of paper dictionaries containing:
                - paper_id: Unique paper identifier
                - title: Paper title
                - authors: List of authors
                - categories: List of categories
                - published_date: Publication date
                - chunk_count: Number of chunks stored
                - abstract: Paper abstract (truncated to 300 chars)

        Raises:
            UnexpectedResponse: If retrieval fails
        """
        try:
            # Scroll through all points in batches
            offset = None
            all_points = []

            logger.info("Starting to retrieve all papers from collection")

            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                points, next_offset = result
                all_points.extend(points)

                logger.debug(
                    "Retrieved batch of points",
                    batch_size=len(points),
                    total_so_far=len(all_points)
                )

                if next_offset is None:
                    break
                offset = next_offset

            # Group by paper_id
            papers_dict = {}
            for point in all_points:
                payload = point.payload
                paper_id = payload.get('paper_id', 'unknown')

                if paper_id not in papers_dict:
                    papers_dict[paper_id] = {
                        'paper_id': paper_id,
                        'title': payload.get('title', 'Unknown'),
                        'authors': payload.get('authors', []),
                        'categories': payload.get('categories', []),
                        'published_date': payload.get('published_date', ''),
                        'indexed_at': payload.get('processing_timestamp', ''),
                        'chunk_count': 0,
                        'abstract': payload.get('abstract', '')[:300] + '...' if payload.get('abstract') else ''
                    }

                papers_dict[paper_id]['chunk_count'] += 1
                # Update indexed_at if we find a newer timestamp
                current_ts = payload.get('processing_timestamp', '')
                if current_ts and current_ts > papers_dict[paper_id].get('indexed_at', ''):
                    papers_dict[paper_id]['indexed_at'] = current_ts

            # Convert to list and sort by title
            papers = list(papers_dict.values())
            papers.sort(key=lambda x: x['title'])

            logger.info(
                "Retrieved all papers",
                total_papers=len(papers),
                total_chunks=len(all_points)
            )

            return papers

        except UnexpectedResponse as e:
            logger.error(
                "Failed to get all papers",
                error=str(e)
            )
            raise
        except Exception as e:
            logger.error(
                "Unexpected error getting all papers",
                error=str(e)
            )
            return []

    def clear_collection(self) -> None:
        """
        Clear all data from the collection by deleting and recreating it.

        This is a destructive operation that removes all papers and vectors
        from the knowledge base. The collection is immediately recreated
        with the same configuration.

        Raises:
            UnexpectedResponse: If clear operation fails
        """
        try:
            logger.warning("Clearing entire collection - this will delete all data")

            # Delete the collection
            self.client.delete_collection(self.collection_name)

            logger.info("Collection deleted")

            # Recreate the collection with same config
            self.create_collection()

            logger.info("Collection cleared and recreated successfully")

        except UnexpectedResponse as e:
            logger.error(
                "Failed to clear collection",
                error=str(e)
            )
            raise

    def initialize_context_engineering_collections(
        self,
        vector_size: int = 1536,
        distance: Distance = Distance.COSINE
    ) -> Dict[str, bool]:
        """
        Initialize all collections needed for context engineering.

        Creates the paper_summaries and linked_papers collections if they
        don't exist. The main research_vectors collection should already exist.

        Args:
            vector_size: Dimension of vectors (default: 1536 for OpenAI embeddings)
            distance: Distance metric (default: COSINE)

        Returns:
            Dictionary mapping collection names to creation status (True if created)
        """
        results = {}

        for collection_type in [CollectionType.PAPER_SUMMARIES, CollectionType.LINKED_PAPERS]:
            collection_name = self._collection_names[collection_type]

            try:
                # Check if collection exists
                collections = self.client.get_collections().collections
                exists = any(col.name == collection_name for col in collections)

                if exists:
                    logger.info(
                        "Collection already exists",
                        collection=collection_name,
                        type=collection_type.value
                    )
                    results[collection_name] = False
                else:
                    # Create the collection
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=vector_size,
                            distance=distance
                        )
                    )
                    logger.info(
                        "Created context engineering collection",
                        collection=collection_name,
                        type=collection_type.value,
                        vector_size=vector_size
                    )
                    results[collection_name] = True

            except UnexpectedResponse as e:
                logger.error(
                    "Failed to create collection",
                    collection=collection_name,
                    error=str(e)
                )
                results[collection_name] = False

        return results

    def add_paper_summary(
        self,
        paper_id: str,
        summary_data: Dict[str, Any],
        embedding: List[float]
    ) -> bool:
        """
        Add a paper summary to the paper_summaries collection.

        Args:
            paper_id: Unique paper identifier (e.g., arXiv ID)
            summary_data: Summary metadata containing:
                - title: Paper title
                - one_line: One-sentence summary
                - abstract_summary: Condensed abstract
                - key_contributions: List of bullet points
                - methodology: Brief methodology description
                - domains: Inferred topics/domains
                - source_paper_id: For linked papers, which KB paper cited this
            embedding: Vector embedding of the summary

        Returns:
            True if successfully added
        """
        summaries_collection = self._collection_names[CollectionType.PAPER_SUMMARIES]

        try:
            # Generate unique point ID
            point_id = abs(hash(f"summary_{paper_id}")) % (2**63 - 1)

            # Build payload
            payload = {
                "paper_id": paper_id,
                "summary_type": "paper_summary",
                **summary_data
            }

            # Create and upload point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )

            self.client.upload_points(
                collection_name=summaries_collection,
                points=[point],
                wait=True
            )

            logger.info(
                "Added paper summary",
                paper_id=paper_id,
                collection=summaries_collection
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to add paper summary",
                paper_id=paper_id,
                error=str(e)
            )
            return False

    def search_summaries(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search paper summaries for fast relevance checking.

        This enables the "fast path" in two-tier retrieval - quickly identify
        which papers are relevant before diving into full chunk retrieval.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of summary results with paper metadata and scores
        """
        summaries_collection = self._collection_names[CollectionType.PAPER_SUMMARIES]

        try:
            results = self.client.search(
                collection_name=summaries_collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )

            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "paper_id": hit.payload.get("paper_id"),
                    "title": hit.payload.get("title"),
                    "one_line": hit.payload.get("one_line"),
                    "domains": hit.payload.get("domains", []),
                    "payload": hit.payload
                }
                for hit in results
            ]

        except Exception as e:
            logger.error("Summary search failed", error=str(e))
            return []

    def add_linked_paper_vectors(
        self,
        paper_id: str,
        chunks: List[Dict[str, Any]],
        source_paper_id: str,
        batch_size: int = 100
    ) -> int:
        """
        Add vectors for a linked (cited) paper to the linked_papers collection.

        This method is similar to add_vectors but stores in the linked_papers
        collection and includes source_paper_id to track which KB paper cited it.

        Args:
            paper_id: Unique identifier for the linked paper
            chunks: List of chunk dictionaries (same format as add_vectors)
            source_paper_id: The KB paper that cited this linked paper
            batch_size: Number of vectors to upload per batch

        Returns:
            Number of vectors added
        """
        linked_collection = self._collection_names[CollectionType.LINKED_PAPERS]

        try:
            points = []

            for chunk in chunks:
                point_id = abs(hash(f"linked_{paper_id}_chunk_{chunk['chunk_index']}")) % (2**63 - 1)

                payload = {
                    "paper_id": paper_id,
                    "source_paper_id": source_paper_id,  # Track citation source
                    "is_linked_paper": True,
                    "chunk_text": chunk["chunk_text"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk.get("total_chunks", len(chunks)),
                }

                # Add optional metadata
                for key in ["title", "authors", "abstract", "categories",
                           "published_date", "pdf_path", "processing_timestamp"]:
                    if key in chunk:
                        payload[key] = chunk[key]

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=chunk["embedding"],
                        payload=payload
                    )
                )

            # Upload in batches
            total_uploaded = 0
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upload_points(
                    collection_name=linked_collection,
                    points=batch,
                    wait=True
                )
                total_uploaded += len(batch)

            logger.info(
                "Added linked paper vectors",
                paper_id=paper_id,
                source_paper_id=source_paper_id,
                num_vectors=total_uploaded,
                collection=linked_collection
            )

            return total_uploaded

        except Exception as e:
            logger.error(
                "Failed to add linked paper vectors",
                paper_id=paper_id,
                error=str(e)
            )
            raise

    def search_linked_papers(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.7,
        source_paper_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search linked papers collection for broader context.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            source_paper_id: Optional filter to only search papers cited by a specific KB paper

        Returns:
            List of search results from linked papers
        """
        linked_collection = self._collection_names[CollectionType.LINKED_PAPERS]

        try:
            query_filter = None
            if source_paper_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="source_paper_id",
                            match=MatchValue(value=source_paper_id)
                        )
                    ]
                )

            results = self.client.search(
                collection_name=linked_collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False
            )

            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                for hit in results
            ]

        except Exception as e:
            logger.error("Linked papers search failed", error=str(e))
            return []

    def get_all_collections_info(self) -> Dict[str, Any]:
        """
        Get information about all context engineering collections.

        Returns:
            Dictionary with info for each collection
        """
        info = {}

        for collection_type, collection_name in self._collection_names.items():
            try:
                coll_info = self.client.get_collection(collection_name)
                info[collection_type.value] = {
                    "name": coll_info.name,
                    "vectors_count": coll_info.vectors_count,
                    "points_count": coll_info.points_count,
                    "status": coll_info.status.value,
                }
            except Exception:
                info[collection_type.value] = {
                    "name": collection_name,
                    "exists": False,
                    "error": "Collection does not exist"
                }

        return info

    def get_all_linked_papers(self) -> List[Dict[str, Any]]:
        """
        Get list of all unique papers in the linked_papers collection.

        Similar to get_all_papers but for the linked papers collection.
        Includes source_paper_id to show which KB paper cited each linked paper.

        Returns:
            List of linked paper dictionaries containing metadata and citation source
        """
        linked_collection = self._collection_names[CollectionType.LINKED_PAPERS]

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if not any(col.name == linked_collection for col in collections):
                logger.debug("Linked papers collection does not exist")
                return []

            # Scroll through all points
            offset = None
            all_points = []

            while True:
                result = self.client.scroll(
                    collection_name=linked_collection,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                points, next_offset = result
                all_points.extend(points)

                if next_offset is None:
                    break
                offset = next_offset

            # Group by paper_id
            papers_dict = {}
            for point in all_points:
                payload = point.payload
                paper_id = payload.get('paper_id', 'unknown')

                if paper_id not in papers_dict:
                    papers_dict[paper_id] = {
                        'paper_id': paper_id,
                        'title': payload.get('title', 'Unknown'),
                        'authors': payload.get('authors', []),
                        'categories': payload.get('categories', []),
                        'published_date': payload.get('published_date', ''),
                        'indexed_at': payload.get('processing_timestamp', ''),
                        'chunk_count': 0,
                        'abstract': payload.get('abstract', '')[:300] + '...' if payload.get('abstract') else '',
                        'source_paper_id': payload.get('source_paper_id', ''),
                        'is_linked': True
                    }

                papers_dict[paper_id]['chunk_count'] += 1
                # Update indexed_at if we find a newer timestamp
                current_ts = payload.get('processing_timestamp', '')
                if current_ts and current_ts > papers_dict[paper_id].get('indexed_at', ''):
                    papers_dict[paper_id]['indexed_at'] = current_ts

            # Convert to list and sort by title
            papers = list(papers_dict.values())
            papers.sort(key=lambda x: x['title'])

            logger.info(
                "Retrieved all linked papers",
                total_papers=len(papers),
                total_chunks=len(all_points)
            )

            return papers

        except Exception as e:
            logger.error(
                "Failed to get linked papers",
                error=str(e)
            )
            return []

    def add_extractions(
        self,
        paper_id: str,
        extractions: List[Dict[str, Any]],
        embeddings: List[List[float]],
        batch_size: int = 100
    ) -> int:
        """
        Add structured extractions to the paper_extractions collection.

        Each extraction is stored with its embedding for semantic search.

        Args:
            paper_id: Unique paper identifier
            extractions: List of extraction dictionaries containing:
                - type: Extraction type (method, dataset, finding, etc.)
                - name: Short identifier
                - content: Detailed description
                - source_span: Source location for grounding
                - confidence: Extraction confidence score
                - attributes: Type-specific attributes
            embeddings: List of embeddings corresponding to each extraction
            batch_size: Number of points to upload per batch

        Returns:
            Number of extractions added
        """
        extractions_collection = self._collection_names[CollectionType.PAPER_EXTRACTIONS]

        try:
            # Ensure collection exists
            self._ensure_extractions_collection()

            points = []
            for i, (extraction, embedding) in enumerate(zip(extractions, embeddings)):
                # Generate unique point ID
                point_id = abs(hash(
                    f"extraction_{paper_id}_{extraction.get('type')}_{i}"
                )) % (2**63 - 1)

                # Build payload
                payload = {
                    "paper_id": paper_id,
                    "extraction_type": extraction.get("type", "unknown"),
                    "name": extraction.get("name", ""),
                    "content": extraction.get("content", ""),
                    "confidence": extraction.get("confidence", 0.8),
                    "attributes": extraction.get("attributes", {}),
                    "source_span": extraction.get("source_span", {}),
                    "extracted_at": extraction.get("extracted_at", ""),
                }

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                )

            # Upload in batches
            total_uploaded = 0
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upload_points(
                    collection_name=extractions_collection,
                    points=batch,
                    wait=True
                )
                total_uploaded += len(batch)

            logger.info(
                "Added extractions",
                paper_id=paper_id,
                num_extractions=total_uploaded,
                collection=extractions_collection
            )

            return total_uploaded

        except Exception as e:
            logger.error(
                "Failed to add extractions",
                paper_id=paper_id,
                error=str(e)
            )
            raise

    def search_extractions(
        self,
        query_vector: List[float],
        extraction_types: Optional[List[str]] = None,
        paper_ids: Optional[List[str]] = None,
        limit: int = 10,
        score_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search structured extractions for precise entity retrieval.

        Args:
            query_vector: Query embedding vector
            extraction_types: Optional filter by extraction types (method, dataset, etc.)
            paper_ids: Optional filter by paper IDs
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of extraction results with metadata and scores
        """
        extractions_collection = self._collection_names[CollectionType.PAPER_EXTRACTIONS]

        try:
            # Build filter conditions
            conditions = []

            if extraction_types:
                from qdrant_client.models import MatchAny
                conditions.append(
                    FieldCondition(
                        key="extraction_type",
                        match=MatchAny(any=extraction_types)
                    )
                )

            if paper_ids:
                from qdrant_client.models import MatchAny
                conditions.append(
                    FieldCondition(
                        key="paper_id",
                        match=MatchAny(any=paper_ids)
                    )
                )

            query_filter = Filter(must=conditions) if conditions else None

            results = self.client.search(
                collection_name=extractions_collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False
            )

            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "paper_id": hit.payload.get("paper_id"),
                    "extraction_type": hit.payload.get("extraction_type"),
                    "name": hit.payload.get("name"),
                    "content": hit.payload.get("content"),
                    "confidence": hit.payload.get("confidence"),
                    "attributes": hit.payload.get("attributes", {}),
                    "source_span": hit.payload.get("source_span", {}),
                    "payload": hit.payload
                }
                for hit in results
            ]

        except Exception as e:
            logger.error("Extractions search failed", error=str(e))
            return []

    def get_paper_extractions(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Get all extractions for a specific paper.

        Args:
            paper_id: Unique paper identifier

        Returns:
            List of extraction payloads
        """
        extractions_collection = self._collection_names[CollectionType.PAPER_EXTRACTIONS]

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if not any(col.name == extractions_collection for col in collections):
                return []

            result = self.client.scroll(
                collection_name=extractions_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="paper_id",
                            match=MatchValue(value=paper_id)
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
                with_vectors=False
            )

            extractions = [
                {
                    "id": point.id,
                    **point.payload
                }
                for point in result[0]
            ]

            logger.info(
                "Retrieved paper extractions",
                paper_id=paper_id,
                num_extractions=len(extractions)
            )

            return extractions

        except Exception as e:
            logger.error(
                "Failed to get paper extractions",
                paper_id=paper_id,
                error=str(e)
            )
            return []

    def delete_paper_extractions(self, paper_id: str) -> None:
        """
        Delete all extractions for a paper.

        Args:
            paper_id: Unique paper identifier
        """
        extractions_collection = self._collection_names[CollectionType.PAPER_EXTRACTIONS]

        try:
            self.client.delete(
                collection_name=extractions_collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="paper_id",
                            match=MatchValue(value=paper_id)
                        )
                    ]
                )
            )

            logger.info(
                "Deleted paper extractions",
                paper_id=paper_id
            )

        except Exception as e:
            logger.error(
                "Failed to delete paper extractions",
                paper_id=paper_id,
                error=str(e)
            )

    def _ensure_extractions_collection(self) -> None:
        """Ensure the extractions collection exists."""
        extractions_collection = self._collection_names[CollectionType.PAPER_EXTRACTIONS]

        try:
            collections = self.client.get_collections().collections
            if not any(col.name == extractions_collection for col in collections):
                self.client.create_collection(
                    collection_name=extractions_collection,
                    vectors_config=VectorParams(
                        size=self.settings.EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE
                    )
                )
                logger.info(
                    "Created extractions collection",
                    collection=extractions_collection
                )
        except Exception as e:
            logger.warning(
                "Could not create extractions collection",
                error=str(e)
            )
