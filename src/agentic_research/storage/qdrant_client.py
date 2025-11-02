"""
Qdrant vector database client for semantic search.

This module provides a client for interacting with Qdrant vector database
for storing and retrieving paper embeddings.
"""

from typing import List, Dict, Any, Optional
from uuid import uuid4
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


class QdrantClientWrapper:
    """
    Wrapper for Qdrant vector database operations.

    I am a Qdrant vector database client who handles embedding storage,
    similarity search, and vector management for research papers.
    """

    def __init__(self, settings=None):
        """
        Initialize Qdrant client.

        Args:
            settings: Optional settings object (uses get_settings() if None)
        """
        self.settings = settings or get_settings()
        self.collection_name = self.settings.QDRANT_COLLECTION_NAME

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.settings.QDRANT_URL,
            api_key=self.settings.QDRANT_API_KEY
        )

        logger.info(
            "Qdrant client initialized",
            url=self.settings.QDRANT_URL,
            collection=self.collection_name
        )

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
                        'chunk_count': 0,
                        'abstract': payload.get('abstract', '')[:300] + '...' if payload.get('abstract') else ''
                    }

                papers_dict[paper_id]['chunk_count'] += 1

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
