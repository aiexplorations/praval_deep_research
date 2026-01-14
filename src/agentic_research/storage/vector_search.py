"""
Vector search using Qdrant for semantic paper search.

Provides embedding generation and similarity search for Q&A.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models
import openai

from ..core.config import get_settings

logger = logging.getLogger(__name__)


class VectorSearchClient:
    """Client for vector similarity search using Qdrant."""

    def __init__(self):
        """Initialize vector search client."""
        self.settings = get_settings()

        # Connect to Qdrant
        qdrant_url = self.settings.QDRANT_URL.replace('http://', '').replace('https://', '')
        host, port = qdrant_url.split(':') if ':' in qdrant_url else (qdrant_url, '6333')

        self.client = QdrantClient(host=host, port=int(port))
        self.collection_name = self.settings.QDRANT_COLLECTION_NAME

        # Set up OpenAI
        openai.api_key = self.settings.OPENAI_API_KEY
        self.embedding_model = self.settings.OPENAI_EMBEDDING_MODEL
        self.embedding_dim = self.settings.EMBEDDING_DIMENSIONS

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the Qdrant collection exists."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        try:
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def add_paper_chunks(self, paper_id: str, title: str, chunks: List[Dict[str, Any]]):
        """
        Add paper chunks to vector store.

        Args:
            paper_id: Unique paper identifier
            title: Paper title
            chunks: List of text chunks with metadata
        """
        try:
            points = []
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('text', '')
                if not chunk_text:
                    continue

                # Generate embedding
                embedding = self.generate_embedding(chunk_text)

                # Create point with unsigned integer ID (Qdrant requirement)
                # Use abs() to ensure positive ID from hash
                point_id = abs(hash(f"{paper_id}_{i}")) % (2**63 - 1)

                # Create point
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "paper_id": paper_id,
                        "title": title,
                        "chunk_index": i,
                        "text": chunk_text[:1000],  # Store first 1000 chars
                        "full_text": chunk_text,
                        **chunk.get('metadata', {})
                    }
                )
                points.append(point)

            # Upload to Qdrant
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Added {len(points)} chunks for paper {paper_id}")

        except Exception as e:
            logger.error(f"Failed to add paper chunks: {e}")
            raise

    def search(self, query: str, top_k: int = 5,
               score_threshold: float = 0.7,
               paper_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using semantic similarity.

        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            paper_ids: Optional list of paper IDs to filter results (for "Chat with Papers")

        Returns:
            List of relevant chunks with metadata
        """
        from qdrant_client.models import Filter, FieldCondition, MatchAny

        logger.info(
            f"Vector search started",
            extra={
                "query": query[:100],
                "top_k": top_k,
                "score_threshold": score_threshold,
                "collection": self.collection_name,
                "paper_ids_filter": paper_ids
            }
        )

        try:
            # Generate query embedding
            logger.debug(f"Generating embedding for query: {query[:50]}")
            query_embedding = self.generate_embedding(query)
            logger.debug(f"Embedding generated, dimension: {len(query_embedding) if query_embedding else 0}")

            # Build filter for paper_ids if specified
            query_filter = None
            if paper_ids:
                logger.info(f"Applying paper_ids filter: {paper_ids}")
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="paper_id",
                            match=MatchAny(any=paper_ids)
                        )
                    ]
                )

            # Search in Qdrant
            logger.debug(f"Searching Qdrant collection: {self.collection_name}")
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold
            )

            logger.info(f"Qdrant search returned {len(results) if results else 0} results")

            # Format results
            chunks = []
            if results:
                for result in results:
                    # Document processor stores text as 'chunk_text'
                    chunk_text = result.payload.get("chunk_text", "")

                    chunk = {
                        "title": result.payload.get("title"),
                        "paper_id": result.payload.get("paper_id"),
                        "chunk_index": result.payload.get("chunk_index"),
                        "text": chunk_text,  # Full text content
                        "excerpt": chunk_text[:500] if chunk_text else "",  # First 500 chars
                        "relevance_score": result.score,
                        "metadata": {
                            k: v for k, v in result.payload.items()
                            if k not in ["title", "paper_id", "chunk_index", "text", "full_text", "chunk_text"]
                        }
                    }
                    chunks.append(chunk)
                    logger.debug(f"Formatted chunk from paper: {chunk['title']}, score: {chunk['relevance_score']}")

            logger.info(f"Vector search completed: {len(chunks)} relevant chunks for query: {query[:50]}")
            return chunks

        except Exception as e:
            logger.error(f"Vector search failed with error: {e}", exc_info=True)
            return []


# Global client instance
_client: Optional[VectorSearchClient] = None


def get_vector_search_client() -> VectorSearchClient:
    """Get or create global vector search client."""
    global _client
    if _client is None:
        _client = VectorSearchClient()
    return _client
