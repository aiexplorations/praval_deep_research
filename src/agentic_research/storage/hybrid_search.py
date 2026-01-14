"""
Hybrid Search: Vajra BM25 + Qdrant Vector with RRF Fusion.

Combines keyword-based BM25 search (Vajra) with semantic vector search (Qdrant)
using Reciprocal Rank Fusion for optimal retrieval.

Why hybrid search?
- BM25 excels at exact keyword/term matches (author names, technical terms)
- Vector search captures semantic similarity (concepts, paraphrases)
- Combined: best of both worlds with adjustable balance

Usage:
    from agentic_research.storage.hybrid_search import get_hybrid_search

    hybrid = get_hybrid_search()
    results = hybrid.search("transformer attention", top_k=20, alpha=0.5)
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
import structlog

from agentic_research.storage.paper_index import get_paper_index
from agentic_research.storage.qdrant_client import QdrantClientWrapper
from agentic_research.storage.embeddings import EmbeddingsGenerator

logger = structlog.get_logger(__name__)

# Cache for query embeddings (avoids repeated API calls for same queries)
@lru_cache(maxsize=100)
def _cached_embedding(query: str, generator_id: int) -> Tuple[float, ...]:
    """Cache embeddings by query string. Returns tuple for hashability."""
    # This function is called by the HybridPaperSearch instance
    # generator_id is used to invalidate cache if generator changes
    return tuple()  # Placeholder - actual implementation below


@dataclass
class HybridSearchResult:
    """Result from hybrid search with component scores."""

    paper_id: str
    title: str
    authors: List[str]
    categories: List[str]
    abstract: str
    combined_score: float
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    vector_score: Optional[float] = None
    vector_rank: Optional[int] = None
    matching_chunks: int = 0


class HybridPaperSearch:
    """
    Hybrid search combining Vajra BM25 and Qdrant vector search.

    Uses Reciprocal Rank Fusion (RRF) to combine results:
    RRF(d) = alpha * 1/(k + bm25_rank) + (1-alpha) * 1/(k + vector_rank)

    Args:
        alpha: Weight for BM25 (0-1). Default 0.5 for balanced.
               - alpha=1.0: Pure keyword/BM25 search
               - alpha=0.5: Balanced hybrid (recommended)
               - alpha=0.0: Pure semantic/vector search
        rrf_k: Constant for RRF formula (default 60)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        self.alpha = alpha
        self.rrf_k = rrf_k

        # Initialize search engines
        self._paper_index = None
        self._qdrant = None
        self._embeddings = None

    @property
    def paper_index(self):
        """Lazy load paper index."""
        if self._paper_index is None:
            self._paper_index = get_paper_index()
        return self._paper_index

    @property
    def qdrant(self):
        """Lazy load Qdrant client."""
        if self._qdrant is None:
            self._qdrant = QdrantClientWrapper()
        return self._qdrant

    @property
    def embeddings(self):
        """Lazy load embeddings generator."""
        if self._embeddings is None:
            self._embeddings = EmbeddingsGenerator()
        return self._embeddings

    def search(
        self,
        query: str,
        top_k: int = 20,
        alpha: Optional[float] = None,
        categories: Optional[List[str]] = None,
        paper_ids: Optional[List[str]] = None,
    ) -> List[HybridSearchResult]:
        """
        Hybrid search over indexed papers.

        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Override default BM25 weight (optional)
            categories: Filter by ArXiv categories (optional)
            paper_ids: Filter to specific papers (optional)

        Returns:
            List of HybridSearchResult sorted by combined score
        """
        alpha = alpha if alpha is not None else self.alpha

        logger.info(
            "Starting hybrid search",
            query=query[:50],
            top_k=top_k,
            alpha=alpha,
        )

        # Fetch more results for fusion
        fetch_k = top_k * 3

        # Get BM25 results from Vajra (skip if pure semantic)
        if alpha > 0:
            bm25_results = self._search_bm25(
                query, fetch_k, categories, paper_ids
            )
        else:
            bm25_results = []

        # Get vector results from Qdrant (skip if pure keyword - saves API call)
        if alpha < 1.0:
            vector_results = self._search_vector(
                query, fetch_k, categories, paper_ids
            )
        else:
            vector_results = []
            logger.info("Skipping vector search for pure keyword mode")

        # Fuse results using RRF
        fused = self._rrf_fusion(bm25_results, vector_results, alpha)

        # Sort by combined score
        sorted_results = sorted(
            fused.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )

        # Convert to HybridSearchResult
        results = []
        for item in sorted_results[:top_k]:
            results.append(HybridSearchResult(
                paper_id=item["paper_id"],
                title=item["title"],
                authors=item["authors"],
                categories=item["categories"],
                abstract=item["abstract"],
                combined_score=item["combined_score"],
                bm25_score=item.get("bm25_score"),
                bm25_rank=item.get("bm25_rank"),
                vector_score=item.get("vector_score"),
                vector_rank=item.get("vector_rank"),
                matching_chunks=item.get("matching_chunks", 0),
            ))

        logger.info(
            "Hybrid search completed",
            query=query[:30],
            bm25_count=len(bm25_results),
            vector_count=len(vector_results),
            fused_count=len(results),
        )

        return results

    def _search_bm25(
        self,
        query: str,
        top_k: int,
        categories: Optional[List[str]] = None,
        paper_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search using Vajra BM25."""
        try:
            hits = self.paper_index.search_papers(
                query=query,
                top_k=top_k,
                categories=categories,
            )

            # Filter by paper_ids if specified
            if paper_ids:
                hits = [h for h in hits if h.metadata.get("paper_id") in paper_ids]

            # Group by paper_id and aggregate scores
            papers: Dict[str, Dict] = {}
            for rank, hit in enumerate(hits, 1):
                paper_id = hit.metadata.get("paper_id")
                if not paper_id:
                    continue

                if paper_id not in papers:
                    papers[paper_id] = {
                        "paper_id": paper_id,
                        "title": hit.metadata.get("title", ""),
                        "authors": hit.metadata.get("authors", []),
                        "categories": hit.metadata.get("categories", []),
                        "abstract": hit.metadata.get("abstract", ""),
                        "bm25_score": hit.score,
                        "bm25_rank": rank,
                        "matching_chunks": 1,
                    }
                else:
                    # Aggregate: keep best score, count chunks
                    papers[paper_id]["matching_chunks"] += 1
                    if hit.score > papers[paper_id]["bm25_score"]:
                        papers[paper_id]["bm25_score"] = hit.score
                        papers[paper_id]["bm25_rank"] = rank

            return list(papers.values())

        except Exception as e:
            logger.warning("BM25 search failed", error=str(e))
            return []

    def _get_cached_embedding(self, query: str) -> List[float]:
        """Get embedding with caching to avoid repeated API calls."""
        # Use a simple dict cache on the instance
        if not hasattr(self, '_embedding_cache'):
            self._embedding_cache = {}

        if query not in self._embedding_cache:
            self._embedding_cache[query] = self.embeddings.generate_embedding(query)
            # Limit cache size
            if len(self._embedding_cache) > 100:
                # Remove oldest entry
                oldest = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest]

        return self._embedding_cache[query]

    def _search_vector(
        self,
        query: str,
        top_k: int,
        categories: Optional[List[str]] = None,
        paper_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search using Qdrant vector search."""
        try:
            # Generate query embedding (cached)
            query_embedding = self._get_cached_embedding(query)

            # Search Qdrant
            hits = self.qdrant.search_similar(
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=0.3,
            )

            # Filter and group by paper_id
            papers: Dict[str, Dict] = {}
            for rank, hit in enumerate(hits, 1):
                paper_id = hit.get("paper_id")
                if not paper_id:
                    continue

                # Apply filters
                if paper_ids and paper_id not in paper_ids:
                    continue

                if categories:
                    hit_cats = hit.get("categories", [])
                    if not any(c in hit_cats for c in categories):
                        continue

                if paper_id not in papers:
                    papers[paper_id] = {
                        "paper_id": paper_id,
                        "title": hit.get("title", ""),
                        "authors": hit.get("authors", []),
                        "categories": hit.get("categories", []),
                        "abstract": hit.get("abstract", ""),
                        "vector_score": hit.get("score", 0.0),
                        "vector_rank": rank,
                    }
                else:
                    # Keep best score
                    if hit.get("score", 0) > papers[paper_id]["vector_score"]:
                        papers[paper_id]["vector_score"] = hit.get("score", 0)
                        papers[paper_id]["vector_rank"] = rank

            return list(papers.values())

        except Exception as e:
            logger.warning("Vector search failed", error=str(e))
            return []

    def _rrf_fusion(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        alpha: float,
    ) -> Dict[str, Dict]:
        """
        Reciprocal Rank Fusion.

        RRF(d) = alpha * 1/(k + bm25_rank) + (1-alpha) * 1/(k + vector_rank)
        """
        fused: Dict[str, Dict] = {}

        # Add BM25 contributions
        for result in bm25_results:
            paper_id = result["paper_id"]
            rank = result.get("bm25_rank", 999)
            rrf_score = alpha / (self.rrf_k + rank)

            fused[paper_id] = {
                **result,
                "combined_score": rrf_score,
            }

        # Add vector contributions
        for result in vector_results:
            paper_id = result["paper_id"]
            rank = result.get("vector_rank", 999)
            rrf_score = (1 - alpha) / (self.rrf_k + rank)

            if paper_id in fused:
                # Merge: add vector score and combine RRF
                fused[paper_id]["combined_score"] += rrf_score
                fused[paper_id]["vector_score"] = result.get("vector_score")
                fused[paper_id]["vector_rank"] = result.get("vector_rank")
            else:
                # New paper from vector search only
                fused[paper_id] = {
                    **result,
                    "combined_score": rrf_score,
                    "bm25_score": None,
                    "bm25_rank": None,
                    "matching_chunks": 0,
                }

        return fused

    def get_search_mode(self, alpha: float) -> str:
        """Get human-readable search mode based on alpha."""
        if alpha >= 0.9:
            return "keyword"
        elif alpha <= 0.1:
            return "semantic"
        else:
            return "hybrid"


# Singleton instance
_hybrid_search: Optional[HybridPaperSearch] = None


def get_hybrid_search() -> HybridPaperSearch:
    """Get the global hybrid search singleton."""
    global _hybrid_search

    if _hybrid_search is None:
        _hybrid_search = HybridPaperSearch()

    return _hybrid_search
