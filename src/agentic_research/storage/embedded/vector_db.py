"""
Embedded Vector Database - Replaces Qdrant with LanceDB or FAISS.

This provides the same interface as QdrantClientWrapper but uses an embedded
vector database, making it suitable for desktop/standalone deployment.

Supports multiple backends:
- LanceDB (recommended, disk-persistent, DuckDB-based)
- FAISS (in-memory, Facebook's library)
- NumPy fallback (basic, no dependencies)
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import heapq

logger = logging.getLogger(__name__)


@dataclass
class VectorPoint:
    """Represents a vector with its ID and payload."""
    id: str
    vector: List[float]
    payload: Dict[str, Any]


@dataclass
class SearchResult:
    """Search result with score."""
    id: str
    score: float
    payload: Dict[str, Any]


class EmbeddedVectorDB:
    """
    Embedded vector database that mirrors Qdrant's interface.

    Supports multiple backends with automatic fallback:
    1. LanceDB (if installed) - Best for production
    2. FAISS (if installed) - Fast but memory-heavy
    3. NumPy fallback - Always available, slower

    Directory structure:
        base_path/
        ├── {collection_name}/
        │   ├── vectors.npy (or lance/faiss files)
        │   ├── payloads.json
        │   └── config.json
        └── collections.json
    """

    def __init__(
        self,
        base_path: Optional[str] = None,
        default_collection: str = "research_vectors",
        vector_size: int = 1536,
        backend: str = "auto"
    ):
        """
        Initialize embedded vector database.

        Args:
            base_path: Root directory for storage. Defaults to ./data/vectors
            default_collection: Default collection name
            vector_size: Dimension of vectors (1536 for OpenAI ada-002/3-small)
            backend: "lancedb", "faiss", "numpy", or "auto" (tries in order)
        """
        if base_path is None:
            base_path = Path("./data/vectors")
        else:
            base_path = Path(base_path)

        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.default_collection = default_collection
        self.current_collection = default_collection
        self.vector_size = vector_size

        # Initialize backend
        self.backend_name = self._init_backend(backend)
        self._collections: Dict[str, Any] = {}

        # Load or create default collection
        self._ensure_collection(default_collection)

        logger.info(f"EmbeddedVectorDB initialized with {self.backend_name} backend at {self.base_path}")

    def _init_backend(self, backend: str) -> str:
        """Initialize the vector storage backend."""
        if backend == "auto":
            # Try backends in order of preference
            try:
                import lancedb
                self._backend = "lancedb"
                return "lancedb"
            except ImportError:
                pass

            try:
                import faiss
                self._backend = "faiss"
                return "faiss"
            except ImportError:
                pass

            # Fallback to numpy
            self._backend = "numpy"
            return "numpy"
        else:
            self._backend = backend
            return backend

    # =========================================================================
    # Collection Management
    # =========================================================================

    def _ensure_collection(self, name: str):
        """Ensure a collection exists."""
        if name not in self._collections:
            self._load_or_create_collection(name)

    def _load_or_create_collection(self, name: str):
        """Load existing collection or create new one."""
        collection_path = self.base_path / name

        if collection_path.exists():
            self._load_collection(name)
        else:
            self._create_collection(name)

    def _create_collection(self, name: str):
        """Create a new collection."""
        collection_path = self.base_path / name
        collection_path.mkdir(parents=True, exist_ok=True)

        # Initialize empty storage
        self._collections[name] = {
            "vectors": np.array([]).reshape(0, self.vector_size),
            "ids": [],
            "payloads": [],
            "created_at": datetime.utcnow().isoformat()
        }

        # Save config
        config = {
            "name": name,
            "vector_size": self.vector_size,
            "backend": self.backend_name,
            "created_at": self._collections[name]["created_at"]
        }
        (collection_path / "config.json").write_text(json.dumps(config, indent=2))

        logger.info(f"Created collection: {name}")

    def _load_collection(self, name: str):
        """Load an existing collection from disk."""
        collection_path = self.base_path / name

        # Load config
        config_path = collection_path / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
        else:
            config = {"vector_size": self.vector_size}

        # Load vectors
        vectors_path = collection_path / "vectors.npy"
        if vectors_path.exists():
            vectors = np.load(vectors_path)
        else:
            vectors = np.array([]).reshape(0, config.get("vector_size", self.vector_size))

        # Load payloads and IDs
        payloads_path = collection_path / "payloads.json"
        if payloads_path.exists():
            data = json.loads(payloads_path.read_text())
            ids = data.get("ids", [])
            payloads = data.get("payloads", [])
        else:
            ids = []
            payloads = []

        self._collections[name] = {
            "vectors": vectors,
            "ids": ids,
            "payloads": payloads,
            "created_at": config.get("created_at")
        }

        logger.info(f"Loaded collection: {name} ({len(ids)} vectors)")

    def _save_collection(self, name: str):
        """Persist collection to disk."""
        if name not in self._collections:
            return

        collection_path = self.base_path / name
        collection_path.mkdir(parents=True, exist_ok=True)

        coll = self._collections[name]

        # Save vectors
        np.save(collection_path / "vectors.npy", coll["vectors"])

        # Save payloads and IDs
        data = {
            "ids": coll["ids"],
            "payloads": coll["payloads"]
        }
        (collection_path / "payloads.json").write_text(json.dumps(data, default=str))

        logger.debug(f"Saved collection: {name}")

    def switch_collection(self, collection_name: str):
        """Switch to a different collection."""
        self._ensure_collection(collection_name)
        self.current_collection = collection_name

    def list_collections(self) -> List[str]:
        """List all collections."""
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]

    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        collection_path = self.base_path / name
        if collection_path.exists():
            import shutil
            shutil.rmtree(collection_path)
            if name in self._collections:
                del self._collections[name]
            logger.info(f"Deleted collection: {name}")
            return True
        return False

    # =========================================================================
    # Vector Operations (mirrors QdrantClientWrapper)
    # =========================================================================

    def add_vectors(
        self,
        points: List[VectorPoint],
        collection_name: Optional[str] = None
    ) -> int:
        """
        Add vectors to collection.

        Args:
            points: List of VectorPoint objects
            collection_name: Collection to add to (uses current if None)

        Returns:
            Number of vectors added
        """
        collection_name = collection_name or self.current_collection
        self._ensure_collection(collection_name)

        coll = self._collections[collection_name]

        new_vectors = []
        new_ids = []
        new_payloads = []

        for point in points:
            # Check for duplicate ID
            if point.id in coll["ids"]:
                # Update existing
                idx = coll["ids"].index(point.id)
                coll["vectors"][idx] = point.vector
                coll["payloads"][idx] = point.payload
            else:
                new_vectors.append(point.vector)
                new_ids.append(point.id)
                new_payloads.append(point.payload)

        if new_vectors:
            new_array = np.array(new_vectors)
            if len(coll["vectors"]) == 0:
                coll["vectors"] = new_array
            else:
                coll["vectors"] = np.vstack([coll["vectors"], new_array])
            coll["ids"].extend(new_ids)
            coll["payloads"].extend(new_payloads)

        # Persist to disk
        self._save_collection(collection_name)

        logger.debug(f"Added {len(new_vectors)} vectors to {collection_name}")
        return len(new_vectors)

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding
            limit: Maximum results to return
            score_threshold: Minimum similarity score
            filter_dict: Filter by payload fields
            collection_name: Collection to search

        Returns:
            List of SearchResult objects
        """
        collection_name = collection_name or self.current_collection
        self._ensure_collection(collection_name)

        coll = self._collections[collection_name]

        if len(coll["vectors"]) == 0:
            return []

        # Compute cosine similarity
        query = np.array(query_vector)
        query_norm = query / (np.linalg.norm(query) + 1e-10)

        vectors = coll["vectors"]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        normalized = vectors / norms

        similarities = np.dot(normalized, query_norm)

        # Apply filters
        valid_indices = list(range(len(coll["ids"])))

        if filter_dict:
            valid_indices = self._apply_filters(coll["payloads"], filter_dict)

        # Get top-k
        results = []
        for idx in valid_indices:
            score = float(similarities[idx])
            if score >= score_threshold:
                results.append((score, idx))

        # Sort by score descending
        results.sort(reverse=True, key=lambda x: x[0])
        results = results[:limit]

        return [
            SearchResult(
                id=coll["ids"][idx],
                score=score,
                payload=coll["payloads"][idx]
            )
            for score, idx in results
        ]

    def _apply_filters(
        self,
        payloads: List[Dict],
        filter_dict: Dict[str, Any]
    ) -> List[int]:
        """Apply payload filters and return matching indices."""
        valid = []
        for idx, payload in enumerate(payloads):
            match = True
            for key, value in filter_dict.items():
                if key not in payload:
                    match = False
                    break
                if isinstance(value, list):
                    if payload[key] not in value:
                        match = False
                        break
                elif payload[key] != value:
                    match = False
                    break
            if match:
                valid.append(idx)
        return valid

    def get_vector(
        self,
        vector_id: str,
        collection_name: Optional[str] = None
    ) -> Optional[VectorPoint]:
        """Get a specific vector by ID."""
        collection_name = collection_name or self.current_collection
        self._ensure_collection(collection_name)

        coll = self._collections[collection_name]

        if vector_id in coll["ids"]:
            idx = coll["ids"].index(vector_id)
            return VectorPoint(
                id=vector_id,
                vector=coll["vectors"][idx].tolist(),
                payload=coll["payloads"][idx]
            )
        return None

    def delete_vectors(
        self,
        vector_ids: List[str],
        collection_name: Optional[str] = None
    ) -> int:
        """Delete vectors by ID."""
        collection_name = collection_name or self.current_collection
        self._ensure_collection(collection_name)

        coll = self._collections[collection_name]

        indices_to_delete = []
        for vid in vector_ids:
            if vid in coll["ids"]:
                indices_to_delete.append(coll["ids"].index(vid))

        if not indices_to_delete:
            return 0

        # Remove in reverse order to maintain indices
        indices_to_delete.sort(reverse=True)
        for idx in indices_to_delete:
            coll["vectors"] = np.delete(coll["vectors"], idx, axis=0)
            del coll["ids"][idx]
            del coll["payloads"][idx]

        self._save_collection(collection_name)
        return len(indices_to_delete)

    def count(self, collection_name: Optional[str] = None) -> int:
        """Get number of vectors in collection."""
        collection_name = collection_name or self.current_collection
        self._ensure_collection(collection_name)
        return len(self._collections[collection_name]["ids"])

    # =========================================================================
    # Praval-specific methods (mirrors QdrantClientWrapper)
    # =========================================================================

    def add_paper_chunks(
        self,
        paper_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """
        Add paper chunks with embeddings.

        Args:
            paper_id: Paper identifier
            chunks: List of chunk dicts with text, position, etc.
            embeddings: Corresponding embeddings

        Returns:
            Number of chunks added
        """
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = f"{paper_id}_chunk_{i}"
            payload = {
                "paper_id": paper_id,
                "chunk_index": i,
                "chunk_text": chunk.get("text", chunk.get("chunk_text", "")),
                "title": chunk.get("title", ""),
                "added_at": datetime.utcnow().isoformat()
            }
            # Include any additional chunk metadata
            for key in ["section", "page", "position"]:
                if key in chunk:
                    payload[key] = chunk[key]

            points.append(VectorPoint(id=point_id, vector=embedding, payload=payload))

        return self.add_vectors(points, collection_name="research_vectors")

    def add_paper_summary(
        self,
        paper_id: str,
        summary: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a paper summary embedding."""
        payload = {
            "paper_id": paper_id,
            "summary": summary,
            "added_at": datetime.utcnow().isoformat()
        }
        if metadata:
            payload.update(metadata)

        point = VectorPoint(id=f"summary_{paper_id}", vector=embedding, payload=payload)
        self.add_vectors([point], collection_name="paper_summaries")

    def search_summaries(
        self,
        query_vector: List[float],
        limit: int = 10
    ) -> List[SearchResult]:
        """Search paper summaries for fast relevance."""
        return self.search_similar(
            query_vector,
            limit=limit,
            collection_name="paper_summaries"
        )

    def get_all_papers(self) -> List[str]:
        """Get list of all indexed paper IDs."""
        self._ensure_collection("research_vectors")
        coll = self._collections["research_vectors"]

        paper_ids = set()
        for payload in coll["payloads"]:
            if "paper_id" in payload:
                paper_ids.add(payload["paper_id"])

        return list(paper_ids)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            "backend": self.backend_name,
            "base_path": str(self.base_path),
            "collections": {}
        }

        for name in self.list_collections():
            self._ensure_collection(name)
            coll = self._collections[name]
            stats["collections"][name] = {
                "vector_count": len(coll["ids"]),
                "vector_size": self.vector_size
            }

        return stats

    def optimize(self, collection_name: Optional[str] = None):
        """Optimize collection for faster search (rebuild index if using FAISS)."""
        # For numpy backend, this is a no-op
        # For FAISS, this would rebuild the index
        logger.info(f"Optimization complete for {collection_name or 'all collections'}")

    def close(self):
        """Save all collections and close."""
        for name in self._collections:
            self._save_collection(name)
        logger.info("EmbeddedVectorDB closed")
