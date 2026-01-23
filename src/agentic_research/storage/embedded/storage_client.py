"""
Embedded Storage Client - Replaces MinIO with local filesystem storage.

This provides the same interface as MinIOClient but stores files directly
on the local filesystem, making it suitable for desktop/standalone deployment.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import shutil
import logging

logger = logging.getLogger(__name__)


class EmbeddedStorageClient:
    """
    Local filesystem storage client that mirrors MinIO's interface.

    Directory structure:
        base_path/
        ├── papers/
        │   └── {paper_id}.pdf
        ├── metadata/
        │   └── {paper_id}.json
        ├── extracted_text/
        │   └── {paper_id}.txt
        └── .index.json  (file listing cache)
    """

    def __init__(self, base_path: Optional[str] = None, bucket_name: str = "research-papers"):
        """
        Initialize embedded storage.

        Args:
            base_path: Root directory for storage. Defaults to ./data/storage
            bucket_name: Logical bucket name (used as subdirectory)
        """
        if base_path is None:
            base_path = Path("./data/storage")
        else:
            base_path = Path(base_path)

        self.base_path = base_path / bucket_name
        self.bucket_name = bucket_name

        # Create directory structure
        self._ensure_directories()

        logger.info(f"EmbeddedStorageClient initialized at {self.base_path}")

    def _ensure_directories(self):
        """Create required directory structure."""
        directories = ["papers", "metadata", "extracted_text", "thumbnails"]
        for dir_name in directories:
            (self.base_path / dir_name).mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PDF Operations (mirrors MinIOClient)
    # =========================================================================

    def upload_pdf(self, paper_id: str, pdf_data: bytes) -> str:
        """
        Upload a PDF file.

        Args:
            paper_id: Unique identifier for the paper
            pdf_data: Raw PDF bytes

        Returns:
            Local file path
        """
        file_path = self.base_path / "papers" / f"{paper_id}.pdf"
        file_path.write_bytes(pdf_data)

        # Update index
        self._update_index(paper_id, "pdf", len(pdf_data))

        logger.info(f"Uploaded PDF: {paper_id} ({len(pdf_data)} bytes)")
        return str(file_path)

    def download_pdf(self, paper_id: str) -> Optional[bytes]:
        """
        Download a PDF file.

        Args:
            paper_id: Unique identifier for the paper

        Returns:
            PDF bytes or None if not found
        """
        file_path = self.base_path / "papers" / f"{paper_id}.pdf"
        if file_path.exists():
            return file_path.read_bytes()
        return None

    def pdf_exists(self, paper_id: str) -> bool:
        """Check if a PDF exists."""
        file_path = self.base_path / "papers" / f"{paper_id}.pdf"
        return file_path.exists()

    def delete_pdf(self, paper_id: str) -> bool:
        """Delete a PDF file."""
        file_path = self.base_path / "papers" / f"{paper_id}.pdf"
        if file_path.exists():
            file_path.unlink()
            self._remove_from_index(paper_id, "pdf")
            logger.info(f"Deleted PDF: {paper_id}")
            return True
        return False

    def generate_presigned_url(self, paper_id: str, expiry_hours: int = 1) -> str:
        """
        Generate a URL for accessing the PDF.

        For embedded storage, this returns a file:// URL.
        In a Tauri app, this would be converted to an asset URL.

        Args:
            paper_id: Unique identifier for the paper
            expiry_hours: Ignored for local files

        Returns:
            Local file URL
        """
        file_path = self.base_path / "papers" / f"{paper_id}.pdf"
        if file_path.exists():
            # Return absolute path as file URL
            return f"file://{file_path.absolute()}"
        return ""

    # =========================================================================
    # Metadata Operations
    # =========================================================================

    def upload_metadata(self, paper_id: str, metadata: Dict[str, Any]) -> str:
        """
        Store paper metadata as JSON.

        Args:
            paper_id: Unique identifier for the paper
            metadata: Dictionary of paper metadata

        Returns:
            Local file path
        """
        file_path = self.base_path / "metadata" / f"{paper_id}.json"

        # Add storage timestamp
        metadata["_stored_at"] = datetime.utcnow().isoformat()

        file_path.write_text(json.dumps(metadata, indent=2, default=str))
        logger.debug(f"Stored metadata: {paper_id}")
        return str(file_path)

    def download_metadata(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve paper metadata.

        Args:
            paper_id: Unique identifier for the paper

        Returns:
            Metadata dictionary or None if not found
        """
        file_path = self.base_path / "metadata" / f"{paper_id}.json"
        if file_path.exists():
            return json.loads(file_path.read_text())
        return None

    def list_all_metadata(self) -> List[Dict[str, Any]]:
        """List metadata for all stored papers."""
        metadata_dir = self.base_path / "metadata"
        results = []
        for json_file in metadata_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text())
                data["_paper_id"] = json_file.stem
                results.append(data)
            except Exception as e:
                logger.warning(f"Failed to read metadata {json_file}: {e}")
        return results

    # =========================================================================
    # Extracted Text Operations
    # =========================================================================

    def upload_extracted_text(self, paper_id: str, text: str) -> str:
        """
        Store extracted text from a paper.

        Args:
            paper_id: Unique identifier for the paper
            text: Extracted text content

        Returns:
            Local file path
        """
        file_path = self.base_path / "extracted_text" / f"{paper_id}.txt"
        file_path.write_text(text, encoding="utf-8")

        self._update_index(paper_id, "text", len(text))
        logger.debug(f"Stored extracted text: {paper_id} ({len(text)} chars)")
        return str(file_path)

    def download_extracted_text(self, paper_id: str) -> Optional[str]:
        """
        Retrieve extracted text.

        Args:
            paper_id: Unique identifier for the paper

        Returns:
            Extracted text or None if not found
        """
        file_path = self.base_path / "extracted_text" / f"{paper_id}.txt"
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        return None

    def text_exists(self, paper_id: str) -> bool:
        """Check if extracted text exists."""
        file_path = self.base_path / "extracted_text" / f"{paper_id}.txt"
        return file_path.exists()

    # =========================================================================
    # Listing and Search
    # =========================================================================

    def list_papers(self) -> List[str]:
        """List all paper IDs that have PDFs stored."""
        papers_dir = self.base_path / "papers"
        return [f.stem for f in papers_dir.glob("*.pdf")]

    def list_objects(self, prefix: str = "") -> List[Dict[str, Any]]:
        """
        List objects with optional prefix filter.
        Mimics MinIO's list_objects interface.

        Args:
            prefix: Filter objects by prefix (e.g., "papers/", "metadata/")

        Returns:
            List of object info dicts
        """
        results = []

        # Determine which directory to search
        if prefix.startswith("papers"):
            search_dir = self.base_path / "papers"
            pattern = "*.pdf"
        elif prefix.startswith("metadata"):
            search_dir = self.base_path / "metadata"
            pattern = "*.json"
        elif prefix.startswith("extracted_text"):
            search_dir = self.base_path / "extracted_text"
            pattern = "*.txt"
        else:
            # Search all
            for subdir in ["papers", "metadata", "extracted_text"]:
                search_dir = self.base_path / subdir
                for f in search_dir.glob("*"):
                    if f.is_file():
                        results.append(self._file_to_object_info(f, subdir))
            return results

        for f in search_dir.glob(pattern):
            results.append(self._file_to_object_info(f, prefix.rstrip("/")))

        return results

    def _file_to_object_info(self, file_path: Path, prefix: str) -> Dict[str, Any]:
        """Convert file info to MinIO-style object info."""
        stat = file_path.stat()
        return {
            "object_name": f"{prefix}/{file_path.name}",
            "size": stat.st_size,
            "last_modified": datetime.fromtimestamp(stat.st_mtime),
            "etag": hashlib.md5(file_path.read_bytes()).hexdigest() if stat.st_size < 10_000_000 else None,
        }

    # =========================================================================
    # Index Management (for fast lookups)
    # =========================================================================

    def _get_index_path(self) -> Path:
        return self.base_path / ".index.json"

    def _load_index(self) -> Dict[str, Any]:
        index_path = self._get_index_path()
        if index_path.exists():
            return json.loads(index_path.read_text())
        return {"papers": {}, "updated_at": None}

    def _save_index(self, index: Dict[str, Any]):
        index["updated_at"] = datetime.utcnow().isoformat()
        self._get_index_path().write_text(json.dumps(index, indent=2))

    def _update_index(self, paper_id: str, file_type: str, size: int):
        index = self._load_index()
        if paper_id not in index["papers"]:
            index["papers"][paper_id] = {}
        index["papers"][paper_id][file_type] = {
            "size": size,
            "added_at": datetime.utcnow().isoformat()
        }
        self._save_index(index)

    def _remove_from_index(self, paper_id: str, file_type: str):
        index = self._load_index()
        if paper_id in index["papers"] and file_type in index["papers"][paper_id]:
            del index["papers"][paper_id][file_type]
            if not index["papers"][paper_id]:
                del index["papers"][paper_id]
            self._save_index(index)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "total_papers": len(self.list_papers()),
            "total_size_bytes": 0,
            "by_type": {}
        }

        for subdir in ["papers", "metadata", "extracted_text"]:
            dir_path = self.base_path / subdir
            size = sum(f.stat().st_size for f in dir_path.glob("*") if f.is_file())
            count = len(list(dir_path.glob("*")))
            stats["by_type"][subdir] = {"count": count, "size_bytes": size}
            stats["total_size_bytes"] += size

        return stats

    def cleanup_orphaned_files(self) -> Dict[str, List[str]]:
        """
        Find and optionally clean up files without corresponding PDFs.

        Returns:
            Dict of orphaned files by type
        """
        pdf_ids = set(self.list_papers())
        orphaned = {"metadata": [], "extracted_text": []}

        for subdir in ["metadata", "extracted_text"]:
            dir_path = self.base_path / subdir
            for f in dir_path.glob("*"):
                if f.stem not in pdf_ids:
                    orphaned[subdir].append(f.stem)

        return orphaned

    def export_to_directory(self, export_path: str) -> int:
        """
        Export all papers to a flat directory structure.

        Args:
            export_path: Destination directory

        Returns:
            Number of files exported
        """
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for pdf_file in (self.base_path / "papers").glob("*.pdf"):
            dest = export_dir / pdf_file.name
            shutil.copy2(pdf_file, dest)
            count += 1

        logger.info(f"Exported {count} papers to {export_path}")
        return count
