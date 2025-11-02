"""
MinIO client for PDF storage and retrieval.

This module provides a client for interacting with MinIO object storage
for storing and retrieving research paper PDFs.
"""

import io
from typing import Optional, BinaryIO
from datetime import timedelta
import structlog
from minio import Minio
from minio.error import S3Error

from agentic_research.core.config import get_settings


logger = structlog.get_logger(__name__)


class MinIOClient:
    """
    Client for MinIO object storage operations.

    I am a MinIO storage client who handles PDF storage, retrieval, and
    presigned URL generation for research papers.
    """

    def __init__(self, settings=None):
        """
        Initialize MinIO client.

        Args:
            settings: Optional settings object (uses get_settings() if None)
        """
        self.settings = settings or get_settings()
        self.bucket_name = self.settings.MINIO_BUCKET_NAME

        # Initialize MinIO client
        self.client = Minio(
            endpoint=self.settings.MINIO_ENDPOINT,
            access_key=self.settings.MINIO_ACCESS_KEY,
            secret_key=self.settings.MINIO_SECRET_KEY,
            secure=self.settings.MINIO_SECURE
        )

        logger.info(
            "MinIO client initialized",
            endpoint=self.settings.MINIO_ENDPOINT,
            bucket=self.bucket_name
        )

    def bucket_init(self) -> None:
        """
        Initialize bucket if it doesn't exist.

        Raises:
            S3Error: If bucket creation fails
        """
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info("Bucket created", bucket=self.bucket_name)
            else:
                logger.debug("Bucket already exists", bucket=self.bucket_name)

        except S3Error as e:
            logger.error(
                "Failed to initialize bucket",
                bucket=self.bucket_name,
                error=str(e)
            )
            raise

    def upload_pdf(
        self,
        paper_id: str,
        pdf_data: bytes,
        content_type: str = "application/pdf"
    ) -> str:
        """
        Upload PDF to MinIO.

        Args:
            paper_id: Unique identifier for the paper (e.g., ArXiv ID)
            pdf_data: PDF file data as bytes
            content_type: MIME type for the PDF

        Returns:
            Object path in MinIO (e.g., "papers/2106.04560.pdf")

        Raises:
            S3Error: If upload fails
        """
        object_name = f"papers/{paper_id}.pdf"

        try:
            # Ensure bucket exists
            self.bucket_init()

            # Upload PDF
            pdf_stream = io.BytesIO(pdf_data)
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=pdf_stream,
                length=len(pdf_data),
                content_type=content_type
            )

            logger.info(
                "PDF uploaded successfully",
                paper_id=paper_id,
                object_name=object_name,
                size_bytes=len(pdf_data)
            )

            return object_name

        except S3Error as e:
            logger.error(
                "Failed to upload PDF",
                paper_id=paper_id,
                object_name=object_name,
                error=str(e)
            )
            raise

    def download_pdf(self, paper_id: str) -> bytes:
        """
        Download PDF from MinIO.

        Args:
            paper_id: Unique identifier for the paper

        Returns:
            PDF file data as bytes

        Raises:
            S3Error: If download fails or object doesn't exist
        """
        object_name = f"papers/{paper_id}.pdf"

        try:
            response = self.client.get_object(
                bucket_name=self.bucket_name,
                object_name=object_name
            )

            pdf_data = response.read()
            response.close()
            response.release_conn()

            logger.info(
                "PDF downloaded successfully",
                paper_id=paper_id,
                object_name=object_name,
                size_bytes=len(pdf_data)
            )

            return pdf_data

        except S3Error as e:
            logger.error(
                "Failed to download PDF",
                paper_id=paper_id,
                object_name=object_name,
                error=str(e)
            )
            raise

    def generate_presigned_url(
        self,
        paper_id: str,
        expiry_hours: int = 24
    ) -> str:
        """
        Generate presigned URL for PDF download.

        Args:
            paper_id: Unique identifier for the paper
            expiry_hours: URL expiration time in hours (default: 24)

        Returns:
            Presigned URL for direct PDF download

        Raises:
            S3Error: If URL generation fails
        """
        object_name = f"papers/{paper_id}.pdf"

        try:
            url = self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                expires=timedelta(hours=expiry_hours)
            )

            logger.info(
                "Presigned URL generated",
                paper_id=paper_id,
                object_name=object_name,
                expiry_hours=expiry_hours
            )

            return url

        except S3Error as e:
            logger.error(
                "Failed to generate presigned URL",
                paper_id=paper_id,
                object_name=object_name,
                error=str(e)
            )
            raise

    def pdf_exists(self, paper_id: str) -> bool:
        """
        Check if PDF exists in MinIO.

        Args:
            paper_id: Unique identifier for the paper

        Returns:
            True if PDF exists, False otherwise
        """
        object_name = f"papers/{paper_id}.pdf"

        try:
            self.client.stat_object(
                bucket_name=self.bucket_name,
                object_name=object_name
            )
            return True

        except S3Error:
            return False

    def upload_metadata(
        self,
        paper_id: str,
        metadata: dict
    ) -> str:
        """
        Upload paper metadata as JSON.

        Args:
            paper_id: Unique identifier for the paper
            metadata: Metadata dictionary to store

        Returns:
            Object path in MinIO (e.g., "metadata/2106.04560.json")

        Raises:
            S3Error: If upload fails
        """
        import json

        object_name = f"metadata/{paper_id}.json"

        try:
            # Ensure bucket exists
            self.bucket_init()

            # Convert metadata to JSON bytes
            metadata_json = json.dumps(metadata, indent=2)
            metadata_bytes = metadata_json.encode('utf-8')

            # Upload metadata
            metadata_stream = io.BytesIO(metadata_bytes)
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=metadata_stream,
                length=len(metadata_bytes),
                content_type="application/json"
            )

            logger.info(
                "Metadata uploaded successfully",
                paper_id=paper_id,
                object_name=object_name
            )

            return object_name

        except S3Error as e:
            logger.error(
                "Failed to upload metadata",
                paper_id=paper_id,
                object_name=object_name,
                error=str(e)
            )
            raise

    def upload_extracted_text(
        self,
        paper_id: str,
        text: str
    ) -> str:
        """
        Upload extracted text from PDF.

        Args:
            paper_id: Unique identifier for the paper
            text: Extracted text content

        Returns:
            Object path in MinIO (e.g., "extracted_text/2106.04560.txt")

        Raises:
            S3Error: If upload fails
        """
        object_name = f"extracted_text/{paper_id}.txt"

        try:
            # Ensure bucket exists
            self.bucket_init()

            # Convert text to bytes
            text_bytes = text.encode('utf-8')

            # Upload text
            text_stream = io.BytesIO(text_bytes)
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=text_stream,
                length=len(text_bytes),
                content_type="text/plain; charset=utf-8"
            )

            logger.info(
                "Extracted text uploaded successfully",
                paper_id=paper_id,
                object_name=object_name,
                size_bytes=len(text_bytes)
            )

            return object_name

        except S3Error as e:
            logger.error(
                "Failed to upload extracted text",
                paper_id=paper_id,
                object_name=object_name,
                error=str(e)
            )
            raise

    def delete_paper(self, paper_id: str) -> None:
        """
        Delete all objects related to a paper (PDF, metadata, extracted text).

        Args:
            paper_id: Unique identifier for the paper

        Raises:
            S3Error: If deletion fails
        """
        objects_to_delete = [
            f"papers/{paper_id}.pdf",
            f"metadata/{paper_id}.json",
            f"extracted_text/{paper_id}.txt"
        ]

        for object_name in objects_to_delete:
            try:
                self.client.remove_object(
                    bucket_name=self.bucket_name,
                    object_name=object_name
                )
                logger.debug(
                    "Object deleted",
                    paper_id=paper_id,
                    object_name=object_name
                )
            except S3Error as e:
                # Log but don't fail if object doesn't exist
                logger.warning(
                    "Failed to delete object (may not exist)",
                    paper_id=paper_id,
                    object_name=object_name,
                    error=str(e)
                )

        logger.info("Paper objects deleted", paper_id=paper_id)
