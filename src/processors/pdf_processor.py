"""
PDF processing module for downloading and extracting text from research papers.

This module handles PDF downloads from ArXiv, text extraction,
and intelligent text chunking for embedding generation.
"""

import io
import re
import time
from typing import List, Dict, Any
import aiohttp
import requests
import structlog
import pdfplumber
from PyPDF2 import PdfReader


logger = structlog.get_logger(__name__)


class PDFProcessingError(Exception):
    """Exception raised when PDF processing fails."""
    pass


class PDFProcessor:
    """
    Processor for PDF download, extraction, and chunking.

    I am a PDF processor who downloads papers from ArXiv, extracts text,
    and chunks content intelligently for embedding generation.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_chunks_per_paper: int = 50
    ):
        """
        Initialize PDF processor.

        Args:
            chunk_size: Target size for each text chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            max_chunks_per_paper: Maximum number of chunks to generate per paper
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks_per_paper = max_chunks_per_paper

        logger.info(
            "PDF processor initialized",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks_per_paper
        )

    async def download_from_arxiv(
        self,
        arxiv_id: str,
        max_retries: int = 3
    ) -> bytes:
        """
        Download PDF from ArXiv.

        Args:
            arxiv_id: ArXiv paper ID (e.g., "2106.04560")
            max_retries: Number of download retries

        Returns:
            PDF file data as bytes

        Raises:
            PDFProcessingError: If download fails after retries
        """
        # Construct PDF URL
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(pdf_url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                        if response.status == 200:
                            pdf_data = await response.read()

                            logger.info(
                                "PDF downloaded successfully",
                                arxiv_id=arxiv_id,
                                size_bytes=len(pdf_data),
                                attempt=attempt + 1
                            )

                            return pdf_data

                        else:
                            logger.warning(
                                "PDF download failed",
                                arxiv_id=arxiv_id,
                                status=response.status,
                                attempt=attempt + 1
                            )

            except aiohttp.ClientError as e:
                logger.warning(
                    "Network error downloading PDF",
                    arxiv_id=arxiv_id,
                    error=str(e),
                    attempt=attempt + 1
                )

            except Exception as e:
                logger.error(
                    "Unexpected error downloading PDF",
                    arxiv_id=arxiv_id,
                    error=str(e),
                    attempt=attempt + 1
                )

            # Wait before retry (exponential backoff)
            if attempt < max_retries - 1:
                import asyncio
                await asyncio.sleep(2 ** attempt)

        raise PDFProcessingError(
            f"Failed to download PDF for {arxiv_id} after {max_retries} attempts"
        )

    def download_from_arxiv_sync(
        self,
        arxiv_id: str,
        max_retries: int = 3
    ) -> bytes:
        """
        Download PDF from ArXiv synchronously.

        Use this method when calling from a synchronous context that's already
        running inside an async event loop (e.g., Praval agents with RabbitMQ).

        Args:
            arxiv_id: ArXiv paper ID (e.g., "2106.04560")
            max_retries: Number of download retries

        Returns:
            PDF file data as bytes

        Raises:
            PDFProcessingError: If download fails after retries
        """
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        for attempt in range(max_retries):
            try:
                response = requests.get(pdf_url, timeout=60)

                if response.status_code == 200:
                    pdf_data = response.content

                    logger.info(
                        "PDF downloaded successfully (sync)",
                        arxiv_id=arxiv_id,
                        size_bytes=len(pdf_data),
                        attempt=attempt + 1
                    )

                    return pdf_data

                else:
                    logger.warning(
                        "PDF download failed",
                        arxiv_id=arxiv_id,
                        status=response.status_code,
                        attempt=attempt + 1
                    )

            except requests.RequestException as e:
                logger.warning(
                    "Network error downloading PDF",
                    arxiv_id=arxiv_id,
                    error=str(e),
                    attempt=attempt + 1
                )

            except Exception as e:
                logger.error(
                    "Unexpected error downloading PDF",
                    arxiv_id=arxiv_id,
                    error=str(e),
                    attempt=attempt + 1
                )

            # Wait before retry (exponential backoff)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        raise PDFProcessingError(
            f"Failed to download PDF for {arxiv_id} after {max_retries} attempts"
        )

    def extract_text(
        self,
        pdf_data: bytes,
        method: str = "pdfplumber"
    ) -> str:
        """
        Extract text from PDF bytes.

        Args:
            pdf_data: PDF file data as bytes
            method: Extraction method ("pdfplumber" or "pypdf2")

        Returns:
            Extracted text as string

        Raises:
            PDFProcessingError: If text extraction fails
        """
        try:
            if method == "pdfplumber":
                text = self._extract_with_pdfplumber(pdf_data)
            elif method == "pypdf2":
                text = self._extract_with_pypdf2(pdf_data)
            else:
                raise ValueError(f"Unknown extraction method: {method}")

            # Clean text
            text = self._clean_text(text)

            logger.info(
                "Text extracted successfully",
                method=method,
                text_length=len(text),
                num_words=len(text.split())
            )

            return text

        except Exception as e:
            logger.error(
                "Text extraction failed",
                method=method,
                error=str(e)
            )
            raise PDFProcessingError(f"Failed to extract text: {str(e)}")

    def _extract_with_pdfplumber(self, pdf_data: bytes) -> str:
        """
        Extract text using pdfplumber (better layout handling).

        Args:
            pdf_data: PDF file data as bytes

        Returns:
            Extracted text
        """
        text_parts = []

        with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        return "\n\n".join(text_parts)

    def _extract_with_pypdf2(self, pdf_data: bytes) -> str:
        """
        Extract text using PyPDF2 (fallback method).

        Args:
            pdf_data: PDF file data as bytes

        Returns:
            Extracted text
        """
        text_parts = []

        reader = PdfReader(io.BytesIO(pdf_data))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        return "\n\n".join(text_parts)

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)

        # Remove headers/footers (repeated patterns)
        lines = text.split('\n')
        if len(lines) > 10:
            # Remove lines that appear too frequently (likely headers/footers)
            line_counts = {}
            for line in lines:
                stripped = line.strip()
                if stripped:
                    line_counts[stripped] = line_counts.get(stripped, 0) + 1

            # Filter out lines appearing more than 10% of pages
            threshold = len(lines) * 0.1
            filtered_lines = [
                line for line in lines
                if line_counts.get(line.strip(), 0) < threshold
            ]
            text = '\n'.join(filtered_lines)

        # Normalize unicode characters
        text = text.replace('\u2019', "'")
        text = text.replace('\u2018', "'")
        text = text.replace('\u201c', '"')
        text = text.replace('\u201d', '"')
        text = text.replace('\u2013', '-')
        text = text.replace('\u2014', '--')

        return text.strip()

    def chunk_text(
        self,
        text: str,
        paper_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text intelligently for embedding generation.

        Args:
            text: Text to chunk
            paper_metadata: Optional metadata to include with each chunk

        Returns:
            List of chunk dictionaries with text and metadata

        Raises:
            PDFProcessingError: If chunking fails
        """
        if not text or not text.strip():
            raise PDFProcessingError("Cannot chunk empty text")

        try:
            # Split into sentences
            sentences = self._split_into_sentences(text)

            chunks = []
            current_chunk = ""
            current_length = 0

            for sentence in sentences:
                sentence_length = len(sentence)

                # If adding this sentence would exceed chunk_size
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append(current_chunk.strip())

                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + sentence
                    current_length = len(current_chunk)
                else:
                    # Add sentence to current chunk
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_length += sentence_length

                # Limit total chunks
                if len(chunks) >= self.max_chunks_per_paper:
                    logger.warning(
                        "Reached max chunks limit",
                        max_chunks=self.max_chunks_per_paper
                    )
                    break

            # Add final chunk
            if current_chunk.strip() and len(chunks) < self.max_chunks_per_paper:
                chunks.append(current_chunk.strip())

            # Build chunk dictionaries
            chunk_dicts = []
            for idx, chunk_text in enumerate(chunks):
                chunk_dict = {
                    "chunk_text": chunk_text,
                    "chunk_index": idx,
                    "total_chunks": len(chunks)
                }

                # Add metadata if provided
                if paper_metadata:
                    chunk_dict.update(paper_metadata)

                chunk_dicts.append(chunk_dict)

            logger.info(
                "Text chunked successfully",
                num_chunks=len(chunk_dicts),
                avg_chunk_size=sum(len(c["chunk_text"]) for c in chunk_dicts) // len(chunk_dicts) if chunk_dicts else 0
            )

            return chunk_dicts

        except Exception as e:
            logger.error("Chunking failed", error=str(e))
            raise PDFProcessingError(f"Failed to chunk text: {str(e)}")

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with NLTK)
        # Split on period, exclamation, question mark followed by space/newline
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter out very short sentences (likely artifacts)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        return sentences

    def _get_overlap_text(self, text: str) -> str:
        """
        Get overlap text from end of chunk.

        Args:
            text: Source text

        Returns:
            Overlap text (last N characters, sentence-aware)
        """
        if len(text) <= self.chunk_overlap:
            return text

        # Get last chunk_overlap characters
        overlap = text[-self.chunk_overlap:]

        # Try to start at sentence boundary
        sentence_start = overlap.find('. ')
        if sentence_start != -1 and sentence_start < len(overlap) // 2:
            overlap = overlap[sentence_start + 2:]

        return overlap.strip()

    def estimate_chunks(self, text_or_length: int | str) -> int:
        """
        Estimate number of chunks for given text.

        Args:
            text_or_length: Either text string or text length

        Returns:
            Estimated number of chunks
        """
        if isinstance(text_or_length, str):
            text_length = len(text_or_length)
        else:
            text_length = text_or_length

        # Rough estimation accounting for overlap
        effective_chunk_size = self.chunk_size - self.chunk_overlap
        estimated = max(1, (text_length + effective_chunk_size - 1) // effective_chunk_size)

        return min(estimated, self.max_chunks_per_paper)
