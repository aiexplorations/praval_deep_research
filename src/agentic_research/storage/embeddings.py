"""
Embeddings generation using OpenAI text-embedding-3-small.

This module provides functions for generating vector embeddings
from text using OpenAI's embedding models.
"""

from typing import List, Dict, Any
import structlog
from openai import OpenAI, APIError, RateLimitError
import tiktoken

from agentic_research.core.config import get_settings


logger = structlog.get_logger(__name__)


class EmbeddingsGenerator:
    """
    Generator for OpenAI text embeddings.

    I am an embeddings generator who creates vector representations
    of text using OpenAI's text-embedding-3-small model.
    """

    def __init__(self, settings=None):
        """
        Initialize embeddings generator.

        Args:
            settings: Optional settings object (uses get_settings() if None)
        """
        self.settings = settings or get_settings()
        self.model = "text-embedding-3-small"
        self.dimensions = 1536

        # Initialize OpenAI client
        if not self.settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for embeddings generation")

        self.client = OpenAI(api_key=self.settings.OPENAI_API_KEY)

        # Initialize tokenizer for cost estimation
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback to cl100k_base if model not found
            self.encoding = tiktoken.get_encoding("cl100k_base")

        logger.info(
            "Embeddings generator initialized",
            model=self.model,
            dimensions=self.dimensions
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning("Token estimation failed, using approximation", error=str(e))
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4

    def estimate_cost(self, text: str) -> float:
        """
        Estimate cost in USD for embedding generation.

        Args:
            text: Text to estimate cost for

        Returns:
            Estimated cost in USD

        Note:
            text-embedding-3-small costs $0.00002 per 1K tokens
        """
        tokens = self.estimate_tokens(text)
        cost_per_1k_tokens = 0.00002
        return (tokens / 1000.0) * cost_per_1k_tokens

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            APIError: If OpenAI API call fails
            RateLimitError: If rate limit is exceeded
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Remove newlines for better embedding quality
            text = text.replace("\n", " ").strip()

            # Estimate tokens
            tokens = self.estimate_tokens(text)

            # OpenAI has max 8191 tokens for this model
            max_tokens = 8191
            if tokens > max_tokens:
                logger.warning(
                    "Text exceeds max tokens, truncating",
                    tokens=tokens,
                    max_tokens=max_tokens
                )
                # Truncate text (rough approximation)
                text = text[:max_tokens * 4]

            # Generate embedding
            response = self.client.embeddings.create(
                model=self.model,
                input=text
                # Note: dimensions parameter removed for compatibility
                # text-embedding-3-small defaults to 1536 dimensions
            )

            embedding = response.data[0].embedding

            logger.debug(
                "Embedding generated",
                text_length=len(text),
                tokens=tokens,
                embedding_dims=len(embedding)
            )

            return embedding

        except RateLimitError as e:
            logger.error("Rate limit exceeded", error=str(e))
            raise

        except APIError as e:
            logger.error("OpenAI API error", error=str(e))
            raise

        except Exception as e:
            logger.error("Unexpected error generating embedding", error=str(e))
            raise

    def batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to embed per API call (max 2048)

        Returns:
            List of embedding vectors

        Raises:
            APIError: If OpenAI API call fails
            RateLimitError: If rate limit is exceeded

        Note:
            OpenAI allows up to 2048 inputs per request for embeddings API.
            We use smaller batches (100) for better error handling.
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [text.replace("\n", " ").strip() for text in texts if text and text.strip()]

        if not valid_texts:
            raise ValueError("All texts are empty")

        embeddings = []

        try:
            # Process in batches
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i + batch_size]

                # Estimate total tokens for batch
                batch_tokens = sum(self.estimate_tokens(text) for text in batch)

                logger.debug(
                    "Processing batch",
                    batch_index=i // batch_size + 1,
                    batch_size=len(batch),
                    estimated_tokens=batch_tokens
                )

                # Generate embeddings for batch
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                    # Note: dimensions parameter removed for compatibility
                    # text-embedding-3-small defaults to 1536 dimensions
                )

                # Extract embeddings in order
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

            logger.info(
                "Batch embeddings generated",
                num_texts=len(valid_texts),
                num_batches=(len(valid_texts) + batch_size - 1) // batch_size,
                num_embeddings=len(embeddings)
            )

            return embeddings

        except RateLimitError as e:
            logger.error(
                "Rate limit exceeded in batch processing",
                error=str(e),
                processed=len(embeddings)
            )
            raise

        except APIError as e:
            logger.error(
                "OpenAI API error in batch processing",
                error=str(e),
                processed=len(embeddings)
            )
            raise

        except Exception as e:
            logger.error(
                "Unexpected error in batch processing",
                error=str(e),
                processed=len(embeddings)
            )
            raise

    def generate_embeddings_with_metadata(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunks and return with metadata.

        Args:
            chunks: List of chunk dictionaries containing 'chunk_text' key

        Returns:
            List of chunks with 'embedding' key added

        Raises:
            APIError: If OpenAI API call fails
            RateLimitError: If rate limit is exceeded
        """
        if not chunks:
            return []

        # Extract texts
        texts = [chunk["chunk_text"] for chunk in chunks]

        # Generate embeddings
        embeddings = self.batch_embeddings(texts)

        # Add embeddings to chunks
        enriched_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            enriched_chunk = chunk.copy()
            enriched_chunk["embedding"] = embedding
            enriched_chunks.append(enriched_chunk)

        logger.info(
            "Embeddings generated with metadata",
            num_chunks=len(enriched_chunks)
        )

        return enriched_chunks

    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate that embedding has correct dimensions.

        Args:
            embedding: Embedding vector to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(embedding, list):
            return False

        if len(embedding) != self.dimensions:
            logger.warning(
                "Invalid embedding dimensions",
                expected=self.dimensions,
                actual=len(embedding)
            )
            return False

        # Check all values are floats
        if not all(isinstance(x, (int, float)) for x in embedding):
            return False

        return True
