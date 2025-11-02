"""
Unit tests for embeddings generation module.

Tests the EmbeddingsGenerator class for OpenAI text embedding generation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from agentic_research.storage.embeddings import EmbeddingsGenerator


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock()
    settings.OPENAI_API_KEY = "test-api-key"
    return settings


@pytest.fixture
def embeddings_generator(mock_settings):
    """Create an EmbeddingsGenerator instance with mocked OpenAI client."""
    with patch('agentic_research.storage.embeddings.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client

        generator = EmbeddingsGenerator(settings=mock_settings)
        generator.client = mock_client

        return generator


class TestEmbeddingsGenerator:
    """Test suite for EmbeddingsGenerator class."""

    def test_initialization_with_settings(self, mock_settings):
        """Test that generator initializes correctly with provided settings."""
        with patch('agentic_research.storage.embeddings.OpenAI'):
            generator = EmbeddingsGenerator(settings=mock_settings)

            assert generator.settings == mock_settings
            assert generator.model == "text-embedding-3-small"
            assert generator.dimensions == 1536

    def test_initialization_without_api_key(self):
        """Test that generator raises error when API key is missing."""
        settings = Mock()
        settings.OPENAI_API_KEY = None

        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            EmbeddingsGenerator(settings=settings)

    def test_estimate_tokens(self, embeddings_generator):
        """Test token estimation for text."""
        test_text = "This is a test sentence for token estimation."

        tokens = embeddings_generator.estimate_tokens(test_text)

        assert isinstance(tokens, int)
        assert tokens > 0
        # Rough approximation: should be around 10-15 tokens
        assert 5 < tokens < 20

    def test_estimate_cost(self, embeddings_generator):
        """Test cost estimation for text embedding."""
        test_text = "This is a test sentence." * 100  # ~100 words

        cost = embeddings_generator.estimate_cost(test_text)

        assert isinstance(cost, float)
        assert cost > 0
        # Should be a very small cost
        assert cost < 0.01

    def test_generate_embedding_success(self, embeddings_generator):
        """Test successful embedding generation."""
        test_text = "This is a test sentence."
        mock_embedding = [0.1] * 1536  # Mock 1536-dimensional embedding

        # Mock the OpenAI API response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=mock_embedding)]
        embeddings_generator.client.embeddings.create.return_value = mock_response

        result = embeddings_generator.generate_embedding(test_text)

        assert result == mock_embedding
        assert len(result) == 1536
        embeddings_generator.client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=test_text
        )

    def test_generate_embedding_empty_text(self, embeddings_generator):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            embeddings_generator.generate_embedding("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            embeddings_generator.generate_embedding("   ")

    def test_generate_embedding_truncates_long_text(self, embeddings_generator):
        """Test that text exceeding max tokens is truncated."""
        # Create text that's definitely over 8191 tokens
        long_text = "word " * 10000
        mock_embedding = [0.1] * 1536

        mock_response = Mock()
        mock_response.data = [Mock(embedding=mock_embedding)]
        embeddings_generator.client.embeddings.create.return_value = mock_response

        result = embeddings_generator.generate_embedding(long_text)

        assert result == mock_embedding
        # Verify truncation happened
        call_args = embeddings_generator.client.embeddings.create.call_args
        assert len(call_args[1]['input']) < len(long_text)

    def test_batch_embeddings_success(self, embeddings_generator):
        """Test successful batch embedding generation."""
        test_texts = ["First text", "Second text", "Third text"]
        mock_embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]

        mock_response = Mock()
        mock_response.data = [Mock(embedding=emb) for emb in mock_embeddings]
        embeddings_generator.client.embeddings.create.return_value = mock_response

        results = embeddings_generator.batch_embeddings(test_texts)

        assert len(results) == 3
        assert results == mock_embeddings
        embeddings_generator.client.embeddings.create.assert_called_once()

    def test_batch_embeddings_empty_list(self, embeddings_generator):
        """Test that empty list returns empty results."""
        results = embeddings_generator.batch_embeddings([])

        assert results == []
        embeddings_generator.client.embeddings.create.assert_not_called()

    def test_batch_embeddings_filters_empty_texts(self, embeddings_generator):
        """Test that empty texts are filtered out."""
        test_texts = ["Valid text", "", "  ", "Another valid"]
        mock_embeddings = [[0.1] * 1536, [0.2] * 1536]

        mock_response = Mock()
        mock_response.data = [Mock(embedding=emb) for emb in mock_embeddings]
        embeddings_generator.client.embeddings.create.return_value = mock_response

        results = embeddings_generator.batch_embeddings(test_texts)

        assert len(results) == 2
        # Verify only valid texts were sent
        call_args = embeddings_generator.client.embeddings.create.call_args
        sent_texts = call_args[1]['input']
        assert "Valid text" in sent_texts
        assert "Another valid" in sent_texts
        assert "" not in sent_texts

    def test_batch_embeddings_respects_batch_size(self, embeddings_generator):
        """Test that batch processing respects batch_size parameter."""
        test_texts = ["Text"] * 150  # 150 texts
        mock_embedding = [0.1] * 1536

        mock_response = Mock()
        mock_response.data = [Mock(embedding=mock_embedding)] * 100
        embeddings_generator.client.embeddings.create.return_value = mock_response

        results = embeddings_generator.batch_embeddings(test_texts, batch_size=100)

        # Should be called twice: 100 + 50
        assert embeddings_generator.client.embeddings.create.call_count == 2
        assert len(results) == 150

    def test_generate_embeddings_with_metadata(self, embeddings_generator):
        """Test embedding generation with metadata preservation."""
        chunks = [
            {"chunk_text": "First chunk", "metadata": {"page": 1}},
            {"chunk_text": "Second chunk", "metadata": {"page": 2}},
        ]
        mock_embeddings = [[0.1] * 1536, [0.2] * 1536]

        mock_response = Mock()
        mock_response.data = [Mock(embedding=emb) for emb in mock_embeddings]
        embeddings_generator.client.embeddings.create.return_value = mock_response

        results = embeddings_generator.generate_embeddings_with_metadata(chunks)

        assert len(results) == 2
        assert results[0]["chunk_text"] == "First chunk"
        assert results[0]["embedding"] == mock_embeddings[0]
        assert results[0]["metadata"] == {"page": 1}
        assert results[1]["chunk_text"] == "Second chunk"
        assert results[1]["embedding"] == mock_embeddings[1]
        assert results[1]["metadata"] == {"page": 2}

    def test_generate_embeddings_with_metadata_empty_chunks(self, embeddings_generator):
        """Test that empty chunks list returns empty results."""
        results = embeddings_generator.generate_embeddings_with_metadata([])

        assert results == []
        embeddings_generator.client.embeddings.create.assert_not_called()

    def test_validate_embedding_valid(self, embeddings_generator):
        """Test validation of valid embedding."""
        valid_embedding = [0.1] * 1536

        result = embeddings_generator.validate_embedding(valid_embedding)

        assert result is True

    def test_validate_embedding_wrong_dimensions(self, embeddings_generator):
        """Test validation fails for wrong dimensions."""
        wrong_embedding = [0.1] * 512  # Wrong dimensions

        result = embeddings_generator.validate_embedding(wrong_embedding)

        assert result is False

    def test_validate_embedding_not_list(self, embeddings_generator):
        """Test validation fails for non-list input."""
        result = embeddings_generator.validate_embedding("not a list")

        assert result is False

    def test_validate_embedding_invalid_values(self, embeddings_generator):
        """Test validation fails for non-numeric values."""
        invalid_embedding = ["string"] * 1536

        result = embeddings_generator.validate_embedding(invalid_embedding)

        assert result is False


@pytest.mark.integration
class TestEmbeddingsIntegration:
    """Integration tests requiring actual OpenAI API calls."""

    @pytest.mark.skip(reason="Requires OpenAI API key and incurs cost")
    def test_real_embedding_generation(self):
        """Test actual embedding generation with OpenAI API."""
        import os
        from agentic_research.core.config import get_settings

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        settings = get_settings()
        generator = EmbeddingsGenerator(settings=settings)

        test_text = "This is a test sentence for real embedding generation."
        embedding = generator.generate_embedding(test_text)

        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
        assert generator.validate_embedding(embedding)
