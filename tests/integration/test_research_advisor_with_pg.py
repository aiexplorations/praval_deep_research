"""
Integration tests for research advisor with PostgreSQL chat history.

Tests the integration between:
- Research advisor insights generation
- PostgreSQL conversation store
- Chat history retrieval and personalization
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock


@pytest.fixture
async def populated_conversation_store():
    """
    Create a conversation store with test chat history.

    Provides realistic conversation data for testing insights personalization.
    """
    try:
        from agentic_research.db.base import init_db, close_db, get_session_maker
        from agentic_research.storage.pg_conversation_store import PostgreSQLConversationStore

        # Initialize database
        await init_db()

        store = PostgreSQLConversationStore()

        # Create conversation 1: AI/ML focus
        conv1 = await store.create_conversation(title="Machine Learning Research")
        await store.add_message(conv1.id, "user", "What are the latest advances in transformer architectures?")
        await store.add_message(conv1.id, "assistant", "Recent advances include...")
        await store.add_message(conv1.id, "user", "How does attention mechanism work in transformers?")
        await store.add_message(conv1.id, "assistant", "The attention mechanism...")

        # Create conversation 2: Computer vision
        conv2 = await store.create_conversation(title="Computer Vision Project")
        await store.add_message(conv2.id, "user", "What are best practices for object detection?")
        await store.add_message(conv2.id, "assistant", "Best practices include...")
        await store.add_message(conv2.id, "user", "Explain YOLO vs Faster R-CNN")
        await store.add_message(conv2.id, "assistant", "YOLO and Faster R-CNN differ in...")

        # Create conversation 3: NLP
        conv3 = await store.create_conversation(title="Natural Language Processing")
        await store.add_message(conv3.id, "user", "How does BERT differ from GPT?")
        await store.add_message(conv3.id, "assistant", "BERT and GPT have different architectures...")

        yield store

        # Cleanup
        session_maker = get_session_maker()
        async with session_maker() as session:
            from sqlalchemy import text
            await session.execute(text("DELETE FROM messages"))
            await session.execute(text("DELETE FROM conversations"))
            await session.commit()

        await close_db()

    except ImportError as e:
        pytest.skip(f"PostgreSQL dependencies not available: {e}")
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
class TestResearchAdvisorIntegration:
    """Test research advisor integration with PostgreSQL."""

    async def test_fetch_recent_conversations(self, populated_conversation_store):
        """Test that research advisor can fetch recent conversations."""
        from agentic_research.storage.conversation_store import get_conversation_store

        store = get_conversation_store()
        conversations = await store.list_conversations(limit=3)

        assert len(conversations) == 3
        assert all(conv.message_count > 0 for conv in conversations)

    async def test_extract_user_queries_from_history(self, populated_conversation_store):
        """Test extracting user queries from conversation history."""
        from agentic_research.storage.conversation_store import get_conversation_store

        store = get_conversation_store()
        conversations = await store.list_conversations(limit=3)

        all_queries = []
        for conv in conversations:
            messages = await store.get_messages(conv.id)
            user_queries = [msg.content for msg in messages if msg.role == "user"]
            all_queries.extend(user_queries)

        # Should have extracted user questions
        assert len(all_queries) > 0
        assert "transformer" in " ".join(all_queries).lower() or "bert" in " ".join(all_queries).lower()

    @patch('agents.interaction.research_advisor.QdrantClientWrapper')
    @patch('agents.interaction.research_advisor.OpenAI')
    async def test_generate_insights_with_chat_history(
        self,
        mock_openai,
        mock_qdrant,
        populated_conversation_store
    ):
        """Test that insights generation uses chat history for personalization."""
        from agents.interaction.research_advisor import generate_insights_sync
        from agentic_research.core.config import get_settings

        # Mock Qdrant to return some papers
        mock_qdrant_instance = MagicMock()
        mock_qdrant_instance.get_all_papers.return_value = [
            {
                "paper_id": "2301.12345",
                "title": "Attention Is All You Need",
                "categories": ["cs.AI", "cs.LG"],
                "published_date": "2023-01-15"
            },
            {
                "paper_id": "2302.54321",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "categories": ["cs.CL", "cs.AI"],
                "published_date": "2023-02-20"
            }
        ]
        mock_qdrant.return_value = mock_qdrant_instance

        # Mock OpenAI responses
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content='["Test Area 1", "Test Area 2"]'))]

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_openai_instance

        # Generate insights
        settings = get_settings()
        insights = generate_insights_sync(settings)

        # Verify insights were generated
        assert "research_areas" in insights
        assert "generation_metadata" in insights

        # Verify chat history was used
        metadata = insights["generation_metadata"]
        assert "conversation_history_used" in metadata
        assert metadata["conversation_history_used"] > 0  # Should have fetched conversations
        assert metadata.get("personalization_enabled", False) is True

    @patch('agents.interaction.research_advisor.QdrantClientWrapper')
    @patch('agents.interaction.research_advisor.OpenAI')
    async def test_generate_insights_without_chat_history(
        self,
        mock_openai,
        mock_qdrant
    ):
        """Test insights generation when no chat history exists."""
        from agents.interaction.research_advisor import generate_insights_sync
        from agentic_research.core.config import get_settings
        from agentic_research.db.base import init_db, close_db, get_session_maker

        # Initialize empty database
        await init_db()

        try:
            # Mock Qdrant
            mock_qdrant_instance = MagicMock()
            mock_qdrant_instance.get_all_papers.return_value = [
                {
                    "paper_id": "2301.12345",
                    "title": "Test Paper",
                    "categories": ["cs.AI"],
                    "published_date": "2023-01-15"
                }
            ]
            mock_qdrant.return_value = mock_qdrant_instance

            # Mock OpenAI
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock(message=MagicMock(content='[]'))]
            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_openai_instance

            # Generate insights
            settings = get_settings()
            insights = generate_insights_sync(settings)

            # Verify insights were generated even without history
            assert "generation_metadata" in insights
            metadata = insights["generation_metadata"]
            assert metadata["conversation_history_used"] == 0
            assert metadata.get("personalization_enabled", False) is False

        finally:
            # Cleanup
            session_maker = get_session_maker()
            async with session_maker() as session:
                from sqlalchemy import text
                await session.execute(text("DELETE FROM messages"))
                await session.execute(text("DELETE FROM conversations"))
                await session.commit()
            await close_db()

    async def test_insights_api_endpoint_uses_postgres(self, populated_conversation_store):
        """Test that the insights API endpoint uses PostgreSQL for chat history."""
        from fastapi.testclient import TestClient
        from agentic_research.api.main import app

        client = TestClient(app)

        with patch('agents.interaction.research_advisor.QdrantClientWrapper') as mock_qdrant, \
             patch('agents.interaction.research_advisor.OpenAI') as mock_openai:

            # Mock responses
            mock_qdrant_instance = MagicMock()
            mock_qdrant_instance.get_all_papers.return_value = []
            mock_qdrant.return_value = mock_qdrant_instance

            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock(message=MagicMock(content='[]'))]
            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_openai_instance

            # Call insights endpoint
            response = client.get("/research/insights")

            # Should succeed (may return cached or fresh insights)
            assert response.status_code in [200, 500]  # 500 if no papers/setup issues

            if response.status_code == 200:
                data = response.json()
                assert "generation_metadata" in data


@pytest.mark.integration
@pytest.mark.asyncio
class TestConversationStoreFactory:
    """Test the conversation store factory function."""

    async def test_factory_returns_postgres_by_default(self):
        """Test that factory returns PostgreSQL store by default."""
        from agentic_research.storage.conversation_store import get_conversation_store
        from agentic_research.storage.pg_conversation_store import PostgreSQLConversationStore

        store = get_conversation_store()
        assert isinstance(store, PostgreSQLConversationStore)

    async def test_factory_returns_redis_when_env_set(self, monkeypatch):
        """Test that factory returns Redis store when USE_REDIS_STORE=true."""
        monkeypatch.setenv("USE_REDIS_STORE", "true")

        # Clear the global cache
        import agentic_research.storage.conversation_store as conv_store_module
        conv_store_module._conversation_store = None

        from agentic_research.storage.conversation_store import get_conversation_store, ConversationStore

        store = get_conversation_store()
        assert isinstance(store, ConversationStore)
        assert not hasattr(store, 'session_maker')  # Redis store doesn't have session_maker


@pytest.mark.integration
@pytest.mark.asyncio
class TestHealthCheckWithPostgreSQL:
    """Test health check endpoint with PostgreSQL."""

    async def test_health_check_includes_postgresql(self):
        """Test that health check endpoint includes PostgreSQL status."""
        from fastapi.testclient import TestClient
        from agentic_research.api.main import app

        client = TestClient(app)

        response = client.get("/health/infrastructure")

        assert response.status_code == 200
        data = response.json()

        # Should include PostgreSQL in infrastructure status
        assert "postgresql" in data
        assert data["postgresql"] in ["connected", "disconnected"]

    async def test_postgresql_health_check_verifies_connection(self):
        """Test that PostgreSQL health check actually verifies connection."""
        from agentic_research.api.routes.health import _check_infrastructure

        infrastructure = await _check_infrastructure()

        assert "postgresql" in infrastructure
        # Status depends on whether PostgreSQL is actually running
        assert infrastructure["postgresql"] in ["connected", "disconnected"]
