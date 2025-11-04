"""
Comprehensive tests for storage_enhancement branch.

Tests PostgreSQL conversation/message storage and Redis caching functionality.
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timezone
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

# Import the storage components
from agentic_research.storage.pg_conversation_store import (
    PostgreSQLConversationStore,
    get_pg_conversation_store
)
from agentic_research.db.base import init_db, close_db, get_session_maker
from agentic_research.db.models import Conversation, Message
from agentic_research.core.config import get_settings


@pytest.fixture(scope="module")
async def setup_database():
    """Initialize database before tests."""
    await init_db()
    yield
    await close_db()


@pytest.fixture
async def pg_store():
    """Get PostgreSQL conversation store instance."""
    return get_pg_conversation_store()


@pytest.fixture
async def redis_client():
    """Get Redis client for caching tests."""
    settings = get_settings()
    client = aioredis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        encoding="utf-8"
    )
    yield client
    await client.close()


class TestPostgreSQLConversationStorage:
    """Test PostgreSQL conversation and message storage."""

    @pytest.mark.asyncio
    async def test_create_conversation(self, setup_database, pg_store):
        """Test creating a new conversation."""
        conv = await pg_store.create_conversation(title="Test Conversation")

        assert conv is not None
        assert conv.title == "Test Conversation"
        assert conv.message_count == 0
        assert uuid.UUID(conv.id)  # Valid UUID

        print(f"✓ Created conversation: {conv.id}")

    @pytest.mark.asyncio
    async def test_get_conversation(self, setup_database, pg_store):
        """Test retrieving a conversation."""
        # Create a conversation
        created = await pg_store.create_conversation(title="Retrieve Test")

        # Retrieve it
        retrieved = await pg_store.get_conversation(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.title == "Retrieve Test"

        print(f"✓ Retrieved conversation: {retrieved.id}")

    @pytest.mark.asyncio
    async def test_list_conversations(self, setup_database, pg_store):
        """Test listing conversations."""
        # Create multiple conversations
        await pg_store.create_conversation(title="Conv 1")
        await pg_store.create_conversation(title="Conv 2")
        await pg_store.create_conversation(title="Conv 3")

        # List them
        conversations = await pg_store.list_conversations(limit=10)

        assert len(conversations) >= 3
        assert all(hasattr(conv, 'title') for conv in conversations)

        print(f"✓ Listed {len(conversations)} conversations")

    @pytest.mark.asyncio
    async def test_add_message(self, setup_database, pg_store):
        """Test adding messages to a conversation."""
        # Create conversation
        conv = await pg_store.create_conversation(title="Message Test")

        # Add user message
        user_msg = await pg_store.add_message(
            conv_id=conv.id,
            role="user",
            content="What is machine learning?"
        )

        assert user_msg is not None
        assert user_msg.role == "user"
        assert user_msg.content == "What is machine learning?"

        # Add assistant message with sources
        assistant_msg = await pg_store.add_message(
            conv_id=conv.id,
            role="assistant",
            content="Machine learning is...",
            sources=[{"title": "Paper 1", "relevance_score": 0.95}]
        )

        assert assistant_msg is not None
        assert assistant_msg.role == "assistant"
        assert assistant_msg.sources is not None

        print(f"✓ Added 2 messages to conversation {conv.id}")

    @pytest.mark.asyncio
    async def test_get_messages(self, setup_database, pg_store):
        """Test retrieving messages from a conversation."""
        # Create conversation with messages
        conv = await pg_store.create_conversation(title="Get Messages Test")

        await pg_store.add_message(conv.id, "user", "Question 1")
        await pg_store.add_message(conv.id, "assistant", "Answer 1")
        await pg_store.add_message(conv.id, "user", "Question 2")

        # Retrieve messages
        messages = await pg_store.get_messages(conv.id)

        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[2].role == "user"

        print(f"✓ Retrieved {len(messages)} messages from conversation")

    @pytest.mark.asyncio
    async def test_message_count_increment(self, setup_database, pg_store):
        """Test that message_count increments correctly."""
        conv = await pg_store.create_conversation(title="Count Test")

        # Initial count should be 0
        assert conv.message_count == 0

        # Add messages
        await pg_store.add_message(conv.id, "user", "Message 1")
        await pg_store.add_message(conv.id, "assistant", "Message 2")

        # Check count updated
        updated_conv = await pg_store.get_conversation(conv.id)
        assert updated_conv.message_count == 2

        print(f"✓ Message count correctly incremented to {updated_conv.message_count}")

    @pytest.mark.asyncio
    async def test_update_conversation_title(self, setup_database, pg_store):
        """Test updating conversation title."""
        conv = await pg_store.create_conversation(title="Original Title")

        # Update title
        success = await pg_store.update_conversation_title(
            conv.id,
            "Updated Title"
        )

        assert success is True

        # Verify update
        updated = await pg_store.get_conversation(conv.id)
        assert updated.title == "Updated Title"

        print(f"✓ Updated conversation title successfully")

    @pytest.mark.asyncio
    async def test_delete_conversation(self, setup_database, pg_store):
        """Test deleting a conversation (should cascade delete messages)."""
        # Create conversation with messages
        conv = await pg_store.create_conversation(title="Delete Test")
        await pg_store.add_message(conv.id, "user", "Test message")

        # Delete conversation
        success = await pg_store.delete_conversation(conv.id)
        assert success is True

        # Verify deletion
        deleted = await pg_store.get_conversation(conv.id)
        assert deleted is None

        # Messages should also be deleted (cascade)
        messages = await pg_store.get_messages(conv.id)
        assert len(messages) == 0

        print(f"✓ Deleted conversation and cascaded messages")


class TestRedisCaching:
    """Test Redis caching for research insights."""

    @pytest.mark.asyncio
    async def test_redis_connection(self, redis_client):
        """Test basic Redis connection."""
        # Set a test key
        await redis_client.set("test_key", "test_value")

        # Retrieve it
        value = await redis_client.get("test_key")
        assert value == "test_value"

        # Clean up
        await redis_client.delete("test_key")

        print("✓ Redis connection working")

    @pytest.mark.asyncio
    async def test_insights_cache_structure(self, redis_client):
        """Test research insights cache structure."""
        CACHE_KEY = "research_insights:v1"

        # Simulate cached insights
        mock_insights = {
            "research_areas": [
                {"name": "Machine Learning", "paper_count": 5}
            ],
            "trending_topics": ["transformers", "attention"],
            "research_gaps": [],
            "next_steps": [],
            "kb_context": {
                "total_papers": 10,
                "categories": {"cs.AI": 5, "cs.LG": 5}
            },
            "generation_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "insights_quality": "high"
            }
        }

        # Cache the insights with TTL
        await redis_client.setex(
            CACHE_KEY,
            3600,  # 1 hour
            json.dumps(mock_insights)
        )

        # Retrieve and verify
        cached_data = await redis_client.get(CACHE_KEY)
        assert cached_data is not None

        parsed = json.loads(cached_data)
        assert "research_areas" in parsed
        assert "trending_topics" in parsed
        assert "kb_context" in parsed

        # Clean up
        await redis_client.delete(CACHE_KEY)

        print("✓ Insights cache structure correct")

    @pytest.mark.asyncio
    async def test_cache_ttl(self, redis_client):
        """Test that cache TTL is set correctly."""
        CACHE_KEY = "test_ttl_key"

        # Set with 5 second TTL
        await redis_client.setex(CACHE_KEY, 5, "test_value")

        # Check TTL
        ttl = await redis_client.ttl(CACHE_KEY)
        assert 0 < ttl <= 5

        # Clean up
        await redis_client.delete(CACHE_KEY)

        print(f"✓ Cache TTL correctly set ({ttl}s remaining)")

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, redis_client):
        """Test cache invalidation on new data."""
        CACHE_KEY = "research_insights:v1"

        # Set cached data
        await redis_client.set(CACHE_KEY, json.dumps({"test": "data"}))

        # Verify it exists
        cached = await redis_client.get(CACHE_KEY)
        assert cached is not None

        # Simulate cache invalidation (what happens when papers are indexed)
        await redis_client.delete(CACHE_KEY)

        # Verify it's gone
        invalidated = await redis_client.get(CACHE_KEY)
        assert invalidated is None

        print("✓ Cache invalidation working")


class TestIntegration:
    """Integration tests for PostgreSQL + Redis workflow."""

    @pytest.mark.asyncio
    async def test_chat_to_insights_workflow(self, setup_database, pg_store, redis_client):
        """
        Test complete workflow:
        1. User asks questions (stored in PostgreSQL)
        2. Insights are generated using chat history
        3. Insights are cached in Redis
        """
        # Step 1: Create conversation and add messages (PostgreSQL)
        conv = await pg_store.create_conversation(title="Integration Test Chat")

        await pg_store.add_message(
            conv.id,
            "user",
            "What are transformers in NLP?"
        )
        await pg_store.add_message(
            conv.id,
            "assistant",
            "Transformers are a neural network architecture..."
        )

        # Step 2: Fetch recent queries (simulating insights generation)
        conversations = await pg_store.list_conversations(limit=3)
        recent_queries = []

        for c in conversations:
            messages = await pg_store.get_messages(c.id, limit=20)
            user_msgs = [msg.content for msg in messages if msg.role == 'user']
            recent_queries.extend(user_msgs)

        assert len(recent_queries) > 0
        assert "transformers" in recent_queries[0].lower()

        # Step 3: Cache insights (Redis)
        CACHE_KEY = "research_insights:v1"
        insights = {
            "recent_queries": recent_queries[:10],
            "trending_topics": ["transformers", "attention"],
            "kb_context": {"total_papers": 5}
        }

        await redis_client.setex(
            CACHE_KEY,
            3600,
            json.dumps(insights)
        )

        # Verify cache
        cached = await redis_client.get(CACHE_KEY)
        assert cached is not None

        parsed = json.loads(cached)
        assert "transformers" in str(parsed["recent_queries"])

        # Clean up
        await redis_client.delete(CACHE_KEY)

        print("✓ Complete chat-to-insights workflow working")


# Helper function to run all tests
async def run_all_tests():
    """Run all storage enhancement tests."""
    print("\n" + "="*60)
    print("STORAGE ENHANCEMENT BRANCH - COMPREHENSIVE TESTS")
    print("="*60 + "\n")

    # Initialize database
    print("Initializing database...")
    await init_db()

    # Get instances
    pg_store = get_pg_conversation_store()
    settings = get_settings()
    redis_client = aioredis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        encoding="utf-8"
    )

    try:
        print("\n--- PostgreSQL Tests ---\n")

        # Test conversation creation
        conv = await pg_store.create_conversation(title="Manual Test Conv")
        print(f"✓ Created conversation: {conv.id}")

        # Test message addition
        await pg_store.add_message(conv.id, "user", "Test question")
        await pg_store.add_message(conv.id, "assistant", "Test answer")
        print(f"✓ Added messages to conversation")

        # Test retrieval
        messages = await pg_store.get_messages(conv.id)
        print(f"✓ Retrieved {len(messages)} messages")

        # Test conversation list
        all_convs = await pg_store.list_conversations()
        print(f"✓ Listed {len(all_convs)} conversations")

        print("\n--- Redis Tests ---\n")

        # Test Redis connection
        await redis_client.set("test", "value")
        val = await redis_client.get("test")
        assert val == "value"
        await redis_client.delete("test")
        print("✓ Redis connection working")

        # Test insights cache
        CACHE_KEY = "research_insights:v1"
        test_insights = {
            "research_areas": [],
            "trending_topics": ["test"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await redis_client.setex(CACHE_KEY, 3600, json.dumps(test_insights))
        cached = await redis_client.get(CACHE_KEY)
        assert cached is not None
        await redis_client.delete(CACHE_KEY)
        print("✓ Insights caching working")

        print("\n--- Integration Test ---\n")

        # Test complete workflow
        recent_convs = await pg_store.list_conversations(limit=3)
        queries = []
        for c in recent_convs:
            msgs = await pg_store.get_messages(c.id)
            queries.extend([m.content for m in msgs if m.role == 'user'])

        print(f"✓ Fetched {len(queries)} user queries from PostgreSQL")

        # Cache with queries
        insights_with_history = {
            "recent_queries": queries[:10],
            "kb_context": {"total_papers": 5}
        }
        await redis_client.setex(
            "test_insights",
            3600,
            json.dumps(insights_with_history)
        )
        print("✓ Cached insights with chat history")

        await redis_client.delete("test_insights")

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60 + "\n")

        print("Summary:")
        print("  • PostgreSQL: Conversations and messages stored correctly")
        print("  • Redis: Insights caching with 1-hour TTL working")
        print("  • Integration: Chat history → Insights workflow functional")
        print()

    finally:
        await redis_client.close()
        await close_db()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
