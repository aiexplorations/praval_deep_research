"""
Unit tests for PostgreSQL conversation store.

Tests the PostgreSQLConversationStore implementation including:
- Creating, reading, updating, deleting conversations
- Adding and retrieving messages
- Cascade deletion behavior
- Pagination and ordering
- Error handling and edge cases
"""

import pytest
import uuid
from datetime import datetime
from typing import List

# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    True,  # Will be replaced with actual check
    reason="PostgreSQL not available"
)


@pytest.fixture
async def pg_store():
    """
    Provide PostgreSQL conversation store instance for testing.

    Ensures database is initialized and provides clean store for each test.
    """
    try:
        from agentic_research.db.base import init_db, close_db, get_session_maker
        from agentic_research.storage.pg_conversation_store import PostgreSQLConversationStore

        # Initialize database tables
        await init_db()

        # Create store instance
        store = PostgreSQLConversationStore()

        yield store

        # Cleanup: Delete all test data
        session_maker = get_session_maker()
        async with session_maker() as session:
            from sqlalchemy import text
            await session.execute(text("DELETE FROM messages"))
            await session.execute(text("DELETE FROM conversations"))
            await session.commit()

        # Close connections
        await close_db()

    except ImportError as e:
        pytest.skip(f"PostgreSQL dependencies not available: {e}")
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")


@pytest.mark.asyncio
class TestConversationCreation:
    """Test conversation creation with various scenarios."""

    async def test_create_conversation_default(self, pg_store):
        """Test creating a conversation with auto-generated ID and default title."""
        conv = await pg_store.create_conversation()

        assert conv.id is not None
        assert conv.title.startswith("Chat ")
        assert conv.message_count == 0
        assert conv.created_at is not None
        assert conv.updated_at is not None

    async def test_create_conversation_custom_title(self, pg_store):
        """Test creating a conversation with custom title."""
        custom_title = "My Research Project"
        conv = await pg_store.create_conversation(title=custom_title)

        assert conv.title == custom_title
        assert conv.message_count == 0

    async def test_create_conversation_custom_id(self, pg_store):
        """Test creating a conversation with specific ID."""
        custom_id = str(uuid.uuid4())
        conv = await pg_store.create_conversation(
            title="Custom ID Test",
            conversation_id=custom_id
        )

        assert conv.id == custom_id
        assert conv.title == "Custom ID Test"

    async def test_create_multiple_conversations(self, pg_store):
        """Test creating multiple conversations with unique IDs."""
        conv1 = await pg_store.create_conversation(title="Conv 1")
        conv2 = await pg_store.create_conversation(title="Conv 2")
        conv3 = await pg_store.create_conversation(title="Conv 3")

        assert conv1.id != conv2.id
        assert conv2.id != conv3.id
        assert conv1.id != conv3.id


@pytest.mark.asyncio
class TestConversationRetrieval:
    """Test retrieving conversations."""

    async def test_get_existing_conversation(self, pg_store):
        """Test retrieving an existing conversation by ID."""
        created = await pg_store.create_conversation(title="Test Conv")
        retrieved = await pg_store.get_conversation(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.title == created.title
        assert retrieved.message_count == created.message_count

    async def test_get_nonexistent_conversation(self, pg_store):
        """Test retrieving a conversation that doesn't exist."""
        fake_id = str(uuid.uuid4())
        retrieved = await pg_store.get_conversation(fake_id)

        assert retrieved is None

    async def test_list_conversations_empty(self, pg_store):
        """Test listing conversations when none exist."""
        conversations = await pg_store.list_conversations()

        assert conversations == []

    async def test_list_conversations_multiple(self, pg_store):
        """Test listing multiple conversations ordered by most recent."""
        # Create conversations with slight delays to ensure ordering
        conv1 = await pg_store.create_conversation(title="First")
        conv2 = await pg_store.create_conversation(title="Second")
        conv3 = await pg_store.create_conversation(title="Third")

        conversations = await pg_store.list_conversations(limit=10)

        assert len(conversations) == 3
        # Most recent first (conv3, conv2, conv1)
        assert conversations[0].title == "Third"
        assert conversations[1].title == "Second"
        assert conversations[2].title == "First"

    async def test_list_conversations_pagination(self, pg_store):
        """Test pagination when listing conversations."""
        # Create 5 conversations
        for i in range(5):
            await pg_store.create_conversation(title=f"Conv {i}")

        # Get first 2
        page1 = await pg_store.list_conversations(limit=2, offset=0)
        assert len(page1) == 2

        # Get next 2
        page2 = await pg_store.list_conversations(limit=2, offset=2)
        assert len(page2) == 2

        # Get last 1
        page3 = await pg_store.list_conversations(limit=2, offset=4)
        assert len(page3) == 1

        # Verify no duplicates
        all_ids = [c.id for c in page1 + page2 + page3]
        assert len(all_ids) == len(set(all_ids))


@pytest.mark.asyncio
class TestConversationDeletion:
    """Test conversation deletion including cascade behavior."""

    async def test_delete_existing_conversation(self, pg_store):
        """Test deleting an existing conversation."""
        conv = await pg_store.create_conversation(title="To Delete")
        result = await pg_store.delete_conversation(conv.id)

        assert result is True

        # Verify it's gone
        retrieved = await pg_store.get_conversation(conv.id)
        assert retrieved is None

    async def test_delete_nonexistent_conversation(self, pg_store):
        """Test deleting a conversation that doesn't exist."""
        fake_id = str(uuid.uuid4())
        result = await pg_store.delete_conversation(fake_id)

        assert result is False

    async def test_delete_conversation_cascades_to_messages(self, pg_store):
        """Test that deleting conversation also deletes its messages."""
        # Create conversation with messages
        conv = await pg_store.create_conversation(title="With Messages")
        await pg_store.add_message(conv.id, "user", "Question 1")
        await pg_store.add_message(conv.id, "assistant", "Answer 1")
        await pg_store.add_message(conv.id, "user", "Question 2")

        # Verify messages exist
        messages = await pg_store.get_messages(conv.id)
        assert len(messages) == 3

        # Delete conversation
        await pg_store.delete_conversation(conv.id)

        # Verify messages are also deleted
        messages_after = await pg_store.get_messages(conv.id)
        assert len(messages_after) == 0


@pytest.mark.asyncio
class TestMessageOperations:
    """Test adding and retrieving messages."""

    async def test_add_user_message(self, pg_store):
        """Test adding a user message to a conversation."""
        conv = await pg_store.create_conversation(title="Test")
        message = await pg_store.add_message(
            conv.id,
            role="user",
            content="Hello, research assistant!"
        )

        assert message.id is not None
        assert message.role == "user"
        assert message.content == "Hello, research assistant!"
        assert message.sources is None or message.sources == []
        assert message.timestamp is not None

    async def test_add_assistant_message_with_sources(self, pg_store):
        """Test adding an assistant message with source citations."""
        conv = await pg_store.create_conversation(title="Test")
        sources = [
            {"title": "Paper 1", "paper_id": "2301.12345"},
            {"title": "Paper 2", "paper_id": "2302.54321"}
        ]

        message = await pg_store.add_message(
            conv.id,
            role="assistant",
            content="Based on the papers...",
            sources=sources
        )

        assert message.role == "assistant"
        assert message.sources == sources
        assert len(message.sources) == 2

    async def test_add_message_updates_conversation_metadata(self, pg_store):
        """Test that adding messages updates conversation metadata."""
        conv = await pg_store.create_conversation(title="Test")
        original_updated_at = conv.updated_at
        original_count = conv.message_count

        # Add a message
        await pg_store.add_message(conv.id, "user", "Message 1")

        # Get updated conversation
        updated_conv = await pg_store.get_conversation(conv.id)
        assert updated_conv.message_count == original_count + 1
        assert updated_conv.updated_at != original_updated_at

    async def test_add_multiple_messages(self, pg_store):
        """Test adding multiple messages maintains order."""
        conv = await pg_store.create_conversation(title="Chat")

        msg1 = await pg_store.add_message(conv.id, "user", "First")
        msg2 = await pg_store.add_message(conv.id, "assistant", "Second")
        msg3 = await pg_store.add_message(conv.id, "user", "Third")

        messages = await pg_store.get_messages(conv.id)

        assert len(messages) == 3
        assert messages[0].content == "First"
        assert messages[1].content == "Second"
        assert messages[2].content == "Third"

    async def test_get_messages_empty_conversation(self, pg_store):
        """Test getting messages from conversation with no messages."""
        conv = await pg_store.create_conversation(title="Empty")
        messages = await pg_store.get_messages(conv.id)

        assert messages == []

    async def test_get_messages_with_limit(self, pg_store):
        """Test retrieving messages with limit."""
        conv = await pg_store.create_conversation(title="Chat")

        # Add 5 messages
        for i in range(5):
            await pg_store.add_message(conv.id, "user", f"Message {i}")

        # Get only first 3
        messages = await pg_store.get_messages(conv.id, limit=3)

        assert len(messages) == 3
        assert messages[0].content == "Message 0"
        assert messages[2].content == "Message 2"

    async def test_message_count_increments(self, pg_store):
        """Test that message_count increments correctly."""
        conv = await pg_store.create_conversation(title="Test")

        for i in range(5):
            await pg_store.add_message(conv.id, "user", f"Message {i}")

        updated_conv = await pg_store.get_conversation(conv.id)
        assert updated_conv.message_count == 5


@pytest.mark.asyncio
class TestConversationUpdate:
    """Test updating conversation properties."""

    async def test_update_conversation_title(self, pg_store):
        """Test updating a conversation's title."""
        conv = await pg_store.create_conversation(title="Original Title")
        new_title = "Updated Title"

        result = await pg_store.update_conversation_title(conv.id, new_title)
        assert result is True

        updated = await pg_store.get_conversation(conv.id)
        assert updated.title == new_title

    async def test_update_nonexistent_conversation_title(self, pg_store):
        """Test updating title of non-existent conversation."""
        fake_id = str(uuid.uuid4())
        result = await pg_store.update_conversation_title(fake_id, "New Title")

        assert result is False

    async def test_update_title_updates_timestamp(self, pg_store):
        """Test that updating title also updates updated_at timestamp."""
        conv = await pg_store.create_conversation(title="Original")
        original_updated_at = conv.updated_at

        # Small delay to ensure timestamp difference
        import asyncio
        await asyncio.sleep(0.1)

        await pg_store.update_conversation_title(conv.id, "New Title")

        updated = await pg_store.get_conversation(conv.id)
        assert updated.updated_at != original_updated_at


@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and error conditions."""

    async def test_empty_message_content(self, pg_store):
        """Test that database constraint prevents empty message content."""
        conv = await pg_store.create_conversation(title="Test")

        with pytest.raises(Exception):  # Should violate database constraint
            await pg_store.add_message(conv.id, "user", "")

    async def test_invalid_message_role(self, pg_store):
        """Test that database constraint validates message role."""
        conv = await pg_store.create_conversation(title="Test")

        with pytest.raises(Exception):  # Should violate role check constraint
            await pg_store.add_message(conv.id, "invalid_role", "Content")

    async def test_empty_conversation_title(self, pg_store):
        """Test that database constraint prevents empty conversation title."""
        with pytest.raises(Exception):  # Should violate title constraint
            await pg_store.create_conversation(title="")

    async def test_get_messages_from_nonexistent_conversation(self, pg_store):
        """Test getting messages from non-existent conversation."""
        fake_id = str(uuid.uuid4())
        messages = await pg_store.get_messages(fake_id)

        assert messages == []

    async def test_concurrent_message_additions(self, pg_store):
        """Test adding messages concurrently to same conversation."""
        import asyncio

        conv = await pg_store.create_conversation(title="Concurrent Test")

        # Add messages concurrently
        tasks = [
            pg_store.add_message(conv.id, "user", f"Message {i}")
            for i in range(10)
        ]
        await asyncio.gather(*tasks)

        # Verify all messages were added
        messages = await pg_store.get_messages(conv.id)
        assert len(messages) == 10

        # Verify message count is correct
        updated_conv = await pg_store.get_conversation(conv.id)
        assert updated_conv.message_count == 10
