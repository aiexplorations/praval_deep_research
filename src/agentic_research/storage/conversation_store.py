"""
Redis-based conversation storage for chat history.

Stores conversations and messages with persistent storage across restarts.
"""

import json
import uuid
from datetime import datetime
from typing import Optional

import redis.asyncio as aioredis
from pydantic import BaseModel

from ..core.config import get_settings


class MessageDict(BaseModel):
    """Message data structure."""
    id: str
    role: str  # 'user' or 'assistant'
    content: str
    sources: Optional[list] = None
    timestamp: str


class ConversationDict(BaseModel):
    """Conversation metadata."""
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class ConversationStore:
    """Redis-based conversation storage."""

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client

    def _conv_key(self, conv_id: str) -> str:
        """Get Redis key for conversation metadata."""
        return f"conversation:{conv_id}:metadata"

    def _messages_key(self, conv_id: str) -> str:
        """Get Redis key for conversation messages."""
        return f"conversation:{conv_id}:messages"

    def _conv_list_key(self) -> str:
        """Get Redis key for conversations list."""
        return "conversations:list"

    async def create_conversation(
        self, title: Optional[str] = None, conversation_id: Optional[str] = None
    ) -> ConversationDict:
        """Create a new conversation with optional specific ID."""
        conv_id = conversation_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        conversation = ConversationDict(
            id=conv_id,
            title=title or f"Chat {now[:10]}",
            created_at=now,
            updated_at=now,
            message_count=0,
        )

        # Store metadata
        await self.redis.set(
            self._conv_key(conv_id), conversation.model_dump_json()
        )

        # Add to conversations list (sorted set with timestamp)
        await self.redis.zadd(
            self._conv_list_key(), {conv_id: datetime.utcnow().timestamp()}
        )

        return conversation

    async def get_conversation(self, conv_id: str) -> Optional[ConversationDict]:
        """Get conversation metadata."""
        data = await self.redis.get(self._conv_key(conv_id))
        if not data:
            return None
        return ConversationDict.model_validate_json(data)

    async def list_conversations(
        self, limit: int = 50, offset: int = 0
    ) -> list[ConversationDict]:
        """List conversations ordered by most recent."""
        # Get conversation IDs (most recent first)
        conv_ids = await self.redis.zrevrange(
            self._conv_list_key(), offset, offset + limit - 1
        )

        conversations = []
        for conv_id in conv_ids:
            # conv_id is already a string because decode_responses=True in redis client
            conv_id_str = conv_id if isinstance(conv_id, str) else conv_id.decode()
            conv = await self.get_conversation(conv_id_str)
            if conv:
                conversations.append(conv)

        return conversations

    async def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and all its messages."""
        # Delete metadata
        await self.redis.delete(self._conv_key(conv_id))

        # Delete messages
        await self.redis.delete(self._messages_key(conv_id))

        # Remove from list
        await self.redis.zrem(self._conv_list_key(), conv_id)

        return True

    async def add_message(
        self, conv_id: str, role: str, content: str, sources: Optional[list] = None
    ) -> MessageDict:
        """Add a message to a conversation."""
        message = MessageDict(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            sources=sources,
            timestamp=datetime.utcnow().isoformat(),
        )

        # Append message to list
        await self.redis.rpush(
            self._messages_key(conv_id), message.model_dump_json()
        )

        # Update conversation metadata
        conv = await self.get_conversation(conv_id)
        if conv:
            conv.updated_at = datetime.utcnow().isoformat()
            conv.message_count += 1
            await self.redis.set(self._conv_key(conv_id), conv.model_dump_json())

            # Update timestamp in sorted set
            await self.redis.zadd(
                self._conv_list_key(), {conv_id: datetime.utcnow().timestamp()}
            )

        return message

    async def get_messages(
        self, conv_id: str, limit: Optional[int] = None
    ) -> list[MessageDict]:
        """Get all messages in a conversation."""
        end = limit - 1 if limit else -1
        messages_data = await self.redis.lrange(self._messages_key(conv_id), 0, end)

        messages = []
        for msg_data in messages_data:
            messages.append(MessageDict.model_validate_json(msg_data))

        return messages

    async def update_conversation_title(self, conv_id: str, title: str) -> bool:
        """Update conversation title."""
        conv = await self.get_conversation(conv_id)
        if not conv:
            return False

        conv.title = title
        conv.updated_at = datetime.utcnow().isoformat()
        await self.redis.set(self._conv_key(conv_id), conv.model_dump_json())

        return True


# Global instance
_conversation_store: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """Get or create the conversation store instance."""
    global _conversation_store
    if _conversation_store is None:
        settings = get_settings()
        redis_client = aioredis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
        _conversation_store = ConversationStore(redis_client)
    return _conversation_store
