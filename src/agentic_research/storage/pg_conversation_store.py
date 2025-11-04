"""
PostgreSQL-based conversation storage for chat history.

Drop-in replacement for Redis-based conversation storage,
using PostgreSQL for persistent, relational data storage.
"""

import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import select, update, delete, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from ..db.base import get_session_maker
from ..db.models import Conversation as ConversationModel, Message as MessageModel


# Reuse Pydantic models from Redis implementation for API compatibility
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


class PostgreSQLConversationStore:
    """
    PostgreSQL-based conversation storage.

    Implements the same interface as Redis ConversationStore for
    drop-in replacement with better data integrity and query flexibility.
    """

    def __init__(self):
        """Initialize with session maker."""
        self.session_maker = get_session_maker()

    async def create_conversation(
        self, title: Optional[str] = None, conversation_id: Optional[str] = None
    ) -> ConversationDict:
        """Create a new conversation with optional specific ID."""
        async with self.session_maker() as session:
            conv_id = uuid.UUID(conversation_id) if conversation_id else uuid.uuid4()
            now = datetime.utcnow()

            conversation = ConversationModel(
                id=conv_id,
                title=title or f"Chat {now.strftime('%Y-%m-%d')}",
                created_at=now,
                updated_at=now,
                message_count=0,
            )

            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)

            return ConversationDict(
                id=str(conversation.id),
                title=conversation.title,
                created_at=conversation.created_at.isoformat(),
                updated_at=conversation.updated_at.isoformat(),
                message_count=conversation.message_count,
            )

    async def get_conversation(self, conv_id: str) -> Optional[ConversationDict]:
        """Get conversation metadata."""
        async with self.session_maker() as session:
            result = await session.execute(
                select(ConversationModel).where(ConversationModel.id == uuid.UUID(conv_id))
            )
            conversation = result.scalar_one_or_none()

            if not conversation:
                return None

            return ConversationDict(
                id=str(conversation.id),
                title=conversation.title,
                created_at=conversation.created_at.isoformat(),
                updated_at=conversation.updated_at.isoformat(),
                message_count=conversation.message_count,
            )

    async def list_conversations(
        self, limit: int = 50, offset: int = 0
    ) -> List[ConversationDict]:
        """List conversations ordered by most recent."""
        async with self.session_maker() as session:
            result = await session.execute(
                select(ConversationModel)
                .order_by(desc(ConversationModel.updated_at))
                .limit(limit)
                .offset(offset)
            )
            conversations = result.scalars().all()

            return [
                ConversationDict(
                    id=str(conv.id),
                    title=conv.title,
                    created_at=conv.created_at.isoformat(),
                    updated_at=conv.updated_at.isoformat(),
                    message_count=conv.message_count,
                )
                for conv in conversations
            ]

    async def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and all its messages (cascade)."""
        async with self.session_maker() as session:
            result = await session.execute(
                delete(ConversationModel).where(ConversationModel.id == uuid.UUID(conv_id))
            )
            await session.commit()
            return result.rowcount > 0

    async def add_message(
        self, conv_id: str, role: str, content: str, sources: Optional[list] = None
    ) -> MessageDict:
        """Add a message to a conversation."""
        async with self.session_maker() as session:
            message = MessageModel(
                id=uuid.uuid4(),
                conversation_id=uuid.UUID(conv_id),
                role=role,
                content=content,
                sources=sources or [],
                timestamp=datetime.utcnow(),
            )

            session.add(message)

            # Update conversation metadata
            await session.execute(
                update(ConversationModel)
                .where(ConversationModel.id == uuid.UUID(conv_id))
                .values(
                    updated_at=datetime.utcnow(),
                    message_count=ConversationModel.message_count + 1,
                )
            )

            await session.commit()
            await session.refresh(message)

            return MessageDict(
                id=str(message.id),
                role=message.role,
                content=message.content,
                sources=message.sources,
                timestamp=message.timestamp.isoformat(),
            )

    async def get_messages(
        self, conv_id: str, limit: Optional[int] = None
    ) -> List[MessageDict]:
        """Get all messages in a conversation."""
        async with self.session_maker() as session:
            query = (
                select(MessageModel)
                .where(MessageModel.conversation_id == uuid.UUID(conv_id))
                .order_by(MessageModel.timestamp)
            )

            if limit:
                query = query.limit(limit)

            result = await session.execute(query)
            messages = result.scalars().all()

            return [
                MessageDict(
                    id=str(msg.id),
                    role=msg.role,
                    content=msg.content,
                    sources=msg.sources,
                    timestamp=msg.timestamp.isoformat(),
                )
                for msg in messages
            ]

    async def update_conversation_title(self, conv_id: str, title: str) -> bool:
        """Update conversation title."""
        async with self.session_maker() as session:
            result = await session.execute(
                update(ConversationModel)
                .where(ConversationModel.id == uuid.UUID(conv_id))
                .values(title=title, updated_at=datetime.utcnow())
            )
            await session.commit()
            return result.rowcount > 0


# Global instance
_pg_conversation_store: Optional[PostgreSQLConversationStore] = None


def get_pg_conversation_store() -> PostgreSQLConversationStore:
    """Get or create the PostgreSQL conversation store instance."""
    global _pg_conversation_store
    if _pg_conversation_store is None:
        _pg_conversation_store = PostgreSQLConversationStore()
    return _pg_conversation_store
