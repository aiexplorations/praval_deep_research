"""
SQLAlchemy models for chat history storage.

Defines database schema for conversations and messages.
"""

import uuid
from datetime import datetime
from typing import List

from sqlalchemy import Column, String, Integer, Text, ForeignKey, DateTime, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func

from .base import Base


class Conversation(Base):
    """
    Conversation model - represents a chat session.

    A conversation contains multiple messages and has metadata like
    title, creation time, and message count.
    """
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    message_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )

    # Relationship to messages (one-to-many)
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",  # Delete messages when conversation is deleted
        order_by="Message.timestamp"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("char_length(title) > 0", name="conversations_title_not_empty"),
    )

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title='{self.title[:30]}...', messages={self.message_count})>"


class Message(Base):
    """
    Message model - represents a single message in a conversation.

    Messages belong to a conversation and have a role (user/assistant),
    content, optional sources, and timestamp.
    """
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False
    )
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    sources: Mapped[dict] = mapped_column(
        JSONB,
        nullable=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationship to conversation (many-to-one)
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="messages_role_check"),
        CheckConstraint("char_length(content) > 0", name="messages_content_not_empty"),
    )

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role='{self.role}', content='{self.content[:30]}...')>"
