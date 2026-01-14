"""
SQLAlchemy models for chat history storage.

Defines database schema for conversations and messages with support
for branched conversations using a thread-based model.

Thread-based branching:
- Each message belongs to a thread_id (integer, starting at 0)
- Thread 0 is the default/original conversation flow
- When user edits a message, a new thread is created
- The new thread copies all messages up to the edit point, then continues independently
- Each thread is a complete, linear conversation path
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, String, Integer, Text, ForeignKey, DateTime, CheckConstraint, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func

from .base import Base


class Conversation(Base):
    """
    Conversation model - represents a chat session.

    A conversation contains multiple threads, where each thread is a complete
    linear conversation path. The active_thread_id tracks which thread is
    currently being displayed.
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
    # Currently active thread (0 = original conversation)
    active_thread_id: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    # Highest thread number created (for generating next thread_id)
    max_thread_id: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    # Context data (e.g., paper_ids for KB search chats)
    # Note: Cannot use 'metadata' as it's reserved by SQLAlchemy
    context_data: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        default=None
    )

    # Relationship to messages (one-to-many)
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.position"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("char_length(title) > 0", name="conversations_title_not_empty"),
        CheckConstraint("active_thread_id >= 0", name="conversations_active_thread_non_negative"),
        CheckConstraint("max_thread_id >= 0", name="conversations_max_thread_non_negative"),
    )

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title='{self.title[:30]}...', threads={self.max_thread_id + 1})>"


class Message(Base):
    """
    Message model - represents a single message in a conversation thread.

    Thread-based branching model:
    - thread_id: Which thread this message belongs to (0 = original)
    - position: Position within the thread (1-indexed for clarity)

    Example:
        Thread 0 (original):
            pos=1: User: "What is ML?"
            pos=2: Assistant: "ML is..."
            pos=3: User: "Tell me more"     <- User edits this
            pos=4: Assistant: "Here's more..."

        Thread 1 (created from edit at pos=3):
            pos=1: User: "What is ML?"      (copied from thread 0)
            pos=2: Assistant: "ML is..."    (copied from thread 0)
            pos=3: User: "Explain neural networks"  (edited version)
            pos=4: Assistant: "Neural networks..."  (new response)

    To find versions of a message at position N:
        SELECT * FROM messages WHERE conversation_id=? AND position=N
        GROUP BY thread_id
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

    # === Thread-based Branching ===
    # Thread identifier (0 = original conversation, 1+ = branches)
    thread_id: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    # Position within the thread (1-indexed)
    position: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    # Relationship to conversation (many-to-one)
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages"
    )

    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="messages_role_check"),
        CheckConstraint("char_length(content) > 0", name="messages_content_not_empty"),
        CheckConstraint("thread_id >= 0", name="messages_thread_non_negative"),
        CheckConstraint("position > 0", name="messages_position_positive"),
        # Index for loading a thread efficiently
        Index("ix_messages_conversation_thread", "conversation_id", "thread_id", "position"),
        # Index for finding all versions at a position
        Index("ix_messages_conversation_position", "conversation_id", "position"),
    )

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, thread={self.thread_id}, pos={self.position}, role='{self.role}')>"
