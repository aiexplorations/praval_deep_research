"""
SQLAlchemy models for chat history storage.

Defines database schema for conversations and messages with support
for branched conversations (edit & resubmit like ChatGPT/Claude).
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

    A conversation contains multiple messages organized in a tree structure
    to support branching (edit & resubmit). The active_branch_id tracks
    which branch is currently being displayed.
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
    # Track the currently active branch (null = main branch)
    active_branch_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        default=None
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

    Messages form a tree structure to support branching:
    - parent_message_id: Points to the message this is a response/edit of
    - branch_id: Groups messages into branches (UUID, null = main branch)
    - branch_index: Position within sibling branches (for < > navigation)

    Example tree structure:
        User: "What is ML?"           (parent=null, branch=null)
        └─ Assistant: "ML is..."      (parent=msg1, branch=null)
           └─ User: "Tell me more"    (parent=msg2, branch=null)  [EDITED]
              ├─ [Branch 0] User: "Tell me more about neural networks"
              │   └─ Assistant: "Neural networks..."
              └─ [Branch 1] User: "Tell me more about decision trees"
                  └─ Assistant: "Decision trees..."
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

    # === Branching Support ===
    # Parent message (for tree structure)
    parent_message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
        default=None
    )
    # Branch identifier (UUID, messages with same branch_id are in same branch)
    branch_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        default=None
    )
    # Index among sibling branches (0 = original, 1 = first edit, etc.)
    branch_index: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )

    # Relationship to conversation (many-to-one)
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages"
    )

    # Self-referential relationship for parent/children
    parent: Mapped[Optional["Message"]] = relationship(
        "Message",
        remote_side="Message.id",
        foreign_keys=[parent_message_id],
        backref="children"
    )

    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="messages_role_check"),
        CheckConstraint("char_length(content) > 0", name="messages_content_not_empty"),
        Index("ix_messages_conversation_branch", "conversation_id", "branch_id"),
        Index("ix_messages_parent", "parent_message_id"),
    )

    def __repr__(self) -> str:
        branch_info = f", branch={self.branch_id}" if self.branch_id else ""
        return f"<Message(id={self.id}, role='{self.role}'{branch_info}, content='{self.content[:30]}...')>"
