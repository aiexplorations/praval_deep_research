"""
PostgreSQL-based conversation storage for chat history.

Uses a thread-based model for branching:
- Each message belongs to a thread_id (0 = original conversation)
- When user edits a message, a new thread is created
- The new thread copies all prior messages and continues independently
- Each thread is a complete, linear conversation path
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import select, update, delete, func, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from ..db.base import get_session_maker
from ..db.models import Conversation as ConversationModel, Message as MessageModel


# Pydantic models for API responses
class MessageDict(BaseModel):
    """Message data structure with thread info."""
    id: str
    role: str  # 'user' or 'assistant'
    content: str
    sources: Optional[list] = None
    timestamp: str
    # Thread-based branching
    thread_id: int = 0
    position: int
    # Computed UI fields for branch navigation
    has_other_versions: bool = False  # True if other threads have a message at this position
    version_count: int = 1  # Total versions at this position
    current_version: int = 1  # Which version this is (1-indexed for display)


class ConversationDict(BaseModel):
    """Conversation metadata."""
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    active_thread_id: int = 0
    max_thread_id: int = 0


class ThreadInfo(BaseModel):
    """Information about threads at a message position."""
    position: int
    thread_count: int
    threads: List[Dict[str, Any]]


class PostgreSQLConversationStore:
    """PostgreSQL-backed conversation storage with thread-based branching."""

    def __init__(self):
        self.session_maker = get_session_maker()

    async def create_conversation(self, title: str) -> ConversationDict:
        """Create a new conversation."""
        async with self.session_maker() as session:
            conversation = ConversationModel(
                id=uuid.uuid4(),
                title=title,
                message_count=0,
                active_thread_id=0,
                max_thread_id=0,
            )
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)

            return ConversationDict(
                id=str(conversation.id),
                title=conversation.title,
                created_at=conversation.created_at.isoformat(),
                updated_at=conversation.updated_at.isoformat(),
                message_count=0,
                active_thread_id=0,
                max_thread_id=0,
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
                active_thread_id=conversation.active_thread_id,
                max_thread_id=conversation.max_thread_id,
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
                    active_thread_id=conv.active_thread_id,
                    max_thread_id=conv.max_thread_id,
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
        self,
        conv_id: str,
        role: str,
        content: str,
        sources: Optional[list] = None,
        thread_id: Optional[int] = None,
        position: Optional[int] = None
    ) -> MessageDict:
        """
        Add a message to a conversation thread.

        Args:
            conv_id: Conversation ID
            role: 'user' or 'assistant'
            content: Message content
            sources: Optional list of source citations
            thread_id: Thread to add to (None = active thread, 0 = original)
            position: Position in thread (auto-calculated if not provided)
        """
        async with self.session_maker() as session:
            # If thread_id not provided, use the conversation's active thread
            if thread_id is None:
                conv_result = await session.execute(
                    select(ConversationModel.active_thread_id)
                    .where(ConversationModel.id == uuid.UUID(conv_id))
                )
                thread_id = conv_result.scalar() or 0

            # If position not provided, calculate next position in thread
            if position is None:
                result = await session.execute(
                    select(func.max(MessageModel.position))
                    .where(
                        and_(
                            MessageModel.conversation_id == uuid.UUID(conv_id),
                            MessageModel.thread_id == thread_id
                        )
                    )
                )
                max_position = result.scalar() or 0
                position = max_position + 1

            message = MessageModel(
                id=uuid.uuid4(),
                conversation_id=uuid.UUID(conv_id),
                role=role,
                content=content,
                sources=sources or [],
                timestamp=datetime.utcnow(),
                thread_id=thread_id,
                position=position,
            )

            session.add(message)

            # Update conversation metadata
            await session.execute(
                update(ConversationModel)
                .where(ConversationModel.id == uuid.UUID(conv_id))
                .values(
                    message_count=ConversationModel.message_count + 1,
                    updated_at=datetime.utcnow()
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
                thread_id=message.thread_id,
                position=message.position,
                has_other_versions=False,
                version_count=1,
                current_version=1,
            )

    async def get_messages(
        self,
        conv_id: str,
        thread_id: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[MessageDict]:
        """
        Get messages for a conversation thread.

        If thread_id is None, uses the conversation's active_thread_id.
        Also computes version info for each message position.

        Args:
            conv_id: Conversation ID
            thread_id: Specific thread to load (None = active thread)
            limit: Optional limit on messages

        Returns:
            List of messages with version navigation info
        """
        async with self.session_maker() as session:
            # Get conversation to find active thread
            conv_result = await session.execute(
                select(ConversationModel).where(ConversationModel.id == uuid.UUID(conv_id))
            )
            conversation = conv_result.scalar_one_or_none()

            if not conversation:
                return []

            # Use provided thread_id or conversation's active thread
            active_thread = thread_id if thread_id is not None else conversation.active_thread_id

            # Get messages for this thread
            query = (
                select(MessageModel)
                .where(
                    and_(
                        MessageModel.conversation_id == uuid.UUID(conv_id),
                        MessageModel.thread_id == active_thread
                    )
                )
                .order_by(MessageModel.position)
            )

            if limit:
                query = query.limit(limit)

            result = await session.execute(query)
            thread_messages = result.scalars().all()

            # Get all messages at each position across all threads (for version counting)
            positions = [msg.position for msg in thread_messages]
            if positions:
                version_result = await session.execute(
                    select(MessageModel.position, MessageModel.thread_id)
                    .where(
                        and_(
                            MessageModel.conversation_id == uuid.UUID(conv_id),
                            MessageModel.position.in_(positions)
                        )
                    )
                )
                all_versions = version_result.all()

                # Count versions per position
                version_counts = {}
                threads_at_position = {}
                for pos, tid in all_versions:
                    if pos not in version_counts:
                        version_counts[pos] = 0
                        threads_at_position[pos] = []
                    version_counts[pos] += 1
                    threads_at_position[pos].append(tid)
            else:
                version_counts = {}
                threads_at_position = {}

            # Build result with version info
            result_messages = []
            for msg in thread_messages:
                threads = sorted(threads_at_position.get(msg.position, [msg.thread_id]))
                version_count = version_counts.get(msg.position, 1)
                current_version = threads.index(msg.thread_id) + 1 if msg.thread_id in threads else 1

                result_messages.append(MessageDict(
                    id=str(msg.id),
                    role=msg.role,
                    content=msg.content,
                    sources=msg.sources,
                    timestamp=msg.timestamp.isoformat(),
                    thread_id=msg.thread_id,
                    position=msg.position,
                    has_other_versions=version_count > 1,
                    version_count=version_count,
                    current_version=current_version,
                ))

            return result_messages

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

    # ========== Thread-based Branching Operations ==========

    async def edit_message(
        self,
        conv_id: str,
        message_id: str,
        new_content: str
    ) -> Dict[str, Any]:
        """
        Edit a message by creating a new thread.

        This creates a new thread that:
        1. Copies all messages from position 1 to position-1 from the source thread
        2. Adds the edited message at the original position
        3. Sets this new thread as active

        Args:
            conv_id: Conversation ID
            message_id: ID of the message to edit (must be a user message)
            new_content: The edited message content

        Returns:
            Dict with new_thread_id, new_message, and position info
        """
        async with self.session_maker() as session:
            # Get the original message
            result = await session.execute(
                select(MessageModel).where(MessageModel.id == uuid.UUID(message_id))
            )
            original_msg = result.scalar_one_or_none()

            if not original_msg:
                raise ValueError(f"Message {message_id} not found")

            if original_msg.role != "user":
                raise ValueError("Can only edit user messages")

            # Get conversation to get max_thread_id
            conv_result = await session.execute(
                select(ConversationModel).where(ConversationModel.id == uuid.UUID(conv_id))
            )
            conversation = conv_result.scalar_one_or_none()

            if not conversation:
                raise ValueError(f"Conversation {conv_id} not found")

            # Create new thread ID
            new_thread_id = conversation.max_thread_id + 1
            source_thread = original_msg.thread_id
            edit_position = original_msg.position

            # Copy all messages from source thread up to (but not including) edit position
            messages_result = await session.execute(
                select(MessageModel)
                .where(
                    and_(
                        MessageModel.conversation_id == uuid.UUID(conv_id),
                        MessageModel.thread_id == source_thread,
                        MessageModel.position < edit_position
                    )
                )
                .order_by(MessageModel.position)
            )
            messages_to_copy = messages_result.scalars().all()

            # Create copies in new thread
            for msg in messages_to_copy:
                new_msg = MessageModel(
                    id=uuid.uuid4(),
                    conversation_id=uuid.UUID(conv_id),
                    role=msg.role,
                    content=msg.content,
                    sources=msg.sources,
                    timestamp=msg.timestamp,  # Preserve original timestamp
                    thread_id=new_thread_id,
                    position=msg.position,
                )
                session.add(new_msg)

            # Add the edited message at the edit position
            edited_message = MessageModel(
                id=uuid.uuid4(),
                conversation_id=uuid.UUID(conv_id),
                role="user",
                content=new_content,
                sources=[],
                timestamp=datetime.utcnow(),
                thread_id=new_thread_id,
                position=edit_position,
            )
            session.add(edited_message)

            # Update conversation: set active thread and increment max_thread_id
            await session.execute(
                update(ConversationModel)
                .where(ConversationModel.id == uuid.UUID(conv_id))
                .values(
                    active_thread_id=new_thread_id,
                    max_thread_id=new_thread_id,
                    message_count=ConversationModel.message_count + len(messages_to_copy) + 1,
                    updated_at=datetime.utcnow(),
                )
            )

            await session.commit()
            await session.refresh(edited_message)

            # Count versions at this position (now includes the new one)
            version_result = await session.execute(
                select(func.count(MessageModel.id))
                .where(
                    and_(
                        MessageModel.conversation_id == uuid.UUID(conv_id),
                        MessageModel.position == edit_position
                    )
                )
            )
            version_count = version_result.scalar() or 1

            return {
                "new_thread_id": new_thread_id,
                "new_message": MessageDict(
                    id=str(edited_message.id),
                    role=edited_message.role,
                    content=edited_message.content,
                    sources=edited_message.sources,
                    timestamp=edited_message.timestamp.isoformat(),
                    thread_id=new_thread_id,
                    position=edit_position,
                    has_other_versions=True,
                    version_count=version_count,
                    current_version=version_count,  # New version is always latest
                ),
                "position": edit_position,
                "source_thread_id": source_thread,
                "copied_messages": len(messages_to_copy),
            }

    async def switch_thread(
        self,
        conv_id: str,
        thread_id: Optional[int] = None,
        position: Optional[int] = None,
        direction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Switch to a different thread.

        Can be called in two ways:
        1. With thread_id: Switch directly to that thread
        2. With position and direction: Navigate to prev/next thread at that position

        Args:
            conv_id: Conversation ID
            thread_id: Direct thread ID to switch to
            position: Message position for direction-based navigation
            direction: 'prev' or 'next' for navigating between threads at a position

        Returns:
            Dict with new active_thread_id and messages
        """
        async with self.session_maker() as session:
            new_thread_id = None

            if thread_id is not None:
                # Direct switch
                new_thread_id = thread_id

            elif position is not None and direction:
                # Navigate between threads at a position
                # Get current active thread
                conv_result = await session.execute(
                    select(ConversationModel).where(ConversationModel.id == uuid.UUID(conv_id))
                )
                conversation = conv_result.scalar_one_or_none()
                if not conversation:
                    raise ValueError(f"Conversation {conv_id} not found")

                current_thread = conversation.active_thread_id

                # Get all threads that have a message at this position
                threads_result = await session.execute(
                    select(MessageModel.thread_id)
                    .where(
                        and_(
                            MessageModel.conversation_id == uuid.UUID(conv_id),
                            MessageModel.position == position
                        )
                    )
                    .distinct()
                    .order_by(MessageModel.thread_id)
                )
                threads = [t[0] for t in threads_result.all()]

                if len(threads) <= 1:
                    # No other threads at this position
                    return {
                        "active_thread_id": current_thread,
                        "messages": await self.get_messages(conv_id, current_thread),
                        "status": "no_change"
                    }

                # Find current position in list
                try:
                    current_idx = threads.index(current_thread)
                except ValueError:
                    current_idx = 0

                # Calculate target
                if direction == "prev":
                    target_idx = max(0, current_idx - 1)
                else:  # next
                    target_idx = min(len(threads) - 1, current_idx + 1)

                new_thread_id = threads[target_idx]

            if new_thread_id is None:
                raise ValueError("Must provide either thread_id or (position + direction)")

            # Update conversation's active thread
            await session.execute(
                update(ConversationModel)
                .where(ConversationModel.id == uuid.UUID(conv_id))
                .values(
                    active_thread_id=new_thread_id,
                    updated_at=datetime.utcnow()
                )
            )
            await session.commit()

            # Return new thread's messages
            messages = await self.get_messages(conv_id, new_thread_id)

            return {
                "active_thread_id": new_thread_id,
                "messages": [m.model_dump() for m in messages],
                "status": "thread_switched"
            }

    async def get_threads_at_position(
        self,
        conv_id: str,
        position: int
    ) -> ThreadInfo:
        """
        Get information about all threads that have a message at a given position.

        Useful for showing "< 1/3 >" navigation at messages with multiple versions.

        Args:
            conv_id: Conversation ID
            position: Message position to query

        Returns:
            ThreadInfo with count and details of threads at this position
        """
        async with self.session_maker() as session:
            # Get all messages at this position
            result = await session.execute(
                select(MessageModel)
                .where(
                    and_(
                        MessageModel.conversation_id == uuid.UUID(conv_id),
                        MessageModel.position == position
                    )
                )
                .order_by(MessageModel.thread_id)
            )
            messages = result.scalars().all()

            threads = []
            for msg in messages:
                threads.append({
                    "thread_id": msg.thread_id,
                    "message_id": str(msg.id),
                    "content_preview": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                })

            return ThreadInfo(
                position=position,
                thread_count=len(threads),
                threads=threads,
            )

    async def delete_thread(self, conv_id: str, thread_id: int) -> Dict[str, Any]:
        """
        Delete a thread and all its messages.

        Cannot delete thread 0 (original conversation).
        If deleting the active thread, switches to thread 0.

        Args:
            conv_id: Conversation ID
            thread_id: Thread ID to delete

        Returns:
            Dict with status and new active thread
        """
        if thread_id == 0:
            raise ValueError("Cannot delete the original thread (thread 0)")

        async with self.session_maker() as session:
            # Count messages to delete
            count_result = await session.execute(
                select(func.count(MessageModel.id))
                .where(
                    and_(
                        MessageModel.conversation_id == uuid.UUID(conv_id),
                        MessageModel.thread_id == thread_id
                    )
                )
            )
            message_count = count_result.scalar() or 0

            # Delete messages
            await session.execute(
                delete(MessageModel)
                .where(
                    and_(
                        MessageModel.conversation_id == uuid.UUID(conv_id),
                        MessageModel.thread_id == thread_id
                    )
                )
            )

            # Check if this was the active thread
            conv_result = await session.execute(
                select(ConversationModel).where(ConversationModel.id == uuid.UUID(conv_id))
            )
            conversation = conv_result.scalar_one_or_none()

            new_active_thread = 0
            if conversation and conversation.active_thread_id == thread_id:
                # Switch to thread 0
                await session.execute(
                    update(ConversationModel)
                    .where(ConversationModel.id == uuid.UUID(conv_id))
                    .values(
                        active_thread_id=0,
                        message_count=ConversationModel.message_count - message_count,
                        updated_at=datetime.utcnow()
                    )
                )
            else:
                # Just update message count
                await session.execute(
                    update(ConversationModel)
                    .where(ConversationModel.id == uuid.UUID(conv_id))
                    .values(
                        message_count=ConversationModel.message_count - message_count,
                        updated_at=datetime.utcnow()
                    )
                )
                new_active_thread = conversation.active_thread_id if conversation else 0

            await session.commit()

            return {
                "deleted_thread_id": thread_id,
                "deleted_message_count": message_count,
                "new_active_thread_id": new_active_thread,
                "status": "thread_deleted"
            }


# Global singleton instance
_pg_conversation_store: Optional[PostgreSQLConversationStore] = None


def get_pg_conversation_store() -> PostgreSQLConversationStore:
    """Get the PostgreSQL conversation store singleton."""
    global _pg_conversation_store
    if _pg_conversation_store is None:
        _pg_conversation_store = PostgreSQLConversationStore()
    return _pg_conversation_store
