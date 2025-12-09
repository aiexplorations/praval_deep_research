"""
PostgreSQL-based conversation storage for chat history.

Drop-in replacement for Redis-based conversation storage,
using PostgreSQL for persistent, relational data storage.

Supports branched conversations (edit & resubmit like ChatGPT/Claude).
"""

import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import select, update, delete, func, desc, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from ..db.base import get_session_maker
from ..db.models import Conversation as ConversationModel, Message as MessageModel


# Reuse Pydantic models from Redis implementation for API compatibility
class MessageDict(BaseModel):
    """Message data structure with branching support."""
    id: str
    role: str  # 'user' or 'assistant'
    content: str
    sources: Optional[list] = None
    timestamp: str
    # Branching fields
    parent_message_id: Optional[str] = None
    branch_id: Optional[str] = None
    branch_index: int = 0
    # Computed fields for UI
    has_branches: bool = False
    sibling_count: int = 1
    sibling_index: int = 0


class ConversationDict(BaseModel):
    """Conversation metadata with branching support."""
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    active_branch_id: Optional[str] = None


class BranchInfo(BaseModel):
    """Information about a branch point in the conversation."""
    message_id: str
    branch_count: int
    branches: List[dict]  # [{branch_id, branch_index, first_message_preview}]


class PostgreSQLConversationStore:
    """
    PostgreSQL-based conversation storage with branching support.

    Implements the same interface as Redis ConversationStore for
    drop-in replacement with better data integrity and query flexibility.

    Branching model:
    - Messages form a tree structure via parent_message_id
    - branch_id groups messages in the same branch
    - branch_index orders sibling branches (for < > navigation)
    - Conversations track active_branch_id for current view
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
                active_branch_id=None,
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
                active_branch_id=str(conversation.active_branch_id) if conversation.active_branch_id else None,
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
                    active_branch_id=str(conv.active_branch_id) if conv.active_branch_id else None,
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
        parent_message_id: Optional[str] = None,
        branch_id: Optional[str] = None,
        branch_index: int = 0
    ) -> MessageDict:
        """
        Add a message to a conversation.

        Args:
            conv_id: Conversation ID
            role: 'user' or 'assistant'
            content: Message content
            sources: Optional list of source citations
            parent_message_id: ID of parent message (for branching)
            branch_id: Branch identifier (for grouping branch messages)
            branch_index: Index among sibling branches
        """
        async with self.session_maker() as session:
            message = MessageModel(
                id=uuid.uuid4(),
                conversation_id=uuid.UUID(conv_id),
                role=role,
                content=content,
                sources=sources or [],
                timestamp=datetime.utcnow(),
                parent_message_id=uuid.UUID(parent_message_id) if parent_message_id else None,
                branch_id=uuid.UUID(branch_id) if branch_id else None,
                branch_index=branch_index,
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
                parent_message_id=str(message.parent_message_id) if message.parent_message_id else None,
                branch_id=str(message.branch_id) if message.branch_id else None,
                branch_index=message.branch_index,
            )

    async def get_messages(
        self, conv_id: str, limit: Optional[int] = None, branch_id: Optional[str] = None
    ) -> List[MessageDict]:
        """
        Get messages in a conversation, optionally filtered by branch.

        For branched conversations, this returns messages following the active branch path.
        """
        async with self.session_maker() as session:
            # Get conversation to check active branch
            conv_result = await session.execute(
                select(ConversationModel).where(ConversationModel.id == uuid.UUID(conv_id))
            )
            conversation = conv_result.scalar_one_or_none()

            # Use provided branch_id or conversation's active branch
            active_branch = uuid.UUID(branch_id) if branch_id else (
                conversation.active_branch_id if conversation else None
            )

            # Build query - get messages in the active branch path
            query = (
                select(MessageModel)
                .where(MessageModel.conversation_id == uuid.UUID(conv_id))
                .order_by(MessageModel.timestamp)
            )

            if limit:
                query = query.limit(limit)

            result = await session.execute(query)
            all_messages = result.scalars().all()

            # Group messages by their parent_message_id to find siblings
            # Siblings share the same parent and are alternatives at the same conversation point
            siblings_by_parent = {}
            for msg in all_messages:
                # Use parent_message_id as key, None for root messages
                key = msg.parent_message_id
                if key not in siblings_by_parent:
                    siblings_by_parent[key] = []
                siblings_by_parent[key].append(msg)

            # For root-level messages (parent=None), we need special handling:
            # Only the first user message and its edits should be considered siblings
            root_siblings = siblings_by_parent.get(None, [])
            user_root_msgs = [m for m in root_siblings if m.role == 'user']

            # Find the first user message (chronologically) and messages that are edits
            first_user_msg = None
            valid_first_siblings = []
            if user_root_msgs:
                # Find the first user message (with branch_id=NULL and branch_index=0)
                original_msgs = [m for m in user_root_msgs if m.branch_id is None and m.branch_index == 0]
                if original_msgs:
                    first_user_msg = min(original_msgs, key=lambda m: m.timestamp)
                    # Siblings are: original first message + any with branch_id set
                    valid_first_siblings = [m for m in user_root_msgs if (
                        m.id == first_user_msg.id or m.branch_id is not None
                    )]

            def get_sibling_info(msg):
                """Get sibling count and index for a message."""
                if msg.parent_message_id:
                    # Non-root message - use parent_message_id to find siblings
                    siblings = [
                        s for s in siblings_by_parent.get(msg.parent_message_id, [])
                        if s.role == msg.role
                    ]
                elif msg.role == 'user':
                    # Root-level user message - special handling
                    # Only the FIRST user message has potential siblings (edits)
                    # Other root-level user messages are different conversation turns
                    if first_user_msg and (msg.id == first_user_msg.id or msg.branch_id is not None):
                        # This is the first message or an edit of it
                        siblings = valid_first_siblings
                    else:
                        # This is a subsequent user message at root level - no siblings
                        siblings = [msg]
                else:
                    # Root-level assistant message - typically no siblings
                    siblings = [msg]

                siblings = sorted(siblings, key=lambda s: s.branch_index)
                sibling_count = len(siblings)
                sibling_index = next(
                    (i for i, s in enumerate(siblings) if s.id == msg.id),
                    0
                )
                has_branches = sibling_count > 1
                return has_branches, sibling_count, sibling_index

            # Filter to active branch path if we have branches
            if active_branch:
                # Get messages that are either:
                # 1. In the main branch (branch_id is None) before the branch point
                # 2. In the active branch
                filtered_messages = [
                    msg for msg in all_messages
                    if msg.branch_id is None or msg.branch_id == active_branch
                ]
            else:
                # No active branch - show main branch (branch_id is None or branch_index 0)
                filtered_messages = [
                    msg for msg in all_messages
                    if msg.branch_id is None or msg.branch_index == 0
                ]

            result_messages = []
            for msg in filtered_messages:
                has_branches, sibling_count, sibling_index = get_sibling_info(msg)
                result_messages.append(MessageDict(
                    id=str(msg.id),
                    role=msg.role,
                    content=msg.content,
                    sources=msg.sources,
                    timestamp=msg.timestamp.isoformat(),
                    parent_message_id=str(msg.parent_message_id) if msg.parent_message_id else None,
                    branch_id=str(msg.branch_id) if msg.branch_id else None,
                    branch_index=msg.branch_index,
                    has_branches=has_branches,
                    sibling_count=sibling_count,
                    sibling_index=sibling_index,
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

    # ========== Branching Operations ==========

    async def edit_message(
        self,
        conv_id: str,
        message_id: str,
        new_content: str
    ) -> MessageDict:
        """
        Edit a message by creating a new branch.

        This creates a new user message as a sibling branch to the original,
        preserving the conversation history. The new branch becomes active.

        Args:
            conv_id: Conversation ID
            message_id: ID of the message to edit (must be a user message)
            new_content: The edited message content

        Returns:
            The newly created message in the new branch
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

            # Find siblings (messages with the same parent)
            parent_id = original_msg.parent_message_id
            sibling_result = await session.execute(
                select(MessageModel).where(
                    and_(
                        MessageModel.conversation_id == uuid.UUID(conv_id),
                        MessageModel.parent_message_id == parent_id if parent_id else MessageModel.parent_message_id.is_(None),
                        MessageModel.role == "user"
                    )
                )
            )
            siblings = sibling_result.scalars().all()

            # Calculate new branch index
            max_branch_index = max((s.branch_index for s in siblings), default=-1)
            new_branch_index = max_branch_index + 1

            # Create new branch ID
            new_branch_id = uuid.uuid4()

            # Create the edited message as a new branch
            new_message = MessageModel(
                id=uuid.uuid4(),
                conversation_id=uuid.UUID(conv_id),
                role="user",
                content=new_content,
                sources=[],
                timestamp=datetime.utcnow(),
                parent_message_id=parent_id,
                branch_id=new_branch_id,
                branch_index=new_branch_index,
            )

            session.add(new_message)

            # Update conversation to use new branch and update timestamp
            await session.execute(
                update(ConversationModel)
                .where(ConversationModel.id == uuid.UUID(conv_id))
                .values(
                    active_branch_id=new_branch_id,
                    updated_at=datetime.utcnow(),
                    message_count=ConversationModel.message_count + 1,
                )
            )

            await session.commit()
            await session.refresh(new_message)

            return MessageDict(
                id=str(new_message.id),
                role=new_message.role,
                content=new_message.content,
                sources=new_message.sources,
                timestamp=new_message.timestamp.isoformat(),
                parent_message_id=str(new_message.parent_message_id) if new_message.parent_message_id else None,
                branch_id=str(new_message.branch_id),
                branch_index=new_message.branch_index,
                has_branches=False,
                sibling_count=new_branch_index + 1,
                sibling_index=new_branch_index,
            )

    async def switch_branch(
        self,
        conv_id: str,
        branch_id: Optional[str] = None,
        message_id: Optional[str] = None,
        direction: Optional[str] = None
    ) -> Optional[str]:
        """
        Switch to a different branch in the conversation.

        Can be called in two ways:
        1. With branch_id: Switch directly to that branch
        2. With message_id and direction: Navigate left/right among sibling branches

        Args:
            conv_id: Conversation ID
            branch_id: Direct branch ID to switch to
            message_id: Message ID to navigate from (for direction-based switching)
            direction: 'left' or 'right' for sibling navigation

        Returns:
            The new active branch_id, or None if switching to main branch
        """
        async with self.session_maker() as session:
            new_branch_id = None

            if branch_id:
                # Direct branch switch
                new_branch_id = uuid.UUID(branch_id)

            elif message_id and direction:
                # Navigate among siblings
                result = await session.execute(
                    select(MessageModel).where(MessageModel.id == uuid.UUID(message_id))
                )
                current_msg = result.scalar_one_or_none()

                if not current_msg:
                    raise ValueError(f"Message {message_id} not found")

                # Find siblings
                parent_id = current_msg.parent_message_id
                sibling_result = await session.execute(
                    select(MessageModel)
                    .where(
                        and_(
                            MessageModel.conversation_id == uuid.UUID(conv_id),
                            MessageModel.parent_message_id == parent_id if parent_id else MessageModel.parent_message_id.is_(None),
                            MessageModel.role == current_msg.role
                        )
                    )
                    .order_by(MessageModel.branch_index)
                )
                siblings = sibling_result.scalars().all()

                if len(siblings) <= 1:
                    return None  # No siblings to navigate to

                # Find current position
                current_index = next(
                    (i for i, s in enumerate(siblings) if s.id == current_msg.id),
                    0
                )

                # Calculate target index
                if direction == "left":
                    target_index = max(0, current_index - 1)
                else:  # right
                    target_index = min(len(siblings) - 1, current_index + 1)

                if target_index == current_index:
                    return str(current_msg.branch_id) if current_msg.branch_id else None

                target_msg = siblings[target_index]
                new_branch_id = target_msg.branch_id

            # Update conversation's active branch
            await session.execute(
                update(ConversationModel)
                .where(ConversationModel.id == uuid.UUID(conv_id))
                .values(
                    active_branch_id=new_branch_id,
                    updated_at=datetime.utcnow()
                )
            )
            await session.commit()

            return str(new_branch_id) if new_branch_id else None

    async def get_branches_at_message(
        self,
        conv_id: str,
        message_id: str
    ) -> BranchInfo:
        """
        Get information about all branches that stem from a message's parent.

        This is used to show branch navigation UI (< 1/3 >) for a message.

        For messages at the root level (parent_message_id=NULL), we only consider
        messages as siblings if they are either:
        1. The original first message (branch_id=NULL, branch_index=0)
        2. Explicit edits of the first message (branch_id is set)

        Args:
            conv_id: Conversation ID
            message_id: Message ID to get branch info for

        Returns:
            BranchInfo with count and details of all sibling branches
        """
        async with self.session_maker() as session:
            # Get the message
            result = await session.execute(
                select(MessageModel).where(MessageModel.id == uuid.UUID(message_id))
            )
            msg = result.scalar_one_or_none()

            if not msg:
                raise ValueError(f"Message {message_id} not found")

            # Find all siblings (messages with same parent and role)
            parent_id = msg.parent_message_id

            if parent_id:
                # Message has a parent - find siblings with same parent
                sibling_result = await session.execute(
                    select(MessageModel)
                    .where(
                        and_(
                            MessageModel.conversation_id == uuid.UUID(conv_id),
                            MessageModel.parent_message_id == parent_id,
                            MessageModel.role == msg.role
                        )
                    )
                    .order_by(MessageModel.branch_index)
                )
            else:
                # Root-level message (parent=NULL)
                # Only consider as siblings:
                # 1. Messages at branch_index=0 AND branch_id=NULL (original messages)
                # 2. Messages with the same branch_id as the queried message (if it has one)
                # 3. The first user message (chronologically) and its edits
                #
                # For now, we'll find all root-level user messages and filter:
                # - The original (min timestamp, branch_id=NULL)
                # - Any with branch_id set (explicit edits)
                sibling_result = await session.execute(
                    select(MessageModel)
                    .where(
                        and_(
                            MessageModel.conversation_id == uuid.UUID(conv_id),
                            MessageModel.parent_message_id.is_(None),
                            MessageModel.role == msg.role,
                            # Only include messages that are either:
                            # - branch_index == 0 (could be original)
                            # - have a branch_id (explicit edit)
                            or_(
                                and_(MessageModel.branch_index == 0, MessageModel.branch_id.is_(None)),
                                MessageModel.branch_id.isnot(None)
                            )
                        )
                    )
                    .order_by(MessageModel.timestamp)
                )

            raw_siblings = sibling_result.scalars().all()

            # For root-level messages, we need to identify the "first" message
            # and only include it along with its edits
            if not parent_id and msg.role == "user":
                # Find the first user message (chronologically with branch_index=0, branch_id=NULL)
                first_msg = next(
                    (s for s in raw_siblings if s.branch_id is None and s.branch_index == 0),
                    None
                )
                if first_msg:
                    # Filter to only include:
                    # - The first message
                    # - Messages that are edits (have branch_id set) created near the first message
                    # For simplicity, just include the first message and any with branch_id
                    first_timestamp = first_msg.timestamp
                    siblings = [s for s in raw_siblings if (
                        s.id == first_msg.id or  # The original first message
                        s.branch_id is not None  # Any explicit edit
                    )]
                else:
                    siblings = raw_siblings
            else:
                siblings = raw_siblings

            # Sort by branch_index for consistent ordering
            siblings = sorted(siblings, key=lambda s: s.branch_index)

            branches = []
            for sibling in siblings:
                branches.append({
                    "branch_id": str(sibling.branch_id) if sibling.branch_id else None,
                    "branch_index": sibling.branch_index,
                    "message_id": str(sibling.id),
                    "first_message_preview": sibling.content[:100] + "..." if len(sibling.content) > 100 else sibling.content,
                    "timestamp": sibling.timestamp.isoformat()
                })

            return BranchInfo(
                message_id=message_id,
                branch_count=len(branches),
                branches=branches
            )

    async def get_message(self, message_id: str) -> Optional[MessageDict]:
        """Get a single message by ID."""
        async with self.session_maker() as session:
            result = await session.execute(
                select(MessageModel).where(MessageModel.id == uuid.UUID(message_id))
            )
            msg = result.scalar_one_or_none()

            if not msg:
                return None

            return MessageDict(
                id=str(msg.id),
                role=msg.role,
                content=msg.content,
                sources=msg.sources,
                timestamp=msg.timestamp.isoformat(),
                parent_message_id=str(msg.parent_message_id) if msg.parent_message_id else None,
                branch_id=str(msg.branch_id) if msg.branch_id else None,
                branch_index=msg.branch_index,
            )

    async def delete_branch(self, conv_id: str, branch_id: str) -> bool:
        """
        Delete all messages in a specific branch.

        This removes the branch and all its messages. If this was the active
        branch, switches to the main branch.

        Args:
            conv_id: Conversation ID
            branch_id: Branch ID to delete

        Returns:
            True if branch was deleted, False if not found
        """
        async with self.session_maker() as session:
            # Count messages in branch
            count_result = await session.execute(
                select(func.count(MessageModel.id))
                .where(
                    and_(
                        MessageModel.conversation_id == uuid.UUID(conv_id),
                        MessageModel.branch_id == uuid.UUID(branch_id)
                    )
                )
            )
            message_count = count_result.scalar()

            if message_count == 0:
                return False

            # Delete messages in the branch
            await session.execute(
                delete(MessageModel)
                .where(
                    and_(
                        MessageModel.conversation_id == uuid.UUID(conv_id),
                        MessageModel.branch_id == uuid.UUID(branch_id)
                    )
                )
            )

            # Get conversation to check if this was the active branch
            conv_result = await session.execute(
                select(ConversationModel).where(ConversationModel.id == uuid.UUID(conv_id))
            )
            conversation = conv_result.scalar_one_or_none()

            if conversation and str(conversation.active_branch_id) == branch_id:
                # Switch back to main branch
                await session.execute(
                    update(ConversationModel)
                    .where(ConversationModel.id == uuid.UUID(conv_id))
                    .values(
                        active_branch_id=None,
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

            await session.commit()
            return True


# Global instance
_pg_conversation_store: Optional[PostgreSQLConversationStore] = None


def get_pg_conversation_store() -> PostgreSQLConversationStore:
    """Get or create the PostgreSQL conversation store instance."""
    global _pg_conversation_store
    if _pg_conversation_store is None:
        _pg_conversation_store = PostgreSQLConversationStore()
    return _pg_conversation_store
