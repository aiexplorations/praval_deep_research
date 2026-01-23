"""
Embedded Cache Store - Replaces Redis with local caching.

This provides the same interface as the Redis-based ConversationStore
but uses local file-based caching via diskcache or a simple dict fallback.
"""

import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class CachedItem:
    """Cached item with metadata."""
    value: Any
    created_at: str
    expires_at: Optional[str] = None
    tags: List[str] = None


class EmbeddedCacheStore:
    """
    Local cache store that mirrors Redis interface.

    Supports two backends:
    1. diskcache (if installed) - Persistent, SQLite-based
    2. dict fallback - In-memory, lost on restart

    Features:
    - TTL support
    - Async interface (for compatibility)
    - Tag-based invalidation
    - Conversation storage helpers
    """

    def __init__(
        self,
        base_path: Optional[str] = None,
        use_disk: bool = True,
        default_ttl: Optional[int] = None
    ):
        """
        Initialize embedded cache.

        Args:
            base_path: Directory for disk cache. Defaults to ./data/cache
            use_disk: Use disk-based cache (requires diskcache package)
            default_ttl: Default TTL in seconds (None = no expiry)
        """
        if base_path is None:
            base_path = Path("./data/cache")
        else:
            base_path = Path(base_path)

        self.base_path = base_path
        self.default_ttl = default_ttl
        self._lock = threading.Lock()

        # Initialize backend
        if use_disk:
            try:
                import diskcache
                self.base_path.mkdir(parents=True, exist_ok=True)
                self._cache = diskcache.Cache(str(self.base_path))
                self._backend = "diskcache"
                logger.info(f"EmbeddedCacheStore using diskcache at {self.base_path}")
            except ImportError:
                logger.warning("diskcache not installed, using in-memory cache")
                self._cache = {}
                self._backend = "dict"
        else:
            self._cache = {}
            self._backend = "dict"

        # Expiry tracking for dict backend
        self._expiry: Dict[str, datetime] = {}

    # =========================================================================
    # Basic Key-Value Operations (Redis-like interface)
    # =========================================================================

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        return self._get_sync(key)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ):
        """Set value with optional TTL."""
        self._set_sync(key, value, ttl, tags)

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        return self._delete_sync(key)

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self._exists_sync(key)

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        return self._keys_sync(pattern)

    # =========================================================================
    # Sync implementations
    # =========================================================================

    def _get_sync(self, key: str) -> Optional[Any]:
        """Synchronous get."""
        with self._lock:
            # Check expiry for dict backend
            if self._backend == "dict":
                if key in self._expiry:
                    if datetime.utcnow() > self._expiry[key]:
                        del self._cache[key]
                        del self._expiry[key]
                        return None
                return self._cache.get(key)
            else:
                # diskcache handles expiry
                try:
                    return self._cache.get(key)
                except KeyError:
                    return None

    def _set_sync(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ):
        """Synchronous set."""
        ttl = ttl or self.default_ttl

        with self._lock:
            if self._backend == "dict":
                self._cache[key] = value
                if ttl:
                    self._expiry[key] = datetime.utcnow() + timedelta(seconds=ttl)
            else:
                # diskcache
                if ttl:
                    self._cache.set(key, value, expire=ttl)
                else:
                    self._cache.set(key, value)

            # Store tags for invalidation
            if tags:
                for tag in tags:
                    tag_key = f"_tag:{tag}"
                    existing = self._get_sync(tag_key) or []
                    if key not in existing:
                        existing.append(key)
                        self._set_sync(tag_key, existing)

    def _delete_sync(self, key: str) -> bool:
        """Synchronous delete."""
        with self._lock:
            if self._backend == "dict":
                if key in self._cache:
                    del self._cache[key]
                    if key in self._expiry:
                        del self._expiry[key]
                    return True
                return False
            else:
                return self._cache.delete(key)

    def _exists_sync(self, key: str) -> bool:
        """Synchronous exists check."""
        return self._get_sync(key) is not None

    def _keys_sync(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        import fnmatch

        with self._lock:
            if self._backend == "dict":
                all_keys = list(self._cache.keys())
            else:
                all_keys = list(self._cache.iterkeys())

            if pattern == "*":
                return all_keys

            return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]

    # =========================================================================
    # Conversation Storage (mirrors ConversationStore interface)
    # =========================================================================

    async def create_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new conversation."""
        conversation = {
            "id": conversation_id,
            "title": title or f"Conversation {conversation_id[:8]}",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "message_count": 0,
            "context_data": context_data or {},
            "messages": []
        }

        await self.set(f"conversation:{conversation_id}", conversation)

        # Add to conversation list
        conv_list = await self.get("conversations:list") or []
        conv_list.insert(0, {
            "id": conversation_id,
            "title": conversation["title"],
            "created_at": conversation["created_at"],
            "message_count": 0
        })
        await self.set("conversations:list", conv_list)

        logger.debug(f"Created conversation: {conversation_id}")
        return conversation

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID."""
        return await self.get(f"conversation:{conversation_id}")

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add a message to a conversation."""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            conversation = await self.create_conversation(conversation_id)

        message = {
            "id": f"{conversation_id}_{len(conversation['messages'])}",
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "sources": sources,
            "metadata": metadata
        }

        conversation["messages"].append(message)
        conversation["message_count"] = len(conversation["messages"])
        conversation["updated_at"] = datetime.utcnow().isoformat()

        await self.set(f"conversation:{conversation_id}", conversation)

        # Update list
        await self._update_conversation_list(conversation_id, conversation)

        logger.debug(f"Added message to {conversation_id}: {role}")
        return message

    async def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get messages from a conversation."""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return []

        messages = conversation.get("messages", [])

        if offset:
            messages = messages[offset:]
        if limit:
            messages = messages[:limit]

        return messages

    async def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List all conversations."""
        conv_list = await self.get("conversations:list") or []

        if offset:
            conv_list = conv_list[offset:]
        if limit:
            conv_list = conv_list[:limit]

        return conv_list

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        await self.delete(f"conversation:{conversation_id}")

        # Remove from list
        conv_list = await self.get("conversations:list") or []
        conv_list = [c for c in conv_list if c["id"] != conversation_id]
        await self.set("conversations:list", conv_list)

        logger.debug(f"Deleted conversation: {conversation_id}")
        return True

    async def _update_conversation_list(
        self,
        conversation_id: str,
        conversation: Dict[str, Any]
    ):
        """Update conversation in the list."""
        conv_list = await self.get("conversations:list") or []

        for i, conv in enumerate(conv_list):
            if conv["id"] == conversation_id:
                conv_list[i] = {
                    "id": conversation_id,
                    "title": conversation["title"],
                    "created_at": conversation["created_at"],
                    "updated_at": conversation["updated_at"],
                    "message_count": conversation["message_count"]
                }
                break

        # Move to top (most recent)
        conv_list = sorted(
            conv_list,
            key=lambda x: x.get("updated_at", x.get("created_at", "")),
            reverse=True
        )

        await self.set("conversations:list", conv_list)

    # =========================================================================
    # Tag-based Operations
    # =========================================================================

    async def invalidate_by_tag(self, tag: str) -> int:
        """Delete all keys with a given tag."""
        tag_key = f"_tag:{tag}"
        keys = await self.get(tag_key) or []

        count = 0
        for key in keys:
            if await self.delete(key):
                count += 1

        await self.delete(tag_key)
        return count

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def clear(self):
        """Clear all cached data."""
        with self._lock:
            if self._backend == "dict":
                self._cache.clear()
                self._expiry.clear()
            else:
                self._cache.clear()

        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            if self._backend == "dict":
                return {
                    "backend": "dict",
                    "keys": len(self._cache),
                    "keys_with_ttl": len(self._expiry)
                }
            else:
                return {
                    "backend": "diskcache",
                    "keys": len(self._cache),
                    "size_bytes": self._cache.volume(),
                    "path": str(self.base_path)
                }

    def close(self):
        """Close the cache."""
        if self._backend == "diskcache":
            self._cache.close()
        logger.info("EmbeddedCacheStore closed")
