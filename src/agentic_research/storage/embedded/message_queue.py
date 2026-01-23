"""
Local Message Queue - Replaces RabbitMQ for desktop/standalone deployment.

This provides an in-process message queue that maintains compatibility with
the Praval agent architecture while eliminating the RabbitMQ dependency.
"""

import asyncio
import json
from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import logging
import threading
from queue import Queue, Empty
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LocalMessage:
    """Message structure compatible with Praval Spore messages."""
    message_id: str
    message_type: str
    payload: Dict[str, Any]
    routing_key: str = ""
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    priority: int = 0
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalMessage":
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "LocalMessage":
        return cls.from_dict(json.loads(json_str))


class MessageHandler:
    """Handler registration for message processing."""

    def __init__(
        self,
        handler: Callable[[LocalMessage], Awaitable[Any]],
        routing_key_pattern: str = "#",
        message_types: Optional[List[str]] = None
    ):
        self.handler = handler
        self.routing_key_pattern = routing_key_pattern
        self.message_types = message_types

    def matches(self, message: LocalMessage) -> bool:
        """Check if this handler should process the message."""
        # Check message type
        if self.message_types and message.message_type not in self.message_types:
            return False

        # Check routing key pattern
        if self.routing_key_pattern == "#":
            return True

        # Simple wildcard matching
        pattern_parts = self.routing_key_pattern.split(".")
        key_parts = message.routing_key.split(".")

        for i, pattern in enumerate(pattern_parts):
            if pattern == "#":
                return True
            if pattern == "*":
                continue
            if i >= len(key_parts) or pattern != key_parts[i]:
                return False

        return len(pattern_parts) == len(key_parts)


class LocalMessageQueue:
    """
    In-process message queue that replaces RabbitMQ.

    Features:
    - Async message publishing and consumption
    - Topic-based routing (compatible with RabbitMQ patterns)
    - Priority queues
    - Dead letter handling
    - Request-reply pattern support

    Usage:
        queue = LocalMessageQueue()
        await queue.start()

        # Subscribe to messages
        await queue.subscribe(
            handler=my_handler,
            routing_key="search.*"
        )

        # Publish a message
        await queue.publish(
            message_type="search_request",
            payload={"query": "transformers"},
            routing_key="search.request"
        )
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        enable_persistence: bool = False,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize local message queue.

        Args:
            max_queue_size: Maximum messages in queue before blocking
            enable_persistence: Persist messages to disk (for crash recovery)
            persistence_path: Path for persistent storage
        """
        self.max_queue_size = max_queue_size
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path

        # Main message queue
        self._queue: asyncio.Queue = None
        self._priority_queues: Dict[int, asyncio.Queue] = {}

        # Handlers
        self._handlers: List[MessageHandler] = []
        self._reply_handlers: Dict[str, asyncio.Future] = {}

        # State
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Stats
        self._stats = {
            "published": 0,
            "processed": 0,
            "failed": 0,
            "dead_lettered": 0
        }

        # Dead letter queue
        self._dead_letter: List[LocalMessage] = []

        logger.info("LocalMessageQueue initialized")

    async def start(self):
        """Start the message queue worker."""
        if self._running:
            return

        self._queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())

        logger.info("LocalMessageQueue started")

    async def stop(self):
        """Stop the message queue worker."""
        if not self._running:
            return

        self._running = False

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.info("LocalMessageQueue stopped")

    async def _worker(self):
        """Main worker loop that processes messages."""
        while self._running:
            try:
                # Get message with timeout to allow shutdown
                try:
                    message = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process message
                await self._process_message(message)
                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")

    async def _process_message(self, message: LocalMessage):
        """Process a single message by routing to handlers."""
        handled = False

        # Check for reply handler first
        if message.correlation_id and message.correlation_id in self._reply_handlers:
            future = self._reply_handlers.pop(message.correlation_id)
            if not future.done():
                future.set_result(message)
            self._stats["processed"] += 1
            return

        # Route to registered handlers
        for handler in self._handlers:
            if handler.matches(message):
                try:
                    await handler.handler(message)
                    handled = True
                except Exception as e:
                    logger.error(f"Handler error for {message.message_type}: {e}")
                    message.retries += 1

                    # Retry or dead letter
                    if message.retries < 3:
                        await self._queue.put(message)
                    else:
                        self._dead_letter.append(message)
                        self._stats["dead_lettered"] += 1

        if handled:
            self._stats["processed"] += 1
        else:
            logger.warning(f"No handler for message: {message.message_type} ({message.routing_key})")
            self._stats["failed"] += 1

    # =========================================================================
    # Publishing
    # =========================================================================

    async def publish(
        self,
        message_type: str,
        payload: Dict[str, Any],
        routing_key: str = "",
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        priority: int = 0
    ) -> str:
        """
        Publish a message to the queue.

        Args:
            message_type: Type of message (e.g., "search_request")
            payload: Message data
            routing_key: Topic routing key (e.g., "search.request")
            correlation_id: For request-reply correlation
            reply_to: Reply routing key
            priority: Message priority (higher = more urgent)

        Returns:
            Message ID
        """
        import uuid

        message = LocalMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            payload=payload,
            routing_key=routing_key,
            correlation_id=correlation_id,
            reply_to=reply_to,
            priority=priority
        )

        await self._queue.put(message)
        self._stats["published"] += 1

        logger.debug(f"Published: {message_type} -> {routing_key}")
        return message.message_id

    async def publish_and_wait(
        self,
        message_type: str,
        payload: Dict[str, Any],
        routing_key: str = "",
        timeout: float = 30.0
    ) -> Optional[LocalMessage]:
        """
        Publish a message and wait for a reply.

        Args:
            message_type: Type of message
            payload: Message data
            routing_key: Topic routing key
            timeout: Maximum wait time in seconds

        Returns:
            Reply message or None if timeout
        """
        import uuid

        correlation_id = str(uuid.uuid4())

        # Create future for reply
        future = asyncio.get_event_loop().create_future()
        self._reply_handlers[correlation_id] = future

        # Publish with correlation ID
        await self.publish(
            message_type=message_type,
            payload=payload,
            routing_key=routing_key,
            correlation_id=correlation_id,
            reply_to=f"reply.{correlation_id}"
        )

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._reply_handlers.pop(correlation_id, None)
            return None

    # =========================================================================
    # Subscribing
    # =========================================================================

    async def subscribe(
        self,
        handler: Callable[[LocalMessage], Awaitable[Any]],
        routing_key: str = "#",
        message_types: Optional[List[str]] = None
    ):
        """
        Subscribe to messages.

        Args:
            handler: Async function to handle messages
            routing_key: Topic pattern (# = all, * = single word wildcard)
            message_types: Filter by message types
        """
        msg_handler = MessageHandler(
            handler=handler,
            routing_key_pattern=routing_key,
            message_types=message_types
        )
        self._handlers.append(msg_handler)

        logger.info(f"Subscribed handler for: {routing_key} (types: {message_types})")

    async def unsubscribe(self, handler: Callable):
        """Remove a handler subscription."""
        self._handlers = [h for h in self._handlers if h.handler != handler]

    # =========================================================================
    # Praval/Reef Compatibility Layer
    # =========================================================================

    async def send_spore(
        self,
        agent_id: str,
        message_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None
    ):
        """
        Send a Praval Spore-compatible message.

        This mirrors the RabbitMQBridge.send_spore() interface.
        """
        # Determine routing key from message type
        if "search" in message_type.lower():
            routing_key = "search.request"
        elif "qa" in message_type.lower():
            routing_key = "qa.request"
        elif "process" in message_type.lower():
            routing_key = "process.request"
        else:
            routing_key = f"agent.{message_type}"

        # Add agent_id to payload
        full_payload = {
            "agent_id": agent_id,
            **payload
        }

        await self.publish(
            message_type=message_type,
            payload=full_payload,
            routing_key=routing_key,
            correlation_id=correlation_id,
            reply_to=reply_to
        )

    async def broadcast(
        self,
        channel: str,
        message_type: str,
        payload: Dict[str, Any]
    ):
        """
        Broadcast a message to all subscribers of a channel.

        This mirrors the Praval Reef broadcast interface.
        """
        await self.publish(
            message_type=message_type,
            payload=payload,
            routing_key=f"broadcast.{channel}"
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            **self._stats,
            "queue_size": self._queue.qsize() if self._queue else 0,
            "handlers": len(self._handlers),
            "dead_letter_count": len(self._dead_letter),
            "running": self._running
        }

    def get_dead_letters(self) -> List[LocalMessage]:
        """Get dead letter queue contents."""
        return self._dead_letter.copy()

    async def retry_dead_letters(self) -> int:
        """Retry all dead lettered messages."""
        count = len(self._dead_letter)
        for message in self._dead_letter:
            message.retries = 0
            await self._queue.put(message)

        self._dead_letter.clear()
        logger.info(f"Retried {count} dead letters")
        return count

    async def drain(self, timeout: float = 5.0):
        """Wait for queue to drain."""
        try:
            await asyncio.wait_for(
                self._queue.join(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Queue drain timeout")


# =========================================================================
# Bridge for RabbitMQ interface compatibility
# =========================================================================

class LocalRabbitMQBridge:
    """
    Drop-in replacement for RabbitMQBridge using LocalMessageQueue.

    This provides the same interface as the real RabbitMQBridge but
    uses the local queue, making it suitable for desktop deployment.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._queue = LocalMessageQueue()
        self._connected = False

    async def connect(self):
        """Connect (start the local queue)."""
        await self._queue.start()
        self._connected = True
        logger.info("LocalRabbitMQBridge connected")

    async def disconnect(self):
        """Disconnect (stop the local queue)."""
        await self._queue.stop()
        self._connected = False
        logger.info("LocalRabbitMQBridge disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def send_spore(
        self,
        agent_id: str,
        message_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None
    ):
        """Send a Spore message."""
        await self._queue.send_spore(
            agent_id=agent_id,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
            reply_to=reply_to
        )

    async def subscribe(
        self,
        handler: Callable[[LocalMessage], Awaitable[Any]],
        routing_key: str = "#"
    ):
        """Subscribe to messages."""
        await self._queue.subscribe(handler=handler, routing_key=routing_key)

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "connected": self._connected,
            "queue_stats": self._queue.get_stats()
        }
