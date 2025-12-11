"""
Simple messaging using Praval's reef.broadcast() on shared "broadcast" channel.

The API container uses the same Reef + RabbitMQ backend as agents.
All agents subscribe to the "broadcast" channel and filter by type
using their responds_to configuration.

This matches the pattern used by start_agents() for local mode.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from praval.core.reef import Reef, get_reef
from praval.core.reef_backend import RabbitMQBackend

from .config import get_settings

logger = logging.getLogger(__name__)

# Must match the channel used in run_agents.py
BROADCAST_CHANNEL = "broadcast"


class MessagePublisher:
    """
    Simple message publisher using Praval's reef.broadcast().

    Shares the same exchange and channel as the agents container.
    """

    def __init__(self):
        self.settings = get_settings()
        self.reef: Optional[Reef] = None
        self._initialized = False

    async def initialize(self):
        """Initialize using the global Reef (InMemory backend, shared with agents)."""
        if self._initialized:
            return

        try:
            # Use the GLOBAL reef - same one used by start_agents() with InMemory backend
            self.reef = get_reef()

            # Create the broadcast channel if not exists
            self.reef.create_channel(BROADCAST_CHANNEL)

            self._initialized = True
            logger.info(f"MessagePublisher initialized with InMemory backend (channel: {BROADCAST_CHANNEL})")

        except Exception as e:
            logger.error(f"Failed to initialize MessagePublisher: {e}")
            raise

    def broadcast_sync(self, knowledge: Dict[str, Any]) -> str:
        """
        Broadcast a message synchronously.

        Uses reef.broadcast() on the shared broadcast channel.
        Agents filter by knowledge["type"] using responds_to.
        """
        if not self._initialized or not self.reef:
            # Try to initialize
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule but can't wait - use global reef as fallback
                    asyncio.ensure_future(self.initialize())
                    reef = get_reef()
                    reef.create_channel(BROADCAST_CHANNEL)
                    return reef.broadcast(
                        from_agent="api",
                        knowledge=knowledge,
                        channel=BROADCAST_CHANNEL
                    )
                else:
                    loop.run_until_complete(self.initialize())
            except RuntimeError:
                asyncio.run(self.initialize())

        return self.reef.broadcast(
            from_agent="api",
            knowledge=knowledge,
            channel=BROADCAST_CHANNEL
        )

    async def broadcast_async(self, knowledge: Dict[str, Any]) -> str:
        """Broadcast a message asynchronously on the shared broadcast channel."""
        await self.initialize()
        return self.reef.broadcast(
            from_agent="api",
            knowledge=knowledge,
            channel=BROADCAST_CHANNEL
        )

    async def publish_search_request(
        self,
        query: str,
        domain: str,
        max_results: int,
        session_id: str,
        **kwargs
    ) -> bool:
        """Publish search request. paper_searcher responds_to: search_request"""
        try:
            await self.broadcast_async({
                "type": "search_request",
                "query": query,
                "domain": domain,
                "max_results": max_results,
                "session_id": session_id,
                **kwargs
            })
            logger.info(f"Published search_request: {query}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish search_request: {e}")
            return False

    async def publish_qa_request(
        self,
        question: str,
        session_id: str,
        **kwargs
    ) -> bool:
        """Publish Q&A request. qa_specialist responds_to: user_query"""
        try:
            await self.broadcast_async({
                "type": "user_query",
                "query": question,
                "session_id": session_id,
                **kwargs
            })
            logger.info(f"Published user_query: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish user_query: {e}")
            return False

    async def publish_index_request(
        self,
        papers: list,
        session_id: str,
        **kwargs
    ) -> bool:
        """Publish papers for indexing. document_processor responds_to: papers_found"""
        try:
            knowledge = {
                "type": "papers_found",
                "papers": papers,
                "original_query": f"Manual selection of {len(papers)} papers",
                "search_metadata": {
                    "domain": "user_selected",
                    "manual_selection": True,
                    "results_count": len(papers)
                },
                "session_id": session_id,
                **kwargs
            }
            result = await self.broadcast_async(knowledge)
            logger.info(f"Published papers_found: {len(papers)} papers")
            return True
        except Exception as e:
            logger.error(f"Failed to publish papers_found: {e}")
            return False

    async def close(self):
        """Close the connection."""
        if self.reef:
            await self.reef.close_backend()
            self.reef.shutdown()


# Global instance
_publisher: Optional[MessagePublisher] = None


def get_publisher() -> MessagePublisher:
    """Get or create global publisher."""
    global _publisher
    if _publisher is None:
        _publisher = MessagePublisher()
    return _publisher
