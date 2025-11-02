"""
RabbitMQ messaging for inter-container communication.

Provides message publishing and consumption for communication
between API and agent containers.
"""

import json
import logging
from typing import Dict, Any, Optional
import pika
from pika.exceptions import AMQPConnectionError

from .config import get_settings

logger = logging.getLogger(__name__)


class MessagePublisher:
    """Publishes messages to RabbitMQ for agent consumption."""

    def __init__(self):
        """Initialize message publisher."""
        self.settings = get_settings()
        self.connection = None
        self.channel = None
        self._connect()

    def _connect(self):
        """Establish connection to RabbitMQ."""
        try:
            # Parse RabbitMQ URL
            params = pika.URLParameters(self.settings.RABBITMQ_URL)
            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()

            # Declare exchange and queue
            self.channel.exchange_declare(
                exchange='research_agents',
                exchange_type='topic',
                durable=True
            )

            logger.info("Connected to RabbitMQ")

        except AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            self.connection = None
            self.channel = None

    def publish_search_request(self, query: str, domain: str, max_results: int,
                               session_id: str, **kwargs) -> bool:
        """
        Publish a search request for paper discovery agents.

        Args:
            query: Search query
            domain: Research domain
            max_results: Maximum number of results
            session_id: Session identifier
            **kwargs: Additional parameters

        Returns:
            True if published successfully, False otherwise
        """
        # Reconnect if channel is closed
        if not self.channel or self.channel.is_closed:
            logger.warning("RabbitMQ channel closed, reconnecting...")
            self._connect()

        if not self.channel:
            logger.error("Failed to establish RabbitMQ connection")
            return False

        message = {
            "type": "search_request",
            "query": query,
            "domain": domain,
            "max_results": max_results,
            "session_id": session_id,
            **kwargs
        }

        try:
            self.channel.basic_publish(
                exchange='research_agents',
                routing_key='search.request',
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Persistent
                    content_type='application/json'
                )
            )
            logger.info(f"✅ Published search request: {session_id} for query: '{query}'")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to publish message: {e}")
            # Try to reconnect for next time
            try:
                self._connect()
            except:
                pass
            return False

    def publish_qa_request(self, question: str, session_id: str, **kwargs) -> bool:
        """
        Publish a Q&A request for QA specialist agents.

        Args:
            question: User question
            session_id: Session identifier
            **kwargs: Additional parameters

        Returns:
            True if published successfully, False otherwise
        """
        # Reconnect if channel is closed
        if not self.channel or self.channel.is_closed:
            logger.warning("RabbitMQ channel closed, reconnecting...")
            self._connect()

        if not self.channel:
            logger.error("Failed to establish RabbitMQ connection")
            return False

        message = {
            "type": "qa_request",
            "question": question,
            "session_id": session_id,
            **kwargs
        }

        try:
            self.channel.basic_publish(
                exchange='research_agents',
                routing_key='qa.request',
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    content_type='application/json'
                )
            )
            logger.info(f"✅ Published Q&A request: {session_id} for question: '{question[:50]}...'")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to publish message: {e}")
            # Try to reconnect for next time
            try:
                self._connect()
            except:
                pass
            return False

    def close(self):
        """Close RabbitMQ connection."""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("Closed RabbitMQ connection")


# Global publisher instance
_publisher: Optional[MessagePublisher] = None


def get_publisher() -> MessagePublisher:
    """Get or create global message publisher."""
    global _publisher
    if _publisher is None:
        _publisher = MessagePublisher()
    return _publisher
