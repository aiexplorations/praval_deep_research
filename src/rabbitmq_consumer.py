"""
RabbitMQ consumer that bridges messages to Praval agents.

Listens to RabbitMQ messages and triggers Praval agents
via the Reef message bus.
"""

import asyncio
import json
import logging
from typing import Callable
import pika
from praval import get_reef

from agentic_research.core.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentMessageConsumer:
    """Consumes RabbitMQ messages and triggers Praval agents."""

    def __init__(self):
        """Initialize consumer."""
        self.settings = get_settings()
        self.connection = None
        self.channel = None
        self.reef = get_reef()

    def connect(self):
        """Connect to RabbitMQ and setup queue."""
        # Connect to RabbitMQ
        params = pika.URLParameters(self.settings.RABBITMQ_URL)
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()

        # Declare exchange
        self.channel.exchange_declare(
            exchange='research_agents',
            exchange_type='topic',
            durable=True
        )

        # Declare queue
        result = self.channel.queue_declare(queue='agent_requests', durable=True)
        queue_name = result.method.queue

        # Bind queue to exchange with routing keys
        self.channel.queue_bind(
            exchange='research_agents',
            queue=queue_name,
            routing_key='search.request'
        )
        self.channel.queue_bind(
            exchange='research_agents',
            queue=queue_name,
            routing_key='qa.request'
        )

        logger.info(f"Connected to RabbitMQ, listening on queue: {queue_name}")

        # Set up consumer
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=queue_name,
            on_message_callback=self.on_message
        )

    def on_message(self, ch, method, properties, body):
        """
        Handle incoming message and trigger appropriate Praval agent.

        Args:
            ch: Channel
            method: Delivery method
            properties: Message properties
            body: Message body
        """
        try:
            logger.info(f"🔔 RAW MESSAGE RECEIVED - Body length: {len(body)} bytes")
            logger.info(f"📦 Message properties: {properties}")

            # Parse message
            message = json.loads(body)
            msg_type = message.get('type')

            logger.info(f"✅ MESSAGE PARSED - Type: {msg_type}")
            logger.info(f"📋 Full message content: {json.dumps(message, indent=2)}")

            # Trigger appropriate agent via Reef broadcast
            if msg_type == 'search_request':
                logger.info(f"🔍 SEARCH REQUEST DETECTED")
                logger.info(f"   Query: {message.get('query')}")
                logger.info(f"   Domain: {message.get('domain')}")
                logger.info(f"   Max results: {message.get('max_results')}")
                logger.info(f"   Session ID: {message.get('session_id')}")

                knowledge_payload = {
                    'type': 'search_request',
                    'query': message.get('query'),
                    'domain': message.get('domain', 'computer_science'),
                    'max_results': message.get('max_results', 10),
                    'quality_threshold': message.get('quality_threshold', 0.7),
                    'session_id': message.get('session_id')
                }

                logger.info(f"📡 BROADCASTING to Reef with knowledge: {json.dumps(knowledge_payload, indent=2)}")

                # Broadcast to paper_searcher agent on its channel
                broadcast_result = self.reef.broadcast(
                    from_agent='rabbitmq_bridge',
                    knowledge=knowledge_payload,
                    channel='paper_searcher_channel'  # Specific channel for this agent
                )

                logger.info(f"✅ BROADCAST COMPLETE - Result: {broadcast_result}")
                logger.info(f"   Channel: paper_searcher_channel")
                logger.info(f"🎯 Triggered paper_searcher agent for query: {message.get('query')}")

            elif msg_type == 'qa_request':
                logger.info(f"💬 QA REQUEST DETECTED")
                logger.info(f"   Question: {message.get('question')}")
                logger.info(f"   Session ID: {message.get('session_id')}")

                knowledge_payload = {
                    'type': 'user_query',
                    'question': message.get('question'),
                    'query': message.get('question'),
                    'user_id': message.get('user_id'),
                    'conversation_id': message.get('conversation_id'),
                    'context': message.get('context', []),
                    'session_id': message.get('session_id')
                }

                logger.info(f"📡 BROADCASTING to Reef with knowledge: {json.dumps(knowledge_payload, indent=2)}")

                # Broadcast to qa_specialist agent on its channel
                broadcast_result = self.reef.broadcast(
                    from_agent='rabbitmq_bridge',
                    knowledge=knowledge_payload,
                    channel='qa_specialist_channel'  # Specific channel for this agent
                )

                logger.info(f"✅ BROADCAST COMPLETE - Result: {broadcast_result}")
                logger.info(f"   Channel: qa_specialist_channel")
                logger.info(f"🎯 Triggered qa_specialist agent for question: {message.get('question')}")
            else:
                logger.warning(f"⚠️ UNKNOWN MESSAGE TYPE: {msg_type}")

            # Acknowledge message
            logger.info(f"✅ ACKNOWLEDGING message delivery tag: {method.delivery_tag}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"✅ MESSAGE PROCESSING COMPLETE")

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON DECODE ERROR: {e}", exc_info=True)
            logger.error(f"   Raw body: {body}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            logger.error(f"❌ ERROR PROCESSING MESSAGE: {e}", exc_info=True)
            logger.error(f"   Message type: {msg_type if 'msg_type' in locals() else 'unknown'}")
            logger.error(f"   Body: {body}")
            # Reject and requeue the message
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def start_consuming(self):
        """Start consuming messages."""
        logger.info("Starting to consume messages...")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
            self.stop()

    def stop(self):
        """Stop consuming and close connection."""
        if self.channel:
            self.channel.stop_consuming()
        if self.connection:
            self.connection.close()
        logger.info("Consumer stopped")


def start_consumer():
    """Start the RabbitMQ consumer."""
    consumer = AgentMessageConsumer()
    consumer.connect()
    consumer.start_consuming()


if __name__ == '__main__':
    start_consumer()
