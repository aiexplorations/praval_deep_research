"""
RabbitMQ Bridge for Praval Agent Communication.

This module provides a bridge between Praval agents and RabbitMQ, enabling
reliable, persistent messaging alongside Praval's native spore communication.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Awaitable, Optional
from uuid import uuid4

import aio_pika
from aio_pika import Message, ExchangeType
from aio_pika.abc import AbstractRobustConnection, AbstractRobustChannel, AbstractExchange

# Type aliases
MessageHandler = Callable[['SporeMessage'], Awaitable[None]]

logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for RabbitMQ connection."""
    host: str = "localhost"
    port: int = 5672
    username: str = "research_user"
    password: str = "research_pass"
    virtual_host: str = "research_vhost"
    exchange_name: str = "agent_spores"
    queue_prefix: str = "agent_"


@dataclass
class SporeMessage:
    """
    Standardized message format for Praval agent communication.
    
    This represents a spore message that can be sent between agents
    through the RabbitMQ messaging infrastructure.
    """
    agent_id: str
    message_type: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def __post_init__(self):
        """Generate correlation ID if not provided."""
        if self.correlation_id is None:
            self.correlation_id = str(uuid4())
    
    def to_json(self) -> str:
        """Serialize message to JSON string."""
        return json.dumps({
            "agent_id": self.agent_id,
            "message_type": self.message_type,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "timestamp": self.timestamp
        })
    
    @classmethod
    def from_json(cls, json_data: str) -> 'SporeMessage':
        """Deserialize message from JSON string."""
        data = json.loads(json_data)
        return cls(
            agent_id=data["agent_id"],
            message_type=data["message_type"],
            payload=data["payload"],
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            timestamp=data["timestamp"]
        )


class RabbitMQBridge:
    """
    Bridge for Praval agent communication through RabbitMQ.
    
    Enables reliable, persistent messaging between agents using RabbitMQ
    as the message broker while maintaining compatibility with Praval's
    agent communication patterns.
    """
    
    def __init__(self, config: ConnectionConfig):
        """Initialize the RabbitMQ bridge."""
        self.config = config
        self._connection: Optional[AbstractRobustConnection] = None
        self._channel: Optional[AbstractRobustChannel] = None
        self._exchange: Optional[AbstractExchange] = None
        self._handlers: Dict[str, MessageHandler] = {}
        self._is_connected: bool = False
        
    @property
    def is_connected(self) -> bool:
        """Check if bridge is connected to RabbitMQ."""
        return self._is_connected and self._connection is not None
    
    async def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        try:
            logger.info(f"Connecting to RabbitMQ at {self.config.host}:{self.config.port}")
            
            # Build connection URL
            connection_url = (
                f"amqp://{self.config.username}:{self.config.password}@"
                f"{self.config.host}:{self.config.port}/{self.config.virtual_host}"
            )
            
            # Establish robust connection
            self._connection = await aio_pika.connect_robust(connection_url)
            self._channel = await self._connection.channel()
            
            # Declare exchange for agent communication
            self._exchange = await self._channel.declare_exchange(
                self.config.exchange_name,
                type=ExchangeType.TOPIC,
                durable=True
            )
            
            self._is_connected = True
            logger.info("Successfully connected to RabbitMQ")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            self._is_connected = False
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        if self._connection:
            await self._connection.close()
        self._connection = None
        self._channel = None
        self._exchange = None
        self._is_connected = False
        logger.info("Disconnected from RabbitMQ")
    
    async def send_spore(self, message: SporeMessage) -> None:
        """Send a spore message to the specified agent."""
        if not self.is_connected:
            raise ConnectionError("Not connected to RabbitMQ")
        
        routing_key = self._get_routing_key(message)
        
        # Create AMQP message
        amqp_message = Message(
            body=message.to_json().encode('utf-8'),
            correlation_id=message.correlation_id,
            reply_to=message.reply_to,
            content_type="application/json",
            delivery_mode=2  # Make message persistent
        )
        
        # Publish message
        await self._exchange.publish(
            amqp_message,
            routing_key=routing_key
        )
        
        logger.debug(f"Sent spore message to {message.agent_id}: {message.message_type}")
    
    async def register_handler(self, agent_id: str, handler: MessageHandler) -> None:
        """Register a message handler for a specific agent."""
        self._handlers[agent_id] = handler
        
        # Set up consumer for this agent
        await self._setup_consumer(agent_id, handler)
        
        logger.info(f"Registered handler for agent: {agent_id}")
    
    async def _setup_consumer(self, agent_id: str, handler: MessageHandler) -> None:
        """Set up message consumer for an agent."""
        if not self.is_connected:
            raise ConnectionError("Not connected to RabbitMQ")
        
        queue_name = self._get_queue_name(agent_id)
        
        # Declare queue for agent
        queue = await self._channel.declare_queue(
            queue_name,
            durable=True,
            auto_delete=False
        )
        
        # Bind queue to exchange with routing patterns
        routing_patterns = [
            f"{agent_id}.*",  # Direct messages to this agent
            "broadcast.*"     # Broadcast messages to all agents
        ]
        
        for pattern in routing_patterns:
            await queue.bind(self._exchange, routing_key=pattern)
        
        # Set up message consumer
        await queue.consume(
            callback=self._handle_incoming_message,
            no_ack=False  # Require explicit acknowledgment
        )
        
        logger.info(f"Set up consumer for agent {agent_id} on queue {queue_name}")
    
    async def _handle_incoming_message(self, message: aio_pika.abc.AbstractIncomingMessage) -> None:
        """Handle incoming messages from RabbitMQ."""
        try:
            # Parse message
            spore_message = SporeMessage.from_json(message.body.decode('utf-8'))
            
            # Find handler for target agent
            handler = self._handlers.get(spore_message.agent_id)
            if handler:
                # Process message
                await handler(spore_message)
                
                # Acknowledge successful processing
                await message.ack()
                
                logger.debug(f"Processed message for {spore_message.agent_id}: {spore_message.message_type}")
            else:
                logger.warning(f"No handler registered for agent: {spore_message.agent_id}")
                await message.ack()  # Acknowledge to remove from queue
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Reject message and requeue for retry
            await message.nack(requeue=True)
    
    def _get_queue_name(self, agent_id: str) -> str:
        """Generate queue name for an agent."""
        return f"{self.config.queue_prefix}{agent_id}"
    
    def _get_routing_key(self, message: SporeMessage) -> str:
        """Generate routing key for a message."""
        return f"{message.agent_id}.{message.message_type}"
    
    async def __aenter__(self) -> 'RabbitMQBridge':
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()