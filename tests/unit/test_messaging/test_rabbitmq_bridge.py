"""
Unit tests for RabbitMQ-Praval bridge integration.

This module tests the messaging bridge that allows Praval agents to communicate
through RabbitMQ for reliable, persistent messaging in addition to Praval's
native spore-based communication.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from agentic_research.messaging.rabbitmq_bridge import (
    RabbitMQBridge,
    SporeMessage,
    MessageHandler,
    ConnectionConfig
)


class TestRabbitMQBridge:
    """Test suite for RabbitMQ bridge functionality."""

    @pytest.fixture
    def connection_config(self) -> ConnectionConfig:
        """Create test connection configuration."""
        return ConnectionConfig(
            host="localhost",
            port=5672,
            username="research_user",
            password="research_pass",
            virtual_host="research_vhost",
            exchange_name="agent_spores",
            queue_prefix="agent_"
        )

    @pytest.fixture
    def bridge(self, connection_config: ConnectionConfig) -> RabbitMQBridge:
        """Create RabbitMQ bridge instance for testing."""
        return RabbitMQBridge(connection_config)

    def test_connection_config_creation(self, connection_config: ConnectionConfig):
        """Test connection configuration is created correctly."""
        assert connection_config.host == "localhost"
        assert connection_config.port == 5672
        assert connection_config.username == "research_user"
        assert connection_config.password == "research_pass"
        assert connection_config.virtual_host == "research_vhost"
        assert connection_config.exchange_name == "agent_spores"
        assert connection_config.queue_prefix == "agent_"

    def test_spore_message_creation(self):
        """Test SporeMessage creation and serialization."""
        message = SporeMessage(
            agent_id="paper_searcher",
            message_type="search_request", 
            payload={"query": "machine learning", "max_results": 10},
            correlation_id="req_123",
            reply_to="requester_agent"
        )
        
        assert message.agent_id == "paper_searcher"
        assert message.message_type == "search_request"
        assert message.payload["query"] == "machine learning"
        assert message.correlation_id == "req_123"
        assert message.reply_to == "requester_agent"
        assert message.timestamp is not None

    def test_spore_message_serialization(self):
        """Test SporeMessage JSON serialization/deserialization."""
        original = SporeMessage(
            agent_id="test_agent",
            message_type="test_message",
            payload={"data": "test"},
            correlation_id="test_123"
        )
        
        # Serialize to JSON
        json_data = original.to_json()
        assert isinstance(json_data, str)
        
        # Deserialize from JSON  
        reconstructed = SporeMessage.from_json(json_data)
        assert reconstructed.agent_id == original.agent_id
        assert reconstructed.message_type == original.message_type
        assert reconstructed.payload == original.payload
        assert reconstructed.correlation_id == original.correlation_id

    @pytest.mark.asyncio
    async def test_bridge_connection(self, bridge: RabbitMQBridge):
        """Test RabbitMQ bridge connection establishment."""
        with patch('aio_pika.connect_robust') as mock_connect:
            mock_connection = AsyncMock()
            mock_channel = AsyncMock()
            mock_exchange = AsyncMock()
            
            mock_connect.return_value = mock_connection
            mock_connection.channel.return_value = mock_channel
            mock_channel.declare_exchange.return_value = mock_exchange
            
            await bridge.connect()
            
            # Verify connection was established
            assert bridge.is_connected
            mock_connect.assert_called_once()
            mock_connection.channel.assert_called_once()
            mock_channel.declare_exchange.assert_called_once_with(
                "agent_spores", type="topic", durable=True
            )

    @pytest.mark.asyncio 
    async def test_bridge_disconnection(self, bridge: RabbitMQBridge):
        """Test RabbitMQ bridge disconnection."""
        # Mock connected state
        mock_connection = AsyncMock()
        bridge._connection = mock_connection
        bridge._channel = AsyncMock()
        bridge._is_connected = True
        
        await bridge.disconnect()
        
        # Verify disconnection
        assert not bridge.is_connected
        assert bridge._connection is None
        assert bridge._channel is None
        assert bridge._exchange is None
        mock_connection.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_spore_message(self, bridge: RabbitMQBridge):
        """Test sending spore messages through RabbitMQ."""
        # Setup mocked connection
        bridge._connection = AsyncMock()
        bridge._channel = AsyncMock()
        bridge._exchange = AsyncMock()
        bridge._is_connected = True
        
        message = SporeMessage(
            agent_id="test_agent",
            message_type="test_message",
            payload={"data": "test"},
            correlation_id="test_123"
        )
        
        await bridge.send_spore(message)
        
        # Verify message was published
        bridge._exchange.publish.assert_called_once()
        call_args = bridge._exchange.publish.call_args
        assert call_args[1]["routing_key"] == "test_agent.test_message"

    @pytest.mark.asyncio
    async def test_register_message_handler(self, bridge: RabbitMQBridge):
        """Test registering message handlers for specific agents."""
        handler = AsyncMock()
        
        with patch.object(bridge, '_setup_consumer') as mock_setup:
            await bridge.register_handler("test_agent", handler)
            
            # Verify handler was registered
            assert "test_agent" in bridge._handlers
            assert bridge._handlers["test_agent"] == handler
            mock_setup.assert_called_once_with("test_agent", handler)

    @pytest.mark.asyncio
    async def test_message_routing(self, bridge: RabbitMQBridge):
        """Test message routing to correct handlers."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        
        bridge._handlers = {
            "agent1": handler1,
            "agent2": handler2
        }
        
        # Create test message for agent1
        message_data = json.dumps({
            "agent_id": "agent1",
            "message_type": "test",
            "payload": {"data": "test"},
            "correlation_id": "123",
            "timestamp": "2023-01-01T00:00:00Z"
        })
        
        # Mock incoming message
        mock_message = MagicMock()
        mock_message.body = message_data.encode('utf-8')
        mock_message.ack = AsyncMock()
        
        await bridge._handle_incoming_message(mock_message)
        
        # Verify only agent1 handler was called
        handler1.assert_called_once()
        handler2.assert_not_called()
        mock_message.ack.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_retry_logic(self, bridge: RabbitMQBridge):
        """Test automatic reconnection on connection failure."""
        with patch('aio_pika.connect_robust') as mock_connect:
            # First call fails, second succeeds
            mock_connect.side_effect = [
                ConnectionError("Connection failed"),
                AsyncMock()
            ]
            
            with pytest.raises(ConnectionError):
                await bridge.connect()
            
            # Should have retried once
            assert mock_connect.call_count == 1

    @pytest.mark.asyncio
    async def test_message_acknowledgment(self, bridge: RabbitMQBridge):
        """Test proper message acknowledgment after processing."""
        handler = AsyncMock()
        bridge._handlers = {"test_agent": handler}
        
        message_data = json.dumps({
            "agent_id": "test_agent", 
            "message_type": "test",
            "payload": {"data": "test"},
            "correlation_id": "123",
            "timestamp": "2023-01-01T00:00:00Z"
        })
        
        mock_message = MagicMock()
        mock_message.body = message_data.encode('utf-8')
        mock_message.ack = AsyncMock()
        
        await bridge._handle_incoming_message(mock_message)
        
        # Verify message was acknowledged after processing
        handler.assert_called_once()
        mock_message.ack.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_in_message_processing(self, bridge: RabbitMQBridge):
        """Test error handling when message processing fails."""
        # Handler that raises an exception
        handler = AsyncMock(side_effect=ValueError("Processing failed"))
        bridge._handlers = {"test_agent": handler}
        
        message_data = json.dumps({
            "agent_id": "test_agent",
            "message_type": "test", 
            "payload": {"data": "test"},
            "correlation_id": "123",
            "timestamp": "2023-01-01T00:00:00Z"
        })
        
        mock_message = MagicMock()
        mock_message.body = message_data.encode('utf-8')
        mock_message.nack = AsyncMock()
        
        await bridge._handle_incoming_message(mock_message)
        
        # Verify message was rejected (nack) due to processing error
        handler.assert_called_once()
        mock_message.nack.assert_called_once_with(requeue=True)

    def test_queue_name_generation(self, bridge: RabbitMQBridge):
        """Test proper queue name generation for agents."""
        queue_name = bridge._get_queue_name("paper_searcher")
        assert queue_name == "agent_paper_searcher"
        
        queue_name = bridge._get_queue_name("document_processor") 
        assert queue_name == "agent_document_processor"

    def test_routing_key_generation(self, bridge: RabbitMQBridge):
        """Test proper routing key generation for messages."""
        message = SporeMessage(
            agent_id="paper_searcher",
            message_type="search_request",
            payload={"query": "test"}
        )
        
        routing_key = bridge._get_routing_key(message)
        assert routing_key == "paper_searcher.search_request"

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, bridge: RabbitMQBridge):
        """Test RabbitMQ bridge as async context manager."""
        with patch.object(bridge, 'connect') as mock_connect:
            with patch.object(bridge, 'disconnect') as mock_disconnect:
                async with bridge:
                    mock_connect.assert_called_once()
                
                mock_disconnect.assert_called_once()