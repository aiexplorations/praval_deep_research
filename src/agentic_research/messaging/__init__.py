"""
Messaging infrastructure for the Praval Deep Research system.

This package provides the messaging bridge between Praval agents and RabbitMQ,
enabling reliable, persistent communication in addition to Praval's native
spore-based messaging system.
"""

from .rabbitmq_bridge import (
    RabbitMQBridge,
    SporeMessage,
    MessageHandler,
    ConnectionConfig
)

__all__ = [
    "RabbitMQBridge",
    "SporeMessage", 
    "MessageHandler",
    "ConnectionConfig"
]