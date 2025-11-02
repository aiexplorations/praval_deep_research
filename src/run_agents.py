"""
Agent Runner - Start all Praval agents.

This script imports and initializes all research agents,
keeping them running to process spore messages.
"""

import asyncio
import logging
import signal
import sys
import threading
from typing import NoReturn

from praval import start_agents
from agentic_research.core.config import get_settings, setup_logging
from rabbitmq_consumer import start_consumer

# Import all agents to register them
from agents.research.paper_discovery import paper_discovery_agent
from agents.research.document_processor import document_processing_agent
from agents.research.semantic_analyzer import semantic_analysis_agent
from agents.research.summarization import summarization_agent
from agents.interaction.qa_specialist import qa_specialist_agent
from agents.interaction.research_advisor import research_advisor_agent


# Setup logging
settings = get_settings()
logger = setup_logging(
    log_level=settings.LOG_LEVEL,
    structured=True
)


class AgentRunner:
    """Runner for Praval agents."""

    def __init__(self):
        """Initialize agent runner."""
        self.running = False
        self.settings = get_settings()

        # Praval configuration is handled via environment variables
        # The agents use @agent decorator which reads config automatically
        praval_config = self.settings.get_praval_config()

        logger.info(
            "Agent runner initialized",
            praval_provider=praval_config["default_provider"],
            praval_model=praval_config["default_model"],
            agents_loaded=6
        )

    async def start(self) -> NoReturn:
        """
        Start the agent runner.

        This starts all Praval agents and keeps them running
        to process spore messages.
        """
        self.running = True

        logger.info("🚀 Starting agent runner...")

        # Start all Praval agents
        logger.info("🤖 Initializing Praval agents...")
        agents_list = [
            paper_discovery_agent,
            document_processing_agent,
            semantic_analysis_agent,
            summarization_agent,
            qa_specialist_agent,
            research_advisor_agent
        ]

        # Start agents using Praval's start_agents function
        start_agents(*agents_list)
        logger.info(f"✅ Started {len(agents_list)} Praval agents")

        # Log channel subscriptions for debugging
        from praval import get_reef
        reef = get_reef()
        logger.info(f"📊 Reef channels: {list(reef.channels.keys())}")
        for channel_name, channel in reef.channels.items():
            subscribers = list(channel.subscribers.keys())
            logger.info(f"   Channel '{channel_name}': {len(subscribers)} subscribers - {subscribers}")

        # Start RabbitMQ consumer in separate thread
        logger.info("📨 Starting RabbitMQ consumer...")
        consumer_thread = threading.Thread(target=start_consumer, daemon=True)
        consumer_thread.start()

        logger.info("📡 Agents active and listening for spores")
        logger.info(
            "Registered agents",
            agents=[
                "paper_discovery_agent",
                "document_processing_agent",
                "semantic_analysis_agent",
                "summarization_agent",
                "qa_specialist_agent",
                "research_advisor_agent"
            ]
        )

        # Keep running
        try:
            while self.running:
                # Agents are event-driven via Praval's internal mechanism
                # This loop just keeps the process alive
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Agent runner cancelled")
            self.running = False

    def stop(self) -> None:
        """Stop the agent runner."""
        logger.info("Stopping agent runner...")
        self.running = False


# Global runner instance
runner = AgentRunner()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    runner.stop()
    sys.exit(0)


async def main():
    """Main entry point."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Validate API keys
    try:
        runner.settings.validate_api_keys()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Start runner
    try:
        await runner.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Agent runner error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
