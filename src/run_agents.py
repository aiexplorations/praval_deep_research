"""
Agent Runner - Start all Praval agents with RabbitMQ backend.

Uses Praval's default channel and routing. Simple pattern:
1. Initialize RabbitMQ backend
2. Subscribe all agents to shared "broadcast" channel
3. Agents filter by message type using responds_to
4. broadcast() sends to all agents on the shared channel

Message Flow:
  search_request → [paper_searcher] → papers_found → [document_processor] → ...

Usage:
    python src/run_agents.py
    python src/run_agents.py --local  # For testing without RabbitMQ
"""

import asyncio
import logging
import signal
import sys

from praval import start_agents, get_reef, get_agent_info
from praval.core.reef_backend import RabbitMQBackend

from agentic_research.core.config import get_settings

# Import all agents
from agents.research.paper_discovery import paper_discovery_agent
from agents.research.document_processor import document_processing_agent
from agents.research.semantic_analyzer import semantic_analysis_agent
from agents.research.summarization import summarization_agent
from agents.interaction.qa_specialist import qa_specialist_agent
from agents.interaction.research_advisor import research_advisor_agent

# Configure logging to show Praval's internal logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

settings = get_settings()

# All agents
AGENTS = [
    paper_discovery_agent,
    document_processing_agent,
    semantic_analysis_agent,
    summarization_agent,
    qa_specialist_agent,
    research_advisor_agent,
]

# Shared channel for all agents (like start_agents uses "startup")
BROADCAST_CHANNEL = "broadcast"


async def run_distributed_agents():
    """Run agents in distributed mode with RabbitMQ."""
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    # Register signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Initialize RabbitMQ backend
    backend = RabbitMQBackend()
    reef = get_reef()
    reef.backend = backend

    try:
        await reef.initialize_backend({
            'url': settings.RABBITMQ_URL,
            'exchange_name': 'praval.research',
        })
        logger.info(f"✓ Backend initialized: {backend.__class__.__name__}")

        # Create broadcast channel and subscribe all agents
        # This mirrors what start_agents() does for local mode
        reef.create_channel(BROADCAST_CHANNEL)

        for agent_func in AGENTS:
            agent_info = get_agent_info(agent_func)
            agent_name = agent_info["name"]
            underlying_agent = agent_info["underlying_agent"]

            # Subscribe agent to broadcast channel
            underlying_agent.subscribe_to_channel(BROADCAST_CHANNEL)

            # Store startup channel so broadcast() defaults to it
            underlying_agent._startup_channel = BROADCAST_CHANNEL

            logger.info(f"  ✓ Agent '{agent_name}' subscribed to '{BROADCAST_CHANNEL}'")

        logger.info("=" * 60)
        logger.info("Agents ready and listening. Press Ctrl+C to shutdown...")
        logger.info("=" * 60)

        # Wait for shutdown
        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"Error running agents: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down...")
        await reef.close_backend()
        reef.shutdown()


def main():
    """Main entry point."""
    local_mode = "--local" in sys.argv

    # Validate API keys
    try:
        settings.validate_api_keys()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Praval Research Agents")
    logger.info("=" * 60)

    for agent_func in AGENTS:
        name = getattr(agent_func, '_praval_name', agent_func.__name__)
        responds_to = getattr(agent_func, '_praval_responds_to', [])
        logger.info(f"  {name} → responds_to: {responds_to}")

    logger.info("=" * 60)

    if local_mode:
        # Local mode - InMemory backend
        logger.info("Mode: LOCAL (InMemory)")
        start_agents(*AGENTS)

        try:
            reef = get_reef()
            reef.wait_for_completion()
        except KeyboardInterrupt:
            pass
        finally:
            get_reef().shutdown()

    else:
        # Distributed mode - RabbitMQ backend
        logger.info("Mode: DISTRIBUTED (RabbitMQ)")
        logger.info(f"RabbitMQ: {settings.RABBITMQ_URL.split('@')[-1]}")

        asyncio.run(run_distributed_agents())

    logger.info("Agents stopped")


if __name__ == "__main__":
    main()
