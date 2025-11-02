"""
Server-Sent Events (SSE) endpoints for real-time updates.

Provides streaming endpoints for agent status and progress updates.
"""

import asyncio
import json
from typing import AsyncGenerator
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/sse", tags=["sse"])


# Global event queue for broadcasting agent updates
agent_events_queue = asyncio.Queue()


async def event_generator(request: Request) -> AsyncGenerator[str, None]:
    """
    Generate SSE events for client.

    Yields formatted SSE messages with agent status updates.
    """
    try:
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                logger.info("Client disconnected from SSE stream")
                break

            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(
                    agent_events_queue.get(),
                    timeout=30.0
                )

                # Format as SSE
                yield f"data: {json.dumps(event)}\n\n"

            except asyncio.TimeoutError:
                # Send keepalive ping
                yield ": keepalive\n\n"

    except asyncio.CancelledError:
        logger.info("SSE stream cancelled")
    except Exception as e:
        logger.error("SSE stream error", error=str(e))


@router.get("/agent-updates")
async def agent_updates(request: Request):
    """
    Stream agent status updates via Server-Sent Events.

    Returns:
        StreamingResponse with text/event-stream content type
    """
    logger.info("Client connected to agent updates SSE stream")

    return StreamingResponse(
        event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


async def broadcast_agent_event(event: dict):
    """
    Broadcast an agent event to all connected SSE clients.

    Args:
        event: Event dictionary to broadcast
    """
    await agent_events_queue.put(event)
