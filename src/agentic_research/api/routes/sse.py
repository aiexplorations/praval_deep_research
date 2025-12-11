"""
Server-Sent Events (SSE) endpoints for real-time updates.

Provides streaming endpoints for agent status and progress updates.
Includes sync helpers for broadcasting from Praval agents.
"""

import asyncio
import json
import threading
import queue
from typing import AsyncGenerator, Optional
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/sse", tags=["sse"])


# Global event queue for broadcasting agent updates
agent_events_queue = asyncio.Queue()

# Thread-safe queue for sync-to-async event bridging
_sync_event_queue: queue.Queue = queue.Queue()
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Set the event loop for sync-to-async bridging."""
    global _event_loop
    _event_loop = loop


def broadcast_agent_event_sync(event: dict) -> None:
    """
    Synchronous wrapper for broadcasting SSE events from Praval agents.

    This function can be called from sync code (like Praval agents) and will
    safely queue the event for async broadcast.

    Args:
        event: Event dictionary to broadcast. Should include:
            - event_type: Type of notification (e.g., 'toast', 'progress', 'status')
            - For toast: toast_type ('success', 'info', 'warning', 'error'), title, message
            - For progress: stage, current, total, details
    """
    global _event_loop

    if _event_loop is None:
        logger.warning("Event loop not set, cannot broadcast SSE event")
        return

    try:
        # Use thread-safe call to put event on async queue
        _event_loop.call_soon_threadsafe(
            lambda: asyncio.create_task(_put_event_async(event))
        )
        logger.debug("SSE event queued for broadcast", event_type=event.get("event_type"))
    except Exception as e:
        logger.error("Failed to queue SSE event", error=str(e))


async def _put_event_async(event: dict) -> None:
    """Helper to put event on async queue."""
    await agent_events_queue.put(event)


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
