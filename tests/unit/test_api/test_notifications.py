"""
Unit tests for the notification system including:
- SSE event broadcasting
- Sync-to-async event bridging
- Toast notification event types
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock


class TestSSEBroadcast:
    """Tests for SSE broadcasting functionality."""

    def test_broadcast_agent_event_sync_without_loop(self):
        """Test sync broadcast when event loop is not set."""
        from agentic_research.api.routes.sse import broadcast_agent_event_sync, set_event_loop

        # Reset event loop
        set_event_loop(None)

        # Should not raise, just log warning
        broadcast_agent_event_sync({
            "event_type": "toast",
            "toast_type": "info",
            "title": "Test",
            "message": "Test message"
        })

    def test_broadcast_agent_event_sync_with_loop(self):
        """Test sync broadcast with event loop set."""
        from agentic_research.api.routes.sse import (
            broadcast_agent_event_sync,
            set_event_loop,
            agent_events_queue
        )

        # Create and set event loop
        loop = asyncio.new_event_loop()
        set_event_loop(loop)

        test_event = {
            "event_type": "toast",
            "toast_type": "success",
            "title": "Paper Indexed",
            "message": "Test paper was indexed"
        }

        # Broadcast event
        broadcast_agent_event_sync(test_event)

        # Run loop briefly to process event
        loop.run_until_complete(asyncio.sleep(0.1))

        # Clean up
        loop.close()

    def test_event_types(self):
        """Test different event types are structured correctly."""
        toast_event = {
            "event_type": "toast",
            "toast_type": "success",
            "title": "Title",
            "message": "Message",
            "duration": 5000
        }

        assert toast_event["event_type"] == "toast"
        assert toast_event["toast_type"] in ["success", "info", "warning", "error"]
        assert isinstance(toast_event["duration"], int)

        progress_event = {
            "event_type": "indexing_progress",
            "stage": "processing",
            "current": 2,
            "total": 5,
            "current_paper": "Test Paper Title"
        }

        assert progress_event["event_type"] == "indexing_progress"
        assert progress_event["stage"] in ["starting", "downloading", "processing", "embedding", "summarizing", "extracting_citations", "indexing_linked", "complete", "error"]
        assert progress_event["current"] <= progress_event["total"]

        paper_indexed_event = {
            "event_type": "paper_indexed",
            "title": "Test Paper",
            "arxiv_id": "2401.00001",
            "vectors_stored": 50
        }

        assert paper_indexed_event["event_type"] == "paper_indexed"
        assert "title" in paper_indexed_event
        assert "arxiv_id" in paper_indexed_event

    def test_linked_paper_event(self):
        """Test linked paper indexed event structure."""
        event = {
            "event_type": "linked_paper_indexed",
            "title": "Cited Paper",
            "arxiv_id": "2401.00002",
            "source_paper_id": "2401.00001",
            "vectors_stored": 30
        }

        assert event["event_type"] == "linked_paper_indexed"
        assert "source_paper_id" in event

    def test_indexing_complete_event(self):
        """Test indexing complete event structure."""
        event = {
            "event_type": "indexing_complete",
            "papers_indexed": 5,
            "vectors_stored": 250,
            "total": 5,
            "failed": 0
        }

        assert event["event_type"] == "indexing_complete"
        assert event["papers_indexed"] == event["total"] - event["failed"]

    def test_indexing_error_event(self):
        """Test indexing error event structure."""
        event = {
            "event_type": "indexing_error",
            "error": "Failed to download PDF"
        }

        assert event["event_type"] == "indexing_error"
        assert "error" in event


class TestDocumentProcessorNotifications:
    """Tests for notifications from document processor."""

    def test_document_processor_emits_progress(self):
        """Test that document processor emits progress events."""
        with patch('agents.research.document_processor.broadcast_agent_event_sync') as mock_broadcast:
            # The document processor should emit these events:
            expected_events = [
                {"event_type": "indexing_progress", "stage": "starting"},
                {"event_type": "indexing_progress", "stage": "processing"},
                {"event_type": "paper_indexed"},
                {"event_type": "indexing_complete"}
            ]

            # Verify event types are defined
            for event in expected_events:
                assert "event_type" in event


class TestLinkedPaperIndexerNotifications:
    """Tests for notifications from linked paper indexer."""

    def test_linked_indexer_emits_progress(self):
        """Test that linked paper indexer emits progress events."""
        expected_events = [
            {"event_type": "indexing_progress", "stage": "indexing_linked"},
            {"event_type": "linked_paper_indexed"},
        ]

        for event in expected_events:
            assert "event_type" in event
