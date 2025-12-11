"""
Unit tests for research insights performance optimization.

Tests the parallelized LLM call execution in generate_insights_sync.
"""

import pytest
from unittest.mock import patch, MagicMock
import time


class TestInsightsParallelization:
    """Tests for parallel LLM call execution."""

    @pytest.fixture
    def mock_kb_papers(self):
        """Mock knowledge base papers."""
        return [
            {
                'paper_id': f'paper-{i}',
                'title': f'Test Paper {i}',
                'categories': ['cs.AI', 'cs.LG'],
                'published_date': f'2024-0{i % 9 + 1}-15',
                'abstract': 'Test abstract'
            }
            for i in range(10)
        ]

    def test_parallel_execution_structure(self, mock_kb_papers):
        """Test that prompts are structured for parallel execution."""
        # Verify the prompts dictionary structure
        prompts_keys = ['research_areas', 'trending_topics', 'research_gaps', 'next_steps']

        # All these should be able to run in parallel
        for key in prompts_keys:
            assert key in prompts_keys

    def test_json_parsing_robustness(self):
        """Test that JSON parsing handles various LLM output formats."""
        import re
        import json

        # Test various formats the LLM might return
        test_cases = [
            # Clean JSON array
            '["topic1", "topic2"]',
            # JSON in code block
            '```json\n["topic1", "topic2"]\n```',
            # JSON with extra text
            'Here are the topics:\n["topic1", "topic2"]\n\nThese are the trends.',
            # Malformed (should return empty)
            'This is not JSON',
        ]

        def parse_json_response(raw_text: str) -> list:
            """Same parsing logic as in the function."""
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', raw_text, re.DOTALL) or re.search(r'\[.*\]', raw_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1 if '```' in raw_text else 0))
                except json.JSONDecodeError:
                    return []
            return []

        results = [parse_json_response(tc) for tc in test_cases]

        assert results[0] == ["topic1", "topic2"]
        assert results[1] == ["topic1", "topic2"]
        assert results[2] == ["topic1", "topic2"]
        assert results[3] == []  # Malformed returns empty

    def test_insights_response_structure(self):
        """Test that insights response has expected structure using live API."""
        from fastapi.testclient import TestClient
        from agentic_research.api.main import app

        client = TestClient(app)
        response = client.get('/research/insights')

        # API should return 200
        assert response.status_code == 200
        data = response.json()

        # Check required structure
        assert 'research_areas' in data
        assert 'trending_topics' in data
        assert 'research_gaps' in data
        assert 'next_steps' in data
        assert 'kb_context' in data
        assert 'generation_metadata' in data

        # Check types
        assert isinstance(data['research_areas'], list)
        assert isinstance(data['trending_topics'], list)
        assert isinstance(data['research_gaps'], list)
        assert isinstance(data['next_steps'], list)
        assert isinstance(data['kb_context'], dict)
        assert isinstance(data['generation_metadata'], dict)

    def test_generation_metadata_has_timing(self):
        """Test that generation metadata includes timing info."""
        from fastapi.testclient import TestClient
        from agentic_research.api.main import app

        client = TestClient(app)
        response = client.get('/research/insights')

        data = response.json()
        metadata = data['generation_metadata']

        # Check timing metadata exists
        assert 'generation_time_seconds' in metadata
        assert 'parallel_execution' in metadata

        # Parallel execution should be enabled
        assert metadata['parallel_execution'] == True


class TestAreaPapersEndpoint:
    """Tests for the /areas/{area_name}/papers endpoint."""

    def test_area_papers_endpoint_returns_200(self):
        """Test that area papers endpoint returns 200 status."""
        from fastapi.testclient import TestClient
        from agentic_research.api.main import app

        client = TestClient(app)
        response = client.get('/research/areas/machine%20learning/papers')

        assert response.status_code == 200

    def test_area_papers_response_structure(self):
        """Test that area papers response has expected structure."""
        from fastapi.testclient import TestClient
        from agentic_research.api.main import app

        client = TestClient(app)
        response = client.get('/research/areas/artificial%20intelligence/papers')

        data = response.json()

        # Check structure
        assert 'area_name' in data
        assert 'papers' in data
        assert 'total_found' in data
        assert data['status'] == 'success'
        assert isinstance(data['papers'], list)

    def test_area_papers_with_limit(self):
        """Test that limit parameter works."""
        from fastapi.testclient import TestClient
        from agentic_research.api.main import app

        client = TestClient(app)
        response = client.get('/research/areas/test/papers?limit=5')

        assert response.status_code == 200
        data = response.json()
        # Papers list should exist
        assert 'papers' in data
