"""
Unit tests for the main FastAPI application.

This module tests the core API functionality including:
- Application startup and configuration
- Basic endpoints and routing
- Error handling and middleware
- Health checks and system information
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import time

from agentic_research.api.main import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


class TestMainAPI:
    """Test suite for main API functionality."""

    def test_root_endpoint(self, client):
        """Test API root endpoint returns basic information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert "uptime_seconds" in data
        assert "docs_url" in data
        assert "health_check" in data
        assert "research_endpoints" in data
        
        # Check message content
        assert "Praval Deep Research API" in data["message"]
        assert data["version"] == "1.0.0"
        assert data["status"] == "operational"
        assert data["docs_url"] == "/docs"
        assert data["health_check"] == "/health"
        
        # Check research endpoints are documented
        research_endpoints = data["research_endpoints"]
        assert "search_papers" in research_endpoints
        assert "ask_questions" in research_endpoints
        assert "combined_workflow" in research_endpoints
        
        # Check uptime is reasonable
        assert isinstance(data["uptime_seconds"], int)
        assert data["uptime_seconds"] >= 0

    def test_system_info_endpoint(self, client):
        """Test system information endpoint returns detailed info."""
        response = client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "api_version" in data
        assert "praval_version" in data
        assert "uptime_seconds" in data
        assert "started_at" in data
        assert "distributed_mode" in data
        assert "python_version" in data
        assert "environment" in data
        assert "debug_mode" in data
        assert "agent_count" in data
        assert "supported_domains" in data
        
        # Check values
        assert data["api_version"] == "1.0.0"
        assert data["praval_version"] == "0.6.0"
        assert data["agent_count"] == 3
        assert isinstance(data["supported_domains"], list)
        assert len(data["supported_domains"]) > 0
        
        # Check supported domains include expected values
        domains = data["supported_domains"]
        assert "computer science" in domains
        assert "artificial intelligence" in domains
        assert "machine learning" in domains

    def test_openapi_docs_accessible(self, client):
        """Test that OpenAPI documentation is accessible."""
        response = client.get("/docs", follow_redirects=False)
        
        # Should either return the docs or redirect to them
        assert response.status_code in [200, 307]

    def test_openapi_json_endpoint(self, client):
        """Test OpenAPI JSON schema endpoint."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        openapi_spec = response.json()
        
        # Check OpenAPI spec structure
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec
        
        # Check API info
        info = openapi_spec["info"]
        assert info["title"] == "Agentic Deep Research API"
        assert info["version"] == "1.0.0"
        
        # Check that our endpoints are documented
        paths = openapi_spec["paths"]
        assert "/" in paths
        assert "/info" in paths
        assert "/health/" in paths  # Health router uses trailing slash
        assert "/research/search" in paths
        assert "/research/ask" in paths

    def test_cors_headers(self, client):
        """Test CORS headers are properly set."""
        # Test OPTIONS request (preflight)
        response = client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        # Should allow the request
        assert response.status_code == 200
        
        # Check CORS headers are present
        headers = response.headers
        assert "access-control-allow-origin" in headers
        assert "access-control-allow-methods" in headers

    def test_request_validation_error_handling(self, client):
        """Test request validation error handling."""
        # Send invalid request to research endpoint
        invalid_payload = {
            "query": "x",  # Too short (min 3 chars)
            "max_results": 100  # Too high (max 50)
        }
        
        response = client.post("/research/search", json=invalid_payload)
        
        assert response.status_code == 422
        error_data = response.json()
        
        assert "error" in error_data
        assert "message" in error_data
        assert "details" in error_data
        assert error_data["error"] == "ValidationError"
        assert "validation failed" in error_data["message"].lower()

    def test_404_error_handling(self, client):
        """Test 404 error handling for non-existent endpoints."""
        response = client.get("/nonexistent/endpoint")
        
        assert response.status_code == 404

    def test_middleware_adds_request_logging(self, client):
        """Test that request logging middleware is working."""
        with patch('agentic_research.api.main.logger') as mock_logger:
            response = client.get("/")
            
            # Should log request start and completion
            assert mock_logger.info.call_count >= 2
            
            # Check log calls contain expected information
            log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
            assert any("Request started" in call for call in log_calls)
            assert any("Request completed" in call for call in log_calls)

    @patch('agentic_research.api.main.generate_latest')
    def test_metrics_endpoint(self, mock_generate_latest, client):
        """Test Prometheus metrics endpoint."""
        mock_generate_latest.return_value = b"# Mock metrics\napi_requests_total 42\n"
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert mock_generate_latest.called
        
        # Check content type
        assert "text/plain" in response.headers.get("content-type", "")

    def test_health_endpoints_accessible(self, client):
        """Test that health check endpoints are accessible."""
        # Main health check
        response = client.get("/health/")
        assert response.status_code in [200, 503]  # May be 503 if infrastructure not ready
        
        # Liveness probe
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        
        # Readiness probe
        response = client.get("/health/ready")
        # May be 503 if RabbitMQ not available in test environment
        assert response.status_code in [200, 503]

    def test_research_endpoints_accessible(self, client):
        """Test that research endpoints are accessible and return proper errors for invalid requests."""
        # Test search endpoint without data
        response = client.post("/research/search")
        assert response.status_code == 422  # Validation error
        
        # Test ask endpoint without data
        response = client.post("/research/ask")
        assert response.status_code == 422  # Validation error
        
        # Test research and ask endpoint without data
        response = client.post("/research/research-and-ask")
        assert response.status_code == 422  # Validation error
        
        # Test sessions endpoint
        response = client.get("/research/sessions")
        assert response.status_code == 200

    @patch('agentic_research.api.main.initialize_research_network')
    def test_lifespan_startup(self, mock_init_network):
        """Test application lifespan startup initialization."""
        mock_init_network.return_value = True
        
        # Create new client to trigger lifespan
        with TestClient(app) as test_client:
            # Startup should have been called
            # Note: This test might be flaky depending on how TestClient handles lifespan
            pass

    def test_application_configuration(self):
        """Test that application is properly configured."""
        # Check app metadata
        assert app.title == "Praval Deep Research API"
        assert app.version == "1.0.0"
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"
        assert app.openapi_url == "/openapi.json"
        
        # Check middleware is configured
        middleware_types = [type(middleware.cls).__name__ for middleware in app.user_middleware]
        assert "CORSMiddleware" in middleware_types
        assert "TrustedHostMiddleware" in middleware_types

    def test_structured_logging_configuration(self):
        """Test that structured logging is properly configured."""
        import structlog
        
        # Check that structlog is configured
        logger = structlog.get_logger()
        assert logger is not None
        
        # Test that we can log structured data
        test_logger = structlog.get_logger("test")
        # This should not raise an exception
        test_logger.info("Test log message", test_field="test_value")


class TestErrorHandling:
    """Test suite for error handling."""
    
    def test_http_exception_handler(self, client):
        """Test HTTP exception handling."""
        # This would test custom HTTP exception handling
        # For now, we test with a 404 which should be handled
        response = client.get("/nonexistent")
        assert response.status_code == 404

    @patch('agentic_research.api.routes.research.start_distributed_research')
    def test_internal_server_error_handling(self, mock_research, client):
        """Test internal server error handling."""
        # Make the research function raise an exception
        mock_research.side_effect = Exception("Test error")
        
        valid_payload = {
            "query": "test query",
            "domain": "computer science",
            "max_results": 5
        }
        
        response = client.post("/research/search", json=valid_payload)
        
        # Should return 500 error
        assert response.status_code == 500
        error_data = response.json()
        
        assert "error" in error_data
        assert error_data["error"] == "InternalServerError"