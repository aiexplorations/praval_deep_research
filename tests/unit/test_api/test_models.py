"""
Unit tests for API models (Pydantic schemas).

This module tests the request/response models to ensure proper validation,
serialization, and example data integrity.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from agentic_research.api.models.research import (
    ResearchQuery, ResearchResponse, PaperResult,
    QuestionRequest, QuestionResponse, SourceCitation,
    AgentStatus, HealthCheck, ErrorResponse,
    ResearchDomain
)


class TestResearchQuery:
    """Test ResearchQuery model validation."""

    def test_valid_research_query(self):
        """Test creating a valid research query."""
        query = ResearchQuery(
            query="machine learning algorithms",
            domain=ResearchDomain.COMPUTER_SCIENCE,
            max_results=10,
            quality_threshold=0.8
        )
        
        assert query.query == "machine learning algorithms"
        assert query.domain == ResearchDomain.COMPUTER_SCIENCE
        assert query.max_results == 10
        assert query.quality_threshold == 0.8
        assert query.filters == {}

    def test_query_validation_min_length(self):
        """Test query minimum length validation."""
        with pytest.raises(ValidationError) as exc_info:
            ResearchQuery(query="x")  # Too short
        
        error = exc_info.value.errors()[0]
        assert "at least 3 characters" in str(error)

    def test_query_validation_max_length(self):
        """Test query maximum length validation."""
        long_query = "x" * 501  # Too long
        
        with pytest.raises(ValidationError) as exc_info:
            ResearchQuery(query=long_query)
        
        error = exc_info.value.errors()[0]
        assert "at most 500 characters" in str(error)

    def test_max_results_validation(self):
        """Test max_results range validation."""
        # Test minimum
        with pytest.raises(ValidationError):
            ResearchQuery(query="test", max_results=0)
        
        # Test maximum
        with pytest.raises(ValidationError):
            ResearchQuery(query="test", max_results=51)
        
        # Test valid range
        query = ResearchQuery(query="test", max_results=25)
        assert query.max_results == 25

    def test_quality_threshold_validation(self):
        """Test quality_threshold range validation."""
        # Test minimum
        with pytest.raises(ValidationError):
            ResearchQuery(query="test", quality_threshold=-0.1)
        
        # Test maximum
        with pytest.raises(ValidationError):
            ResearchQuery(query="test", quality_threshold=1.1)
        
        # Test valid range
        query = ResearchQuery(query="test", quality_threshold=0.5)
        assert query.quality_threshold == 0.5

    def test_query_whitespace_stripping(self):
        """Test that query whitespace is stripped."""
        query = ResearchQuery(query="  machine learning  ")
        assert query.query == "machine learning"

    def test_default_values(self):
        """Test default values are applied correctly."""
        query = ResearchQuery(query="test query")
        
        assert query.domain == ResearchDomain.COMPUTER_SCIENCE
        assert query.max_results == 10
        assert query.quality_threshold == 0.8
        assert query.filters == {}

    def test_domain_enum_validation(self):
        """Test that only valid domains are accepted."""
        # Valid domain
        query = ResearchQuery(query="test", domain=ResearchDomain.PHYSICS)
        assert query.domain == ResearchDomain.PHYSICS
        
        # Invalid domain should raise validation error
        with pytest.raises(ValidationError):
            ResearchQuery(query="test", domain="invalid_domain")


class TestPaperResult:
    """Test PaperResult model validation."""

    def test_valid_paper_result(self):
        """Test creating a valid paper result."""
        paper = PaperResult(
            title="Test Paper",
            authors=["Dr. Test", "Prof. Example"],
            abstract="This is a test abstract.",
            arxiv_id="2024.test001",
            url="https://arxiv.org/abs/2024.test001",
            published_date="2024-01-01",
            venue="Test Conference",
            relevance_score=0.95,
            categories=["cs.AI", "cs.LG"]
        )
        
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.relevance_score == 0.95
        assert len(paper.categories) == 2

    def test_relevance_score_validation(self):
        """Test relevance score range validation."""
        # Test minimum
        with pytest.raises(ValidationError):
            PaperResult(
                title="Test",
                authors=["Author"],
                abstract="Abstract",
                relevance_score=-0.1
            )
        
        # Test maximum
        with pytest.raises(ValidationError):
            PaperResult(
                title="Test",
                authors=["Author"],
                abstract="Abstract",
                relevance_score=1.1
            )

    def test_required_fields(self):
        """Test that required fields are enforced."""
        # Missing title
        with pytest.raises(ValidationError):
            PaperResult(
                authors=["Author"],
                abstract="Abstract",
                relevance_score=0.8
            )
        
        # Missing authors
        with pytest.raises(ValidationError):
            PaperResult(
                title="Test",
                abstract="Abstract",
                relevance_score=0.8
            )

    def test_optional_fields_defaults(self):
        """Test that optional fields have proper defaults."""
        paper = PaperResult(
            title="Test",
            authors=["Author"],
            abstract="Abstract",
            relevance_score=0.8
        )
        
        assert paper.arxiv_id is None
        assert paper.url is None
        assert paper.published_date is None
        assert paper.venue is None
        assert paper.categories == []


class TestQuestionRequest:
    """Test QuestionRequest model validation."""

    def test_valid_question_request(self):
        """Test creating a valid question request."""
        request = QuestionRequest(
            question="What are transformers?",
            context="I'm studying neural networks",
            conversation_id="conv_123",
            user_id="user_456",
            include_sources=True
        )
        
        assert request.question == "What are transformers?"
        assert request.context == "I'm studying neural networks"
        assert request.include_sources is True

    def test_question_length_validation(self):
        """Test question length validation."""
        # Too short
        with pytest.raises(ValidationError):
            QuestionRequest(question="Hi")
        
        # Too long
        long_question = "x" * 1001
        with pytest.raises(ValidationError):
            QuestionRequest(question=long_question)

    def test_context_length_validation(self):
        """Test context length validation."""
        long_context = "x" * 2001
        with pytest.raises(ValidationError):
            QuestionRequest(
                question="Valid question?",
                context=long_context
            )

    def test_question_whitespace_stripping(self):
        """Test that question whitespace is stripped."""
        request = QuestionRequest(question="  What is AI?  ")
        assert request.question == "What is AI?"

    def test_default_values(self):
        """Test default values."""
        request = QuestionRequest(question="Test question?")
        
        assert request.context is None
        assert request.conversation_id is None
        assert request.user_id is None
        assert request.include_sources is True


class TestQuestionResponse:
    """Test QuestionResponse model validation."""

    def test_valid_question_response(self):
        """Test creating a valid question response."""
        sources = [
            SourceCitation(
                title="Test Paper",
                paper_id="123",
                chunk_index=0,
                relevance_score=0.9
            )
        ]
        
        response = QuestionResponse(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            sources=sources,
            followup_questions=["How does AI work?"],
            confidence_score=0.92,
            response_time_ms=500,
            personalization_applied=True
        )
        
        assert response.question == "What is AI?"
        assert response.confidence_score == 0.92
        assert len(response.sources) == 1
        assert len(response.followup_questions) == 1

    def test_confidence_score_validation(self):
        """Test confidence score range validation."""
        with pytest.raises(ValidationError):
            QuestionResponse(
                question="Test",
                answer="Answer",
                confidence_score=1.5,  # Too high
                response_time_ms=100,
                personalization_applied=False
            )

    def test_response_time_validation(self):
        """Test response time validation."""
        with pytest.raises(ValidationError):
            QuestionResponse(
                question="Test",
                answer="Answer",
                confidence_score=0.8,
                response_time_ms=-1,  # Negative not allowed
                personalization_applied=False
            )

    def test_timestamp_auto_generation(self):
        """Test that timestamp is automatically generated."""
        response = QuestionResponse(
            question="Test",
            answer="Answer",
            confidence_score=0.8,
            response_time_ms=100,
            personalization_applied=False
        )
        
        assert response.timestamp is not None
        assert isinstance(response.timestamp, datetime)


class TestSourceCitation:
    """Test SourceCitation model validation."""

    def test_valid_source_citation(self):
        """Test creating a valid source citation."""
        citation = SourceCitation(
            title="Test Paper",
            paper_id="2024.test001",
            chunk_index=5,
            relevance_score=0.87,
            excerpt="This is a relevant excerpt."
        )
        
        assert citation.title == "Test Paper"
        assert citation.chunk_index == 5
        assert citation.relevance_score == 0.87

    def test_chunk_index_validation(self):
        """Test chunk index validation."""
        with pytest.raises(ValidationError):
            SourceCitation(
                title="Test",
                paper_id="123",
                chunk_index=-1,  # Must be >= 0
                relevance_score=0.8
            )

    def test_relevance_score_validation(self):
        """Test relevance score validation."""
        with pytest.raises(ValidationError):
            SourceCitation(
                title="Test",
                paper_id="123",
                chunk_index=0,
                relevance_score=1.5  # Must be <= 1.0
            )


class TestAgentStatus:
    """Test AgentStatus model validation."""

    def test_valid_agent_status(self):
        """Test creating valid agent status."""
        status = AgentStatus(
            agent_name="paper_searcher",
            status="active",
            uptime_seconds=3600,
            messages_processed=42,
            memory_items=150,
            last_activity=datetime.now(),
            performance_metrics={"avg_response_time": 200.5}
        )
        
        assert status.agent_name == "paper_searcher"
        assert status.status == "active"
        assert status.uptime_seconds == 3600

    def test_non_negative_validations(self):
        """Test that counts must be non-negative."""
        with pytest.raises(ValidationError):
            AgentStatus(
                agent_name="test",
                status="active",
                uptime_seconds=-1,  # Must be >= 0
                messages_processed=0,
                memory_items=0,
                last_activity=datetime.now()
            )


class TestHealthCheck:
    """Test HealthCheck model validation."""

    def test_valid_health_check(self):
        """Test creating valid health check."""
        agents = [
            AgentStatus(
                agent_name="test_agent",
                status="active",
                uptime_seconds=100,
                messages_processed=5,
                memory_items=10,
                last_activity=datetime.now()
            )
        ]
        
        health = HealthCheck(
            status="healthy",
            version="1.0.0",
            uptime_seconds=7200,
            agents=agents,
            infrastructure={"rabbitmq": "connected"},
            memory_usage={"used_mb": 256.0}
        )
        
        assert health.status == "healthy"
        assert len(health.agents) == 1
        assert health.infrastructure["rabbitmq"] == "connected"


class TestErrorResponse:
    """Test ErrorResponse model validation."""

    def test_valid_error_response(self):
        """Test creating valid error response."""
        error = ErrorResponse(
            error="ValidationError",
            message="Request validation failed",
            details={"field": "query", "issue": "too short"},
            trace_id="trace_123"
        )
        
        assert error.error == "ValidationError"
        assert error.message == "Request validation failed"
        assert error.details is not None
        assert error.trace_id == "trace_123"

    def test_timestamp_auto_generation(self):
        """Test that timestamp is automatically generated."""
        error = ErrorResponse(
            error="TestError",
            message="Test message"
        )
        
        assert error.timestamp is not None
        assert isinstance(error.timestamp, datetime)


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_research_query_json_schema(self):
        """Test ResearchQuery JSON schema generation."""
        schema = ResearchQuery.model_json_schema()
        
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "domain" in schema["properties"]
        assert "example" in schema

    def test_model_json_serialization(self):
        """Test model JSON serialization."""
        query = ResearchQuery(
            query="test query",
            domain=ResearchDomain.PHYSICS
        )
        
        json_data = query.model_dump()
        
        assert json_data["query"] == "test query"
        assert json_data["domain"] == "physics"
        assert json_data["max_results"] == 10

    def test_model_json_deserialization(self):
        """Test model JSON deserialization."""
        json_data = {
            "query": "test query",
            "domain": "mathematics",
            "max_results": 15
        }
        
        query = ResearchQuery(**json_data)
        
        assert query.query == "test query"
        assert query.domain == ResearchDomain.MATHEMATICS
        assert query.max_results == 15