"""
Test suite for core configuration management.

Following TDD principles - these tests define the expected behavior
of our configuration system before implementation.
"""

import pytest
from unittest.mock import patch
import os


class TestSettings:
    """Test the application settings configuration."""
    
    def test_settings_loads_from_environment(self):
        """Settings should load configuration from environment variables."""
        with patch.dict(os.environ, {
            'APP_NAME': 'test_app',
            'DEBUG': 'true',
            'LOG_LEVEL': 'DEBUG',
            'OPENAI_API_KEY': 'test-key-123'
        }):
            from agentic_research.core.config import Settings
            settings = Settings()
            
            assert settings.APP_NAME == 'test_app'
            assert settings.DEBUG is True
            assert settings.LOG_LEVEL == 'DEBUG'
            assert settings.OPENAI_API_KEY == 'test-key-123'
    
    def test_settings_has_default_values(self):
        """Settings should provide sensible defaults."""
        from agentic_research.core.config import Settings
        settings = Settings()
        
        assert settings.APP_NAME == 'agentic_deep_research'
        assert settings.DEBUG is False
        assert settings.LOG_LEVEL == 'INFO'
        assert settings.PRAVAL_MAX_THREADS == 10
        assert settings.ARXIV_MAX_RESULTS == 50
    
    def test_settings_validates_required_fields(self):
        """Settings should validate that required API keys are present."""
        with patch.dict(os.environ, {}, clear=True):
            from agentic_research.core.config import Settings
            settings = Settings()
            
            # Should have validation for required API keys
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                settings.validate_api_keys()
    
    def test_database_url_construction(self):
        """Settings should construct proper database URLs."""
        with patch.dict(os.environ, {
            'QDRANT_URL': 'http://localhost:6333',
            'MINIO_ENDPOINT': 'localhost:9000',
            'REDIS_URL': 'redis://localhost:6379'
        }):
            from agentic_research.core.config import Settings
            settings = Settings()
            
            assert 'localhost:6333' in settings.QDRANT_URL
            assert 'localhost:9000' in settings.MINIO_ENDPOINT
            assert 'redis://localhost:6379' in settings.REDIS_URL
    
    def test_praval_configuration(self):
        """Settings should provide proper Praval framework configuration."""
        from agentic_research.core.config import Settings
        settings = Settings()
        
        praval_config = settings.get_praval_config()
        
        assert praval_config['default_provider'] == settings.PRAVAL_DEFAULT_PROVIDER
        assert praval_config['default_model'] == settings.PRAVAL_DEFAULT_MODEL
        assert praval_config['max_threads'] == settings.PRAVAL_MAX_THREADS
        assert 'memory_config' in praval_config
        assert 'reef_config' in praval_config


class TestLoggingConfiguration:
    """Test logging configuration setup."""
    
    def test_logging_setup_with_structured_logs(self):
        """Logging should be configured with structured output."""
        from agentic_research.core.config import setup_logging
        
        # Should not raise any exceptions
        logger = setup_logging(log_level='INFO', structured=True)
        
        # Should return a logger instance
        assert logger is not None
        
        # Should be able to log structured messages
        logger.info("Test message", component="test", action="validation")
    
    def test_logging_setup_with_different_levels(self):
        """Logging should support different log levels."""
        from agentic_research.core.config import setup_logging
        
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            logger = setup_logging(log_level=level)
            assert logger is not None


class TestConfigurationValidation:
    """Test configuration validation functions."""
    
    def test_validate_rabbitmq_url(self):
        """Should validate RabbitMQ URL format."""
        from agentic_research.core.config import validate_rabbitmq_url
        
        # Valid URLs
        assert validate_rabbitmq_url('amqp://user:pass@localhost:5672/vhost')
        assert validate_rabbitmq_url('amqps://user:pass@remote:5671/')
        
        # Invalid URLs
        assert not validate_rabbitmq_url('invalid-url')
        assert not validate_rabbitmq_url('http://localhost:8080')
    
    def test_validate_api_endpoints(self):
        """Should validate external API endpoint configurations."""
        from agentic_research.core.config import validate_api_endpoints
        
        config = {
            'QDRANT_URL': 'http://localhost:6333',
            'MINIO_ENDPOINT': 'localhost:9000',
            'ARXIV_BASE_URL': 'http://export.arxiv.org/api/query'
        }
        
        # Should validate all endpoints successfully
        assert validate_api_endpoints(config)
        
        # Should fail with invalid endpoints
        invalid_config = {
            'QDRANT_URL': 'invalid-url',
            'MINIO_ENDPOINT': '',
            'ARXIV_BASE_URL': 'not-a-url'
        }
        
        assert not validate_api_endpoints(invalid_config)


@pytest.fixture
def clean_environment():
    """Fixture to provide clean environment for testing."""
    original_env = os.environ.copy()
    # Clear environment variables that might affect tests
    test_vars = [
        'APP_NAME', 'DEBUG', 'LOG_LEVEL', 'OPENAI_API_KEY',
        'QDRANT_URL', 'MINIO_ENDPOINT', 'REDIS_URL'
    ]
    
    for var in test_vars:
        if var in os.environ:
            del os.environ[var]
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_config():
    """Fixture providing sample configuration for tests."""
    return {
        'APP_NAME': 'test_research_app',
        'DEBUG': True,
        'LOG_LEVEL': 'DEBUG',
        'OPENAI_API_KEY': 'test-key-12345',
        'PRAVAL_DEFAULT_PROVIDER': 'openai',
        'PRAVAL_DEFAULT_MODEL': 'gpt-4o-mini',
        'QDRANT_URL': 'http://test-qdrant:6333',
        'MINIO_ENDPOINT': 'test-minio:9000',
        'REDIS_URL': 'redis://test-redis:6379',
        'RABBITMQ_URL': 'amqp://test:pass@test-rabbit:5672/test'
    }