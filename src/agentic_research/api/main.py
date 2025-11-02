"""
Main FastAPI application for Praval Deep Research.

This module creates a production-ready FastAPI application with:
- Comprehensive health checks
- Research and Q&A endpoints
- Proper error handling and logging
- CORS configuration
- API documentation
- Monitoring and observability
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
import structlog
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import Counter, Histogram, Gauge
import uvicorn

from .routes import health_router, research_router
from .routes.sse import router as sse_router
from .models.research import ErrorResponse
from ..core.config import get_settings
# Removed old research_agent import - now using distributed Praval agents

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'api_active_connections',
    'Number of active connections'
)

# Application state
_app_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown tasks including:
    - Research agent initialization
    - Infrastructure connections
    - Graceful shutdown
    """
    # Startup
    logger.info("Starting Praval Deep Research API")
    
    try:
        # Initialize settings
        settings = get_settings()
        logger.info("Initializing Praval research agents")
        
        # Note: Praval agents are initialized on-demand in routes/research.py
        # This allows for better scalability and resource management
        
        # Store startup metadata
        app.state.started_at = time.time()
        app.state.distributed_mode = True  # Using Praval distributed agents
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error("Failed to initialize API", error=str(e))
        logger.warning("Continuing with degraded mode")
        app.state.distributed_mode = False
        
    yield
    
    # Shutdown
    logger.info("Shutting down Praval Deep Research API")
    
    try:
        # Note: Praval agents handle their own lifecycle
        # No explicit shutdown needed for distributed agents
        logger.info("API shutdown completed")
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))
    
    logger.info("API shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="Praval Deep Research API",
    description="""
    🤖 **Intelligent Research Assistant API**
    
    This API provides access to Praval-powered research agents that can:
    
    - **Discover Academic Papers**: Search ArXiv and other databases with intelligent query optimization
    - **Answer Research Questions**: Get comprehensive answers with source citations  
    - **Learn and Adapt**: Agents improve performance through memory and experience
    - **Distributed Processing**: Scale across multiple nodes with RabbitMQ transport
    
    ## Features
    
    - 🧠 **Memory-Enhanced Agents**: Agents learn from interactions to improve results
    - 🔍 **Smart Query Optimization**: Automatic query enhancement based on domain expertise  
    - 📚 **Source Citations**: All answers include relevant paper citations with relevance scores
    - 🚀 **Distributed Architecture**: Agents can run across multiple processes/nodes
    - 📊 **Comprehensive Monitoring**: Full observability with health checks and metrics
    - 🔒 **Production Ready**: Proper error handling, validation, and security
    
    ## Agent Architecture
    
    The system uses autonomous Praval agents that self-organize through spore communication:
    
    - **Paper Searcher**: Discovers and retrieves academic papers with domain expertise
    - **Document Processor**: Analyzes and extracts insights from research documents  
    - **Q&A Specialist**: Provides intelligent answers using retrieved knowledge
    
    ## Getting Started
    
    1. **Search Papers**: Use `/research/search` to find relevant academic papers
    2. **Ask Questions**: Use `/research/ask` to get expert answers with citations
    3. **Combined Workflow**: Use `/research/research-and-ask` for end-to-end research
    4. **Monitor Health**: Check `/health` for system status and agent performance
    """,
    version="1.0.0",
    contact={
        "name": "Praval Deep Research Team",
        "email": "research@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add trusted host middleware
# Note: In testing, TestClient may use 'testserver' as hostname
allowed_hosts = settings.ALLOWED_HOSTS + ["testserver"]
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=allowed_hosts
)


@app.middleware("http")
async def add_request_metrics(request: Request, call_next):
    """Add Prometheus metrics for all requests."""
    start_time = time.time()
    
    # Increment active connections
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(time.time() - start_time)
        
        return response
        
    finally:
        # Decrement active connections
        ACTIVE_CONNECTIONS.dec()


@app.middleware("http") 
async def add_request_logging(request: Request, call_next):
    """Add structured logging for all requests."""
    start_time = time.time()
    
    # Generate request ID
    request_id = f"req_{int(time.time() * 1000000)}"
    
    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else None
    )
    
    try:
        response = await call_next(request)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            "Request completed",
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=duration_ms
        )
        
        return response
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        
        logger.error(
            "Request failed",
            request_id=request_id,
            error=str(e),
            duration_ms=duration_ms
        )
        raise


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": " -> ".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.warning(
        "Request validation error",
        path=request.url.path,
        errors=error_details
    )
    
    error_response = ErrorResponse(
        error="ValidationError",
        message="Request validation failed",
        details={"errors": error_details}
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=json.loads(error_response.model_dump_json())
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(
        "HTTP exception",
        path=request.url.path,
        status_code=exc.status_code,
        detail=exc.detail
    )
    
    error_response = ErrorResponse(
        error="HTTPException",
        message=exc.detail or "HTTP error occurred"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=json.loads(error_response.model_dump_json())
    )


@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Handle internal server errors."""
    logger.error(
        "Internal server error",
        path=request.url.path,
        error=str(exc),
        exc_info=True
    )
    
    error_response = ErrorResponse(
        error="InternalServerError",
        message="An internal server error occurred"
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=json.loads(error_response.model_dump_json())
    )


# Include routers
app.include_router(health_router)
app.include_router(research_router)
app.include_router(sse_router)


@app.get("/", summary="API root")
async def root() -> Dict[str, Any]:
    """
    API root endpoint with basic information.
    """
    uptime_seconds = int(time.time() - _app_start_time)
    
    return {
        "message": "🤖 Praval Deep Research API",
        "version": "1.0.0",
        "status": "operational",
        "uptime_seconds": uptime_seconds,
        "distributed_mode": getattr(app.state, 'distributed_mode', False),
        "docs_url": "/docs",
        "health_check": "/health",
        "research_endpoints": {
            "search_papers": "/research/search",
            "ask_questions": "/research/ask", 
            "combined_workflow": "/research/research-and-ask"
        }
    }


@app.get("/metrics", summary="Prometheus metrics")
async def metrics():
    """
    Prometheus metrics endpoint for monitoring.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/info", summary="System information")
async def system_info() -> Dict[str, Any]:
    """
    Get detailed system information.
    """
    uptime_seconds = int(time.time() - _app_start_time)
    
    return {
        "api_version": "1.0.0",
        "praval_version": "0.6.0",
        "uptime_seconds": uptime_seconds,
        "started_at": getattr(app.state, 'started_at', _app_start_time),
        "distributed_mode": getattr(app.state, 'distributed_mode', False),
        "python_version": "3.13+",
        "environment": settings.ENVIRONMENT,
        "debug_mode": settings.DEBUG,
        "agent_count": 3,
        "supported_domains": [
            "computer science",
            "artificial intelligence", 
            "machine learning",
            "physics",
            "mathematics",
            "biology",
            "chemistry",
            "general"
        ]
    }


# Configure application for development
if __name__ == "__main__":
    uvicorn.run(
        "agentic_research.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )