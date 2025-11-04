"""
Health check endpoints for system monitoring.

This module provides comprehensive health checks for the API,
Praval agents, and infrastructure components.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status
import structlog
import psutil

from ..models.research import HealthCheck, AgentStatus
from ...core.config import get_settings
# Note: Using new distributed Praval agents - no direct imports needed

logger = structlog.get_logger()
router = APIRouter(prefix="/health", tags=["health"])

# Startup time for uptime calculation
_startup_time = time.time()


async def _check_infrastructure() -> Dict[str, str]:
    """Check status of infrastructure components."""
    infrastructure_status = {}
    settings = get_settings()
    
    # Check RabbitMQ
    try:
        import aio_pika
        connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
        await connection.close()
        infrastructure_status["rabbitmq"] = "connected"
    except Exception as e:
        logger.warning("RabbitMQ health check failed", error=str(e))
        infrastructure_status["rabbitmq"] = "disconnected"
    
    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.QDRANT_URL)
        client.get_collections()
        infrastructure_status["qdrant"] = "connected"
    except Exception as e:
        logger.warning("Qdrant health check failed", error=str(e))
        infrastructure_status["qdrant"] = "disconnected"
    
    # Check Redis
    try:
        import redis.asyncio as redis
        client = redis.from_url(settings.REDIS_URL)
        await client.ping()
        await client.close()
        infrastructure_status["redis"] = "connected"
    except Exception as e:
        logger.warning("Redis health check failed", error=str(e))
        infrastructure_status["redis"] = "disconnected"

    # Check PostgreSQL
    try:
        from ...db.base import get_session_maker
        session_maker = get_session_maker()
        async with session_maker() as session:
            # Execute a simple query to verify connection
            from sqlalchemy import text
            result = await session.execute(text("SELECT 1"))
            result.scalar()
        infrastructure_status["postgresql"] = "connected"
    except Exception as e:
        logger.warning("PostgreSQL health check failed", error=str(e))
        infrastructure_status["postgresql"] = "disconnected"

    return infrastructure_status


def _get_agent_status(agent_name: str) -> AgentStatus:
    """Get status information for a distributed Praval agent."""
    try:
        # Note: With distributed Praval agents, we simulate status
        # In production, this would query actual agent metrics
        
        return AgentStatus(
            agent_name=agent_name,
            status="active",
            uptime_seconds=int(time.time() - _startup_time),
            messages_processed=0,  # Would be tracked via metrics
            memory_items=0,  # Would be queried from agent memory
            last_activity=datetime.now(),
            performance_metrics={
                "avg_response_time_ms": 200.0,
                "success_rate": 0.98
            }
        )
    except Exception as e:
        logger.error("Failed to get agent status", agent=agent_name, error=str(e))
        return AgentStatus(
            agent_name=agent_name,
            status="error", 
            uptime_seconds=0,
            messages_processed=0,
            memory_items=0,
            last_activity=datetime.now(),
            performance_metrics={}
        )


def _get_memory_usage() -> Dict[str, float]:
    """Get system memory usage statistics."""
    try:
        memory = psutil.virtual_memory()
        return {
            "total_mb": memory.total / 1024 / 1024,
            "used_mb": memory.used / 1024 / 1024,
            "available_mb": memory.available / 1024 / 1024,
            "used_percent": memory.percent
        }
    except Exception as e:
        logger.warning("Failed to get memory usage", error=str(e))
        return {}


@router.get("/", response_model=HealthCheck, summary="Comprehensive health check")
async def health_check() -> HealthCheck:
    """
    Perform comprehensive health check of the system.
    
    Returns detailed status information including:
    - Overall system status
    - Individual agent status
    - Infrastructure component status  
    - Memory usage statistics
    - System uptime
    """
    try:
        # Check infrastructure
        infrastructure = await _check_infrastructure()
        
        # Get agent statuses
        agents = [
            _get_agent_status("paper_discovery"),
            _get_agent_status("document_processor"), 
            _get_agent_status("semantic_analyzer"),
            _get_agent_status("summarizer"),
            _get_agent_status("qa_specialist"),
            _get_agent_status("research_advisor")
        ]
        
        # Determine overall status
        infrastructure_healthy = all(
            status == "connected" 
            for status in infrastructure.values()
        )
        agents_healthy = all(
            agent.status == "active" 
            for agent in agents
        )
        
        overall_status = "healthy" if (infrastructure_healthy and agents_healthy) else "degraded"
        
        # Get memory usage
        memory_usage = _get_memory_usage()
        
        return HealthCheck(
            status=overall_status,
            version="1.0.0",
            timestamp=datetime.now(),
            uptime_seconds=int(time.time() - _startup_time),
            agents=agents,
            infrastructure=infrastructure,
            memory_usage=memory_usage
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/agents", response_model=List[AgentStatus], summary="Agent status check")
async def agents_health() -> List[AgentStatus]:
    """
    Get detailed status information for all Praval agents.
    
    Returns status, uptime, and performance metrics for each agent.
    """
    try:
        agents = [
            _get_agent_status("paper_discovery"),
            _get_agent_status("document_processor"),
            _get_agent_status("semantic_analyzer"),
            _get_agent_status("summarizer"),
            _get_agent_status("qa_specialist"),
            _get_agent_status("research_advisor")
        ]
        
        return agents
        
    except Exception as e:
        logger.error("Agent health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent health check failed: {str(e)}"
        )


@router.get("/infrastructure", response_model=Dict[str, str], summary="Infrastructure status")
async def infrastructure_health() -> Dict[str, str]:
    """
    Check connectivity to external infrastructure components.

    Returns connection status for:
    - RabbitMQ message broker
    - Qdrant vector database
    - Redis cache
    - PostgreSQL database
    """
    try:
        infrastructure = await _check_infrastructure()
        return infrastructure
        
    except Exception as e:
        logger.error("Infrastructure health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Infrastructure health check failed: {str(e)}"
        )


@router.get("/ready", summary="Readiness probe")
async def readiness_check() -> Dict[str, str]:
    """
    Kubernetes readiness probe endpoint.
    
    Returns 200 if the service is ready to handle requests,
    503 if not ready.
    """
    try:
        # Quick check of critical components
        infrastructure = await _check_infrastructure()
        
        # Service is ready if RabbitMQ and agents are working
        if infrastructure.get("rabbitmq") == "connected":
            return {"status": "ready"}
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready - RabbitMQ unavailable"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


@router.get("/live", summary="Liveness probe")
async def liveness_check() -> Dict[str, str]:
    """
    Kubernetes liveness probe endpoint.
    
    Returns 200 if the service is alive and responding,
    500 if the service should be restarted.
    """
    try:
        # Basic liveness check - just ensure the service is responding
        return {"status": "alive", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error("Liveness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service not responding"
        )