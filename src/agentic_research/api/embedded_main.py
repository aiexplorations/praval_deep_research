"""
Embedded Mode Entry Point for Praval Deep Research

This module provides a FastAPI application configured for desktop/embedded deployment,
replacing Docker-based services with local embedded alternatives.

Usage:
    # Start embedded server
    uvicorn agentic_research.api.embedded_main:app --host 127.0.0.1 --port 8000

    # Or programmatically
    from agentic_research.api.embedded_main import create_embedded_app
    app = create_embedded_app()
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Set embedded mode before importing other modules
os.environ["PRAVAL_EMBEDDED_MODE"] = "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_embedded_app() -> FastAPI:
    """
    Create FastAPI application configured for embedded/desktop mode.

    This replaces Docker services with embedded alternatives:
    - MinIO -> EmbeddedStorageClient (filesystem)
    - Qdrant -> EmbeddedVectorDB (LanceDB/numpy)
    - Redis -> EmbeddedCacheStore (diskcache)
    - RabbitMQ -> LocalMessageQueue (in-process)
    - PostgreSQL -> SQLite
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager."""
        logger.info("Starting Praval Deep Research (Embedded Mode)")

        # Initialize embedded services
        from agentic_research.storage.embedded import (
            EmbeddedConfig,
            EmbeddedServices,
            get_embedded_config,
        )

        # Get configuration
        if os.environ.get("PRAVAL_DATA_DIR"):
            config = EmbeddedConfig(data_dir=Path(os.environ["PRAVAL_DATA_DIR"]))
        else:
            config = EmbeddedConfig.for_desktop("Praval")

        logger.info(f"Data directory: {config.data_dir}")

        # Initialize services
        services = EmbeddedServices(config)
        await services.initialize()

        # Store in app state
        app.state.services = services
        app.state.config = config

        # Verify API key
        if not config.openai_api_key:
            logger.warning("OPENAI_API_KEY not set - embeddings will fail")

        logger.info("Embedded services initialized")

        yield

        # Cleanup
        logger.info("Shutting down...")
        await services.shutdown()

    # Create app
    app = FastAPI(
        title="Praval Deep Research",
        description="AI-Powered Research Paper Assistant (Desktop Edition)",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS for local frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "tauri://localhost",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    register_routes(app)

    return app


def register_routes(app: FastAPI):
    """Register API routes."""

    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        services = app.state.services
        return {
            "status": "healthy",
            "mode": "embedded",
            "services": services.health_check()
        }

    # Shutdown endpoint (for graceful shutdown from Tauri)
    @app.post("/shutdown")
    async def shutdown():
        """Graceful shutdown endpoint."""
        import asyncio
        asyncio.get_event_loop().stop()
        return {"status": "shutting_down"}

    # Import main API routers
    try:
        from agentic_research.api.routers import (
            papers_router,
            search_router,
            chat_router,
            # research_router,  # Requires RabbitMQ - use embedded version
        )

        app.include_router(papers_router, prefix="/papers", tags=["Papers"])
        app.include_router(search_router, prefix="/search", tags=["Search"])
        app.include_router(chat_router, prefix="/chat", tags=["Chat"])

    except ImportError as e:
        logger.warning(f"Could not import some routers: {e}")

    # Embedded-specific routes
    register_embedded_routes(app)


def register_embedded_routes(app: FastAPI):
    """Register routes specific to embedded mode."""

    @app.get("/config")
    async def get_config():
        """Get current configuration."""
        config = app.state.config
        return {
            "data_dir": str(config.data_dir),
            "embedding_model": config.embedding_model,
            "llm_model": config.llm_model,
            "features": {
                "knowledge_graph": config.enable_knowledge_graph,
                "bm25_search": config.enable_bm25_search,
            }
        }

    @app.get("/stats")
    async def get_stats():
        """Get storage and service statistics."""
        services = app.state.services
        return {
            "storage": services.storage.get_storage_stats(),
            "vectors": services.vectors.get_stats(),
            "cache": services.cache.get_stats(),
            "queue": services.queue.get_stats(),
        }

    @app.post("/papers/upload")
    async def upload_paper(file: bytes, filename: str = "paper.pdf"):
        """Upload a paper (simplified for embedded mode)."""
        from uuid import uuid4

        services = app.state.services
        paper_id = str(uuid4())[:8]

        # Store PDF
        services.storage.upload_pdf(paper_id, file)

        # Store basic metadata
        services.storage.upload_metadata(paper_id, {
            "filename": filename,
            "paper_id": paper_id,
        })

        return {"paper_id": paper_id, "status": "uploaded"}

    @app.post("/papers/export")
    async def export_papers(path: str):
        """Export all papers to a directory."""
        services = app.state.services
        count = services.storage.export_to_directory(path)
        return {"count": count, "path": path}

    @app.get("/papers")
    async def list_papers():
        """List all papers."""
        services = app.state.services
        paper_ids = services.storage.list_papers()

        papers = []
        for paper_id in paper_ids:
            metadata = services.storage.download_metadata(paper_id)
            papers.append({
                "paper_id": paper_id,
                "metadata": metadata or {}
            })

        return {"papers": papers, "count": len(papers)}


# Create the app instance
app = create_embedded_app()


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Praval Deep Research Backend")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()

    logger.info(f"Starting Praval backend on {args.host}:{args.port}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level
    )
