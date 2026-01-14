#!/usr/bin/env python3
"""
Initialize Vajra BM25 indexes from existing data.

This script builds BM25 indexes from:
1. Paper chunks stored in Qdrant vector database
2. Conversation history stored in PostgreSQL

Run this at application startup to ensure indexes are ready for search.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog
from typing import Dict, Any, List

logger = structlog.get_logger(__name__)


async def init_paper_index() -> int:
    """
    Initialize paper index from Qdrant vectors.

    Returns:
        Number of papers indexed
    """
    from agentic_research.core.config import get_settings
    from agentic_research.storage.qdrant_client import QdrantClientWrapper
    from agentic_research.storage.paper_index import get_paper_index

    settings = get_settings()
    paper_index = get_paper_index()

    # Check if index already has data
    existing_count = paper_index.get_paper_count()
    if existing_count > 0:
        logger.info(
            "Paper index already populated",
            paper_count=existing_count,
            chunk_count=paper_index.document_count,
        )
        return existing_count

    logger.info("Building paper index from Qdrant...")

    try:
        # Connect to Qdrant
        qdrant = QdrantClientWrapper()

        # Get all papers from Qdrant
        papers = qdrant.get_all_papers()

        if not papers:
            logger.info("No papers found in Qdrant, skipping paper indexing")
            return 0

        indexed_count = 0

        for paper in papers:
            paper_id = paper.get("paper_id")
            if not paper_id:
                continue

            # Get all chunks for this paper
            chunks = qdrant.get_paper_chunks(paper_id)

            if not chunks:
                continue

            # Extract chunk texts and metadata from payload
            chunk_texts = []
            metadata = {}

            for chunk in chunks:
                # Qdrant returns chunks with 'payload' containing the data
                payload = chunk.get("payload", {})
                content = payload.get("chunk_text", "")
                if content and content.strip():
                    chunk_texts.append(content)

                # Get metadata from first chunk's payload
                if not metadata and payload:
                    metadata = {
                        "title": payload.get("title", ""),
                        "authors": payload.get("authors", []),
                        "categories": payload.get("categories", []),
                        "abstract": payload.get("abstract", ""),
                        "published_date": payload.get("published_date", ""),
                    }

            if chunk_texts:
                paper_index.index_paper(
                    paper_id=paper_id,
                    title=metadata.get("title", ""),
                    chunks=chunk_texts,
                    authors=metadata.get("authors"),
                    categories=metadata.get("categories"),
                    abstract=metadata.get("abstract"),
                    published_date=metadata.get("published_date"),
                )
                indexed_count += 1

        # Build the BM25 index from added documents
        if indexed_count > 0:
            paper_index.rebuild_index()

        # Save index to disk
        paper_index.save_index()

        logger.info(
            "Paper index built successfully",
            papers_indexed=indexed_count,
            total_chunks=paper_index.document_count,
        )

        return indexed_count

    except Exception as e:
        logger.error("Failed to build paper index", error=str(e))
        return 0


async def init_conversation_index() -> int:
    """
    Initialize conversation index from PostgreSQL.

    Returns:
        Number of messages indexed
    """
    from agentic_research.core.config import get_settings
    from agentic_research.storage.conversation_index import get_conversation_index

    settings = get_settings()
    conv_index = get_conversation_index()

    # Check if index already has data
    existing_count = conv_index.document_count
    if existing_count > 0:
        logger.info(
            "Conversation index already populated",
            message_count=existing_count,
        )
        return existing_count

    logger.info("Building conversation index from PostgreSQL...")

    try:
        # Import async database components
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import text

        # Create async engine
        engine = create_async_engine(settings.DATABASE_URL)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as session:
            # Query all messages
            result = await session.execute(
                text("""
                    SELECT
                        m.id,
                        m.conversation_id,
                        m.role,
                        m.content,
                        m.created_at,
                        c.user_id
                    FROM messages m
                    JOIN conversations c ON m.conversation_id = c.id
                    ORDER BY m.created_at ASC
                """)
            )

            messages = result.fetchall()

            if not messages:
                logger.info("No messages found in PostgreSQL, skipping conversation indexing")
                return 0

            indexed_count = 0

            for msg in messages:
                msg_id, conv_id, role, content, created_at, user_id = msg

                if not content or not content.strip():
                    continue

                conv_index.index_message(
                    message_id=str(msg_id),
                    conversation_id=str(conv_id),
                    role=role,
                    content=content,
                    user_id=str(user_id) if user_id else None,
                    timestamp=created_at.isoformat() if created_at else None,
                )
                indexed_count += 1

            # Save index to disk
            conv_index.save_index()

            logger.info(
                "Conversation index built successfully",
                messages_indexed=indexed_count,
            )

            return indexed_count

    except Exception as e:
        logger.error("Failed to build conversation index", error=str(e))
        return 0
    finally:
        await engine.dispose()


async def main():
    """Main entry point for index initialization."""
    logger.info("Starting Vajra BM25 index initialization...")

    # Ensure index directory exists
    from agentic_research.core.config import get_settings
    settings = get_settings()

    index_path = Path(settings.BM25_INDEX_PATH)
    index_path.mkdir(parents=True, exist_ok=True)

    logger.info("Index directory ready", path=str(index_path))

    # Initialize indexes
    paper_count = await init_paper_index()
    message_count = await init_conversation_index()

    logger.info(
        "Vajra BM25 index initialization complete",
        papers=paper_count,
        messages=message_count,
    )

    return paper_count, message_count


if __name__ == "__main__":
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    import logging
    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
