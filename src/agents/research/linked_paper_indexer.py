"""
Linked Paper Indexer Agent - Research Domain.

I am a linked paper indexing specialist who fetches cited papers from arXiv,
downloads their PDFs, generates embeddings, and stores them in the linked_papers
collection for broader context in Q&A.

This agent is part of the context engineering pipeline:
document_processor -> paper_summarizer -> citation_extractor -> linked_paper_indexer

Unlike regular paper indexing, linked papers are:
- Fully indexed (chunks + embeddings) same as KB papers
- Stored in a separate collection (linked_papers)
- Tracked with source_paper_id to know which KB paper cited them
- Also summarized for fast retrieval
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timezone
from praval import agent, chat, broadcast
from praval import Spore

from agentic_research.core.config import get_settings
from agentic_research.storage.minio_client import MinIOClient
from agentic_research.storage.qdrant_client import QdrantClientWrapper, CollectionType
from agentic_research.storage.embeddings import EmbeddingsGenerator
from processors.pdf_processor import PDFProcessor, PDFProcessingError
from processors.arxiv_client import ArXivClient

# Import SSE broadcast helper for real-time notifications
try:
    from agentic_research.api.routes.sse import broadcast_agent_event_sync
except ImportError:
    # Fallback if SSE module not available
    def broadcast_agent_event_sync(event: dict) -> None:
        pass


logger = logging.getLogger(__name__)
settings = get_settings()


def _generate_linked_paper_summary(
    paper: Dict[str, Any],
    source_paper_id: str
) -> Dict[str, Any]:
    """Generate a structured summary for a linked paper."""
    title = paper.get("title", "Unknown")
    abstract = paper.get("abstract", "")
    authors = paper.get("authors", [])

    summary_prompt = f"""
Analyze this research paper (cited by another paper) and generate a concise summary:

Title: {title}
Authors: {', '.join(authors[:3])}
Abstract: {abstract[:1500]}

Generate:
1. ONE_LINE: A single sentence summary (max 80 chars)
2. KEY_INSIGHT: The main contribution or finding
3. DOMAINS: 2-3 research domains (comma-separated)

Format:
ONE_LINE: [summary]
KEY_INSIGHT: [main insight]
DOMAINS: [domain1], [domain2]
"""

    response = chat(summary_prompt)

    summary = {
        "title": title,
        "one_line": title[:80],
        "key_insight": "",
        "domains": [],
        "source_paper_id": source_paper_id
    }

    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("ONE_LINE:"):
            summary["one_line"] = line.replace("ONE_LINE:", "").strip()[:80]
        elif line.startswith("KEY_INSIGHT:"):
            summary["key_insight"] = line.replace("KEY_INSIGHT:", "").strip()
        elif line.startswith("DOMAINS:"):
            domains_str = line.replace("DOMAINS:", "").strip()
            summary["domains"] = [d.strip() for d in domains_str.split(",") if d.strip()]

    return summary


@agent("linked_paper_indexer", channel="broadcast", responds_to=["citations_extracted"], memory=True)
def linked_paper_indexer_agent(spore: Spore) -> None:
    """
    I am a linked paper indexing specialist who fetches cited papers from arXiv,
    performs full indexing (PDF download, chunking, embedding generation), and
    stores them in the linked_papers collection for broader Q&A context.

    My expertise:
    - Fetching papers from arXiv by ID
    - Full PDF processing and indexing
    - Linked paper collection management
    - Summary generation for linked papers
    - Context expansion for knowledge base

    I respond to 'citations_extracted' events and broadcast 'linked_papers_indexed'
    when complete.
    """
    logger.info("=" * 80)
    logger.info("📚 LINKED PAPER INDEXER AGENT TRIGGERED!")
    logger.info("=" * 80)

    citation_candidates = spore.knowledge.get("citation_candidates", [])
    source_papers = spore.knowledge.get("source_papers", [])
    query_context = spore.knowledge.get("original_query", "")

    logger.info(f"📊 Received {len(citation_candidates)} citation candidates")
    logger.info(f"   Source papers: {len(source_papers)}")
    logger.info(f"   Query context: {query_context}")

    if not citation_candidates:
        logger.warning("⚠️ No citation candidates to index - EXITING")
        return

    # Emit SSE event for linked paper indexing start
    broadcast_agent_event_sync({
        "event_type": "indexing_progress",
        "stage": "indexing_linked",
        "current": 0,
        "total": len(citation_candidates),
        "current_paper": "Starting linked paper indexing..."
    })

    # Initialize storage and processing clients
    try:
        minio_client = MinIOClient(settings)
        qdrant_client = QdrantClientWrapper(settings)
        embeddings_gen = EmbeddingsGenerator(settings)
        pdf_processor = PDFProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            max_chunks_per_paper=settings.MAX_CHUNKS_PER_PAPER
        )
        arxiv_client = ArXivClient()

        # Ensure collections exist
        init_results = qdrant_client.initialize_context_engineering_collections(
            vector_size=settings.EMBEDDING_DIMENSIONS
        )
        logger.info(f"📦 Collection initialization: {init_results}")

    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        broadcast({
            "type": "linked_indexing_error",
            "knowledge": {
                "error": f"Client initialization failed: {str(e)}",
                "candidates_count": len(citation_candidates)
            }
        })
        return

    # Remember indexing session
    linked_paper_indexer_agent.remember(
        f"indexing_session: {len(citation_candidates)} candidates from '{query_context}'",
        importance=0.7
    )

    # Recall past indexing patterns
    past_patterns = linked_paper_indexer_agent.recall("indexing_patterns", limit=5)

    # Check which papers are already indexed
    already_indexed = set()
    # TODO: Could query linked_papers collection to check existing papers

    indexing_stats = {
        "total_candidates": len(citation_candidates),
        "papers_fetched": 0,
        "papers_indexed": 0,
        "summaries_stored": 0,
        "vectors_stored": 0,
        "skipped_already_indexed": 0,
        "errors": []
    }

    indexed_papers = []

    for i, candidate in enumerate(citation_candidates, 1):
        arxiv_id = candidate.get("arxiv_id")
        source_paper_id = candidate.get("source_paper_id")
        citation_title = candidate.get("title", "Unknown")

        if not arxiv_id:
            logger.warning(f"Skipping candidate without arXiv ID: {citation_title}")
            continue

        if arxiv_id in already_indexed:
            logger.debug(f"Skipping already indexed: {arxiv_id}")
            indexing_stats["skipped_already_indexed"] += 1
            continue

        logger.info(f"📚 Indexing linked paper {i}/{len(citation_candidates)}: {arxiv_id}")

        # Emit SSE progress event
        broadcast_agent_event_sync({
            "event_type": "indexing_progress",
            "stage": "indexing_linked",
            "current": i - 1,
            "total": len(citation_candidates),
            "current_paper": citation_title[:80] if citation_title else arxiv_id
        })

        try:
            # STEP 1: Fetch paper metadata from arXiv
            papers = arxiv_client.search(arxiv_id, max_results=1)
            if not papers:
                logger.warning(f"Paper not found on arXiv: {arxiv_id}")
                continue

            paper = papers[0]
            title = paper.get("title", citation_title)
            indexing_stats["papers_fetched"] += 1

            logger.info(f"   Title: {title[:60]}...")

            # STEP 2: Download and process PDF
            chunks_with_embeddings = []

            try:
                # Download PDF
                logger.debug(f"Downloading PDF for {arxiv_id}...")
                pdf_data = pdf_processor.download_from_arxiv_sync(
                    arxiv_id,
                    max_retries=2  # Fewer retries for linked papers
                )

                # Upload to MinIO (with linked_ prefix to distinguish)
                pdf_path = minio_client.upload_pdf(f"linked_{arxiv_id}", pdf_data)

                # Extract text
                extracted_text = pdf_processor.extract_text(
                    pdf_data,
                    method=settings.PDF_EXTRACTION_METHOD
                )

                # Store extracted text
                minio_client.upload_extracted_text(f"linked_{arxiv_id}", extracted_text)

                # Chunk text
                chunk_metadata = {
                    "title": title,
                    "authors": paper.get("authors", []),
                    "abstract": paper.get("abstract", ""),
                    "categories": paper.get("categories", []),
                    "published_date": paper.get("published_date", ""),
                    "pdf_path": pdf_path,
                    "source_paper_id": source_paper_id,
                    "is_linked_paper": True,
                    "processing_timestamp": datetime.now(timezone.utc).isoformat()
                }

                chunks = pdf_processor.chunk_text(extracted_text, chunk_metadata)

                # Generate embeddings
                chunks_with_embeddings = embeddings_gen.generate_embeddings_with_metadata(chunks)

                logger.info(f"   Generated {len(chunks_with_embeddings)} chunks with embeddings")

            except PDFProcessingError as e:
                logger.warning(f"PDF processing failed for {arxiv_id}, using abstract: {e}")

            except Exception as e:
                logger.warning(f"Unexpected error processing PDF for {arxiv_id}: {e}")

            # STEP 3: Fallback to abstract if PDF failed
            if not chunks_with_embeddings and paper.get("abstract"):
                logger.debug(f"Using abstract-only for linked paper {arxiv_id}")

                abstract_chunk = {
                    "chunk_text": f"Title: {title}\n\nAbstract: {paper.get('abstract', '')}",
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "title": title,
                    "authors": paper.get("authors", []),
                    "abstract": paper.get("abstract", ""),
                    "categories": paper.get("categories", []),
                    "published_date": paper.get("published_date", ""),
                    "pdf_path": None,
                    "source_paper_id": source_paper_id,
                    "is_linked_paper": True,
                    "processing_timestamp": datetime.now(timezone.utc).isoformat()
                }

                embedding = embeddings_gen.generate_embedding(abstract_chunk["chunk_text"])
                abstract_chunk["embedding"] = embedding
                chunks_with_embeddings = [abstract_chunk]

            # STEP 4: Store vectors in linked_papers collection
            if chunks_with_embeddings:
                vectors_added = qdrant_client.add_linked_paper_vectors(
                    paper_id=arxiv_id,
                    chunks=chunks_with_embeddings,
                    source_paper_id=source_paper_id
                )
                indexing_stats["vectors_stored"] += vectors_added
                indexing_stats["papers_indexed"] += 1

                logger.info(f"   ✅ Stored {vectors_added} vectors in linked_papers collection")

            # STEP 5: Generate and store summary
            try:
                summary = _generate_linked_paper_summary(paper, source_paper_id)

                # Create summary embedding
                summary_text = f"Title: {summary['title']}\nSummary: {summary['one_line']}\nInsight: {summary['key_insight']}"
                summary_embedding = embeddings_gen.generate_embedding(summary_text)

                # Store in paper_summaries collection
                summary_success = qdrant_client.add_paper_summary(
                    paper_id=arxiv_id,
                    summary_data={
                        **summary,
                        "is_linked_paper": True,
                        "source_paper_id": source_paper_id
                    },
                    embedding=summary_embedding
                )

                if summary_success:
                    indexing_stats["summaries_stored"] += 1
                    logger.info(f"   ✅ Summary stored for linked paper {arxiv_id}")

            except Exception as e:
                logger.warning(f"Failed to generate summary for {arxiv_id}: {e}")

            # Track indexed paper
            indexed_papers.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "source_paper_id": source_paper_id,
                "vectors_stored": len(chunks_with_embeddings),
                "has_summary": indexing_stats["summaries_stored"] > 0
            })

            already_indexed.add(arxiv_id)

            # Emit SSE event for linked paper indexed
            broadcast_agent_event_sync({
                "event_type": "linked_paper_indexed",
                "title": title,
                "arxiv_id": arxiv_id,
                "source_paper_id": source_paper_id,
                "vectors_stored": len(chunks_with_embeddings)
            })

            # Remember successful pattern
            linked_paper_indexer_agent.remember(
                f"indexing_patterns: linked paper {arxiv_id} -> {len(chunks_with_embeddings)} vectors",
                importance=0.5
            )

        except Exception as e:
            logger.error(f"Failed to index linked paper {arxiv_id}: {e}")
            indexing_stats["errors"].append(f"{arxiv_id}: {str(e)}")

    logger.info("=" * 40)
    logger.info(f"✅ Linked paper indexing complete:")
    logger.info(f"   Papers fetched: {indexing_stats['papers_fetched']}/{indexing_stats['total_candidates']}")
    logger.info(f"   Papers indexed: {indexing_stats['papers_indexed']}")
    logger.info(f"   Summaries stored: {indexing_stats['summaries_stored']}")
    logger.info(f"   Vectors stored: {indexing_stats['vectors_stored']}")
    logger.info("=" * 40)

    # Remember session summary
    linked_paper_indexer_agent.remember(
        f"session_complete: {indexing_stats['papers_indexed']} linked papers, {indexing_stats['vectors_stored']} vectors",
        importance=0.8
    )

    # Broadcast completion
    broadcast_payload = {
        "type": "linked_papers_indexed",
        "knowledge": {
            "indexed_papers": indexed_papers,
            "source_papers": source_papers,
            "original_query": query_context,
            "indexing_stats": indexing_stats,
            "processing_metadata": {
                "agent": "linked_paper_indexer",
                "session_timestamp": datetime.now(timezone.utc).isoformat(),
                "memory_contexts_used": len(past_patterns)
            }
        }
    }

    logger.info(f"📡 BROADCASTING linked_papers_indexed event")
    logger.info(f"   Indexed papers: {len(indexed_papers)}")

    broadcast_result = broadcast(broadcast_payload)
    logger.info(f"✅ BROADCAST COMPLETE - Result: {broadcast_result}")
    logger.info("=" * 80)


# Agent metadata for system introspection
AGENT_METADATA = {
    "identity": "linked paper indexing specialist",
    "domain": "research",
    "capabilities": [
        "ArXiv paper fetching",
        "Full PDF processing and indexing",
        "Linked papers collection management",
        "Summary generation for cited papers",
        "Context expansion for knowledge base"
    ],
    "responds_to": ["citations_extracted"],
    "broadcasts": ["linked_papers_indexed", "linked_indexing_error"],
    "memory_enabled": True,
    "learning_focus": "linked paper indexing patterns and citation relationships",
    "storage_integrations": ["MinIO", "Qdrant (linked_papers collection)", "OpenAI Embeddings"]
}
