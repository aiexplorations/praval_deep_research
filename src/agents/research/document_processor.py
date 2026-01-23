"""
Document Processing Agent - Research Domain.

I am a document processing specialist who handles downloading, parsing,
and storing research papers with intelligent extraction and metadata
organization.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timezone
from praval import agent, chat, broadcast
from praval import Spore

from agentic_research.core.config import get_settings
from agentic_research.storage.minio_client import MinIOClient
from agentic_research.storage.qdrant_client import QdrantClientWrapper
from agentic_research.storage.embeddings import EmbeddingsGenerator
from processors.pdf_processor import PDFProcessor, PDFProcessingError

# LangExtract integration for structured entity extraction
try:
    from agentic_research.extraction import LangExtractProcessor
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    LangExtractProcessor = None

# Import SSE broadcast helper for real-time notifications
try:
    from agentic_research.api.routes.sse import broadcast_agent_event_sync
except ImportError:
    # Fallback if SSE module not available
    def broadcast_agent_event_sync(event: dict) -> None:
        pass


logger = logging.getLogger(__name__)
settings = get_settings()


@agent("document_processor", channel="broadcast", responds_to=["papers_found"], memory=True)
def document_processing_agent(spore: Spore) -> None:
    """
    I am a document processing specialist who handles downloading, parsing,
    and storing research papers with intelligent extraction and metadata
    organization.

    TRIGGERED: spore.knowledge.type == 'papers_found'

    My expertise:
    - PDF download and secure storage in MinIO
    - Intelligent text extraction and parsing
    - Structured metadata extraction with LLM
    - Embedding generation with OpenAI
    - Vector storage in Qdrant for semantic search
    - Quality assessment and validation
    """
    logger.info("=" * 80)
    logger.info("📄 DOCUMENT PROCESSOR AGENT TRIGGERED!")
    logger.info("=" * 80)
    logger.info(f"📥 Received spore: {spore}")
    logger.info(f"📦 Spore knowledge keys: {list(spore.knowledge.keys())}")

    papers = spore.knowledge.get("papers", [])
    query = spore.knowledge.get("original_query", "")
    search_metadata = spore.knowledge.get("search_metadata", {})

    logger.info(f"📊 Extracted from spore:")
    logger.info(f"   Papers count: {len(papers)}")
    logger.info(f"   Original query: {query}")
    logger.info(f"   Search metadata: {search_metadata}")

    if not papers:
        logger.warning("⚠️ Document processor received no papers - EXITING")
        return

    logger.info(f"✅ STARTING Document Processing: {len(papers)} papers for query '{query}'")

    # Emit SSE event for indexing start
    broadcast_agent_event_sync({
        "event_type": "indexing_progress",
        "stage": "starting",
        "current": 0,
        "total": len(papers),
        "current_paper": None
    })

    # Initialize storage clients
    try:
        logger.info("Initializing MinIO client...")
        minio_client = MinIOClient(settings)
        logger.info("Initializing Qdrant client...")
        qdrant_client = QdrantClientWrapper(settings)
        logger.info("Initializing embeddings generator...")
        embeddings_gen = EmbeddingsGenerator(settings)
        logger.info("Initializing PDF processor...")
        pdf_processor = PDFProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            max_chunks_per_paper=settings.MAX_CHUNKS_PER_PAPER
        )

        # Ensure Qdrant collection exists
        logger.info("Creating/verifying Qdrant collection...")
        qdrant_client.create_collection(
            vector_size=settings.EMBEDDING_DIMENSIONS,
            recreate=False
        )

        # Initialize LangExtract processor for structured extraction
        langextract_processor = None
        if LANGEXTRACT_AVAILABLE and settings.LANGEXTRACT_ENABLED:
            try:
                langextract_processor = LangExtractProcessor()
                logger.info("LangExtract processor initialized for structured extraction")
            except Exception as le_error:
                logger.warning(f"LangExtract initialization failed, skipping structured extraction: {le_error}")
        else:
            logger.info("LangExtract disabled or unavailable, skipping structured extraction")

        logger.info("Storage clients initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize storage clients: {e}", exc_info=True)
        broadcast({
            "type": "processing_error",
            "knowledge": {
                "error": f"Storage initialization failed: {str(e)}",
                "papers_count": len(papers)
            }
        })
        return

    # Remember processing session with context
    processing_info = f"Processing session: {len(papers)} papers for '{query}'"
    document_processing_agent.remember(processing_info, importance=0.7)

    # Leverage memory for processing patterns
    past_processing = document_processing_agent.recall("processing_patterns", limit=5)
    domain_experience = document_processing_agent.recall(f"domain_processing", limit=3)

    processed_papers = []
    processing_stats = {
        "total_papers": len(papers),
        "successful": 0,
        "failed": 0,
        "pdf_downloaded": 0,
        "vectors_stored": 0,
        "extractions_generated": 0,
        "errors": []
    }

    for i, paper in enumerate(papers, 1):
        paper_title = paper.get('title', 'Untitled')
        arxiv_id = paper.get('arxiv_id', '')

        # DEBUG: Log full paper content to diagnose 'unknown' issue
        logger.info(f"🔍 Processing Paper Data: {paper}")
        logger.info(f"   Title: '{paper_title}'")
        logger.info(f"   ID: '{arxiv_id}'")
        logger.info(f"   Authors: {paper.get('authors')}")

        logger.info(f"⚙️ Processing {i}/{len(papers)}: {paper_title[:60]}...")

        # Emit SSE progress event
        broadcast_agent_event_sync({
            "event_type": "indexing_progress",
            "stage": "processing",
            "current": i - 1,
            "total": len(papers),
            "current_paper": paper_title[:80]
        })

        try:
            # STEP 1: Enhanced LLM Analysis (optional - should not block indexing)
            analysis = None
            extracted_metadata = None

            try:
                analysis_prompt = f"""
                Analyze this research paper briefly:

                Title: {paper.get('title', '')}
                Abstract: {paper.get('abstract', '')[:500]}

                Provide a 2-3 sentence analysis of key contributions.
                """

                analysis = chat(analysis_prompt)

                # STEP 2: Extract key metadata (optional)
                metadata_prompt = f"""
                Extract 3-5 key themes from: {paper.get('title', '')}
                """

                extracted_metadata = chat(metadata_prompt)

            except Exception as llm_error:
                # LLM analysis is optional - log and continue with indexing
                logger.warning(f"LLM analysis skipped for '{paper_title[:50]}': {str(llm_error)[:100]}")
                analysis = "LLM analysis unavailable"
                extracted_metadata = "Metadata extraction skipped"

            # STEP 3: Download and process PDF (CORE FUNCTIONALITY - must work)
            pdf_path = None
            extracted_text = None
            chunks_with_embeddings = []

            if arxiv_id:
                try:
                    # Download PDF from ArXiv using synchronous method
                    # (agent runs in sync context inside async event loop)
                    logger.debug(f"Downloading PDF for {arxiv_id}...")
                    pdf_data = pdf_processor.download_from_arxiv_sync(
                        arxiv_id,
                        max_retries=settings.PDF_MAX_RETRIES
                    )
                    processing_stats["pdf_downloaded"] += 1

                    # Upload to MinIO
                    logger.debug(f"Uploading PDF to MinIO for {arxiv_id}...")
                    pdf_path = minio_client.upload_pdf(arxiv_id, pdf_data)

                    # Extract text
                    logger.debug(f"Extracting text from PDF for {arxiv_id}...")
                    extracted_text = pdf_processor.extract_text(
                        pdf_data,
                        method=settings.PDF_EXTRACTION_METHOD
                    )

                    # Store extracted text in MinIO
                    minio_client.upload_extracted_text(arxiv_id, extracted_text)

                    # STEP 4: Chunk text
                    logger.debug(f"Chunking text for {arxiv_id}...")
                    chunk_metadata = {
                        "title": paper.get('title', ''),
                        "authors": paper.get('authors', []),
                        "abstract": paper.get('abstract', ''),
                        "categories": paper.get('categories', []),
                        "published_date": paper.get('published_date', ''),
                        "pdf_path": pdf_path,
                        "search_context": query,
                        "processing_timestamp": datetime.now(timezone.utc).isoformat()
                    }

                    chunks = pdf_processor.chunk_text(extracted_text, chunk_metadata)

                    # STEP 5: Generate embeddings
                    logger.debug(f"Generating embeddings for {len(chunks)} chunks...")
                    chunks_with_embeddings = embeddings_gen.generate_embeddings_with_metadata(chunks)

                    # STEP 6: Store vectors in Qdrant
                    logger.debug(f"Storing {len(chunks_with_embeddings)} vectors in Qdrant...")
                    vectors_added = qdrant_client.add_vectors(
                        paper_id=arxiv_id,
                        chunks=chunks_with_embeddings
                    )
                    processing_stats["vectors_stored"] += vectors_added

                    logger.info(f"✅ PDF pipeline complete for {arxiv_id}: {vectors_added} vectors stored")

                    # STEP 7: LangExtract structured extraction (optional)
                    if langextract_processor and extracted_text:
                        try:
                            logger.debug(f"Running LangExtract for {arxiv_id}...")
                            paper_extractions = langextract_processor.extract_from_paper(
                                text=extracted_text,
                                paper_id=arxiv_id,
                                title=paper.get('title', '')
                            )

                            if paper_extractions.extractions:
                                # Generate embeddings for extractions
                                extraction_texts = [
                                    f"{e.name}: {e.content}"
                                    for e in paper_extractions.extractions
                                ]
                                extraction_embeddings = embeddings_gen.generate_embeddings_batch(
                                    extraction_texts
                                )

                                # Store extractions in Qdrant
                                extraction_dicts = [
                                    e.to_dict() for e in paper_extractions.extractions
                                ]
                                extractions_added = qdrant_client.add_extractions(
                                    paper_id=arxiv_id,
                                    extractions=extraction_dicts,
                                    embeddings=extraction_embeddings
                                )
                                processing_stats["extractions_generated"] += extractions_added

                                logger.info(
                                    f"✅ LangExtract complete for {arxiv_id}: "
                                    f"{len(paper_extractions.extractions)} entities extracted "
                                    f"({paper_extractions.to_dict()['summary']})"
                                )

                                # Emit SSE event for extraction completion
                                broadcast_agent_event_sync({
                                    "event_type": "extractions_complete",
                                    "arxiv_id": arxiv_id,
                                    "extraction_count": len(paper_extractions.extractions),
                                    "summary": paper_extractions.to_dict()['summary']
                                })

                        except Exception as le_error:
                            logger.warning(
                                f"LangExtract failed for {arxiv_id}, continuing: {le_error}"
                            )

                except PDFProcessingError as e:
                    logger.warning(f"PDF processing failed for {arxiv_id}, using abstract only: {e}")
                    # Continue with abstract-only processing (graceful degradation)

                except Exception as e:
                    logger.error(f"Unexpected error in PDF pipeline for {arxiv_id}: {e}")
                    # Continue with abstract-only processing

            # STEP 7: Fallback - If no PDF or PDF failed, embed abstract
            if not chunks_with_embeddings and paper.get('abstract'):
                try:
                    logger.debug(f"Using abstract-only embedding for {arxiv_id or paper_title}")

                    abstract_chunk = {
                        "chunk_text": f"Title: {paper.get('title', '')}\\n\\nAbstract: {paper.get('abstract', '')}",
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "title": paper.get('title', ''),
                        "authors": paper.get('authors', []),
                        "abstract": paper.get('abstract', ''),
                        "categories": paper.get('categories', []),
                        "published_date": paper.get('published_date', ''),
                        "pdf_path": None,
                        "search_context": query,
                        "processing_timestamp": datetime.now(timezone.utc).isoformat()
                    }

                    # Generate embedding for abstract
                    embedding = embeddings_gen.generate_embedding(abstract_chunk["chunk_text"])
                    abstract_chunk["embedding"] = embedding

                    # Store in Qdrant (use title hash as ID if no arxiv_id)
                    paper_id = arxiv_id or f"paper_{hash(paper.get('title', ''))}"
                    vectors_added = qdrant_client.add_vectors(
                        paper_id=paper_id,
                        chunks=[abstract_chunk]
                    )
                    processing_stats["vectors_stored"] += vectors_added

                    logger.info(f"✅ Abstract-only embedding stored for {paper_id}")

                except Exception as e:
                    logger.error(f"Failed to embed abstract for {paper_title}: {e}")

            # Create processed paper object
            processed_paper = {
                **paper,  # Original paper data
                "processing": {
                    "analysis": analysis,
                    "extracted_metadata": extracted_metadata,
                    "pdf_path": pdf_path,
                    "text_extracted": extracted_text is not None,
                    "chunks_generated": len(chunks_with_embeddings),
                    "vectors_stored": len(chunks_with_embeddings) > 0,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "processing_agent": "document_processor",
                    "query_context": query,
                    "relevance_score": paper.get('relevance', 0.0)
                }
            }

            processed_papers.append(processed_paper)
            processing_stats["successful"] += 1

            # Emit SSE event for successful paper indexing
            broadcast_agent_event_sync({
                "event_type": "paper_indexed",
                "title": paper_title,
                "arxiv_id": arxiv_id,
                "vectors_stored": len(chunks_with_embeddings)
            })

            # Remember successful processing patterns
            success_pattern = f"processing_patterns: Successfully processed {paper.get('categories', ['unknown'])[0] if paper.get('categories') else 'unknown'} paper with {len(chunks_with_embeddings)} vectors"
            document_processing_agent.remember(success_pattern, importance=0.6)

        except Exception as e:
            error_message = str(e)
            logger.error(f"Failed to process paper '{paper_title}': {error_message}")
            processing_stats["failed"] += 1
            processing_stats["errors"].append(f"{paper_title}: {error_message}")

            # Emit SSE error event so frontend knows about the failure
            broadcast_agent_event_sync({
                "event_type": "indexing_error",
                "title": paper_title[:80],
                "arxiv_id": arxiv_id,
                "error": error_message[:200],
                "paper_index": i,
                "total_papers": len(papers)
            })

            # Remember processing failures for learning
            error_pattern = f"processing_error: {paper.get('categories', ['unknown'])[0] if paper.get('categories') else 'unknown'} - {error_message[:100]}"
            document_processing_agent.remember(error_pattern, importance=0.4)
            continue

    logger.info(f"✅ Document processing complete: {processing_stats['successful']}/{processing_stats['total_papers']} successful, {processing_stats['vectors_stored']} vectors stored")

    # Emit SSE event for indexing completion
    broadcast_agent_event_sync({
        "event_type": "indexing_complete",
        "papers_indexed": processing_stats["successful"],
        "vectors_stored": processing_stats["vectors_stored"],
        "total": processing_stats["total_papers"],
        "failed": processing_stats["failed"],
        "errors": processing_stats["errors"][:5] if processing_stats["errors"] else []  # Send first 5 errors
    })

    # Remember overall processing session
    session_summary = f"domain_processing: {search_metadata.get('domain', 'unknown')} - {processing_stats['successful']}/{processing_stats['total_papers']} papers, {processing_stats['vectors_stored']} vectors stored"
    document_processing_agent.remember(session_summary, importance=0.8)

    broadcast_payload = {
        "type": "documents_processed",
        "knowledge": {
            "processed_papers": processed_papers,
            "original_query": query,
            "processing_stats": processing_stats,
            "processing_metadata": {
                "agent": "document_processor",
                "session_timestamp": datetime.now(timezone.utc).isoformat(),
                "memory_contexts_used": len(past_processing) + len(domain_experience),
                "quality_metrics": {
                    "success_rate": processing_stats["successful"] / processing_stats["total_papers"] if processing_stats["total_papers"] > 0 else 0,
                    "error_count": processing_stats["failed"],
                    "pdf_success_rate": processing_stats["pdf_downloaded"] / processing_stats["total_papers"] if processing_stats["total_papers"] > 0 else 0,
                    "vectors_per_paper": processing_stats["vectors_stored"] / processing_stats["successful"] if processing_stats["successful"] > 0 else 0
                }
            }
        }
    }

    logger.info(f"📡 BROADCASTING documents_processed event")
    logger.info(f"   Processed papers: {len(processed_papers)}")
    logger.info(f"   Vectors stored: {processing_stats['vectors_stored']}")

    # Broadcast processed papers via spore
    broadcast_result = broadcast(broadcast_payload)

    logger.info(f"✅ BROADCAST COMPLETE - Result: {broadcast_result}")
    logger.info("=" * 80)


# Agent metadata for system introspection
AGENT_METADATA = {
    "identity": "document processing specialist",
    "domain": "research",
    "capabilities": [
        "PDF download from ArXiv",
        "PDF storage in MinIO",
        "Text extraction with pdfplumber/PyPDF2",
        "Intelligent text chunking",
        "Embedding generation with OpenAI",
        "Vector storage in Qdrant",
        "Metadata extraction with LLM",
        "Quality assessment",
        "Structured entity extraction with LangExtract",
        "Knowledge graph entity preparation"
    ],
    "responds_to": ["papers_found"],
    "broadcasts": ["documents_processed", "processing_error"],
    "memory_enabled": True,
    "learning_focus": "processing patterns and domain-specific extraction techniques",
    "storage_integrations": ["MinIO", "Qdrant", "OpenAI Embeddings", "LangExtract"],
    "extraction_types": ["method", "dataset", "finding", "citation", "metric", "limitation"]
}
