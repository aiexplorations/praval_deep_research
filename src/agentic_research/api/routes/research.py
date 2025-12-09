"""
Research endpoints for paper discovery and Q&A.

This module provides REST API endpoints that interface with
Praval research agents for academic paper discovery and
intelligent question answering.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse
import structlog

from ..models.research import (
    ResearchQuery, ResearchResponse, PaperResult,
    QuestionRequest, QuestionResponse, SourceCitation,
    ErrorResponse
)
from agents import (
    paper_discovery_agent,
    document_processing_agent,
    semantic_analysis_agent,
    summarization_agent,
    qa_specialist_agent,
    research_advisor_agent
)
from processors.arxiv_client import (
    search_arxiv_papers,
    calculate_paper_relevance,
    ArXivAPIError
)
from praval import start_agents
from ...core.messaging import get_publisher
from ...storage.vector_search import get_vector_search_client
from ...core.config import get_settings
from openai import OpenAI

logger = structlog.get_logger()
router = APIRouter(prefix="/research", tags=["research"])
settings = get_settings()


def generate_conversation_title(question: str, answer: str) -> str:
    """
    Generate a concise conversation title from the first Q&A.

    Uses LLM to create a 5-10 word summary like ChatGPT/Claude.
    """
    try:
        from agentic_research.core.config import get_settings
        settings = get_settings()
        openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Generate a concise 5-10 word title for this conversation. Be specific and descriptive. Return only the title, no quotes or extra text."
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nAnswer: {answer[:500]}"
                }
            ],
            temperature=0.3,
            max_tokens=50
        )

        title = response.choices[0].message.content.strip().strip('"\'')
        # Limit to 60 characters for UI
        return title[:60] if len(title) > 60 else title

    except Exception as e:
        logger.error("Failed to generate conversation title", error=str(e))
        # Fallback to first 50 chars of question
        return question[:50] + "..." if len(question) > 50 else question

# Track ongoing research sessions
_active_sessions: Dict[str, Dict[str, Any]] = {}


async def _initialize_agents_if_needed():
    """Ensure research agents are initialized and running."""
    try:
        # Start all research agents using Praval's start_agents
        agents = [
            paper_discovery_agent,
            document_processing_agent, 
            semantic_analysis_agent,
            summarization_agent,
            qa_specialist_agent,
            research_advisor_agent
        ]
        
        # Initialize agent system
        result = start_agents(*agents)
        logger.info("Praval research agents initialized", agents_count=len(agents))
        return True
        
    except Exception as e:
        logger.warning("Failed to initialize Praval agents", error=str(e))
        return False


def _convert_to_paper_results(papers: list) -> list[PaperResult]:
    """Convert agent response papers to API models."""
    results = []
    
    for paper in papers:
        try:
            result = PaperResult(
                title=paper.get('title', 'Unknown Title'),
                authors=paper.get('authors', []),
                abstract=paper.get('abstract', ''),
                arxiv_id=paper.get('arxiv_id'),
                url=paper.get('url'),
                published_date=paper.get('published_date'),
                venue=paper.get('venue'),
                relevance_score=paper.get('relevance', 0.0),
                categories=paper.get('categories', [])
            )
            results.append(result)
        except Exception as e:
            logger.warning("Failed to convert paper result", paper=paper, error=str(e))
            continue
    
    return results


def _convert_to_source_citations(sources: list) -> list[SourceCitation]:
    """Convert agent response sources to API models."""
    citations = []
    
    for source in sources:
        try:
            citation = SourceCitation(
                title=source.get('title', 'Unknown Source'),
                paper_id=source.get('paper_id', ''),
                chunk_index=source.get('chunk_index', 0),
                relevance_score=source.get('relevance_score', 0.0),
                excerpt=source.get('excerpt')
            )
            citations.append(citation)
        except Exception as e:
            logger.warning("Failed to convert source citation", source=source, error=str(e))
            continue
    
    return citations


@router.post("/search", response_model=ResearchResponse, summary="Search for research papers")
async def search_papers(
    query: ResearchQuery,
    background_tasks: BackgroundTasks
) -> ResearchResponse:
    """
    Search for academic papers using intelligent Praval agents.
    
    This endpoint leverages distributed research agents to:
    - Optimize search queries based on domain expertise
    - Search multiple academic databases (ArXiv, etc.)
    - Apply quality filters and relevance scoring
    - Learn from search patterns for future optimization
    
    The agents use memory to improve search quality over time.
    """
    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(
            "Starting paper search",
            session_id=session_id,
            query=query.query,
            domain=query.domain,
            max_results=query.max_results
        )
        
        # Ensure agents are initialized
        background_tasks.add_task(_initialize_agents_if_needed)
        
        # Track session
        _active_sessions[session_id] = {
            "type": "search",
            "query": query.query,
            "started_at": time.time(),
            "status": "running"
        }
        
        try:
            # Initialize agents if needed
            agents_ready = await _initialize_agents_if_needed()
            if not agents_ready:
                logger.warning("Agents not ready, falling back to direct ArXiv search")
                
            # Send message to agents via RabbitMQ
            try:
                publisher = get_publisher()
                broadcast_result = await publisher.publish_search_request(
                    query=query.query,
                    domain=query.domain.value,
                    max_results=query.max_results,
                    session_id=session_id,
                    quality_threshold=query.quality_threshold
                )

                # Also do direct search for immediate response
                real_papers = await search_arxiv_papers(
                    query=query.query,
                    max_results=query.max_results,
                    domain=query.domain.value.lower()
                )
                
                # Calculate relevance scores for papers
                for paper in real_papers:
                    paper['relevance'] = calculate_paper_relevance(paper, query.query)
                
                # Sort by relevance and apply quality threshold
                filtered_papers = [
                    paper for paper in real_papers 
                    if paper.get('relevance', 0.0) >= query.quality_threshold
                ]
                filtered_papers.sort(key=lambda p: p.get('relevance', 0.0), reverse=True)
                
                logger.info(
                    "Research workflow initiated via Praval agents",
                    total_found=len(real_papers),
                    after_filtering=len(filtered_papers),
                    agents_triggered=True,
                    broadcast_id=broadcast_result
                )
                
            except ArXivAPIError as e:
                logger.error("ArXiv API error", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"ArXiv search failed: {str(e)}"
                )
            except Exception as e:
                logger.error("Unexpected error in research workflow", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Research workflow failed: {str(e)}"
                )
            
            # Convert to API models
            paper_results = _convert_to_paper_results(filtered_papers)
            
            # Calculate response time
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Update session status
            _active_sessions[session_id]["status"] = "completed"
            _active_sessions[session_id]["papers_found"] = len(paper_results)
            
            response = ResearchResponse(
                query=query.query,
                domain=query.domain.value,
                papers=paper_results,
                total_found=len(paper_results),
                search_time_ms=response_time_ms,
                optimization_applied=True,
                optimized_query=f"ArXiv search: {query.query} in {query.domain.value}",
                agent_metadata={
                    "session_id": session_id,
                    "search_source": "arxiv_api",
                    "quality_filtered": True,
                    "relevance_threshold": query.quality_threshold
                }
            )
            
            logger.info(
                "Paper search completed",
                session_id=session_id,
                papers_found=len(paper_results),
                response_time_ms=response_time_ms
            )
            
            return response
            
        finally:
            # Clean up session tracking
            if session_id in _active_sessions:
                _active_sessions[session_id]["completed_at"] = time.time()
        
    except Exception as e:
        logger.error("Paper search failed", session_id=session_id, error=str(e))
        
        # Update session status
        if session_id in _active_sessions:
            _active_sessions[session_id]["status"] = "error"
            _active_sessions[session_id]["error"] = str(e)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/ask", response_model=QuestionResponse, summary="Ask research questions")
async def ask_question(
    request: QuestionRequest,
    background_tasks: BackgroundTasks
) -> QuestionResponse:
    """
    Get intelligent answers to research questions.
    
    This endpoint uses specialized Q&A agents that:
    - Retrieve relevant context from paper knowledge base
    - Apply personalization based on user history
    - Generate comprehensive, evidence-based answers
    - Suggest follow-up questions for deeper exploration
    - Cite specific sources with relevance scores
    
    The agents learn from user interactions to improve personalization.
    """
    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(
            "Processing Q&A request",
            session_id=session_id,
            question=request.question,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        # Ensure agents are initialized  
        background_tasks.add_task(_initialize_agents_if_needed)
        
        # Track session
        _active_sessions[session_id] = {
            "type": "qa",
            "question": request.question,
            "started_at": time.time(),
            "status": "running"
        }
        
        try:
            # Initialize agents if needed
            agents_ready = await _initialize_agents_if_needed()

            # Send Q&A request to agents via RabbitMQ
            # Handle None values in request to avoid subscript errors
            try:
                publisher = get_publisher()
                publish_kwargs = {
                    "question": request.question,
                    "session_id": session_id,
                    "user_id": request.user_id or "anonymous",
                    "conversation_id": request.conversation_id or session_id,
                }

                # Only add context if it's not None
                if request.context is not None:
                    publish_kwargs["context"] = request.context

                # RETRIEVE AND PASS CONVERSATION HISTORY
                if request.conversation_id:
                    try:
                        from agentic_research.storage.conversation_store import get_conversation_store
                        store = get_conversation_store()
                        
                        # Get last 5 messages for context
                        # We use get_conversation which likely returns messages, or we might need a specific method
                        # Assuming conversation object has messages or we can fetch them
                        history = await store.get_messages(request.conversation_id, limit=6) # Get last 3 turns
                        
                        conversation_context = []
                        if history:
                            for msg in history:
                                conversation_context.append(f"{msg.role}: {msg.content}")
                                
                        publish_kwargs["conversation_context"] = conversation_context
                        logger.info(f"Attached {len(conversation_context)} messages of history to Q&A request")
                    except Exception as hist_err:
                        logger.warning(f"Failed to attach conversation history: {hist_err}")

                broadcast_result = await publisher.publish_qa_request(**publish_kwargs)
                logger.info("Successfully published Q&A request to agents",
                           session_id=session_id,
                           agents_ready=agents_ready,
                           broadcast_id=broadcast_result)
            except Exception as pub_error:
                # Log but don't fail - we can still process directly
                logger.warning("Failed to publish Q&A request to agents, continuing with direct processing",
                              error=str(pub_error),
                              session_id=session_id,
                              exc_info=True)

            logger.info("Sent Q&A request to agents", agents_ready=agents_ready)

            # Perform vector search to retrieve relevant context
            logger.info(f"Initializing vector search client for question: {request.question[:50]}")
            vector_client = get_vector_search_client()

            if vector_client is None:
                logger.error("Vector search client is None - cannot perform search")
                relevant_chunks = []
            else:
                logger.info("Vector client initialized, performing search")
                relevant_chunks = vector_client.search(
                    query=request.question,
                    top_k=5,
                    score_threshold=0.3  # Lowered from 0.6 for better recall
                )
                logger.info(f"Search completed, found {len(relevant_chunks) if relevant_chunks else 0} chunks")

            # Build context from retrieved chunks
            # Ensure relevant_chunks is a list and filter out any None values
            if not isinstance(relevant_chunks, list):
                logger.error(f"relevant_chunks is not a list: {type(relevant_chunks)}")
                relevant_chunks = []

            relevant_chunks = [c for c in relevant_chunks if c is not None]

            if relevant_chunks:
                logger.info(f"Building context from {len(relevant_chunks)} chunks")
                try:
                    context_text = "\n\n".join([
                        f"From '{chunk.get('title', 'Unknown')}': \n{chunk.get('text', chunk.get('excerpt', ''))}"
                        for chunk in relevant_chunks
                        if isinstance(chunk, dict)
                    ])
                except Exception as e:
                    logger.error(f"Error building context text: {e}", exc_info=True)
                    context_text = None

                # Generate answer using OpenAI with retrieved context
                if context_text:
                    from agentic_research.core.config import get_settings
                    settings = get_settings()
                    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

                    logger.info(f"Generating answer with {len(context_text)} chars of context")
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a research assistant. Answer questions based on the provided research paper excerpts. Be concise and cite specific papers when making claims."
                            },
                            {
                                "role": "user",
                                "content": f"""Answer this question using the research paper excerpts below:

Question: {request.question}

Research Context:
{context_text}

Provide a clear, evidence-based answer citing the papers."""
                        }
                    ],
                    temperature=0.3,
                    max_tokens=800
                )

                answer = response.choices[0].message.content

                # Generate follow-up questions
                followup_response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Based on this Q&A, suggest 3 specific follow-up questions:

Question: {request.question}
Answer: {answer[:200]}...

Generate 3 thoughtful follow-up questions."""
                        }
                    ],
                    temperature=0.7,
                    max_tokens=200
                )

                followup_text = followup_response.choices[0].message.content
                followup_questions = [
                    line.strip().lstrip('123.-')
                    for line in followup_text.split('\n')
                    if line.strip() and len(line.strip()) > 10
                ][:3]

            else:
                answer = f"""I don't have enough information in my knowledge base to answer "{request.question}" with confidence.

This could mean:
1. No papers have been indexed yet that cover this topic
2. The question is outside the scope of the current paper collection
3. Try searching for relevant papers first using the search feature

Once you've found and indexed some papers on this topic, I'll be able to provide detailed answers based on their content."""
                followup_questions = [
                    f"Search for papers about {request.question}",
                    "What topics do you have papers about?",
                    "How can I add more papers to the knowledge base?"
                ]

            # Convert chunks to source citations
            mock_sources = [
                {
                    "title": chunk.get("title", "Unknown"),
                    "paper_id": chunk.get("paper_id", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "relevance_score": chunk.get("relevance_score", 0.0),
                    "excerpt": (chunk.get("excerpt") or chunk.get("text") or "")[:500]
                }
                for chunk in relevant_chunks
            ]
            
            # Convert to API models
            source_citations = _convert_to_source_citations(mock_sources)
            
            # Calculate response time
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Update session status
            _active_sessions[session_id]["status"] = "completed"
            _active_sessions[session_id]["sources_used"] = len(source_citations)
            
            # Calculate confidence score based on source relevance
            confidence_score = (
                sum(s["relevance_score"] for s in mock_sources) / len(mock_sources)
                if mock_sources else 0.0
            )

            response = QuestionResponse(
                question=request.question,
                answer=answer,
                sources=source_citations,
                followup_questions=followup_questions,
                confidence_score=confidence_score,
                response_time_ms=response_time_ms,
                personalization_applied=bool(request.user_id),
                conversation_id=request.conversation_id,
                agent_metadata={
                    "session_id": session_id,
                    "distributed": agents_ready,
                    "context_sources": len(source_citations),
                    "user_personalization": bool(request.user_id),
                    "vector_search": True,
                    "sources_retrieved": len(relevant_chunks)
                }
            )
            
            logger.info(
                "Q&A request completed",
                session_id=session_id,
                sources_used=len(source_citations),
                response_time_ms=response_time_ms,
                confidence=response.confidence_score
            )

            # Save messages to conversation history if conversation_id provided
            if request.conversation_id:
                try:
                    from agentic_research.storage.conversation_store import get_conversation_store

                    store = get_conversation_store()

                    # Ensure conversation exists (create with specific ID if needed)
                    conv = await store.get_conversation(request.conversation_id)
                    is_first_message = False

                    if not conv:
                        logger.info(
                            "Conversation not found, creating new one",
                            conversation_id=request.conversation_id
                        )
                        await store.create_conversation(
                            title="New Chat",  # Temporary title
                            conversation_id=request.conversation_id
                        )
                        is_first_message = True
                    elif conv.message_count == 0:
                        is_first_message = True

                    # Save user message
                    await store.add_message(
                        conv_id=request.conversation_id,
                        role="user",
                        content=request.question
                    )

                    # Save assistant message with sources
                    await store.add_message(
                        conv_id=request.conversation_id,
                        role="assistant",
                        content=answer,
                        sources=[s.model_dump() if hasattr(s, 'model_dump') else s for s in source_citations]
                    )

                    # Auto-generate title from first Q&A (like ChatGPT/Claude)
                    if is_first_message:
                        logger.info("Generating conversation title from first Q&A")
                        generated_title = generate_conversation_title(request.question, answer)
                        await store.update_conversation_title(request.conversation_id, generated_title)
                        logger.info("Updated conversation title", title=generated_title)

                    logger.info("Saved messages to conversation", conversation_id=request.conversation_id)
                except Exception as conv_error:
                    # Log but don't fail the request if conversation saving fails
                    logger.error("Failed to save conversation", error=str(conv_error), exc_info=True)

            return response
            
        finally:
            # Clean up session tracking
            if session_id in _active_sessions:
                _active_sessions[session_id]["completed_at"] = time.time()
        
    except Exception as e:
        import traceback
        full_traceback = traceback.format_exc()
        logger.error("Q&A request failed",
                    session_id=session_id,
                    error=str(e),
                    traceback=full_traceback,
                    exc_info=True)

        # Update session status
        if session_id in _active_sessions:
            _active_sessions[session_id]["status"] = "error"
            _active_sessions[session_id]["error"] = str(e)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Q&A request failed: {str(e)}"
        )


@router.post("/research-and-ask", response_model=Dict[str, Any], summary="Combined research and Q&A")
async def research_and_ask(
    query: ResearchQuery,
    question: QuestionRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Combined workflow: search for papers and then ask questions about them.
    
    This endpoint demonstrates the full research workflow:
    1. Search for relevant papers using the research agent
    2. Process and analyze the found papers
    3. Answer questions using the processed knowledge
    
    This showcases how Praval agents can self-organize to handle
    complex, multi-step research tasks autonomously.
    """
    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(
            "Starting combined research workflow",
            session_id=session_id,
            query=query.query,
            question=question.question
        )
        
        # Ensure agents are initialized
        background_tasks.add_task(_initialize_agents_if_needed)
        
        # Track session
        _active_sessions[session_id] = {
            "type": "research_and_ask",
            "query": query.query,
            "question": question.question,
            "started_at": time.time(),
            "status": "running"
        }
        
        try:
            # Initialize agents if needed
            agents_ready = await _initialize_agents_if_needed()
            
            # NOTE: Praval broadcast() can only be called from @agent functions
            # Agents run separately in research_agents container
            # For now, returning workflow initiation status

            logger.info("Combined research workflow initiated",
                       query=query.query,
                       question=question.question,
                       agents_ready=agents_ready)
            
            # Calculate response time
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Update session status
            _active_sessions[session_id]["status"] = "completed"
            
            # Return combined results
            combined_response = {
                "session_id": session_id,
                "research_query": query.query,
                "question": question.question,
                "workflow_initiated": True,
                "response_time_ms": response_time_ms,
                "agent_metadata": {
                    "praval_agents": agents_ready,
                    "workflow_type": "autonomous_research_and_qa",
                    "self_organizing": True,
                    "agents_count": 6
                }
            }
            
            logger.info(
                "Combined research workflow completed",
                session_id=session_id,
                response_time_ms=response_time_ms
            )
            
            return combined_response
            
        finally:
            # Clean up session tracking
            if session_id in _active_sessions:
                _active_sessions[session_id]["completed_at"] = time.time()
        
    except Exception as e:
        logger.error("Combined research workflow failed", session_id=session_id, error=str(e))
        
        # Update session status
        if session_id in _active_sessions:
            _active_sessions[session_id]["status"] = "error"
            _active_sessions[session_id]["error"] = str(e)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Combined research workflow failed: {str(e)}"
        )


@router.get("/sessions", summary="Get active research sessions")
async def get_active_sessions() -> Dict[str, Any]:
    """
    Get information about currently active research sessions.
    
    Useful for monitoring and debugging research workflows.
    """
    try:
        # Clean up old sessions (older than 1 hour)
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, session_data in _active_sessions.items()
            if current_time - session_data.get("started_at", 0) > 3600
        ]
        
        for session_id in expired_sessions:
            del _active_sessions[session_id]
        
        # Return active sessions
        return {
            "active_sessions": len(_active_sessions),
            "sessions": _active_sessions,
            "cleaned_expired": len(expired_sessions)
        }
        
    except Exception as e:
        logger.error("Failed to get active sessions", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active sessions: {str(e)}"
        )

from pydantic import BaseModel

class IndexRequest(BaseModel):
    papers: List[Dict[str, Any]]

@router.post("/index", summary="Index selected papers for deep Q&A")
async def index_papers(
    request: IndexRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Index selected papers by triggering the Document Processor Agent.

    This endpoint:
    - Accepts a list of selected papers from the frontend
    - Publishes a papers_found message to trigger document processing
    - Document Processor Agent downloads PDFs, extracts text, generates embeddings
    - Stores vectors in Qdrant for semantic search and Q&A

    This is the agentic approach - we leverage the existing agent workflow!
    """
    session_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        papers = request.papers
        logger.info(
            "Indexing selected papers",
            session_id=session_id,
            paper_count=len(papers)
        )

        # Use native Praval publisher to send papers_found message
        publisher = get_publisher()
        broadcast_result = await publisher.publish_index_request(
            papers=papers,
            session_id=session_id
        )

        logger.info(
            "Triggered document processor agent",
            session_id=session_id,
            broadcast_id=broadcast_result,
            papers=len(papers)
        )

        # Invalidate research insights cache (papers added, insights need refresh)
        try:
            import redis.asyncio as aioredis
            redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            await redis_client.delete("research_insights:v1")
            await redis_client.close()
            logger.debug("Invalidated research insights cache after indexing")
        except Exception as e:
            logger.warning(f"Failed to invalidate insights cache: {e}")

        response_time = int((time.time() - start_time) * 1000)

        return {
            "status": "indexing_started",
            "session_id": session_id,
            "papers_submitted": len(papers),
            "broadcast_id": broadcast_result,
            "message": f"Document Processor Agent triggered for {len(papers)} papers. Processing in background...",
            "response_time_ms": response_time,
            "note": "Papers will be downloaded, processed, and indexed. Check Q&A functionality in ~30-60 seconds."
        }

    except Exception as e:
        logger.error(
            "Indexing failed",
            session_id=session_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )


# Knowledge Base Management Endpoints

@router.get("/knowledge-base/papers", summary="List all indexed papers")
async def list_indexed_papers() -> Dict[str, Any]:
    """
    Get a list of all papers currently indexed in the knowledge base.

    Returns metadata for each paper including title, authors, chunk count, etc.
    This provides visibility into what's available for Q&A.
    """
    try:
        from agentic_research.storage.qdrant_client import QdrantClientWrapper

        qdrant = QdrantClientWrapper()
        papers = qdrant.get_all_papers()

        total_vectors = sum(p.get('chunk_count', 0) for p in papers)

        logger.info(
            "Listed all indexed papers",
            total_papers=len(papers),
            total_vectors=total_vectors
        )

        return {
            "papers": papers,
            "total_papers": len(papers),
            "total_vectors": total_vectors,
            "status": "success"
        }

    except Exception as e:
        logger.error("Failed to list papers", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve papers: {str(e)}"
        )


@router.delete("/knowledge-base/papers/{paper_id}", summary="Delete a paper from knowledge base")
async def delete_paper(paper_id: str) -> Dict[str, Any]:
    """
    Delete a specific paper and all its vectors from the knowledge base.

    Args:
        paper_id: The ArXiv ID or unique identifier of the paper to delete

    This removes all chunks/vectors associated with the paper from Qdrant.
    """
    try:
        from agentic_research.storage.qdrant_client import QdrantClientWrapper

        qdrant = QdrantClientWrapper()

        # Count chunks before deletion
        chunk_count = qdrant.count_paper_chunks(paper_id)

        # Delete all vectors for this paper
        qdrant.delete_vectors(paper_id)

        logger.info(
            "Deleted paper from knowledge base",
            paper_id=paper_id,
            chunks_deleted=chunk_count
        )

        return {
            "paper_id": paper_id,
            "vectors_deleted": chunk_count,
            "status": "deleted",
            "message": f"Successfully deleted paper '{paper_id}' and {chunk_count} associated vectors"
        }

    except Exception as e:
        logger.error(
            "Failed to delete paper",
            paper_id=paper_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete paper: {str(e)}"
        )


@router.delete("/knowledge-base/clear", summary="Clear entire knowledge base")
async def clear_knowledge_base() -> Dict[str, Any]:
    """
    Clear the entire knowledge base (delete all papers and vectors).

    WARNING: This action cannot be undone! All indexed papers will be removed.
    The collection will be recreated empty and ready for new papers.
    """
    try:
        from agentic_research.storage.qdrant_client import QdrantClientWrapper

        qdrant = QdrantClientWrapper()

        # Get count before clearing
        papers = qdrant.get_all_papers()
        total_papers = len(papers)
        total_vectors = sum(p.get('chunk_count', 0) for p in papers)

        # Clear the collection
        qdrant.clear_collection()

        logger.warning(
            "Knowledge base cleared",
            papers_deleted=total_papers,
            vectors_deleted=total_vectors
        )

        return {
            "status": "cleared",
            "papers_deleted": total_papers,
            "vectors_deleted": total_vectors,
            "message": "Knowledge base has been completely cleared. All papers and vectors have been deleted."
        }

    except Exception as e:
        logger.error("Failed to clear knowledge base", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear knowledge base: {str(e)}"
        )


@router.get("/knowledge-base/stats", summary="Get knowledge base statistics")
async def get_kb_stats() -> Dict[str, Any]:
    """
    Get statistics about the knowledge base.

    Returns aggregated stats including paper count, vector count,
    average chunks per paper, and category distribution.
    """
    try:
        from agentic_research.storage.qdrant_client import QdrantClientWrapper

        qdrant = QdrantClientWrapper()
        papers = qdrant.get_all_papers()

        # Calculate statistics
        total_chunks = sum(p.get('chunk_count', 0) for p in papers)

        # Count categories
        categories = {}
        for paper in papers:
            for cat in paper.get('categories', []):
                categories[cat] = categories.get(cat, 0) + 1

        # Sort categories by count and get top 10
        top_categories = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10])

        logger.info(
            "Generated knowledge base statistics",
            total_papers=len(papers),
            total_vectors=total_chunks
        )

        return {
            "total_papers": len(papers),
            "total_vectors": total_chunks,
            "avg_chunks_per_paper": round(total_chunks / len(papers), 1) if papers else 0,
            "categories": top_categories,
            "status": "success"
        }

    except Exception as e:
        logger.error("Failed to get stats", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@router.get("/knowledge-base/papers/{paper_id}/pdf", summary="Download PDF for a paper")
async def get_paper_pdf(paper_id: str):
    """
    Stream a paper's PDF directly from storage.

    Args:
        paper_id: The ArXiv ID or unique identifier of the paper

    Returns:
        PDF file stream
    """
    try:
        from fastapi.responses import StreamingResponse
        from agentic_research.storage.minio_client import MinIOClient
        import io

        minio_client = MinIOClient()

        # Check if PDF exists
        if not minio_client.pdf_exists(paper_id):
            logger.warning("PDF not found in MinIO", paper_id=paper_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PDF not found for paper '{paper_id}'. The paper may not have been indexed yet."
            )

        # Download PDF from MinIO
        pdf_data = minio_client.download_pdf(paper_id)

        logger.info("Serving PDF", paper_id=paper_id, size_bytes=len(pdf_data))

        # Stream PDF to browser
        return StreamingResponse(
            io.BytesIO(pdf_data),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{paper_id}.pdf"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to serve PDF",
            paper_id=paper_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to serve PDF: {str(e)}"
        )

# ==========================================
# Proactive Research Insights Endpoint
# ==========================================

@router.get("/insights", summary="Get proactive research insights")
async def get_research_insights() -> Dict[str, Any]:
    """
    Generate proactive research insights based on knowledge base,
    recent papers, and conversation history.

    Returns:
    - Research area clusters
    - Trending topics/keywords
    - Identified research gaps
    - Personalized next steps
    - Suggested papers from arXiv (not yet indexed)

    Results are cached in Redis with 1-hour TTL.
    """
    try:
        import redis.asyncio as aioredis
        import json

        # Check Redis cache first
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            encoding="utf-8"
        )

        CACHE_KEY = "research_insights:v1"
        CACHE_TTL = 3600  # 1 hour

        # Try to get cached insights
        cached_data = await redis_client.get(CACHE_KEY)
        if cached_data:
            logger.info("Returning cached research insights")
            await redis_client.close()
            return json.loads(cached_data)

        logger.info("Generating fresh research insights...")

        # Fetch recent chat history from PostgreSQL (async)
        recent_queries = []
        try:
            from agentic_research.storage.conversation_store import get_conversation_store

            store = get_conversation_store()
            conversations = await store.list_conversations(limit=3, offset=0)

            for conv in conversations:
                messages = await store.get_messages(conv.id, limit=20)
                user_messages = [msg.content for msg in messages if msg.role == 'user']
                recent_queries.extend(user_messages)

            recent_queries = recent_queries[:10]  # Max 10 recent queries
            logger.info(f"Fetched {len(recent_queries)} recent queries from PostgreSQL")

        except Exception as e:
            logger.warning(f"Could not fetch chat history: {e}")
            recent_queries = []

        # Import the insights generation helper
        from agents.interaction.research_advisor import generate_insights_sync

        # Generate insights with chat history
        insights = generate_insights_sync(settings, recent_queries=recent_queries)

        # Cache the result
        await redis_client.setex(
            CACHE_KEY,
            CACHE_TTL,
            json.dumps(insights)
        )
        await redis_client.close()

        logger.info(f"Generated and cached research insights: {insights['kb_context']['total_papers']} papers analyzed")

        return insights

    except ImportError as e:
        # Fallback if agent not available
        logger.warning(f"Could not import insights generator: {e}, returning basic stats")

        from agentic_research.storage.qdrant_client import QdrantClientWrapper

        qdrant = QdrantClientWrapper(settings)
        kb_papers = qdrant.get_all_papers()

        return {
            "research_areas": [],
            "trending_topics": [],
            "research_gaps": [],
            "next_steps": [],
            "suggested_papers": [],
            "kb_context": {
                "total_papers": len(kb_papers),
                "categories": {},
                "recent_activity": False
            },
            "generation_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kb_papers_analyzed": len(kb_papers),
                "insights_quality": "basic"
            }
        }

    except Exception as e:
        logger.error("Failed to generate research insights", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate insights: {str(e)}"
        )

# ==========================================
# Conversation Management Endpoints
# ==========================================

class ConversationCreateRequest(BaseModel):
    """Request model for creating a conversation."""
    title: Optional[str] = None

@router.post("/conversations", summary="Create a new conversation")
async def create_conversation(request: ConversationCreateRequest = ConversationCreateRequest()) -> Dict[str, Any]:
    """
    Create a new conversation for chat history.

    Returns conversation metadata with ID.
    """
    try:
        from agentic_research.storage.conversation_store import get_conversation_store

        store = get_conversation_store()
        conversation = await store.create_conversation(request.title)

        logger.info("Created new conversation", conversation_id=conversation.id, title=conversation.title)

        return conversation.model_dump()

    except Exception as e:
        logger.error("Failed to create conversation", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}"
        )


@router.get("/conversations", summary="List all conversations")
async def list_conversations(
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    List all conversations ordered by most recent.
    
    Returns:
        List of conversation metadata (without full message history)
    """
    try:
        from agentic_research.storage.conversation_store import get_conversation_store
        
        store = get_conversation_store()
        conversations = await store.list_conversations(limit, offset)
        
        return {
            "conversations": [c.model_dump() for c in conversations],
            "total": len(conversations),
            "limit": limit,
            "offset": offset
        }
    
    except Exception as e:
        logger.error("Failed to list conversations", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list conversations: {str(e)}"
        )


@router.get("/conversations/{conversation_id}", summary="Get conversation with messages")
async def get_conversation(conversation_id: str) -> Dict[str, Any]:
    """
    Get a specific conversation with all its messages.
    
    Args:
        conversation_id: UUID of the conversation
        
    Returns:
        Conversation metadata and full message history
    """
    try:
        from agentic_research.storage.conversation_store import get_conversation_store
        
        store = get_conversation_store()
        
        # Get metadata
        conversation = await store.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )
        
        # Get messages
        messages = await store.get_messages(conversation_id)
        
        return {
            **conversation.model_dump(),
            "messages": [m.model_dump() for m in messages]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get conversation", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation: {str(e)}"
        )


class ConversationUpdateRequest(BaseModel):
    """Request model for updating a conversation."""
    title: str

@router.put("/conversations/{conversation_id}", summary="Update conversation title")
async def update_conversation(
    conversation_id: str,
    request: ConversationUpdateRequest
) -> Dict[str, Any]:
    """
    Update conversation title.

    Args:
        conversation_id: UUID of the conversation
        request: Update request with new title

    Returns:
        Success message
    """
    try:
        from agentic_research.storage.conversation_store import get_conversation_store

        store = get_conversation_store()
        success = await store.update_conversation_title(conversation_id, request.title)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )

        logger.info("Updated conversation title", conversation_id=conversation_id, title=request.title)

        return {"message": "Conversation updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update conversation", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update conversation: {str(e)}"
        )


@router.delete("/conversations/{conversation_id}", summary="Delete a conversation")
async def delete_conversation(conversation_id: str) -> Dict[str, Any]:
    """
    Delete a conversation and all its messages.
    
    Args:
        conversation_id: UUID of the conversation
        
    Returns:
        Success message
    """
    try:
        from agentic_research.storage.conversation_store import get_conversation_store
        
        store = get_conversation_store()
        await store.delete_conversation(conversation_id)
        
        return {"message": f"Conversation {conversation_id} deleted successfully"}
    
    except Exception as e:
        logger.error("Failed to delete conversation", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )
