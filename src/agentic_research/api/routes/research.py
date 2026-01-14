"""
Research endpoints for paper discovery and Q&A.

This module provides REST API endpoints that interface with
Praval research agents for academic paper discovery and
intelligent question answering.
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
import structlog

from ..models.research import (
    ResearchQuery, ResearchResponse, PaperResult,
    QuestionRequest, QuestionResponse, SourceCitation,
    ErrorResponse,
    ContentFormat, ContentStyle, ContentGenerationRequest,
    ContentGenerationResponse, Tweet, BlogPost
)
from agents import (
    paper_discovery_agent,
    document_processing_agent,
    semantic_analysis_agent,
    summarization_agent,
    qa_specialist_agent,
    research_advisor_agent,
    # Context Engineering agents - DISABLED (auto-indexes without consent)
    # paper_summarizer_agent,
    # citation_extractor_agent,
    # linked_paper_indexer_agent
)
from processors.arxiv_client import (
    search_arxiv_papers,
    calculate_paper_relevance,
    ArXivAPIError
)
from praval import start_agents
from ...core.messaging import get_publisher, BROADCAST_CHANNEL
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


_agents_initialized = False

async def _initialize_agents_if_needed():
    """Ensure research agents are initialized using InMemory backend (simple, reliable)."""
    global _agents_initialized

    if _agents_initialized:
        return True

    try:
        # Use simple start_agents with InMemory backend - reliable and works in same process
        # NOTE: Context engineering agents (paper_summarizer, citation_extractor, linked_paper_indexer)
        # are DISABLED to prevent automatic background indexing without user consent.
        # These can be re-enabled when user preference controls are implemented.
        agents = [
            paper_discovery_agent,
            document_processing_agent,
            semantic_analysis_agent,
            summarization_agent,
            # Context Engineering agents - DISABLED (auto-indexes without consent)
            # paper_summarizer_agent,
            # citation_extractor_agent,
            # linked_paper_indexer_agent,
            # Interaction agents
            qa_specialist_agent,
            research_advisor_agent
        ]

        # Initialize with InMemory backend using 'broadcast' channel to match publisher
        result = start_agents(*agents, channel=BROADCAST_CHANNEL)
        _agents_initialized = True
        logger.info("Praval research agents initialized (InMemory)", agents_count=len(agents), channel=BROADCAST_CHANNEL)
        return True

    except Exception as e:
        logger.warning("Failed to initialize Praval agents", error=str(e), exc_info=True)
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

                # RETRIEVE AND PASS CONVERSATION HISTORY AND PAPER CONTEXT
                if request.conversation_id:
                    try:
                        from agentic_research.storage.conversation_store import get_conversation_store
                        store = get_conversation_store()

                        # Get conversation metadata (includes paper_ids for KB search chats)
                        conversation = await store.get_conversation(request.conversation_id)
                        if conversation and conversation.metadata:
                            paper_ids = conversation.metadata.get("paper_ids", [])
                            if paper_ids:
                                publish_kwargs["paper_ids"] = paper_ids
                                logger.info(f"Chat with papers: filtering to {len(paper_ids)} papers: {paper_ids}")

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
            # Use paper_ids from conversation context if available (for "Chat with Papers" feature)
            paper_ids_filter = publish_kwargs.get("paper_ids", None)
            logger.info(f"Initializing vector search client for question: {request.question[:50]}")
            if paper_ids_filter:
                logger.info(f"Filtering search to {len(paper_ids_filter)} papers: {paper_ids_filter}")
            vector_client = get_vector_search_client()

            if vector_client is None:
                logger.error("Vector search client is None - cannot perform search")
                relevant_chunks = []
            else:
                logger.info("Vector client initialized, performing search")
                # Pass paper_ids directly to Qdrant for server-side filtering
                relevant_chunks = vector_client.search(
                    query=request.question,
                    top_k=10 if paper_ids_filter else 5,  # More results when filtering to specific papers
                    score_threshold=0.2 if paper_ids_filter else 0.3,  # Lower threshold for focused search
                    paper_ids=paper_ids_filter  # Filter at Qdrant level
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

                    # Save user message (unless it was already saved, e.g., from an edit)
                    if not request.skip_user_message:
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

        # Ensure agents are initialized (same as search/qa endpoints)
        await _initialize_agents_if_needed()

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

        # Invalidate research insights cache and trigger background refresh
        try:
            import redis.asyncio as aioredis
            redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            await redis_client.delete("research_insights:v1")
            await redis_client.close()
            logger.debug("Invalidated research insights cache after indexing")

            # Trigger background refresh after a delay (let indexing complete first)
            async def delayed_refresh():
                await asyncio.sleep(60)  # Wait 60s for indexing to complete
                await _generate_insights_background()

            background_tasks.add_task(delayed_refresh)
            logger.info("Scheduled insights refresh after indexing")
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
async def list_indexed_papers(
    search: Optional[str] = Query(None, description="Search term to filter by title"),
    category: Optional[str] = Query(None, description="Filter by arXiv category (e.g., cs.AI, cs.LG)"),
    source: Optional[str] = Query(None, description="Filter by source: 'kb' (main), 'linked' (cited), or 'all'"),
    sort: Optional[str] = Query("title", description="Sort by: 'title', 'date', 'date_added', 'chunks'"),
    sort_order: Optional[str] = Query("asc", description="Sort order: 'asc' or 'desc'"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of papers per page")
) -> Dict[str, Any]:
    """
    Get a list of all papers currently indexed in the knowledge base.

    Supports filtering by search term, category, and source.
    Supports sorting by title, date, or chunk count.
    Supports pagination.
    """
    try:
        from agentic_research.storage.qdrant_client import QdrantClientWrapper

        qdrant = QdrantClientWrapper()
        all_papers = qdrant.get_all_papers()

        # Also get linked papers if needed
        linked_papers = []
        if source in [None, 'all', 'linked']:
            try:
                linked_papers = qdrant.get_all_linked_papers()
                # Mark them as linked
                for p in linked_papers:
                    p['is_linked'] = True
            except Exception:
                pass  # Linked papers collection may not exist

        # Combine based on source filter
        if source == 'kb':
            papers = all_papers
        elif source == 'linked':
            papers = linked_papers
        else:  # 'all' or None
            papers = all_papers + linked_papers

        # Apply search filter (case-insensitive title search)
        if search:
            search_lower = search.lower()
            papers = [p for p in papers if search_lower in p.get('title', '').lower()]

        # Apply category filter
        if category:
            papers = [p for p in papers if category in p.get('categories', [])]

        # Calculate totals before pagination
        total_papers = len(papers)
        total_vectors = sum(p.get('chunk_count', 0) for p in papers)

        # Apply sorting
        sort_key = {
            'title': lambda p: p.get('title', '').lower(),
            'date': lambda p: p.get('published_date', ''),
            'date_added': lambda p: p.get('indexed_at', ''),
            'chunks': lambda p: p.get('chunk_count', 0)
        }.get(sort, lambda p: p.get('title', '').lower())

        papers = sorted(papers, key=sort_key, reverse=(sort_order == 'desc'))

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_papers = papers[start_idx:end_idx]

        # Get available categories for filter dropdown
        all_categories = set()
        for p in all_papers + linked_papers:
            all_categories.update(p.get('categories', []))

        logger.info(
            "Listed indexed papers with filters",
            total_papers=total_papers,
            filtered_count=len(paginated_papers),
            search=search,
            category=category,
            source=source
        )

        return {
            "papers": paginated_papers,
            "total_papers": total_papers,
            "total_vectors": total_vectors,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_papers + page_size - 1) // page_size,
            "available_categories": sorted(list(all_categories)),
            "filters_applied": {
                "search": search,
                "category": category,
                "source": source,
                "sort": sort,
                "sort_order": sort_order
            },
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


@router.get("/knowledge-base/papers/{paper_id}/related", summary="Find related/cited papers")
async def get_related_papers(paper_id: str) -> Dict[str, Any]:
    """
    Extract citations and find related papers for a specific KB paper.

    This endpoint:
    1. Retrieves the paper's text from MinIO (or uses abstract)
    2. Uses LLM to extract key citations
    3. Searches ArXiv to find arXiv IDs for citations
    4. Returns a list of related papers that can be indexed

    The user can then select which papers to index via POST /research/index.
    """
    import re
    from openai import OpenAI

    try:
        from agentic_research.storage.qdrant_client import QdrantClientWrapper
        from agentic_research.storage.minio_client import MinIOClient

        qdrant = QdrantClientWrapper()
        minio = MinIOClient(settings)

        # Get paper metadata from Qdrant
        papers = qdrant.get_all_papers()
        paper_meta = next((p for p in papers if p.get('paper_id') == paper_id), None)

        if not paper_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Paper '{paper_id}' not found in knowledge base"
            )

        paper_title = paper_meta.get('title', 'Unknown')
        paper_abstract = paper_meta.get('abstract', '')
        paper_categories = paper_meta.get('categories', [])

        # Try to get full text from PDF in MinIO
        paper_text = ""
        text_source = "none"
        try:
            import pdfplumber
            import io

            # Download PDF from MinIO
            pdf_bytes = minio.download_pdf(paper_id)
            if pdf_bytes:
                # Extract text from PDF
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    pages_text = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            pages_text.append(page_text)
                    paper_text = "\n\n".join(pages_text)

                if paper_text:
                    text_source = "pdf"
                    logger.info(f"Extracted text from PDF for {paper_id}, length: {len(paper_text)}")
        except Exception as e:
            logger.info(f"Could not extract text from PDF for {paper_id}: {e}")

        # Use abstract if no full text
        if not paper_text:
            paper_text = paper_abstract
            if paper_text:
                text_source = "abstract"
                logger.info(f"Using abstract for {paper_id}, length: {len(paper_text)}")

        if not paper_text:
            return {
                "paper_id": paper_id,
                "paper_title": paper_title,
                "related_papers": [],
                "message": "No text available for citation extraction"
            }

        # Extract citations using LLM
        openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

        # Limit text to avoid token limits - but try to include the References section
        # References are usually at the end, so prioritize that
        if len(paper_text) > 15000:
            # Take first 5000 chars (intro/abstract) + last 10000 chars (likely has References)
            text_sample = paper_text[:5000] + "\n\n[...middle content omitted...]\n\n" + paper_text[-10000:]
        else:
            text_sample = paper_text

        extraction_prompt = f"""You are a citation extractor. Your ONLY job is to copy citations VERBATIM from the References section.

Paper Title: {paper_title}

Paper Text:
{text_sample}

STRICT INSTRUCTIONS:
1. Find the "References" or "Bibliography" section at the end of the paper
2. Copy EXACTLY 5 citations from that section - do not paraphrase or modify
3. Only extract references that have the format: Author. Year. Title. Venue.

For each reference, extract:
- TITLE: Copy the EXACT title from the reference (word for word)
- AUTHORS: The first author's last name EXACTLY as written
- YEAR: The year EXACTLY as written
- RELEVANCE: Brief description based on where it's cited in the paper

Output format:
CITATION 1:
TITLE: [copy exact title]
AUTHORS: [first author last name]
YEAR: [year]
RELEVANCE: [brief reason]

CITATION 2:
...

RULES:
- If you cannot find a References section, output ONLY: "NO REFERENCES FOUND"
- Do NOT make up titles that don't appear in the text
- Do NOT guess or hallucinate - only extract what's written
- Preserve EXACT spacing in titles - copy word by word with proper spaces
- Prefer arXiv papers (often have IDs like arXiv:XXXX.XXXXX)
- Prefer citations that mention AI, machine learning, neural networks, agents
- Avoid textbooks, books, and conference proceedings without arXiv versions

Maximum 5 citations."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.3,
            max_tokens=1500
        )

        llm_response = response.choices[0].message.content
        logger.info(f"LLM response for {paper_id} (text_source={text_source}): {llm_response[:500]}...")

        # Parse citations from LLM response
        citations = []
        current_citation = {}

        for line in llm_response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("CITATION"):
                if current_citation and current_citation.get("title"):
                    citations.append(current_citation)
                current_citation = {}
            elif line.startswith("TITLE:"):
                current_citation["title"] = line.replace("TITLE:", "").strip()
            elif line.startswith("AUTHORS:"):
                current_citation["authors"] = line.replace("AUTHORS:", "").strip()
            elif line.startswith("YEAR:"):
                year = line.replace("YEAR:", "").strip()
                if year.lower() != "unknown":
                    current_citation["year"] = year
            elif line.startswith("RELEVANCE:"):
                current_citation["relevance"] = line.replace("RELEVANCE:", "").strip()

        # Don't forget the last citation
        if current_citation and current_citation.get("title"):
            citations.append(current_citation)

        # Search ArXiv to find paper details for each citation
        import urllib.parse
        import urllib.request
        import xml.etree.ElementTree as ET

        related_papers = []

        for citation in citations[:5]:
            title = citation.get("title", "")
            if not title:
                continue

            try:
                # Strip any existing quotes from title, then add quotes for exact match
                clean_title = title.strip('"\'')
                search_query = urllib.parse.quote(f'"{clean_title}"')
                url = f"{settings.ARXIV_BASE_URL}?search_query=ti:{search_query}&max_results=1"

                with urllib.request.urlopen(url, timeout=20) as resp:
                    data = resp.read().decode('utf-8')

                root = ET.fromstring(data)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}

                for entry in root.findall('atom:entry', ns):
                    entry_id = entry.find('atom:id', ns)
                    entry_title = entry.find('atom:title', ns)
                    entry_summary = entry.find('atom:summary', ns)
                    entry_published = entry.find('atom:published', ns)

                    if entry_id is not None:
                        id_text = entry_id.text
                        match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', id_text)
                        if match:
                            arxiv_id = match.group(1)
                            arxiv_title = entry_title.text.strip() if entry_title is not None else ""

                            # Get authors
                            authors = []
                            for author in entry.findall('atom:author', ns):
                                name = author.find('atom:name', ns)
                                if name is not None:
                                    authors.append(name.text)

                            # Get categories
                            categories = []
                            for cat in entry.findall('atom:category', ns):
                                term = cat.get('term')
                                if term:
                                    categories.append(term)

                            related_papers.append({
                                "arxiv_id": arxiv_id,
                                "title": arxiv_title,
                                "authors": authors[:5],  # Limit to 5 authors
                                "abstract": entry_summary.text.strip()[:500] if entry_summary is not None else "",
                                "published_date": entry_published.text[:10] if entry_published is not None else None,
                                "categories": categories[:3],
                                "url": f"http://arxiv.org/abs/{arxiv_id}",
                                "relevance": citation.get("relevance", ""),
                                "source_paper_id": paper_id,
                                "source_paper_title": paper_title
                            })
                            break  # Only take first match

            except Exception as e:
                logger.warning(f"ArXiv search failed for '{title}': {e}")
                continue

        # Check which papers are already in KB
        existing_ids = {p.get('paper_id') for p in papers}
        for paper in related_papers:
            paper["already_indexed"] = paper.get("arxiv_id") in existing_ids

        logger.info(
            "Related papers extracted",
            paper_id=paper_id,
            citations_found=len(citations),
            papers_resolved=len(related_papers)
        )

        return {
            "paper_id": paper_id,
            "paper_title": paper_title,
            "related_papers": related_papers,
            "citations_extracted": len(citations),
            "papers_found": len(related_papers),
            "message": f"Found {len(related_papers)} related papers from {len(citations)} citations"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to extract related papers",
            paper_id=paper_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract related papers: {str(e)}"
        )


# ==========================================
# Proactive Research Insights Endpoint
# ==========================================

async def _generate_insights_background():
    """Background task to generate and cache insights."""
    import redis.asyncio as aioredis
    import json
    from datetime import datetime, timezone

    CACHE_KEY = "research_insights:v1"
    CACHE_TTL = 3600  # 1 hour

    try:
        logger.info("Background: Generating fresh research insights...")

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

            recent_queries = recent_queries[:10]
            logger.info(f"Background: Fetched {len(recent_queries)} recent queries")

        except Exception as e:
            logger.warning(f"Background: Could not fetch chat history: {e}")
            recent_queries = []

        # Import the insights generation helper
        from agents.interaction.research_advisor import generate_insights_sync

        # Generate insights with chat history
        insights = generate_insights_sync(settings, recent_queries=recent_queries)

        # Cache the result
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            encoding="utf-8"
        )
        await redis_client.setex(CACHE_KEY, CACHE_TTL, json.dumps(insights))
        await redis_client.close()

        logger.info(f"Background: Cached research insights ({insights['kb_context']['total_papers']} papers)")

    except Exception as e:
        logger.error(f"Background: Failed to generate insights: {e}")


def _get_basic_insights() -> Dict[str, Any]:
    """Get basic stats without LLM generation - fast fallback."""
    from agentic_research.storage.qdrant_client import QdrantClientWrapper

    try:
        qdrant = QdrantClientWrapper()
        kb_papers = qdrant.get_all_papers()

        # Extract categories
        categories = {}
        for paper in kb_papers:
            for cat in paper.get("categories", []):
                categories[cat] = categories.get(cat, 0) + 1

        return {
            "research_areas": [],
            "trending_topics": [],
            "research_gaps": [],
            "next_steps": [],
            "suggested_papers": [],
            "kb_context": {
                "total_papers": len(kb_papers),
                "categories": categories,
                "recent_activity": False
            },
            "generation_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kb_papers_analyzed": len(kb_papers),
                "insights_quality": "basic",
                "is_cached": False,
                "refresh_in_progress": True
            }
        }
    except Exception as e:
        logger.error(f"Failed to get basic insights: {e}")
        return {
            "research_areas": [],
            "trending_topics": [],
            "research_gaps": [],
            "next_steps": [],
            "suggested_papers": [],
            "kb_context": {"total_papers": 0, "categories": {}, "recent_activity": False},
            "generation_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kb_papers_analyzed": 0,
                "insights_quality": "error",
                "is_cached": False,
                "refresh_in_progress": False
            }
        }


@router.get("/insights", summary="Get proactive research insights")
async def get_research_insights(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Get research insights - returns cached data immediately.

    If no cache exists, returns basic stats and triggers background generation.
    Insights are cached in Redis with 1-hour TTL.

    Returns:
    - Research area clusters
    - Trending topics/keywords
    - Identified research gaps
    - Personalized next steps
    """
    try:
        import redis.asyncio as aioredis
        import json

        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            encoding="utf-8"
        )

        CACHE_KEY = "research_insights:v1"

        # Try to get cached insights - return immediately if available
        cached_data = await redis_client.get(CACHE_KEY)
        await redis_client.close()

        if cached_data:
            logger.info("Returning cached research insights")
            result = json.loads(cached_data)
            result["generation_metadata"]["is_cached"] = True
            result["generation_metadata"]["refresh_in_progress"] = False
            return result

        # No cache - return basic stats immediately and generate in background
        logger.info("No cached insights, returning basic stats and triggering background refresh")
        background_tasks.add_task(_generate_insights_background)

        return _get_basic_insights()

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


@router.post("/insights/refresh", summary="Trigger insights refresh")
async def refresh_research_insights(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Manually trigger a refresh of research insights in the background.

    Returns immediately with status, insights will be updated in Redis cache.
    """
    logger.info("Manual insights refresh triggered")
    background_tasks.add_task(_generate_insights_background)

    return {
        "status": "refresh_started",
        "message": "Insights refresh triggered in background. Check /insights endpoint for updated data."
    }


def _is_arxiv_category(name: str) -> bool:
    """Check if name looks like an ArXiv category (e.g., cs.AI, physics.comp-ph)."""
    import re
    # ArXiv categories: prefix.suffix where prefix is letters, suffix is letters/digits/hyphens
    return bool(re.match(r'^[a-z]+(-[a-z]+)?\.[A-Za-z]{2}(-[a-zA-Z]+)?$', name))


@router.get("/areas/{area_name}/papers", summary="Get papers by research area")
async def get_papers_by_area(
    area_name: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum papers to return")
) -> Dict[str, Any]:
    """
    Get papers that match a research area.

    Fast path: If area_name is an ArXiv category (e.g., cs.AI, cs.LG), uses
    Vajra BM25 index with category filter (instant, no API calls).

    Slow path: For descriptive names, falls back to semantic search.
    """
    import time
    start_time = time.time()

    try:
        # Fast path: ArXiv category filter using Vajra BM25 index
        if _is_arxiv_category(area_name):
            logger.info(f"Fast path: Using Vajra category filter for {area_name}")
            from agentic_research.storage.paper_index import get_paper_index

            paper_index = get_paper_index()

            # Get all indexed papers and filter by category
            all_papers = paper_index.get_indexed_papers()

            # Filter by category
            matching_papers = [
                p for p in all_papers
                if area_name in p.get("categories", [])
            ]

            # Sort by chunk count (papers with more chunks = more content)
            matching_papers.sort(key=lambda p: p.get("chunk_count", 0), reverse=True)

            # Format response
            papers = []
            for p in matching_papers[:limit]:
                papers.append({
                    "paper_id": p["paper_id"],
                    "title": p.get("title", "Unknown"),
                    "authors": p.get("authors", []),
                    "abstract": p.get("abstract", "")[:500],
                    "categories": p.get("categories", []),
                    "relevance_score": 1.0  # Category match is exact
                })

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "Fast path: Retrieved papers by category",
                area_name=area_name,
                papers_found=len(papers),
                elapsed_ms=elapsed_ms
            )

            return {
                "area_name": area_name,
                "papers": papers,
                "total_found": len(papers),
                "search_mode": "category_filter",
                "elapsed_ms": elapsed_ms,
                "status": "success"
            }

        # Slow path: Semantic search for descriptive area names
        logger.info(f"Slow path: Using semantic search for {area_name}")
        from agentic_research.storage.qdrant_client import QdrantClientWrapper
        from agentic_research.storage.embeddings import EmbeddingsGenerator

        settings = get_settings()
        qdrant_client = QdrantClientWrapper(settings)
        embeddings_gen = EmbeddingsGenerator(settings)

        # Generate embedding for the area name/description
        area_embedding = embeddings_gen.generate_embedding(area_name)

        # Search for similar papers - use a lower threshold for broad area searches
        search_results = qdrant_client.search_similar(
            area_embedding,
            limit=limit * 3,  # Get more results since we'll collapse chunks to papers
            score_threshold=0.3  # Lower threshold for area-based searches
        )

        # Get unique papers (collapse chunks to papers)
        seen_papers = set()
        papers = []

        for result in search_results:
            paper_id = result.get("payload", {}).get("paper_id")
            if paper_id and paper_id not in seen_papers:
                seen_papers.add(paper_id)
                papers.append({
                    "paper_id": paper_id,
                    "title": result.get("payload", {}).get("title", "Unknown"),
                    "authors": result.get("payload", {}).get("authors", []),
                    "abstract": result.get("payload", {}).get("abstract", "")[:500],
                    "categories": result.get("payload", {}).get("categories", []),
                    "relevance_score": round(result.get("score", 0), 3)
                })

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "Slow path: Retrieved papers by semantic search",
            area_name=area_name,
            papers_found=len(papers),
            elapsed_ms=elapsed_ms
        )

        return {
            "area_name": area_name,
            "papers": papers,
            "total_found": len(papers),
            "search_mode": "semantic",
            "elapsed_ms": elapsed_ms,
            "status": "success"
        }

    except Exception as e:
        logger.error("Failed to get papers by area", area_name=area_name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve papers: {str(e)}"
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
        from datetime import datetime

        store = get_conversation_store()
        # Provide default title if none given
        title = request.title or f"New Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        conversation = await store.create_conversation(title)

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


# ==========================================
# Thread-Based Conversation Branching Endpoints
# ==========================================

class EditMessageRequest(BaseModel):
    """Request model for editing a message (creates a new thread)."""
    new_content: str


class SwitchThreadRequest(BaseModel):
    """Request model for switching threads."""
    thread_id: Optional[int] = None
    position: Optional[int] = None
    direction: Optional[str] = None  # 'prev' or 'next'


@router.post("/conversations/{conversation_id}/messages/{message_id}/edit", summary="Edit a message (creates new thread)")
async def edit_message(
    conversation_id: str,
    message_id: str,
    request: EditMessageRequest
) -> Dict[str, Any]:
    """
    Edit a user message by creating a new thread.

    Thread-based branching model:
    - Creates a new thread with incremented thread_id
    - Copies all messages from the original thread up to the edit point
    - Adds the edited message as the new version at that position
    - The new thread becomes active

    Args:
        conversation_id: UUID of the conversation
        message_id: UUID of the message to edit (must be a user message)
        request: The new content for the message

    Returns:
        The newly created message and thread info
    """
    try:
        from agentic_research.storage.conversation_store import get_conversation_store

        store = get_conversation_store()

        # Edit the message (creates a new thread)
        result = await store.edit_message(
            conv_id=conversation_id,
            message_id=message_id,
            new_content=request.new_content
        )

        # Extract the new message from the result
        new_message = result["new_message"]

        logger.info(
            "Created new thread from message edit",
            conversation_id=conversation_id,
            original_message_id=message_id,
            new_message_id=new_message.id,
            thread_id=new_message.thread_id,
            position=new_message.position
        )

        return {
            "message": new_message.model_dump(),
            "thread_id": new_message.thread_id,
            "position": new_message.position,
            "version_count": new_message.version_count,
            "current_version": new_message.current_version,
            "status": "thread_created"
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Failed to edit message",
            conversation_id=conversation_id,
            message_id=message_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to edit message: {str(e)}"
        )


@router.post("/conversations/{conversation_id}/switch-thread", summary="Switch active thread")
async def switch_thread(
    conversation_id: str,
    request: SwitchThreadRequest
) -> Dict[str, Any]:
    """
    Switch to a different thread in the conversation.

    Can be called in two ways:
    1. With thread_id: Switch directly to that thread
    2. With position and direction: Navigate prev/next among thread versions at that position

    Args:
        conversation_id: UUID of the conversation
        request: Thread switch parameters

    Returns:
        The new active thread ID and updated messages
    """
    try:
        from agentic_research.storage.conversation_store import get_conversation_store

        store = get_conversation_store()

        # Switch thread
        result = await store.switch_thread(
            conv_id=conversation_id,
            thread_id=request.thread_id,
            position=request.position,
            direction=request.direction
        )

        logger.info(
            "Switched conversation thread",
            conversation_id=conversation_id,
            new_thread_id=result.get("active_thread_id")
        )

        # Return result directly as it already has the correct structure
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Failed to switch thread",
            conversation_id=conversation_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to switch thread: {str(e)}"
        )


@router.get("/conversations/{conversation_id}/threads/{position}", summary="Get thread versions at position")
async def get_threads_at_position(
    conversation_id: str,
    position: int
) -> Dict[str, Any]:
    """
    Get information about all thread versions at a specific message position.

    This shows how many thread versions exist at that position and provides
    navigation info for the < 1/3 > style version selector UI.

    Args:
        conversation_id: UUID of the conversation
        position: The message position (1-indexed)

    Returns:
        Thread version information including count and previews
    """
    try:
        from agentic_research.storage.conversation_store import get_conversation_store

        store = get_conversation_store()

        thread_info = await store.get_threads_at_position(
            conv_id=conversation_id,
            position=position
        )

        # Return as dict (ThreadInfo is a Pydantic model)
        return thread_info.model_dump() if hasattr(thread_info, 'model_dump') else thread_info

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Failed to get threads at position",
            conversation_id=conversation_id,
            position=position,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get threads: {str(e)}"
        )


@router.delete("/conversations/{conversation_id}/threads/{thread_id}", summary="Delete a thread")
async def delete_thread(
    conversation_id: str,
    thread_id: int
) -> Dict[str, Any]:
    """
    Delete a specific thread and all its messages.

    Cannot delete thread 0 (the original conversation).
    If the deleted thread was active, switches back to thread 0.

    Args:
        conversation_id: UUID of the conversation
        thread_id: ID of the thread to delete (must be > 0)

    Returns:
        Success message
    """
    try:
        from agentic_research.storage.conversation_store import get_conversation_store

        store = get_conversation_store()

        success = await store.delete_thread(
            conv_id=conversation_id,
            thread_id=thread_id
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thread {thread_id} not found"
            )

        logger.info(
            "Deleted conversation thread",
            conversation_id=conversation_id,
            thread_id=thread_id
        )

        return {
            "message": f"Thread {thread_id} deleted successfully",
            "status": "deleted"
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Failed to delete thread",
            conversation_id=conversation_id,
            thread_id=thread_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete thread: {str(e)}"
        )


# Keep backwards compatibility for old endpoint (redirect to new)
@router.post("/conversations/{conversation_id}/switch-branch", summary="[Deprecated] Switch branch - use switch-thread")
async def switch_branch_deprecated(
    conversation_id: str,
    request: SwitchThreadRequest
) -> Dict[str, Any]:
    """Deprecated: Use /switch-thread instead. This redirects to the new thread-based endpoint."""
    return await switch_thread(conversation_id, request)


@router.get("/conversations/{conversation_id}/messages/{message_id}/branches", summary="[Deprecated] Get branches - use threads endpoint")
async def get_branches_at_message_deprecated(
    conversation_id: str,
    message_id: str
) -> Dict[str, Any]:
    """
    Deprecated: Use /threads/{position} instead.

    This endpoint is maintained for backwards compatibility but returns
    thread-based data in the old format.
    """
    try:
        from agentic_research.storage.conversation_store import get_conversation_store

        store = get_conversation_store()

        # Get the message to find its position
        messages = await store.get_messages(conversation_id)
        target_msg = None
        for msg in messages:
            if msg.id == message_id:
                target_msg = msg
                break

        if not target_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Message {message_id} not found"
            )

        # Get thread info at that position
        thread_info = await store.get_threads_at_position(
            conv_id=conversation_id,
            position=target_msg.position
        )

        # Convert Pydantic model to dict if needed
        info_dict = thread_info.model_dump() if hasattr(thread_info, 'model_dump') else thread_info

        # Convert to old format for backwards compatibility
        return {
            "message_id": message_id,
            "branch_count": info_dict.get("thread_count", 1),
            "branches": [
                {
                    "branch_id": str(v["thread_id"]),
                    "branch_index": i,
                    "message_id": v["message_id"],
                    "first_message_preview": v["content_preview"],
                    "timestamp": v["timestamp"]
                }
                for i, v in enumerate(info_dict.get("threads", []))
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get branches (deprecated)",
            conversation_id=conversation_id,
            message_id=message_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get branches: {str(e)}"
        )


# =============================================================================
# Content Generation Endpoints (Twitter/X + Blog Posts)
# =============================================================================

@router.post(
    "/conversations/{conversation_id}/generate-content",
    response_model=ContentGenerationResponse,
    summary="Generate shareable content from conversation"
)
async def generate_content(
    conversation_id: str,
    request: ContentGenerationRequest
) -> ContentGenerationResponse:
    """
    Generate shareable content (Twitter thread or blog post) from a conversation.

    This endpoint uses the Content Generator Agent (Praval) to:
    - Extract key insights from the Q&A conversation
    - Format them based on the requested output format
    - Include proper citations to referenced papers
    - Learn from successful generation patterns over time

    Args:
        conversation_id: The conversation to generate content from
        request: Content generation options (format, style, etc.)

    Returns:
        Generated content (tweets or blog post) with citations
    """
    start_time = time.time()

    try:
        from agentic_research.storage.conversation_store import get_conversation_store
        from agents.interaction.content_generator import content_generator_agent

        store = get_conversation_store()

        # Fetch conversation messages
        conversation = await store.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )

        messages = await store.get_messages(conversation_id)
        if not messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Conversation has no messages to generate content from"
            )

        # Build conversation context and extract citations
        conversation_text = []
        all_sources = []
        paper_ids = set()

        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            conversation_text.append(f"{role}: {msg.content}")

            # Extract sources from assistant messages
            if msg.role == "assistant" and msg.sources:
                for source in msg.sources:
                    # Handle both dict and object access (sources may be dicts from DB)
                    if isinstance(source, dict):
                        paper_id = source.get("paper_id", "")
                        title = source.get("title", "")
                        relevance = source.get("relevance_score", 0.0)
                    else:
                        paper_id = source.paper_id
                        title = source.title
                        relevance = source.relevance_score

                    if paper_id:
                        paper_ids.add(paper_id)
                        all_sources.append({
                            "title": title,
                            "paper_id": paper_id,
                            "relevance": relevance
                        })

        # Format papers for agent
        papers_list = "\n".join([
            f"- {s['title']} (https://arxiv.org/abs/{s['paper_id']})"
            for s in all_sources[:10]  # Limit to top 10 sources
        ]) if all_sources else "No papers cited"

        # Create a simple spore-like object for the agent
        # The agent accesses spore.knowledge.get() so we use a simple namespace
        class SimpleSpore:
            """Simple spore-like object for direct agent invocation."""
            def __init__(self, knowledge: dict):
                self.knowledge = knowledge

        spore = SimpleSpore(knowledge={
            "conversation_text": conversation_text,
            "papers_list": papers_list,
            "paper_ids": list(paper_ids),
            "format": request.format.value,
            "style": request.style.value,
            "max_tweets": request.max_tweets,
            "include_toc": request.include_toc,
            "custom_prompt": request.custom_prompt or ""
        })

        # Call content generator agent directly for synchronous response
        logger.info(
            "Invoking content generator agent",
            conversation_id=conversation_id,
            format=request.format.value,
            style=request.style.value
        )

        agent_result = content_generator_agent(spore)

        # Check for errors from agent
        if agent_result.get("error"):
            logger.error(
                "Content generator agent returned error",
                error=agent_result["error"]
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Content generation failed: {agent_result['error']}"
            )

        generation_time = int((time.time() - start_time) * 1000)

        # Build response based on format
        if request.format == ContentFormat.TWITTER:
            tweets = []
            for t in agent_result.get("tweets", []):
                tweets.append(Tweet(
                    position=t.get("position", len(tweets) + 1),
                    content=t.get("content", ""),
                    char_count=t.get("char_count", len(t.get("content", ""))),
                    has_citation=t.get("has_citation", False),
                    citation_url=t.get("citation_url")
                ))

            return ContentGenerationResponse(
                format=ContentFormat.TWITTER,
                style=request.style,
                tweets=tweets,
                blog_post=None,
                papers_cited=agent_result.get("paper_ids", list(paper_ids)),
                generation_time_ms=generation_time
            )

        else:
            # Blog format
            blog_data = agent_result.get("blog_post", {})

            # Build references list from sources
            references = []
            for source in all_sources[:10]:
                references.append(
                    f"[{len(references) + 1}] {source['title']}. https://arxiv.org/abs/{source['paper_id']}"
                )

            return ContentGenerationResponse(
                format=ContentFormat.BLOG,
                style=request.style,
                tweets=None,
                blog_post=BlogPost(
                    title=blog_data.get("title", "Research Summary"),
                    content=blog_data.get("content", ""),
                    word_count=blog_data.get("word_count", 0),
                    references=references
                ),
                papers_cited=agent_result.get("paper_ids", list(paper_ids)),
                generation_time_ms=generation_time
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to generate content",
            conversation_id=conversation_id,
            format=request.format.value,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate content: {str(e)}"
        )
