"""
Q&A Specialist Agent - Interaction Domain.

I am a research Q&A expert who provides comprehensive, accurate answers
about research papers using retrieved context and accumulated knowledge
to deliver personalized, insightful responses.

Two-Tier Retrieval Strategy (Context Engineering):
1. Fast Path: Search paper_summaries first for quick relevance check
2. Deep Path: For relevant papers, search main collection for detailed chunks
3. Expanded Context: Also search linked_papers for broader context
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from praval import agent, chat, broadcast
from praval import Spore

from agentic_research.core.config import get_settings
from agentic_research.storage.qdrant_client import QdrantClientWrapper, CollectionType
from agentic_research.storage.embeddings import EmbeddingsGenerator


logger = logging.getLogger(__name__)
settings = get_settings()


def _two_tier_retrieval(
    qdrant_client: QdrantClientWrapper,
    query_embedding: List[float],
    top_k_summaries: int = 10,
    top_k_chunks: int = 5,
    summary_threshold: float = 0.6,
    chunk_threshold: float = 0.7,
    paper_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Implement two-tier retrieval for context engineering.

    Tier 1 (Fast): Search summaries to identify relevant papers
    Tier 2 (Deep): For top papers, retrieve detailed chunks
    Tier 3 (Expanded): Search linked papers for broader context

    Args:
        paper_ids: If provided, filter all results to only these papers (for "Chat with Papers")

    Returns:
        Dictionary with summary_matches, chunk_matches, linked_matches
    """
    results = {
        "summary_matches": [],
        "chunk_matches": [],
        "linked_matches": [],
        "relevant_paper_ids": set(),
        "filtered_to_papers": paper_ids or []
    }

    # Helper to filter results by paper_ids
    def filter_by_papers(items: List[Dict], id_key: str = "paper_id") -> List[Dict]:
        if not paper_ids:
            return items
        return [item for item in items if item.get(id_key) in paper_ids or
                item.get("payload", {}).get("paper_id") in paper_ids]

    # Tier 1: Fast path - search summaries
    try:
        # Get more results if filtering, to ensure we have enough after filter
        fetch_limit = top_k_summaries * 3 if paper_ids else top_k_summaries
        summary_results = qdrant_client.search_summaries(
            query_vector=query_embedding,
            limit=fetch_limit,
            score_threshold=summary_threshold
        )

        # Filter by paper_ids if specified
        summary_results = filter_by_papers(summary_results)[:top_k_summaries]

        for summary in summary_results:
            results["summary_matches"].append(summary)
            if summary.get("paper_id"):
                results["relevant_paper_ids"].add(summary["paper_id"])

        logger.debug(f"Tier 1: Found {len(summary_results)} relevant papers via summaries" +
                    (f" (filtered to {len(paper_ids)} papers)" if paper_ids else ""))
    except Exception as e:
        logger.warning(f"Summary search failed, falling back to chunks only: {e}")

    # Tier 2: Deep path - search main collection for chunks
    try:
        # Get more results if filtering
        fetch_limit = top_k_chunks * 5 if paper_ids else top_k_chunks
        chunk_results = qdrant_client.search_similar(
            query_vector=query_embedding,
            limit=fetch_limit,
            score_threshold=chunk_threshold
        )

        # Filter by paper_ids if specified
        chunk_results = filter_by_papers(chunk_results)[:top_k_chunks]

        for chunk in chunk_results:
            results["chunk_matches"].append(chunk)
            paper_id = chunk.get("payload", {}).get("paper_id")
            if paper_id:
                results["relevant_paper_ids"].add(paper_id)

        logger.debug(f"Tier 2: Found {len(chunk_results)} relevant chunks" +
                    (f" (filtered to {len(paper_ids)} papers)" if paper_ids else ""))
    except Exception as e:
        logger.warning(f"Chunk search failed: {e}")

    # Tier 3: Expanded context - search linked papers (only if not filtering)
    # When filtering to specific papers, we focus on those papers only
    if not paper_ids:
        try:
            linked_results = qdrant_client.search_linked_papers(
                query_vector=query_embedding,
                limit=3,  # Fewer linked results to avoid overwhelming
                score_threshold=chunk_threshold
            )

            results["linked_matches"] = linked_results
            logger.debug(f"Tier 3: Found {len(linked_results)} relevant linked paper chunks")
        except Exception as e:
            logger.debug(f"Linked papers search skipped or failed: {e}")
    else:
        logger.debug("Tier 3: Skipped linked papers search (filtering to selected papers)")

    return results


@agent("qa_specialist", channel="broadcast", responds_to=["user_query", "summaries_complete"], memory=True)
def qa_specialist_agent(spore: Spore) -> None:
    """
    I am a research Q&A expert who provides comprehensive, accurate answers
    about research papers using retrieved context and accumulated knowledge
    to deliver personalized, insightful responses.

    My expertise:
    - Contextual question answering using vector search
    - Semantic retrieval from research paper embeddings
    - Personalized responses based on user interaction history
    - Source citation with paper references
    - Follow-up question generation
    - Research guidance and recommendations
    """
    spore_type = spore.knowledge.get("type")

    if spore_type == "summaries_complete":
        # Store research summaries for Q&A context
        _store_research_context(spore)
        return

    # Handle user queries
    user_query = spore.knowledge.get("query", spore.knowledge.get("question", ""))
    user_id = spore.knowledge.get("user_id", "anonymous")
    conversation_context = spore.knowledge.get("conversation_context", [])
    include_sources = spore.knowledge.get("include_sources", True)
    paper_ids = spore.knowledge.get("paper_ids", None)  # For "Chat with Papers" filtering

    if not user_query:
        logger.warning("Q&A specialist received empty query")
        return

    if paper_ids:
        logger.info(f"❓ Q&A Specialist: Answering '{user_query}' for user {user_id} (FILTERED to {len(paper_ids)} papers)")
    else:
        logger.info(f"❓ Q&A Specialist: Answering '{user_query}' for user {user_id}")

    # Initialize storage clients
    try:
        qdrant_client = QdrantClientWrapper(settings)
        embeddings_gen = EmbeddingsGenerator(settings)
    except Exception as e:
        logger.error(f"Failed to initialize Q&A clients: {e}")
        broadcast({
            "type": "qa_error",
            "knowledge": {
                "user_query": user_query,
                "error": f"Client initialization failed: {str(e)}",
                "error_type": "initialization_error"
            }
        })
        return

    # Personalization through memory
    user_interests = qa_specialist_agent.recall(f"user:{user_id}", limit=10)
    similar_questions = qa_specialist_agent.recall(user_query, limit=5)

    try:
        import time
        start_time = time.time()

        # STEP 1: Contextualize Query (if history exists)
        search_query = user_query
        
        if conversation_context and len(conversation_context) > 0:
            logger.info("Contextualizing query using conversation history...")
            context_str = "\n".join(conversation_context[-4:]) # Last 2 turns
            
            rewrite_prompt = f"""
            Given the conversation history, rewrite the last user question to be a standalone search query.
            Resolve any coreferences (it, that, he, she) to their actual subjects.
            Keep the query concise and focused on the research topic.
            
            History:
            {context_str}
            
            Last User Question: {user_query}
            
            Standalone Search Query (just the text):
            """
            
            rewritten = chat(rewrite_prompt)
            if rewritten and len(rewritten) < 200:
                search_query = rewritten.strip().strip('"')
                logger.info(f"Rewrote query: '{user_query}' -> '{search_query}'")
            else:
                logger.warning("Rewriting failed or produced long output, using original query")

        # STEP 2: Generate query embedding
        logger.debug(f"Generating query embedding for: {search_query}")
        query_embedding = embeddings_gen.generate_embedding(search_query)

        # STEP 2: Two-Tier Retrieval (Context Engineering)
        if paper_ids:
            logger.debug(f"Performing two-tier retrieval (filtered to {len(paper_ids)} papers)...")
        else:
            logger.debug("Performing two-tier retrieval...")
        retrieval_results = _two_tier_retrieval(
            qdrant_client=qdrant_client,
            query_embedding=query_embedding,
            top_k_summaries=10,
            top_k_chunks=8 if paper_ids else 5,  # More chunks when filtered to specific papers
            summary_threshold=0.5 if paper_ids else 0.6,  # Lower threshold when filtered
            chunk_threshold=0.5 if paper_ids else 0.7,  # Lower threshold when filtered
            paper_ids=paper_ids
        )

        # Collect all search results
        search_results = retrieval_results["chunk_matches"]
        summary_matches = retrieval_results["summary_matches"]
        linked_matches = retrieval_results["linked_matches"]

        # STEP 3: Build context from retrieved chunks and summaries
        context_pieces = []
        sources = []
        summary_context = []

        # Add paper summaries for high-level context (from Tier 1)
        for summary in summary_matches[:5]:
            summary_context.append({
                "title": summary.get("title", "Unknown"),
                "one_line": summary.get("one_line", ""),
                "paper_id": summary.get("paper_id", ""),
                "relevance": summary.get("score", 0.0)
            })

        # Add main chunks (from Tier 2)
        for result in search_results:
            payload = result["payload"]

            context_pieces.append({
                "content": payload.get("chunk_text", ""),
                "source": payload.get("title", "Unknown Paper"),
                "relevance": result["score"],
                "is_linked": False
            })

            # Build source citation
            source_citation = {
                "title": payload.get("title", "Unknown Paper"),
                "paper_id": payload.get("paper_id", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "relevance_score": result["score"],
                "excerpt": payload.get("chunk_text", "")[:200],
                "is_linked_paper": False
            }

            # Avoid duplicate sources
            if not any(s["paper_id"] == source_citation["paper_id"] and
                      s["chunk_index"] == source_citation["chunk_index"]
                      for s in sources):
                sources.append(source_citation)

        # Add linked paper chunks (from Tier 3 - expanded context)
        for result in linked_matches:
            payload = result.get("payload", {})

            context_pieces.append({
                "content": payload.get("chunk_text", ""),
                "source": f"[Linked] {payload.get('title', 'Unknown Paper')}",
                "relevance": result.get("score", 0.0),
                "is_linked": True,
                "source_paper_id": payload.get("source_paper_id", "")
            })

            # Build linked source citation
            source_citation = {
                "title": payload.get("title", "Unknown Paper"),
                "paper_id": payload.get("paper_id", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "relevance_score": result.get("score", 0.0),
                "excerpt": payload.get("chunk_text", "")[:200],
                "is_linked_paper": True,
                "source_paper_id": payload.get("source_paper_id", "")
            }

            # Avoid duplicate sources
            if not any(s["paper_id"] == source_citation["paper_id"] and
                      s["chunk_index"] == source_citation["chunk_index"]
                      for s in sources):
                sources.append(source_citation)

        logger.info(f"Two-tier retrieval: {len(summary_matches)} summaries, {len(search_results)} chunks, {len(linked_matches)} linked chunks")

        # STEP 4: Generate personalized response with retrieved context
        # Build context string for LLM

        # First, add high-level paper summaries for quick context
        summary_str = ""
        if summary_context:
            summary_str = "\n".join([
                f"- {s['title']}: {s['one_line']}"
                for s in summary_context[:5]
            ])

        # Then, add detailed chunks
        context_str = ""
        if context_pieces:
            # Separate main and linked papers
            main_chunks = [c for c in context_pieces if not c.get("is_linked", False)]
            linked_chunks = [c for c in context_pieces if c.get("is_linked", False)]

            if main_chunks:
                context_str = "\n\n".join([
                    f"[Relevance: {c['relevance']:.2f}] From '{c['source']}':\n{c['content'][:800]}..."
                    for c in main_chunks[:3]
                ])

            if linked_chunks:
                linked_context = "\n\n".join([
                    f"[Linked Paper - Relevance: {c['relevance']:.2f}] From '{c['source']}':\n{c['content'][:500]}..."
                    for c in linked_chunks[:2]
                ])
                context_str += f"\n\n--- Related Work (from cited papers) ---\n{linked_context}"

        if not context_str:
            context_str = "No directly relevant research papers found in the database."

        # Build focused context note if filtering to specific papers
        focused_note = ""
        if paper_ids:
            focused_note = f"""
        IMPORTANT: This is a focused chat about {len(paper_ids)} specific papers selected by the user.
        Focus your answer ONLY on content from these papers. Do not bring in information from other papers.
        If the question cannot be fully answered from these papers, acknowledge this limitation.
        """

        qa_prompt = f"""
        Answer this research question with expertise and precision:
        {focused_note}
        Question: {user_query}

        Relevant Papers Overview (summaries):
        {summary_str if summary_str else "No paper summaries available"}

        Detailed Research Context:
        {context_str}

        User's Research Profile:
        {[mem.content for mem in user_interests[:3]] if user_interests else "No prior interaction history"}

        Conversation Context:
        {conversation_context[-3:] if conversation_context else "Fresh conversation"}

        Provide a comprehensive answer that:
        1. Directly addresses the question using the retrieved research context
        2. Cites specific papers and findings from the context
        3. {"Focus ONLY on the selected papers" if paper_ids else "When relevant, mention insights from linked/cited papers (marked as [Linked])"}
        4. Acknowledges any limitations if context is insufficient
        5. Considers the user's apparent research interests
        6. Maintains academic rigor while being accessible
        7. Provides actionable insights for researchers

        If the retrieved context doesn't fully answer the question, be honest about
        limitations and suggest how to get better information.
        """

        comprehensive_answer = chat(qa_prompt)

        # STEP 5: Generate insightful follow-ups
        followup_prompt = f"""
        Based on this research Q&A interaction:

        Question: {user_query}
        Answer: {comprehensive_answer[:500]}...
        Retrieved Sources: {len(sources)} papers
        User Context: {user_interests[:3] if user_interests else "new user"}

        Suggest 3 specific, insightful follow-up questions that would:
        1. Deepen understanding of the topic
        2. Explore related research areas
        3. Connect to practical applications or future research

        Make them research-oriented, thought-provoking, and tailored to this user's interests.
        Return just the 3 questions, numbered.
        """

        followup_response = chat(followup_prompt)

        # Parse follow-up questions
        followup_questions = []
        for line in followup_response.split('\n'):
            if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 4)):
                question = line.strip()
                # Remove numbering
                for i in range(1, 4):
                    if question.startswith(f"{i}."):
                        question = question[2:].strip()
                        break
                followup_questions.append(question)

        # STEP 6: Calculate confidence based on retrieval quality
        confidence_score = 0.0

        # Base confidence from retrieval quality
        if search_results:
            avg_relevance = sum(r["score"] for r in search_results) / len(search_results)
            confidence_score += avg_relevance * 0.4  # Up to 0.4 from retrieval

            # Bonus for multiple relevant sources
            if len(search_results) >= 3:
                confidence_score += 0.2
            elif len(search_results) >= 1:
                confidence_score += 0.1

        # Bonus for user history match
        if user_interests:
            confidence_score += 0.1

        # Bonus for similar past questions
        if similar_questions:
            confidence_score += 0.1

        confidence_score = min(confidence_score, 0.95)  # Cap at 0.95

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # STEP 7: Remember interaction for personalization
        interaction_memory = f"user:{user_id} asked '{user_query}' -> answered using {len(sources)} sources"
        qa_specialist_agent.remember(interaction_memory, importance=0.8)

        # Track research interests
        interest_memory = f"user:{user_id} interest: {user_query}"
        qa_specialist_agent.remember(interest_memory, importance=0.6)

        # Remember the Q&A pattern
        qa_pattern = f"qa_pattern: {user_query} -> {len(sources)} sources, confidence {confidence_score:.2f}"
        qa_specialist_agent.remember(qa_pattern, importance=0.5)

        logger.info(f"💡 Q&A complete: answered '{user_query}' with {len(sources)} sources, confidence {confidence_score:.2f}")

        # STEP 8: Broadcast Q&A response
        broadcast({
            "type": "qa_response",
            "knowledge": {
                "user_query": user_query,
                "comprehensive_answer": comprehensive_answer,
                "followup_questions": followup_questions,
                "sources": sources if include_sources else [],
                "confidence_score": confidence_score,
                "response_time_ms": response_time_ms,
                "personalization_applied": bool(user_interests),
                "conversation_id": spore.knowledge.get("conversation_id"),
                "user_id": user_id,
                "response_metadata": {
                    "context_sources_used": len(context_pieces),
                    "unique_papers_cited": len(sources),
                    "avg_relevance_score": sum(r["score"] for r in search_results) / len(search_results) if search_results else 0.0,
                    "personalization_applied": bool(user_interests),
                    "conversation_length": len(conversation_context),
                    "similar_questions_found": len(similar_questions),
                    "vector_search_performed": True,
                    # Paper filtering for "Chat with Papers"
                    "paper_ids_filter": paper_ids,
                    "filtered_chat": bool(paper_ids),
                    # Two-tier retrieval stats
                    "two_tier_retrieval": {
                        "summaries_found": len(summary_matches),
                        "main_chunks_found": len(search_results),
                        "linked_chunks_found": len(linked_matches),
                        "relevant_papers_identified": len(retrieval_results.get("relevant_paper_ids", set()))
                    }
                }
            }
        })

    except Exception as e:
        logger.error(f"Q&A processing failed: {e}")

        # Remember failure for learning
        error_memory = f"qa_error: {user_query} - {str(e)[:100]}"
        qa_specialist_agent.remember(error_memory, importance=0.3)

        broadcast({
            "type": "qa_error",
            "knowledge": {
                "user_query": user_query,
                "error": str(e),
                "error_type": type(e).__name__,
                "user_id": user_id,
                "recovery_suggestion": "rephrase_question_or_provide_more_context"
            }
        })


def _store_research_context(spore: Spore) -> None:
    """Store research summaries for future Q&A context."""
    executive_summary = spore.knowledge.get("executive_summary", "")
    thematic_synthesis = spore.knowledge.get("thematic_synthesis", "")
    key_takeaways = spore.knowledge.get("key_takeaways", "")
    original_query = spore.knowledge.get("original_query", "")

    logger.info(f"📚 Q&A Specialist: Storing research context for '{original_query}'")

    # Store executive summary
    if executive_summary:
        summary_memory = f"research_summaries: {original_query} -> {executive_summary[:500]}..."
        qa_specialist_agent.remember(summary_memory, importance=0.9)

    # Store thematic synthesis
    if thematic_synthesis:
        themes_memory = f"research_themes: {original_query} -> {thematic_synthesis[:500]}..."
        qa_specialist_agent.remember(themes_memory, importance=0.8)

    # Store key takeaways
    if key_takeaways:
        takeaways_memory = f"research_takeaways: {original_query} -> {key_takeaways[:500]}..."
        qa_specialist_agent.remember(takeaways_memory, importance=0.8)

    # Signal Q&A readiness
    broadcast({
        "type": "qa_ready",
        "knowledge": {
            "original_query": original_query,
            "research_context_stored": True,
            "context_types": ["executive_summary", "thematic_synthesis", "key_takeaways"],
            "ready_for_questions": True
        }
    })


# Agent metadata for system introspection
AGENT_METADATA = {
    "identity": "research Q&A expert",
    "domain": "interaction",
    "capabilities": [
        "two-tier retrieval (summaries -> chunks)",
        "semantic vector search",
        "contextual question answering",
        "personalized responses",
        "source citation with paper references",
        "linked papers integration",
        "follow-up generation",
        "research guidance",
        "confidence scoring"
    ],
    "responds_to": ["user_query", "summaries_complete"],
    "broadcasts": ["qa_response", "qa_ready", "qa_error"],
    "memory_enabled": True,
    "learning_focus": "user interests and successful Q&A patterns",
    "storage_integrations": [
        "Qdrant (research_papers collection)",
        "Qdrant (paper_summaries collection)",
        "Qdrant (linked_papers collection)",
        "OpenAI Embeddings"
    ],
    "context_engineering": {
        "two_tier_retrieval": True,
        "linked_papers_integration": True
    }
}
