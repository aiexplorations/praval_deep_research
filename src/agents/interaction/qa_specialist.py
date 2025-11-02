"""
Q&A Specialist Agent - Interaction Domain.

I am a research Q&A expert who provides comprehensive, accurate answers
about research papers using retrieved context and accumulated knowledge
to deliver personalized, insightful responses.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from praval import agent, chat, broadcast
from praval import Spore

from agentic_research.core.config import get_settings
from agentic_research.storage.qdrant_client import QdrantClientWrapper
from agentic_research.storage.embeddings import EmbeddingsGenerator


logger = logging.getLogger(__name__)
settings = get_settings()


@agent("qa_specialist", responds_to=["user_query", "summaries_complete"], memory=True)
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

    if not user_query:
        logger.warning("Q&A specialist received empty query")
        return

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

        # STEP 1: Generate query embedding
        logger.debug(f"Generating query embedding for: {user_query}")
        query_embedding = embeddings_gen.generate_embedding(user_query)

        # STEP 2: Search Qdrant for relevant chunks
        logger.debug("Searching Qdrant for relevant paper chunks...")
        search_results = qdrant_client.search_similar(
            query_vector=query_embedding,
            limit=5,
            score_threshold=0.7
        )

        # STEP 3: Build context from retrieved chunks
        context_pieces = []
        sources = []

        for result in search_results:
            payload = result["payload"]

            context_pieces.append({
                "content": payload.get("chunk_text", ""),
                "source": payload.get("title", "Unknown Paper"),
                "relevance": result["score"]
            })

            # Build source citation
            source_citation = {
                "title": payload.get("title", "Unknown Paper"),
                "paper_id": payload.get("paper_id", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "relevance_score": result["score"],
                "excerpt": payload.get("chunk_text", "")[:200]  # First 200 chars
            }

            # Avoid duplicate sources (same paper, different chunks)
            if not any(s["paper_id"] == source_citation["paper_id"] and
                      s["chunk_index"] == source_citation["chunk_index"]
                      for s in sources):
                sources.append(source_citation)

        logger.info(f"Retrieved {len(context_pieces)} relevant chunks from {len(sources)} papers")

        # STEP 4: Generate personalized response with retrieved context
        # Build context string for LLM
        context_str = ""
        if context_pieces:
            context_str = "\n\n".join([
                f"[Relevance: {c['relevance']:.2f}] From '{c['source']}':\n{c['content'][:800]}..."
                for c in context_pieces[:3]  # Top 3 most relevant
            ])
        else:
            context_str = "No directly relevant research papers found in the database."

        qa_prompt = f"""
        Answer this research question with expertise and precision:

        Question: {user_query}

        Retrieved Research Context:
        {context_str}

        User's Research Profile:
        {[mem.content for mem in user_interests[:3]] if user_interests else "No prior interaction history"}

        Conversation Context:
        {conversation_context[-3:] if conversation_context else "Fresh conversation"}

        Provide a comprehensive answer that:
        1. Directly addresses the question using the retrieved research context
        2. Cites specific papers and findings from the context
        3. Acknowledges any limitations if context is insufficient
        4. Considers the user's apparent research interests
        5. Maintains academic rigor while being accessible
        6. Provides actionable insights for researchers

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
                    "vector_search_performed": True
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
        "semantic vector search",
        "contextual question answering",
        "personalized responses",
        "source citation with paper references",
        "follow-up generation",
        "research guidance",
        "confidence scoring"
    ],
    "responds_to": ["user_query", "summaries_complete"],
    "broadcasts": ["qa_response", "qa_ready", "qa_error"],
    "memory_enabled": True,
    "learning_focus": "user interests and successful Q&A patterns",
    "storage_integrations": ["Qdrant", "OpenAI Embeddings"]
}
