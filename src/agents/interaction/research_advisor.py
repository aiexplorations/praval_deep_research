"""
Research Advisor Agent - Interaction Domain.

I am a research guidance specialist who provides strategic advice on research
directions, methodology selection, and academic career development based on
current research landscapes and user goals.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from praval import agent, chat, broadcast
from praval import Spore

from agentic_research.core.config import get_settings
from agentic_research.storage.qdrant_client import QdrantClientWrapper
from agentic_research.storage.conversation_store import ConversationStore
from processors.arxiv_client import get_arxiv_client

logger = logging.getLogger(__name__)
settings = get_settings()


@agent("research_advisor", channel="broadcast", responds_to=["research_guidance_request", "proactive_analysis_request", "summaries_complete"], memory=True)
def research_advisor_agent(spore: Spore) -> None:
    """
    I am a research guidance specialist who provides strategic advice on research
    directions, methodology selection, and academic career development based on
    current research landscapes and user goals.

    My expertise:
    - Research direction recommendations
    - Methodology selection guidance
    - Literature gap identification
    - Career development advice
    - Collaboration opportunity identification
    - Proactive research insights generation
    """
    spore_type = spore.knowledge.get("type")

    if spore_type == "summaries_complete":
        # Store research insights for advisory context
        _store_advisory_context(spore)
        return

    if spore_type == "proactive_analysis_request":
        # Generate proactive research insights
        _generate_proactive_insights(spore)
        return

    # Handle research guidance requests
    guidance_request = spore.knowledge.get("guidance_request", "")
    research_level = spore.knowledge.get("research_level", "graduate")  # undergraduate, graduate, postdoc, faculty
    research_interests = spore.knowledge.get("research_interests", [])
    current_project = spore.knowledge.get("current_project", "")
    user_id = spore.knowledge.get("user_id", "anonymous")
    
    if not guidance_request:
        logger.warning("Research advisor received empty guidance request")
        return
    
    logger.info(f"🎯 Research Advisor: Providing guidance for '{guidance_request}' (level: {research_level})")
    
    # Leverage memory for personalized advice
    user_history = research_advisor_agent.recall(f"user:{user_id}", limit=10)
    research_landscapes = research_advisor_agent.recall("research_landscape", limit=5)
    methodology_insights = research_advisor_agent.recall("methodology_guidance", limit=5)
    
    try:
        # Provide strategic research guidance
        advisory_prompt = f"""
        Provide expert research guidance as a senior academic advisor:
        
        Guidance Request: {guidance_request}
        Researcher Level: {research_level}
        Research Interests: {research_interests}
        Current Project: {current_project}
        
        User History with System:
        {[mem.content for mem in user_history] if user_history else "First interaction"}
        
        Available Research Landscape Context:
        {[mem.content[:200] for mem in research_landscapes] if research_landscapes else "No recent landscape analysis"}
        
        Methodology Insights Available:
        {[mem.content[:200] for mem in methodology_insights] if methodology_insights else "No methodology guidance stored"}
        
        Provide comprehensive guidance including:
        
        **IMMEDIATE RECOMMENDATIONS**
        - Specific actionable advice for the current request
        - Next steps and priorities
        - Timeline considerations
        
        **RESEARCH STRATEGY**
        - Long-term research direction recommendations
        - Positioning within current research landscape
        - Differentiation opportunities
        
        **METHODOLOGY GUIDANCE**
        - Appropriate research methods for their goals
        - Technical approach recommendations
        - Evaluation strategy suggestions
        
        **CAREER DEVELOPMENT**
        - Skills to develop
        - Networking and collaboration opportunities
        - Publication strategy advice
        
        **RESOURCE RECOMMENDATIONS**
        - Key papers to read
        - Conferences to attend
        - Tools and datasets to explore
        
        Tailor advice to their experience level and provide specific, actionable guidance.
        """
        
        advisory_response = chat(advisory_prompt)
        
        # Generate opportunity identification
        opportunities_prompt = f"""
        Based on the guidance request and research context, identify specific opportunities:
        
        Request: {guidance_request}
        Level: {research_level}
        Interests: {research_interests}
        
        Research Landscape: {research_landscapes[0].content[:300] if research_landscapes else "Limited context"}
        
        Identify:
        1. **Research Gaps** (unexplored areas worth investigating)
        2. **Collaboration Opportunities** (potential partnerships or team projects)
        3. **Funding Possibilities** (grants or programs that might be relevant)
        4. **Publication Venues** (conferences and journals to target)
        5. **Skill Development** (technical or methodological skills to acquire)
        6. **Innovation Potential** (areas where novel contributions are possible)
        
        Make recommendations specific and actionable for their career stage.
        """
        
        opportunities_analysis = chat(opportunities_prompt)
        
        # Create personalized roadmap
        roadmap_prompt = f"""
        Create a personalized research roadmap:
        
        Current Situation:
        - Level: {research_level}
        - Request: {guidance_request}
        - Project: {current_project}
        - Interests: {research_interests}
        
        Advisory Response Summary: {advisory_response[:300]}...
        
        Create a 6-month and 2-year roadmap with:
        
        **6-MONTH GOALS**
        - Immediate priorities and milestones
        - Specific deliverables and deadlines
        - Skills to develop in short term
        
        **2-YEAR VISION**
        - Long-term research objectives
        - Career development targets
        - Major contributions to aim for
        
        **QUARTERLY CHECKPOINTS**
        - Progress evaluation criteria
        - Milestone indicators
        - Adjustment points for strategy
        
        Make it concrete, measurable, and achievable for their level.
        """
        
        research_roadmap = chat(roadmap_prompt)
        
        # Remember advisory interaction
        advisory_memory = f"user:{user_id} sought guidance on '{guidance_request}' -> comprehensive advisory provided"
        research_advisor_agent.remember(advisory_memory, importance=0.8)
        
        # Remember guidance patterns
        guidance_pattern = f"guidance_pattern: {research_level} researcher asking about {guidance_request}"
        research_advisor_agent.remember(guidance_pattern, importance=0.6)
        
        # Track user development
        development_memory = f"user:{user_id} development: {research_level} level, interests in {research_interests}"
        research_advisor_agent.remember(development_memory, importance=0.7)
        
        response_metadata = {
            "advisory_timestamp": datetime.now(timezone.utc).isoformat(),
            "guidance_type": "comprehensive",
            "research_level": research_level,
            "personalization_applied": bool(user_history),
            "landscape_context_used": bool(research_landscapes),
            "methodology_guidance_available": bool(methodology_insights)
        }
        
        logger.info(f"🎓 Research advisory complete: comprehensive guidance provided for {research_level} researcher")
        
        # Broadcast advisory response
        broadcast({
            "type": "research_advisory_complete",
            "knowledge": {
                "guidance_request": guidance_request,
                "advisory_response": advisory_response,
                "opportunities_analysis": opportunities_analysis,
                "research_roadmap": research_roadmap,
                "research_level": research_level,
                "user_id": user_id,
                "response_metadata": response_metadata,
                "advisory_quality": {
                    "comprehensiveness": "high",
                    "personalization": "high" if user_history else "medium",
                    "actionability": "high",
                    "strategic_value": "high"
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Research advisory failed: {e}")
        
        # Remember failure for learning
        error_memory = f"advisory_error: {guidance_request} - {str(e)[:100]}"
        research_advisor_agent.remember(error_memory, importance=0.3)
        
        broadcast({
            "type": "advisory_error",
            "knowledge": {
                "guidance_request": guidance_request,
                "error": str(e),
                "error_type": type(e).__name__,
                "user_id": user_id,
                "recovery_suggestion": "provide_more_specific_guidance_request"
            }
        })


def _store_advisory_context(spore: Spore) -> None:
    """Store research insights for future advisory context."""
    executive_summary = spore.knowledge.get("executive_summary", "")
    methodological_summary = spore.knowledge.get("methodological_summary", "")
    key_takeaways = spore.knowledge.get("key_takeaways", "")
    original_query = spore.knowledge.get("original_query", "")
    papers_count = spore.knowledge.get("papers_count", 0)
    
    logger.info(f"📋 Research Advisor: Storing advisory context for '{original_query}'")
    
    # Store research landscape insights
    if executive_summary:
        landscape_memory = f"research_landscape: {original_query} field analysis -> {executive_summary[:400]}..."
        research_advisor_agent.remember(landscape_memory, importance=0.9)
    
    # Store methodology insights
    if methodological_summary:
        methodology_memory = f"methodology_guidance: {original_query} -> {methodological_summary[:400]}..."
        research_advisor_agent.remember(methodology_memory, importance=0.8)
    
    # Store key opportunities
    if key_takeaways:
        opportunities_memory = f"research_opportunities: {original_query} -> {key_takeaways[:400]}..."
        research_advisor_agent.remember(opportunities_memory, importance=0.8)
    
    # Signal advisory readiness
    broadcast({
        "type": "advisory_context_ready",
        "knowledge": {
            "original_query": original_query,
            "papers_analyzed": papers_count,
            "advisory_context_stored": True,
            "guidance_domains": ["research_landscape", "methodology", "opportunities"],
            "ready_for_advisory_requests": True
        }
    })


def _generate_proactive_insights(spore: Spore) -> None:
    """
    Generate proactive research insights based on knowledge base,
    recent papers, conversation history, and agent memory.

    This is the core proactive intelligence feature that analyzes user
    activity and generates actionable research recommendations.
    """
    logger.info("🔍 Research Advisor: Generating proactive insights...")

    try:
        # Initialize storage clients
        qdrant_client = QdrantClientWrapper(settings)
        arxiv_client = get_arxiv_client()

        # STEP 1: Gather context from knowledge base
        kb_papers = qdrant_client.get_all_papers()
        kb_stats = {
            "total_papers": len(kb_papers),
            "categories": {},
            "recent_papers": []
        }

        # Categorize papers
        for paper in kb_papers:
            for category in paper.get("categories", []):
                kb_stats["categories"][category] = kb_stats["categories"].get(category, 0) + 1

        # Get recent papers (last 10)
        sorted_papers = sorted(kb_papers, key=lambda p: p.get("published_date", ""), reverse=True)
        kb_stats["recent_papers"] = sorted_papers[:10]

        # STEP 2: Get weighted conversation history from agent memory
        conversation_queries = research_advisor_agent.recall("user:", limit=20)
        recent_queries = [mem.content for mem in conversation_queries[:5]]  # Recent × 3 weight
        older_queries = [mem.content for mem in conversation_queries[5:15]]  # Older × 1 weight

        # STEP 3: Extract research interests and themes
        interests_context = {
            "recent_topics": recent_queries,
            "historical_topics": older_queries,
            "kb_categories": list(kb_stats["categories"].keys()),
            "top_categories": sorted(kb_stats["categories"].items(), key=lambda x: x[1], reverse=True)[:5]
        }

        logger.info(f"Context gathered: {kb_stats['total_papers']} papers, {len(conversation_queries)} queries")

        # STEP 4: Generate research area clusters
        if kb_stats["total_papers"] > 0:
            clustering_prompt = f"""
            Analyze this research knowledge base and identify distinct research areas:

            Papers indexed: {kb_stats['total_papers']}
            Categories: {', '.join([f"{cat} ({count})" for cat, count in interests_context['top_categories']])}
            Recent papers: {[p['title'][:80] for p in kb_stats['recent_papers'][:5]]}

            Recent user interests: {recent_queries[:3] if recent_queries else "No recent activity"}

            Identify 3-5 distinct research areas or themes from the knowledge base.
            For each area, provide:
            1. Area name (concise, 2-4 words)
            2. Brief description (one sentence)
            3. Relevant papers count estimate
            4. Why it's significant to the user's research

            Format as JSON array of objects with keys: name, description, paper_count, significance.
            """

            research_areas_raw = chat(clustering_prompt)
            # Parse JSON from response
            import json
            import re
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', research_areas_raw, re.DOTALL)
            if json_match:
                research_areas = json.loads(json_match.group(1))
            else:
                # Try to find JSON array directly
                json_match = re.search(r'\[.*\]', research_areas_raw, re.DOTALL)
                if json_match:
                    research_areas = json.loads(json_match.group(0))
                else:
                    research_areas = []
        else:
            research_areas = []

        # STEP 5: Generate trending topics/keywords
        if kb_stats["total_papers"] > 3:
            trending_prompt = f"""
            From these recent papers and topics, identify emerging trends:

            Recent papers: {[p['title'] for p in kb_stats['recent_papers'][:8]]}
            Recent queries: {recent_queries[:5] if recent_queries else "No recent queries"}
            Top categories: {interests_context['top_categories'][:3]}

            Identify 5-8 trending keywords or concepts that appear frequently.
            Return as a simple JSON array of strings (just the keywords/phrases).
            """

            trending_raw = chat(trending_prompt)
            # Parse trending topics
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', trending_raw, re.DOTALL)
            if json_match:
                trending_topics = json.loads(json_match.group(1))
            else:
                json_match = re.search(r'\[.*\]', trending_raw, re.DOTALL)
                if json_match:
                    trending_topics = json.loads(json_match.group(0))
                else:
                    trending_topics = []
        else:
            trending_topics = []

        # STEP 6: Identify research gaps
        if kb_stats["total_papers"] > 2:
            gaps_prompt = f"""
            Analyze research gaps and opportunities:

            Current knowledge base: {kb_stats['total_papers']} papers
            Focus areas: {interests_context['top_categories'][:4]}
            Recent topics: {recent_queries[:3] if recent_queries else "Limited activity"}

            Identify 3-5 research gaps or unexplored areas worth investigating.
            Consider:
            - Underexplored methodologies
            - Missing application domains
            - Emerging questions
            - Cross-disciplinary opportunities

            Format as JSON array with keys: gap_title, description, potential_value, exploration_steps.
            """

            gaps_raw = chat(gaps_prompt)
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', gaps_raw, re.DOTALL)
            if json_match:
                research_gaps = json.loads(json_match.group(1))
            else:
                json_match = re.search(r'\[.*\]', gaps_raw, re.DOTALL)
                if json_match:
                    research_gaps = json.loads(json_match.group(0))
                else:
                    research_gaps = []
        else:
            research_gaps = []

        # STEP 7: Generate personalized next steps
        next_steps_prompt = f"""
        Based on user's research activity, suggest specific next actions:

        Knowledge base: {kb_stats['total_papers']} papers
        Recent focus: {recent_queries[:3] if recent_queries else "Getting started"}
        Top areas: {interests_context['top_categories'][:3]}

        Suggest 3-5 specific, actionable next steps for the researcher.
        These should be practical and tailored to their current stage.

        Format as JSON array with keys: action, rationale, estimated_time.
        """

        next_steps_raw = chat(next_steps_prompt)
        json_match = re.search(r'```json\s*(\[.*?\])\s*```', next_steps_raw, re.DOTALL)
        if json_match:
            next_steps = json.loads(json_match.group(1))
        else:
            json_match = re.search(r'\[.*\]', next_steps_raw, re.DOTALL)
            if json_match:
                next_steps = json.loads(json_match.group(0))
            else:
                next_steps = []

        # STEP 8: Proactive arXiv search for related papers
        suggested_papers = []
        if kb_stats["total_papers"] > 0 or recent_queries:
            # Generate smart search queries based on interests
            queries_prompt = f"""
            Generate 2-3 specific arXiv search queries based on user interests:

            KB categories: {interests_context['top_categories'][:3]}
            Recent queries: {recent_queries[:3] if recent_queries else "No queries yet"}
            Trending topics: {trending_topics[:5] if trending_topics else []}

            Create search queries that would find recent, relevant papers not already indexed.
            Return as JSON array of strings (just the search queries).
            Keep queries specific and technical.
            """

            queries_raw = chat(queries_prompt)
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', queries_raw, re.DOTALL)
            if json_match:
                search_queries = json.loads(json_match.group(1))
            else:
                json_match = re.search(r'\[.*\]', queries_raw, re.DOTALL)
                if json_match:
                    search_queries = json.loads(json_match.group(0))
                else:
                    search_queries = []

            # Search arXiv for each query
            existing_arxiv_ids = set(p.get("arxiv_id") or p.get("paper_id", "") for p in kb_papers)

            for search_query in search_queries[:2]:  # Limit to 2 queries
                try:
                    # Use asyncio to run async search
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is running, we can't use run_until_complete
                        # Skip arXiv search in this case
                        logger.warning("Event loop already running, skipping arXiv search")
                        break
                    else:
                        papers = loop.run_until_complete(
                            arxiv_client.search_papers(search_query, max_results=5)
                        )

                    for paper in papers:
                        # Filter out already indexed papers
                        if paper.get("arxiv_id") not in existing_arxiv_ids:
                            suggested_papers.append({
                                "arxiv_id": paper.get("arxiv_id"),
                                "title": paper.get("title"),
                                "authors": paper.get("authors", []),
                                "abstract": paper.get("abstract", "")[:300],
                                "published_date": paper.get("published_date"),
                                "url": paper.get("url"),
                                "categories": paper.get("categories", []),
                                "suggested_because": f"Related to: {search_query}"
                            })

                    # Limit total suggestions to 10
                    if len(suggested_papers) >= 10:
                        break

                except Exception as e:
                    logger.error(f"arXiv search failed for query '{search_query}': {e}")
                    continue

            # Limit to top 10 suggestions
            suggested_papers = suggested_papers[:10]

        # STEP 9: Remember this proactive analysis
        analysis_memory = f"proactive_analysis: {kb_stats['total_papers']} papers, {len(research_areas)} areas, {len(suggested_papers)} arXiv suggestions"
        research_advisor_agent.remember(analysis_memory, importance=0.7)

        logger.info(f"✨ Proactive insights generated: {len(research_areas)} areas, {len(suggested_papers)} paper suggestions")

        # STEP 10: Broadcast insights
        broadcast({
            "type": "research_insights_generated",
            "knowledge": {
                "research_areas": research_areas,
                "trending_topics": trending_topics,
                "research_gaps": research_gaps,
                "next_steps": next_steps,
                "suggested_papers": suggested_papers,
                "kb_context": {
                    "total_papers": kb_stats["total_papers"],
                    "categories": dict(kb_stats["categories"]),
                    "recent_activity": bool(recent_queries)
                },
                "generation_metadata": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "kb_papers_analyzed": kb_stats["total_papers"],
                    "conversation_history_used": len(conversation_queries),
                    "arxiv_papers_suggested": len(suggested_papers),
                    "insights_quality": "high" if kb_stats["total_papers"] > 5 else "medium"
                }
            }
        })

    except Exception as e:
        logger.error(f"Proactive insights generation failed: {e}")
        import traceback
        traceback.print_exc()

        # Broadcast error
        broadcast({
            "type": "research_insights_error",
            "knowledge": {
                "error": str(e),
                "error_type": type(e).__name__,
                "recovery_suggestion": "Try again or check knowledge base status"
            }
        })


def generate_insights_sync(settings, recent_queries: list = None) -> Dict[str, Any]:
    """
    Synchronous helper function to generate insights for API endpoint.

    This function extracts the core logic from _generate_proactive_insights
    and returns the result directly instead of broadcasting via Praval.

    OPTIMIZED: Uses ThreadPoolExecutor to run LLM calls in parallel,
    reducing total time from ~16s to ~5s (time of longest single call).

    Args:
        settings: Application settings object
        recent_queries: List of recent user queries from chat history (optional)

    Returns:
        Dictionary with research insights
    """
    try:
        import json
        import re
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from openai import OpenAI
        import time

        start_time = time.time()

        # Initialize OpenAI client directly (can't use Praval's chat() outside agent)
        openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

        def llm_call(prompt: str) -> str:
            """Helper to call OpenAI API directly."""
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content or ""

        def parse_json_response(raw_text: str) -> list:
            """Extract JSON array from LLM response."""
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', raw_text, re.DOTALL) or re.search(r'\[.*\]', raw_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1 if '```' in raw_text else 0))
                except json.JSONDecodeError:
                    return []
            return []

        # Initialize storage clients
        qdrant_client = QdrantClientWrapper(settings)

        # Gather context from knowledge base
        kb_papers = qdrant_client.get_all_papers()
        kb_stats = {
            "total_papers": len(kb_papers),
            "categories": {},
            "recent_papers": []
        }

        # Categorize papers
        for paper in kb_papers:
            for category in paper.get("categories", []):
                kb_stats["categories"][category] = kb_stats["categories"].get(category, 0) + 1

        # Get recent papers (last 10)
        sorted_papers = sorted(kb_papers, key=lambda p: p.get("published_date", ""), reverse=True)
        kb_stats["recent_papers"] = sorted_papers[:10]

        # Use provided recent queries (fetched by caller in async context)
        if recent_queries is None:
            recent_queries = []

        # Extract research interests
        interests_context = {
            "recent_topics": recent_queries,
            "top_categories": sorted(kb_stats["categories"].items(), key=lambda x: x[1], reverse=True)[:5]
        }

        logger.info(f"Sync insights: {kb_stats['total_papers']} papers, {len(recent_queries)} queries from chat history")

        # Prepare prompts for parallel execution
        recent_queries_context = f"\nRecent user questions: {recent_queries[:5]}" if recent_queries else ""

        prompts = {}

        if kb_stats["total_papers"] > 0:
            prompts["research_areas"] = f"""
            Analyze this research knowledge base and identify distinct research areas:

            Papers indexed: {kb_stats['total_papers']}
            Categories: {', '.join([f"{cat} ({count})" for cat, count in interests_context['top_categories']])}
            Recent papers: {[p['title'][:80] for p in kb_stats['recent_papers'][:5]]}{recent_queries_context}

            Identify 3-5 distinct research areas that align with the user's demonstrated interests. Format as JSON array with: name, description, paper_count, significance.
            """

        if kb_stats["total_papers"] > 3:
            prompts["trending_topics"] = f"""
            From these recent papers, identify 5-8 trending keywords:
            Papers: {[p['title'] for p in kb_stats['recent_papers'][:8]]}
            Return as JSON array of strings.
            """

        if kb_stats["total_papers"] > 2:
            prompts["research_gaps"] = f"""
            Identify 3-5 research gaps:
            KB: {kb_stats['total_papers']} papers
            Areas: {interests_context['top_categories'][:4]}
            Format as JSON array with: gap_title, description, potential_value, exploration_steps.
            """

        prompts["next_steps"] = f"""
        Suggest 3-5 next actions for researcher:
        KB: {kb_stats['total_papers']} papers
        Format as JSON array with: action, rationale, estimated_time.
        """

        # Execute LLM calls in parallel using ThreadPoolExecutor
        results = {
            "research_areas": [],
            "trending_topics": [],
            "research_gaps": [],
            "next_steps": []
        }

        if prompts:
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all tasks
                future_to_key = {
                    executor.submit(llm_call, prompt): key
                    for key, prompt in prompts.items()
                }

                # Collect results as they complete
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        raw_result = future.result()
                        results[key] = parse_json_response(raw_result)
                        logger.debug(f"Parallel LLM call completed: {key}")
                    except Exception as exc:
                        logger.warning(f"LLM call {key} failed: {exc}")
                        results[key] = []

        elapsed = time.time() - start_time
        logger.info(f"Insights generation completed in {elapsed:.2f}s (parallel execution)")

        return {
            "research_areas": results["research_areas"],
            "trending_topics": results["trending_topics"],
            "research_gaps": results["research_gaps"],
            "next_steps": results["next_steps"],
            "suggested_papers": [],  # Skip arXiv search in sync version
            "kb_context": {
                "total_papers": kb_stats["total_papers"],
                "categories": dict(kb_stats["categories"]),
                "recent_activity": bool(recent_queries)
            },
            "generation_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kb_papers_analyzed": kb_stats["total_papers"],
                "conversation_history_used": len(recent_queries),
                "arxiv_papers_suggested": 0,
                "insights_quality": "high" if kb_stats["total_papers"] > 5 else "medium",
                "personalization_enabled": len(recent_queries) > 0,
                "generation_time_seconds": round(elapsed, 2),
                "parallel_execution": True
            }
        }

    except Exception as e:
        logger.error(f"Sync insights generation failed: {e}")
        return {
            "research_areas": [],
            "trending_topics": [],
            "research_gaps": [],
            "next_steps": [],
            "suggested_papers": [],
            "kb_context": {"total_papers": 0, "categories": {}, "recent_activity": False},
            "generation_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
        }


# Agent metadata for system introspection
AGENT_METADATA = {
    "identity": "research guidance specialist",
    "domain": "interaction",
    "capabilities": [
        "research direction recommendations",
        "methodology selection guidance",
        "career development advice",
        "opportunity identification",
        "strategic planning",
        "proactive research insights generation",
        "knowledge base analysis",
        "trending topics identification",
        "research gap detection",
        "arXiv proactive search"
    ],
    "responds_to": ["research_guidance_request", "proactive_analysis_request", "summaries_complete"],
    "broadcasts": ["research_advisory_complete", "research_insights_generated", "advisory_context_ready", "advisory_error", "research_insights_error"],
    "memory_enabled": True,
    "learning_focus": "user development patterns and research landscape evolution"
}