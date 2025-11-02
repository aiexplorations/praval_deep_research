"""
Paper Discovery Agent - Research Domain.

I am a research paper discovery specialist who excels at finding relevant 
academic papers by understanding research contexts and optimizing queries 
based on past successful searches.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from praval import agent, chat, broadcast, Spore

# Import processors with proper path handling
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', '..')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from processors.arxiv_client import search_arxiv_papers, calculate_paper_relevance, ArXivAPIError

logger = logging.getLogger(__name__)


@agent("paper_searcher", responds_to=["search_request"], memory=True)
def paper_discovery_agent(spore: Spore) -> None:
    """
    I am a research paper discovery specialist. I excel at finding relevant
    academic papers by understanding research contexts and optimizing queries
    based on past successful searches.

    My expertise:
    - Intelligent query expansion using past searches
    - Domain-specific search optimization
    - Quality filtering and relevance scoring
    - Learning from successful search patterns
    """
    logger.info("=" * 80)
    logger.info("🚀 PAPER DISCOVERY AGENT TRIGGERED!")
    logger.info("=" * 80)
    logger.info(f"📥 Received spore: {spore}")
    logger.info(f"📦 Spore knowledge: {spore.knowledge}")
    logger.info(f"🏷️  Spore type: {type(spore)}")

    query = spore.knowledge.get("query")
    domain = spore.knowledge.get("domain", "computer_science")
    max_results = spore.knowledge.get("max_results", 10)

    logger.info(f"🔍 Extracted from spore:")
    logger.info(f"   Query: {query}")
    logger.info(f"   Domain: {domain}")
    logger.info(f"   Max results: {max_results}")

    if not query:
        logger.warning("⚠️ Paper discovery agent received empty query - EXITING")
        return

    logger.info(f"✅ STARTING Paper Discovery: Processing search for '{query}' in {domain}")
    
    # Memory-driven query optimization
    past_searches = paper_discovery_agent.recall(f"domain:{domain}", limit=5)
    successful_patterns = paper_discovery_agent.recall("successful_query", limit=10)
    
    optimized_query = query  # Default to original
    
    if past_searches or successful_patterns:
        optimization_context = {
            "domain_history": [mem.content for mem in past_searches],
            "successful_patterns": [mem.content for mem in successful_patterns],
            "original_query": query
        }
        
        logger.info(f"📚 Using memory context: {len(past_searches)} domain searches, {len(successful_patterns)} successful patterns")
        
        try:
            optimized_query = chat(f"""
            As a research query optimization expert, enhance this query:
            
            Original: {query}
            Domain: {domain}
            Historical context: {optimization_context}
            
            Provide an optimized query that will find more relevant papers.
            Consider synonyms, related terms, and field-specific terminology.
            Return only the enhanced query.
            """)
            
            # Clean up the response
            optimized_query = optimized_query.strip().strip('"\'')
            
        except Exception as e:
            logger.warning(f"Query optimization failed: {e}, using original query")
            optimized_query = query
    
    # Execute search with error handling
    try:
        # Run async search in sync context (Praval agents are sync)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        papers = loop.run_until_complete(
            search_arxiv_papers(
                query=optimized_query.strip(),
                max_results=max_results,
                domain=domain.lower().replace(" ", "_")
            )
        )
        loop.close()
        
        # Calculate relevance scores and apply quality threshold
        quality_threshold = spore.knowledge.get("quality_threshold", 0.0)
        for paper in papers:
            paper['relevance'] = calculate_paper_relevance(paper, query)
        
        # Filter by quality and sort by relevance
        filtered_papers = [
            paper for paper in papers 
            if paper.get('relevance', 0.0) >= quality_threshold
        ]
        filtered_papers.sort(key=lambda p: p.get('relevance', 0.0), reverse=True)
        
        # Learn from successful searches
        if filtered_papers:
            success_memory = f"successful_query: {query} -> {len(filtered_papers)} quality papers"
            paper_discovery_agent.remember(success_memory, importance=0.8)
            
            domain_memory = f"domain:{domain} -> optimized: {optimized_query}"
            paper_discovery_agent.remember(domain_memory, importance=0.7)
            
            # Remember top papers for future reference
            top_papers = filtered_papers[:3]
            for paper in top_papers:
                paper_memory = f"high_quality_paper: {paper.get('title', '')} (relevance: {paper.get('relevance', 0.0):.2f})"
                paper_discovery_agent.remember(paper_memory, importance=0.6)
        
        logger.info(f"📄 Found {len(papers)} papers, {len(filtered_papers)} after quality filtering")

        # Prepare broadcast payload - flatten structure for document processor
        broadcast_payload = {
            "type": "papers_found",
            "papers": filtered_papers,
            "original_query": query,
            "optimized_query": optimized_query,
            "search_metadata": {
                "domain": domain,
                "optimization_used": optimized_query != query,
                "results_count": len(filtered_papers),
                "quality_threshold": quality_threshold,
                "memory_contexts_used": len(past_searches) + len(successful_patterns)
            }
        }

        logger.info(f"📡 BROADCASTING papers_found event")
        logger.info(f"   Papers count: {len(filtered_papers)}")
        logger.info(f"   Broadcast type: papers_found")
        logger.info(f"   Target channel: document_processor_channel")

        # Broadcast results via spore communication to document processor
        from praval import get_reef
        reef = get_reef()
        broadcast_result = reef.broadcast(
            from_agent='paper_searcher',
            knowledge=broadcast_payload,
            channel='document_processor_channel'
        )

        logger.info(f"✅ BROADCAST COMPLETE - Result: {broadcast_result}")
        logger.info("=" * 80)
        
    except ArXivAPIError as e:
        logger.error(f"ArXiv API error: {e}")
        paper_discovery_agent.remember(f"failed_query: {query} - ArXiv API error: {str(e)}", importance=0.3)
        
        broadcast({
            "type": "search_error",
            "knowledge": {
                "query": query,
                "error": str(e),
                "error_type": "ArXivAPIError",
                "recovery_suggestion": "retry_with_different_terms"
            }
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in paper discovery: {e}")
        paper_discovery_agent.remember(f"failed_query: {query} - Unexpected error: {str(e)}", importance=0.3)
        
        broadcast({
            "type": "search_error", 
            "knowledge": {
                "query": query,
                "error": str(e),
                "error_type": type(e).__name__,
                "recovery_suggestion": "contact_support"
            }
        })


# Agent metadata for system introspection
AGENT_METADATA = {
    "identity": "research paper discovery specialist",
    "domain": "research",
    "capabilities": [
        "academic paper search",
        "query optimization", 
        "relevance scoring",
        "quality filtering",
        "memory-driven learning"
    ],
    "responds_to": ["search_request"],
    "memory_enabled": True,
    "learning_focus": "successful search patterns and domain expertise"
}