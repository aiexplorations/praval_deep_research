"""
Semantic Analysis Agent - Research Domain.

I am a semantic analysis specialist who analyzes research content for themes, 
patterns, and relationships between ideas across multiple papers.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timezone
from praval import agent, chat, broadcast
from praval import Spore

logger = logging.getLogger(__name__)


@agent("semantic_analyzer", responds_to=["documents_processed"], memory=True)
def semantic_analysis_agent(spore: Spore) -> None:
    """
    I am a semantic analysis specialist who analyzes research content for themes, 
    patterns, and relationships between ideas across multiple papers.
    
    My expertise:
    - Cross-paper theme identification
    - Research trend analysis
    - Conceptual relationship mapping
    - Methodological pattern recognition
    - Knowledge gap identification
    """
    processed_papers = spore.knowledge.get("processed_papers", [])
    original_query = spore.knowledge.get("original_query", "")
    processing_stats = spore.knowledge.get("processing_stats", {})
    
    if not processed_papers:
        logger.warning("Semantic analyzer received no processed papers")
        return
    
    logger.info(f"🧠 Semantic Analyzer: Analyzing {len(processed_papers)} papers for '{original_query}'")
    
    # Remember analysis session
    analysis_info = f"Semantic analysis: {len(processed_papers)} papers for '{original_query}'"
    semantic_analysis_agent.remember(analysis_info, importance=0.8)
    
    # Leverage memory for analysis patterns
    past_analyses = semantic_analysis_agent.recall("theme_patterns", limit=5)
    domain_expertise = semantic_analysis_agent.recall(f"domain_analysis", limit=3)
    
    try:
        # Extract themes across all papers
        themes_prompt = f"""
        Analyze these research papers to identify common themes and patterns:
        
        Query Context: {original_query}
        Paper Count: {len(processed_papers)}
        
        Papers Summary:
        {chr(10).join([f"- {paper.get('title', 'Untitled')[:80]}..." for paper in processed_papers[:10]])}
        
        Previous Analysis Experience: {len(past_analyses)} sessions
        Domain Expertise: {len(domain_expertise)} related analyses
        
        Identify:
        1. **Major Research Themes** (3-5 primary themes)
        2. **Methodological Patterns** (common approaches)
        3. **Conceptual Relationships** (how ideas connect)
        4. **Research Trends** (emerging directions)
        5. **Knowledge Gaps** (unexplored areas)
        6. **Cross-Paper Insights** (synthesis opportunities)
        
        Provide structured analysis focusing on actionable research insights.
        """
        
        thematic_analysis = chat(themes_prompt)
        
        # Analyze methodological approaches
        papers_list = chr(10).join([
            f"Title: {paper.get('title', '')}{chr(10)}Categories: {', '.join(paper.get('categories', []))}{chr(10)}Analysis: {paper.get('processing', {}).get('analysis', '')[:200]}...{chr(10)}"
            for paper in processed_papers[:5]
        ])
        methodology_prompt = f"""
        Focus on methodological analysis of these papers:

        {papers_list}

        Analyze:
        1. **Dominant Methodologies** (experimental, theoretical, computational, etc.)
        2. **Technical Approaches** (algorithms, models, frameworks)
        3. **Data Sources and Types** (datasets, benchmarks, evaluation methods)
        4. **Innovation Patterns** (what makes approaches novel)
        5. **Reproducibility Factors** (code availability, experimental design)
        6. **Methodological Gaps** (missing approaches or evaluations)
        
        Focus on technical depth and methodological rigor assessment.
        """
        
        methodological_analysis = chat(methodology_prompt)
        
        # Identify relationships and clusters
        relationship_prompt = f"""
        Map conceptual relationships between these research papers:
        
        Query: {original_query}
        Themes Identified: {thematic_analysis[:300]}...
        
        Create relationship mapping:
        1. **Paper Clusters** (group related papers)
        2. **Concept Networks** (how ideas connect across papers)
        3. **Citation Potential** (which papers complement each other)
        4. **Research Lineage** (building upon prior work)
        5. **Synthesis Opportunities** (papers that could be combined)
        6. **Contradiction Analysis** (conflicting findings or approaches)
        
        Focus on creating actionable insights for literature synthesis.
        """
        
        relationship_analysis = chat(relationship_prompt)
        
        # Generate semantic metadata
        semantic_metadata = {
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "papers_analyzed": len(processed_papers),
            "query_context": original_query,
            "memory_contexts_used": len(past_analyses) + len(domain_expertise),
            "analysis_depth": "comprehensive",
            "confidence_score": min(0.9, 0.5 + (len(processed_papers) * 0.05))  # Higher confidence with more papers
        }
        
        # Remember analysis patterns for future use
        theme_pattern = f"theme_patterns: {original_query} -> {len(processed_papers)} papers analyzed"
        semantic_analysis_agent.remember(theme_pattern, importance=0.8)
        
        # Extract domain for future reference
        domain = ""
        if processed_papers and processed_papers[0].get('categories'):
            domain = processed_papers[0]['categories'][0]
            domain_analysis = f"domain_analysis: {domain} analysis completed with {len(processed_papers)} papers"
            semantic_analysis_agent.remember(domain_analysis, importance=0.7)
        
        logger.info(f"🎯 Semantic analysis complete: themes, methodologies, and relationships identified")
        
        # Broadcast semantic analysis results
        broadcast({
            "type": "semantic_analysis_complete",
            "knowledge": {
                "thematic_analysis": thematic_analysis,
                "methodological_analysis": methodological_analysis,
                "relationship_analysis": relationship_analysis,
                "original_query": original_query,
                "processed_papers_count": len(processed_papers),
                "semantic_metadata": semantic_metadata,
                "analysis_summary": {
                    "themes_identified": True,
                    "methodologies_analyzed": True,
                    "relationships_mapped": True,
                    "synthesis_ready": True
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Semantic analysis failed: {e}")
        
        # Remember failure for learning
        error_memory = f"analysis_error: {original_query} - {str(e)[:100]}"
        semantic_analysis_agent.remember(error_memory, importance=0.4)
        
        broadcast({
            "type": "analysis_error",
            "knowledge": {
                "original_query": original_query,
                "error": str(e),
                "error_type": type(e).__name__,
                "papers_count": len(processed_papers),
                "recovery_suggestion": "retry_with_smaller_batch"
            }
        })


# Agent metadata for system introspection
AGENT_METADATA = {
    "identity": "semantic analysis specialist",
    "domain": "research",
    "capabilities": [
        "thematic analysis",
        "methodological pattern recognition", 
        "conceptual relationship mapping",
        "research trend identification",
        "knowledge gap analysis"
    ],
    "responds_to": ["documents_processed"],
    "memory_enabled": True,
    "learning_focus": "theme patterns and domain-specific analysis techniques"
}