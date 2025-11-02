"""
Summarization Agent - Research Domain.

I am a summarization specialist who creates coherent, comprehensive summaries 
that synthesize insights across multiple research papers.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timezone
from praval import agent, chat, broadcast
from praval import Spore

logger = logging.getLogger(__name__)


@agent("summarizer", responds_to=["semantic_analysis_complete"], memory=True)
def summarization_agent(spore: Spore) -> None:
    """
    I am a summarization specialist who creates coherent, comprehensive summaries 
    that synthesize insights across multiple research papers.
    
    My expertise:
    - Multi-paper synthesis and integration
    - Thematic summarization across domains
    - Executive summary generation
    - Key insight extraction and organization
    - Research narrative construction
    """
    thematic_analysis = spore.knowledge.get("thematic_analysis", "")
    methodological_analysis = spore.knowledge.get("methodological_analysis", "")
    relationship_analysis = spore.knowledge.get("relationship_analysis", "")
    original_query = spore.knowledge.get("original_query", "")
    papers_count = spore.knowledge.get("processed_papers_count", 0)
    semantic_metadata = spore.knowledge.get("semantic_metadata", {})
    
    if not thematic_analysis or not methodological_analysis:
        logger.warning("Summarization agent received incomplete analysis")
        return
    
    logger.info(f"📝 Summarization: Creating synthesis for '{original_query}' ({papers_count} papers)")
    
    # Remember summarization session
    summary_info = f"Summarization: {papers_count} papers for '{original_query}'"
    summarization_agent.remember(summary_info, importance=0.8)
    
    # Leverage memory for summarization patterns
    past_summaries = summarization_agent.recall("summary_patterns", limit=5)
    domain_summaries = summarization_agent.recall("domain_synthesis", limit=3)
    
    try:
        # Create executive summary
        executive_summary_prompt = f"""
        Create a comprehensive executive summary for this research analysis:
        
        Research Query: {original_query}
        Papers Analyzed: {papers_count}
        
        Analysis Components:
        1. Thematic Analysis: {thematic_analysis[:500]}...
        2. Methodological Analysis: {methodological_analysis[:500]}...
        3. Relationship Analysis: {relationship_analysis[:500]}...
        
        Previous Summarization Experience: {len(past_summaries)} sessions
        Domain Expertise: {len(domain_summaries)} domain syntheses
        
        Create an executive summary that includes:
        
        **RESEARCH LANDSCAPE OVERVIEW**
        - Current state of research in this area
        - Major themes and research directions
        - Key methodological approaches
        
        **KEY FINDINGS & INSIGHTS**
        - Most significant discoveries and contributions
        - Methodological innovations and advances
        - Cross-paper insights and patterns
        
        **RESEARCH GAPS & OPPORTUNITIES**
        - Identified knowledge gaps
        - Unexplored research directions
        - Methodological improvements needed
        
        **PRACTICAL IMPLICATIONS**
        - Real-world applications and impact
        - Implementation considerations
        - Future research priorities
        
        Make it comprehensive yet accessible, focusing on actionable insights.
        """
        
        executive_summary = chat(executive_summary_prompt)
        
        # Create thematic synthesis
        thematic_synthesis_prompt = f"""
        Create detailed thematic synthesis from the analysis:
        
        Thematic Analysis: {thematic_analysis}
        Query Context: {original_query}
        
        For each major theme identified, provide:
        1. **Theme Description** (what it encompasses)
        2. **Key Contributions** (major findings in this theme)
        3. **Methodological Approaches** (how research is conducted)
        4. **Research Evolution** (how the theme has developed)
        5. **Open Questions** (what remains to be explored)
        6. **Interconnections** (how this theme relates to others)
        
        Create a scholarly synthesis that researchers can use for literature reviews.
        """
        
        thematic_synthesis = chat(thematic_synthesis_prompt)
        
        # Create methodological summary
        methodological_summary_prompt = f"""
        Summarize methodological landscape from the analysis:
        
        Methodological Analysis: {methodological_analysis}
        Paper Count: {papers_count}
        
        Provide:
        **DOMINANT METHODOLOGIES**
        - Most common research approaches
        - Experimental vs theoretical vs computational balance
        
        **TECHNICAL INNOVATION**
        - Novel algorithms, models, or frameworks
        - Breakthrough methodological contributions
        
        **EVALUATION STANDARDS**
        - Common benchmarks and datasets
        - Evaluation methodologies and metrics
        
        **REPRODUCIBILITY LANDSCAPE**
        - Code availability and replication efforts
        - Experimental design quality
        
        **METHODOLOGICAL GAPS**
        - Missing evaluation approaches
        - Underexplored methodological directions
        
        Focus on practical guidance for researchers planning future work.
        """
        
        methodological_summary = chat(methodological_summary_prompt)
        
        # Generate key takeaways
        takeaways_prompt = f"""
        Extract the most important takeaways for researchers:
        
        Executive Summary: {executive_summary[:300]}...
        Query: {original_query}
        Papers: {papers_count}
        
        Generate:
        1. **Top 5 Key Insights** (most important findings)
        2. **Top 3 Research Opportunities** (highest-impact future work)
        3. **Top 3 Methodological Recommendations** (best practices)
        4. **Critical Knowledge Gaps** (urgent research needs)
        5. **Practical Next Steps** (actionable recommendations)
        
        Make each point specific, actionable, and research-oriented.
        """
        
        key_takeaways = chat(takeaways_prompt)
        
        # Create summary metadata
        summary_metadata = {
            "summary_timestamp": datetime.now(timezone.utc).isoformat(),
            "papers_synthesized": papers_count,
            "query_context": original_query,
            "analysis_sources": ["thematic", "methodological", "relationship"],
            "memory_contexts_used": len(past_summaries) + len(domain_summaries),
            "synthesis_type": "comprehensive",
            "confidence_score": semantic_metadata.get("confidence_score", 0.7)
        }
        
        # Remember summary patterns for future use
        summary_pattern = f"summary_patterns: {original_query} -> {papers_count} papers synthesized"
        summarization_agent.remember(summary_pattern, importance=0.8)
        
        # Remember domain synthesis
        domain_synthesis = f"domain_synthesis: comprehensive summary created for {papers_count} papers"
        summarization_agent.remember(domain_synthesis, importance=0.7)
        
        logger.info(f"📚 Summarization complete: executive summary, thematic synthesis, and methodological summary created")
        
        # Broadcast complete summaries
        broadcast({
            "type": "summaries_complete",
            "knowledge": {
                "executive_summary": executive_summary,
                "thematic_synthesis": thematic_synthesis,
                "methodological_summary": methodological_summary,
                "key_takeaways": key_takeaways,
                "original_query": original_query,
                "papers_count": papers_count,
                "summary_metadata": summary_metadata,
                "synthesis_quality": {
                    "comprehensiveness": "high",
                    "actionability": "high",
                    "research_value": "high",
                    "synthesis_ready": True
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        
        # Remember failure for learning
        error_memory = f"summary_error: {original_query} - {str(e)[:100]}"
        summarization_agent.remember(error_memory, importance=0.4)
        
        broadcast({
            "type": "summary_error",
            "knowledge": {
                "original_query": original_query,
                "error": str(e),
                "error_type": type(e).__name__,
                "papers_count": papers_count,
                "recovery_suggestion": "retry_with_focused_scope"
            }
        })


# Agent metadata for system introspection
AGENT_METADATA = {
    "identity": "summarization specialist",
    "domain": "research",
    "capabilities": [
        "multi-paper synthesis",
        "thematic summarization",
        "executive summary generation",
        "key insight extraction",
        "research narrative construction"
    ],
    "responds_to": ["semantic_analysis_complete"],
    "memory_enabled": True,
    "learning_focus": "summary patterns and domain-specific synthesis techniques"
}