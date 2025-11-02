"""
Research Advisor Agent - Interaction Domain.

I am a research guidance specialist who provides strategic advice on research 
directions, methodology selection, and academic career development based on 
current research landscapes and user goals.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timezone
from praval import agent, chat, broadcast
from praval import Spore

logger = logging.getLogger(__name__)


@agent("research_advisor", responds_to=["research_guidance_request", "summaries_complete"], memory=True)
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
    """
    spore_type = spore.knowledge.get("type")
    
    if spore_type == "summaries_complete":
        # Store research insights for advisory context
        _store_advisory_context(spore)
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


# Agent metadata for system introspection
AGENT_METADATA = {
    "identity": "research guidance specialist", 
    "domain": "interaction",
    "capabilities": [
        "research direction recommendations",
        "methodology selection guidance",
        "career development advice",
        "opportunity identification",
        "strategic planning"
    ],
    "responds_to": ["research_guidance_request", "summaries_complete"],
    "memory_enabled": True,
    "learning_focus": "user development patterns and research landscape evolution"
}