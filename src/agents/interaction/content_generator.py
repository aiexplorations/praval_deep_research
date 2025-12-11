"""
Content Generator Agent - Interaction Domain.

I am a content generation specialist who transforms research conversations into
shareable content formats like Twitter/X threads and blog posts.

NOTE: This agent uses direct OpenAI calls instead of Praval's chat() because
it needs to be callable from both the agent runner AND directly from API endpoints.
The agent pattern is still used for memory and learning capabilities.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from praval import agent
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client
_openai_client = None


def _get_openai_client() -> OpenAI:
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _llm_chat(prompt: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """Direct OpenAI chat completion call."""
    client = _get_openai_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content


@agent("content_generator", channel="broadcast", responds_to=["content_generation_request"], memory=True)
def content_generator_agent(spore) -> Dict[str, Any]:
    """
    I am a content generation specialist who transforms research conversations
    into shareable content formats with proper citations.

    My expertise:
    - Twitter/X thread creation with character limits
    - Blog post drafting in markdown
    - Academic citation formatting
    - Multi-style content adaptation (academic, casual, narrative)
    - Research insight distillation for social media
    """
    # Extract knowledge from spore
    conversation_text = spore.knowledge.get("conversation_text", [])
    papers_list = spore.knowledge.get("papers_list", "No papers cited")
    paper_ids = spore.knowledge.get("paper_ids", [])
    format_type = spore.knowledge.get("format", "twitter")
    style = spore.knowledge.get("style", "academic")
    max_tweets = spore.knowledge.get("max_tweets", 10)
    include_toc = spore.knowledge.get("include_toc", True)
    custom_prompt = spore.knowledge.get("custom_prompt", "")

    logger.info(f"Content Generator: Creating {format_type} content in {style} style")

    # Remember content generation session
    session_info = f"content_generation: {format_type}/{style} from conversation"
    content_generator_agent.remember(session_info, importance=0.7)

    # Recall past successful content generation patterns
    past_twitter_patterns = content_generator_agent.recall("twitter_success", limit=3)
    past_blog_patterns = content_generator_agent.recall("blog_success", limit=3)
    style_patterns = content_generator_agent.recall(f"style:{style}", limit=3)

    # Build custom instructions section
    custom_instructions = ""
    if custom_prompt and custom_prompt.strip():
        custom_instructions = f"""
**CUSTOM INSTRUCTIONS FROM USER:**
{custom_prompt}

Apply these instructions to shape the content appropriately.
"""

    try:
        if format_type == "twitter":
            result = _generate_twitter_thread(
                conversation_text=conversation_text,
                papers_list=papers_list,
                style=style,
                max_tweets=max_tweets,
                custom_instructions=custom_instructions,
                past_patterns=past_twitter_patterns,
                style_patterns=style_patterns
            )

            # Remember successful pattern
            if result.get("tweets"):
                pattern = f"twitter_success: {style} style, {len(result['tweets'])} tweets generated"
                content_generator_agent.remember(pattern, importance=0.8)

        else:  # blog
            result = _generate_blog_post(
                conversation_text=conversation_text,
                papers_list=papers_list,
                style=style,
                include_toc=include_toc,
                custom_instructions=custom_instructions,
                past_patterns=past_blog_patterns,
                style_patterns=style_patterns
            )

            # Remember successful pattern
            if result.get("blog_post"):
                pattern = f"blog_success: {style} style, {result['blog_post'].get('word_count', 0)} words"
                content_generator_agent.remember(pattern, importance=0.8)

        # Remember style usage
        style_memory = f"style:{style} -> successfully applied to {format_type}"
        content_generator_agent.remember(style_memory, importance=0.6)

        result["paper_ids"] = list(paper_ids) if paper_ids else []
        result["format"] = format_type
        result["style"] = style

        logger.info(f"Content Generator: Successfully created {format_type} content")
        return result

    except Exception as e:
        logger.error(f"Content generation failed: {e}")

        # Remember failure for learning
        error_memory = f"content_error: {format_type}/{style} - {str(e)[:100]}"
        content_generator_agent.remember(error_memory, importance=0.4)

        return {
            "error": str(e),
            "format": format_type,
            "style": style
        }


def _generate_twitter_thread(
    conversation_text: List[str],
    papers_list: str,
    style: str,
    max_tweets: int,
    custom_instructions: str,
    past_patterns: List[str],
    style_patterns: List[str]
) -> Dict[str, Any]:
    """Generate a Twitter/X thread from conversation."""

    # Build style guidance
    style_guidance = {
        "academic": "Focus on methodology, findings, and technical insights. Use precise language.",
        "casual": "More accessible language, broader appeal, conversational tone. Use analogies.",
        "narrative": "Storytelling format with a narrative arc, engaging flow. Build curiosity."
    }

    past_learning = ""
    if past_patterns:
        past_learning = f"\nPrevious successful patterns: {len(past_patterns)} twitter threads created"

    twitter_prompt = f"""You are creating a Twitter/X thread summarizing a research conversation.
The thread should be informative, engaging, and cite sources properly.

**CONSTRAINTS:**
- Each tweet MUST be 280 characters or fewer (including URLs)
- ArXiv URLs are ~32 characters: https://arxiv.org/abs/XXXX.XXXXX
- Use X/N format at the end of each tweet for thread navigation (e.g., 1/5, 2/5)
- First tweet should hook the reader and introduce the topic
- Last tweet should summarize key takeaways or call to action

**STYLE: {style}**
{style_guidance.get(style, style_guidance['academic'])}
{past_learning}

{custom_instructions}

**CONVERSATION TO SUMMARIZE:**
{chr(10).join(conversation_text[:5000])}

**PAPERS CITED (use arXiv URLs for citations):**
{papers_list}

Generate a thread of {max_tweets} tweets maximum. Each tweet must be a complete thought.
Return ONLY a valid JSON array of objects with this exact format:
[{{"position": 1, "content": "tweet text here 1/N", "has_citation": false, "citation_url": null}}]

Ensure each tweet content is under 280 characters. Count carefully!"""

    response = _llm_chat(
        twitter_prompt,
        system_prompt="You are a social media content expert who creates engaging Twitter threads about research topics. Always return valid JSON.",
        max_tokens=2000
    )

    # Parse the response
    content = response.strip()
    # Handle potential markdown code blocks
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    tweets_data = json.loads(content)

    # Build validated tweets
    tweets = []
    for t in tweets_data:
        tweet_content = t.get("content", "")
        char_count = len(tweet_content)

        # Truncate if over limit
        if char_count > 280:
            tweet_content = tweet_content[:277] + "..."
            char_count = 280

        tweets.append({
            "position": t.get("position", len(tweets) + 1),
            "content": tweet_content,
            "char_count": char_count,
            "has_citation": t.get("has_citation", False),
            "citation_url": t.get("citation_url")
        })

    return {"tweets": tweets}


def _generate_blog_post(
    conversation_text: List[str],
    papers_list: str,
    style: str,
    include_toc: bool,
    custom_instructions: str,
    past_patterns: List[str],
    style_patterns: List[str]
) -> Dict[str, Any]:
    """Generate a blog post from conversation."""

    # Build style guidance
    style_guidance = {
        "academic": "Formal tone, focus on methodology and findings, technical depth. Use precise terminology.",
        "casual": "Accessible language, explain concepts simply, engaging tone. Use analogies and examples.",
        "narrative": "Tell a story, use analogies, create a compelling flow. Build intrigue and connection."
    }

    toc_instruction = "Include a table of contents after the introduction." if include_toc else "Do not include a table of contents."

    past_learning = ""
    if past_patterns:
        past_learning = f"\nPrevious successful patterns: {len(past_patterns)} blog posts created"

    blog_prompt = f"""You are creating a blog post summarizing a research conversation.
The post should be well-structured, informative, and properly cite sources.

**REQUIREMENTS:**
- Write in markdown format
- Include a compelling title
- {toc_instruction}
- Use proper heading hierarchy (##, ###)
- Include inline citations as markdown links: [Author et al., Year](arxiv_url)
- End with a "References" section listing all cited papers
- Target length: 800-1500 words

**STYLE: {style}**
{style_guidance.get(style, style_guidance['academic'])}
{past_learning}

{custom_instructions}

**CONVERSATION TO SUMMARIZE:**
{chr(10).join(conversation_text[:8000])}

**PAPERS CITED:**
{papers_list}

Write the complete blog post in markdown format. Start with the title as a # heading."""

    blog_content = _llm_chat(
        blog_prompt,
        system_prompt="You are a technical writer who creates engaging blog posts about research topics. Write in clean markdown format.",
        max_tokens=4000
    )

    # Extract title from first line
    lines = blog_content.split("\n")
    title = "Research Summary"
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    # Count words
    word_count = len(blog_content.split())

    return {
        "blog_post": {
            "title": title,
            "content": blog_content,
            "word_count": word_count
        }
    }


# Agent metadata for system introspection
AGENT_METADATA = {
    "identity": "content generation specialist",
    "domain": "interaction",
    "capabilities": [
        "twitter thread creation",
        "blog post generation",
        "multi-style content adaptation",
        "academic citation formatting",
        "research insight distillation"
    ],
    "responds_to": ["content_generation_request"],
    "memory_enabled": True,
    "learning_focus": "content generation patterns and style-specific techniques"
}
