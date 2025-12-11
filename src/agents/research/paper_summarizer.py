"""
Paper Summarizer Agent - Research Domain.

I am a summarization specialist who generates structured summaries of research
papers and stores them with embeddings for fast two-tier retrieval.

This agent is part of the context engineering pipeline:
document_processor -> paper_summarizer -> citation_extractor -> linked_paper_indexer
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


def _generate_paper_summary(
    paper: Dict[str, Any],
    query_context: str = ""
) -> Dict[str, Any]:
    """
    Generate a structured summary of a paper using LLM.

    Args:
        paper: Paper data with title, abstract, etc.
        query_context: Original search query for context

    Returns:
        Structured summary dictionary
    """
    title = paper.get("title", "Unknown")
    abstract = paper.get("abstract", "")
    authors = paper.get("authors", [])
    categories = paper.get("categories", [])

    summary_prompt = f"""
Analyze this research paper and generate a structured summary:

Title: {title}
Authors: {', '.join(authors[:5])}{'...' if len(authors) > 5 else ''}
Categories: {', '.join(categories)}
Abstract: {abstract}
Query Context: {query_context}

Generate a structured summary with these exact fields:

1. ONE_LINE: A single sentence (max 100 chars) capturing the paper's core contribution.

2. ABSTRACT_SUMMARY: A condensed 2-3 sentence version of the abstract focusing on key points.

3. KEY_CONTRIBUTIONS: 3-5 bullet points of the main contributions (start each with "-").

4. METHODOLOGY: A brief 1-2 sentence description of the technical approach.

5. DOMAINS: 3-5 research domains/topics this paper relates to (comma-separated).

Format your response EXACTLY like this:
ONE_LINE: [your one-liner]
ABSTRACT_SUMMARY: [condensed abstract]
KEY_CONTRIBUTIONS:
- [contribution 1]
- [contribution 2]
- [contribution 3]
METHODOLOGY: [brief methodology]
DOMAINS: [domain1], [domain2], [domain3]
"""

    response = chat(summary_prompt)

    # Parse the structured response
    summary = {
        "title": title,
        "one_line": "",
        "abstract_summary": "",
        "key_contributions": [],
        "methodology": "",
        "domains": []
    }

    current_field = None
    lines = response.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("ONE_LINE:"):
            summary["one_line"] = line.replace("ONE_LINE:", "").strip()
        elif line.startswith("ABSTRACT_SUMMARY:"):
            summary["abstract_summary"] = line.replace("ABSTRACT_SUMMARY:", "").strip()
            current_field = "abstract_summary"
        elif line.startswith("KEY_CONTRIBUTIONS:"):
            current_field = "key_contributions"
        elif line.startswith("METHODOLOGY:"):
            summary["methodology"] = line.replace("METHODOLOGY:", "").strip()
            current_field = "methodology"
        elif line.startswith("DOMAINS:"):
            domains_str = line.replace("DOMAINS:", "").strip()
            summary["domains"] = [d.strip() for d in domains_str.split(",") if d.strip()]
        elif line.startswith("-") and current_field == "key_contributions":
            summary["key_contributions"].append(line[1:].strip())
        elif current_field == "abstract_summary" and not line.startswith(("KEY_CONTRIBUTIONS", "METHODOLOGY", "DOMAINS")):
            summary["abstract_summary"] += " " + line

    # Ensure we have at least basic content
    if not summary["one_line"]:
        summary["one_line"] = title[:100]
    if not summary["abstract_summary"]:
        summary["abstract_summary"] = abstract[:500] if abstract else title
    if not summary["domains"]:
        summary["domains"] = categories[:5] if categories else ["general"]

    return summary


@agent("paper_summarizer", channel="broadcast", responds_to=["documents_processed"], memory=True)
def paper_summarizer_agent(spore: Spore) -> None:
    """
    I am a summarization specialist who generates structured summaries of
    research papers and stores them with embeddings for fast two-tier retrieval.

    My expertise:
    - Structured summary generation from paper content
    - Key contribution extraction
    - Domain/topic identification
    - Summary embedding generation for fast retrieval
    - Context engineering for improved Q&A

    I respond to 'documents_processed' events from the document processor and
    broadcast 'papers_summarized' when complete.
    """
    logger.info("=" * 80)
    logger.info("📝 PAPER SUMMARIZER AGENT TRIGGERED!")
    logger.info("=" * 80)

    processed_papers = spore.knowledge.get("processed_papers", [])
    query_context = spore.knowledge.get("original_query", "")
    processing_stats = spore.knowledge.get("processing_stats", {})

    logger.info(f"📊 Received {len(processed_papers)} papers to summarize")
    logger.info(f"   Query context: {query_context}")

    if not processed_papers:
        logger.warning("⚠️ No papers to summarize - EXITING")
        return

    # Initialize storage clients
    try:
        qdrant_client = QdrantClientWrapper(settings)
        embeddings_gen = EmbeddingsGenerator(settings)

        # Initialize context engineering collections
        init_results = qdrant_client.initialize_context_engineering_collections(
            vector_size=settings.EMBEDDING_DIMENSIONS
        )
        logger.info(f"📦 Collection initialization: {init_results}")

    except Exception as e:
        logger.error(f"Failed to initialize storage: {e}")
        broadcast({
            "type": "summarization_error",
            "knowledge": {
                "error": f"Storage initialization failed: {str(e)}",
                "papers_count": len(processed_papers)
            }
        })
        return

    # Remember summarization session
    paper_summarizer_agent.remember(
        f"summarization_session: {len(processed_papers)} papers from query '{query_context}'",
        importance=0.7
    )

    # Recall past summarization patterns
    past_patterns = paper_summarizer_agent.recall("summarization_patterns", limit=5)

    summarization_stats = {
        "total_papers": len(processed_papers),
        "summaries_generated": 0,
        "summaries_stored": 0,
        "errors": []
    }

    summarized_papers = []

    for i, paper in enumerate(processed_papers, 1):
        paper_id = paper.get("arxiv_id", "")
        title = paper.get("title", "Unknown")

        logger.info(f"📝 Summarizing {i}/{len(processed_papers)}: {title[:60]}...")

        try:
            # Generate structured summary
            summary = _generate_paper_summary(paper, query_context)
            summarization_stats["summaries_generated"] += 1

            # Create summary text for embedding
            summary_text = f"""
Title: {summary['title']}
Summary: {summary['one_line']}
{summary['abstract_summary']}
Key contributions: {'; '.join(summary['key_contributions'])}
Methodology: {summary['methodology']}
Domains: {', '.join(summary['domains'])}
"""

            # Generate embedding for summary
            summary_embedding = embeddings_gen.generate_embedding(summary_text)

            # Store summary in paper_summaries collection
            success = qdrant_client.add_paper_summary(
                paper_id=paper_id or f"paper_{hash(title)}",
                summary_data=summary,
                embedding=summary_embedding
            )

            if success:
                summarization_stats["summaries_stored"] += 1
                logger.info(f"✅ Summary stored for {paper_id or title[:30]}")
            else:
                logger.warning(f"⚠️ Failed to store summary for {paper_id}")

            # Add summary to paper data for downstream agents
            summarized_paper = {
                **paper,
                "summary": summary,
                "summary_stored": success
            }
            summarized_papers.append(summarized_paper)

            # Remember successful patterns
            paper_summarizer_agent.remember(
                f"summarization_patterns: {summary['domains'][0] if summary['domains'] else 'general'} paper summarized",
                importance=0.5
            )

        except Exception as e:
            logger.error(f"Failed to summarize '{title}': {e}")
            summarization_stats["errors"].append(f"{title}: {str(e)}")
            # Still pass the paper along without summary
            summarized_papers.append(paper)

    logger.info(f"✅ Summarization complete: {summarization_stats['summaries_stored']}/{summarization_stats['total_papers']} stored")

    # Remember session summary
    paper_summarizer_agent.remember(
        f"session_complete: {summarization_stats['summaries_stored']} summaries for '{query_context}'",
        importance=0.8
    )

    # Broadcast for citation extractor
    broadcast_payload = {
        "type": "papers_summarized",
        "knowledge": {
            "summarized_papers": summarized_papers,
            "original_query": query_context,
            "summarization_stats": summarization_stats,
            "processing_metadata": {
                "agent": "paper_summarizer",
                "session_timestamp": datetime.now(timezone.utc).isoformat(),
                "memory_contexts_used": len(past_patterns)
            }
        }
    }

    logger.info(f"📡 BROADCASTING papers_summarized event")
    logger.info(f"   Summarized papers: {len(summarized_papers)}")
    logger.info(f"   Summaries stored: {summarization_stats['summaries_stored']}")

    broadcast_result = broadcast(broadcast_payload)
    logger.info(f"✅ BROADCAST COMPLETE - Result: {broadcast_result}")
    logger.info("=" * 80)


# Agent metadata for system introspection
AGENT_METADATA = {
    "identity": "paper summarization specialist",
    "domain": "research",
    "capabilities": [
        "Structured summary generation",
        "Key contribution extraction",
        "Domain/topic identification",
        "Summary embedding generation",
        "Two-tier retrieval support",
        "Context engineering"
    ],
    "responds_to": ["documents_processed"],
    "broadcasts": ["papers_summarized", "summarization_error"],
    "memory_enabled": True,
    "learning_focus": "summarization patterns and domain-specific extraction",
    "storage_integrations": ["Qdrant (paper_summaries collection)", "OpenAI Embeddings"]
}
