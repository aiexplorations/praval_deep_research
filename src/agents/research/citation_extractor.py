"""
Citation Extractor Agent - Research Domain.

I am a citation extraction specialist who parses references from papers,
identifies arXiv IDs, and selects the top 3-5 most relevant cited papers
for full indexing.

This agent is part of the context engineering pipeline:
document_processor -> paper_summarizer -> citation_extractor -> linked_paper_indexer
"""

import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from praval import agent, chat, broadcast
from praval import Spore

from agentic_research.core.config import get_settings
from agentic_research.storage.minio_client import MinIOClient


logger = logging.getLogger(__name__)
settings = get_settings()


def _extract_arxiv_ids_from_text(text: str) -> List[str]:
    """
    Extract arXiv IDs from text using regex patterns.

    Handles various formats:
    - arXiv:2106.04560
    - arxiv.org/abs/2106.04560
    - [2106.04560]
    """
    patterns = [
        r'arXiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?)',  # arXiv:2106.04560 or arXiv: 2106.04560
        r'arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)',  # arxiv.org/abs/2106.04560
        r'arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)',  # arxiv.org/pdf/2106.04560
        r'\[(\d{4}\.\d{4,5}(?:v\d+)?)\]',  # [2106.04560]
    ]

    found_ids = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found_ids.update(matches)

    return list(found_ids)


def _extract_citations_with_llm(
    paper_text: str,
    paper_title: str,
    paper_domains: List[str]
) -> List[Dict[str, Any]]:
    """
    Use LLM to extract and identify key citations from paper text.

    Returns structured citation information including potential arXiv IDs.
    """
    # Limit text to avoid token limits
    text_sample = paper_text[:15000] if len(paper_text) > 15000 else paper_text

    extraction_prompt = f"""
Analyze this research paper text and extract the most important citations.

Paper Title: {paper_title}
Paper Domains: {', '.join(paper_domains)}

Paper Text (excerpt):
{text_sample}

Extract the TOP 5 most important cited papers that are:
1. Foundational works in the same domain
2. Methods or approaches this paper builds upon
3. Key comparative baselines

For each citation, provide:
- TITLE: The paper title (as accurately as possible)
- AUTHORS: First author's last name
- ARXIV_ID: If you can identify an arXiv ID from the text (format: XXXX.XXXXX), otherwise "unknown"
- RELEVANCE: Brief reason why this citation is important (1 sentence)
- YEAR: Publication year if mentioned

Format as:
CITATION 1:
TITLE: [title]
AUTHORS: [author]
ARXIV_ID: [id or unknown]
RELEVANCE: [reason]
YEAR: [year or unknown]

CITATION 2:
...

Only include citations you're confident about. Skip if fewer than 5 clear citations exist.
"""

    response = chat(extraction_prompt)

    # Parse the response
    citations = []
    current_citation = {}

    for line in response.strip().split("\n"):
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
        elif line.startswith("ARXIV_ID:"):
            arxiv_id = line.replace("ARXIV_ID:", "").strip()
            if arxiv_id.lower() != "unknown":
                current_citation["arxiv_id"] = arxiv_id
        elif line.startswith("RELEVANCE:"):
            current_citation["relevance"] = line.replace("RELEVANCE:", "").strip()
        elif line.startswith("YEAR:"):
            year = line.replace("YEAR:", "").strip()
            if year.lower() != "unknown":
                current_citation["year"] = year

    # Don't forget the last citation
    if current_citation and current_citation.get("title"):
        citations.append(current_citation)

    return citations[:5]  # Max 5 citations


def _search_arxiv_for_citation(title: str, authors: str = "") -> Optional[str]:
    """
    Search arXiv API to find the arXiv ID for a citation.

    Returns the arXiv ID if found, None otherwise.
    """
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    try:
        # Build search query
        search_terms = []
        if title:
            # Clean title for search
            clean_title = re.sub(r'[^\w\s]', ' ', title)
            words = clean_title.split()[:6]  # First 6 words
            search_terms.append(f"ti:{'+'.join(words)}")

        if authors:
            author_last = authors.split()[0] if authors else ""
            if author_last:
                search_terms.append(f"au:{author_last}")

        if not search_terms:
            return None

        query = "+AND+".join(search_terms)
        url = f"{settings.ARXIV_BASE_URL}?search_query={query}&max_results=3"

        # Make request
        with urllib.request.urlopen(url, timeout=10) as response:
            data = response.read().decode('utf-8')

        # Parse XML
        root = ET.fromstring(data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        for entry in root.findall('atom:entry', ns):
            entry_id = entry.find('atom:id', ns)
            if entry_id is not None:
                # Extract arXiv ID from URL
                id_text = entry_id.text
                match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', id_text)
                if match:
                    return match.group(1)

    except Exception as e:
        logger.warning(f"ArXiv search failed for '{title}': {e}")

    return None


@agent("citation_extractor", channel="broadcast", responds_to=["papers_summarized"], memory=True)
def citation_extractor_agent(spore: Spore) -> None:
    """
    I am a citation extraction specialist who parses references from research
    papers, identifies arXiv IDs, and selects the top 3-5 most relevant cited
    papers for full indexing as linked papers.

    My expertise:
    - Reference parsing from PDF text
    - ArXiv ID identification and validation
    - Citation relevance ranking
    - Cross-paper relationship mapping
    - Knowledge graph building

    I respond to 'papers_summarized' events and broadcast 'citations_extracted'
    with candidate papers for the linked_paper_indexer.
    """
    logger.info("=" * 80)
    logger.info("🔗 CITATION EXTRACTOR AGENT TRIGGERED!")
    logger.info("=" * 80)

    summarized_papers = spore.knowledge.get("summarized_papers", [])
    query_context = spore.knowledge.get("original_query", "")

    logger.info(f"📊 Received {len(summarized_papers)} papers to extract citations from")
    logger.info(f"   Query context: {query_context}")

    if not summarized_papers:
        logger.warning("⚠️ No papers to extract citations from - EXITING")
        return

    # Initialize MinIO client to read extracted text
    try:
        minio_client = MinIOClient(settings)
    except Exception as e:
        logger.error(f"Failed to initialize MinIO client: {e}")
        minio_client = None

    # Remember extraction session
    citation_extractor_agent.remember(
        f"extraction_session: {len(summarized_papers)} papers from query '{query_context}'",
        importance=0.7
    )

    # Recall past extraction patterns
    past_patterns = citation_extractor_agent.recall("citation_patterns", limit=5)

    extraction_stats = {
        "total_papers": len(summarized_papers),
        "papers_with_citations": 0,
        "total_citations_found": 0,
        "arxiv_ids_found": 0,
        "arxiv_ids_searched": 0,
        "errors": []
    }

    all_citation_candidates = []

    for i, paper in enumerate(summarized_papers, 1):
        paper_id = paper.get("arxiv_id", "")
        title = paper.get("title", "Unknown")
        summary = paper.get("summary", {})
        domains = summary.get("domains", [])

        logger.info(f"🔗 Extracting citations from {i}/{len(summarized_papers)}: {title[:50]}...")

        try:
            # Try to get full text from MinIO
            paper_text = ""
            if minio_client and paper_id:
                try:
                    paper_text = minio_client.get_extracted_text(paper_id)
                except Exception as e:
                    logger.debug(f"No extracted text found for {paper_id}: {e}")

            # If no full text, use abstract
            if not paper_text:
                paper_text = paper.get("abstract", "")

            # Method 1: Regex extraction for arXiv IDs directly in text
            regex_ids = _extract_arxiv_ids_from_text(paper_text)
            logger.debug(f"Found {len(regex_ids)} arXiv IDs via regex")

            # Method 2: LLM-based citation extraction
            llm_citations = []
            if paper_text:
                llm_citations = _extract_citations_with_llm(
                    paper_text=paper_text,
                    paper_title=title,
                    paper_domains=domains
                )
                logger.debug(f"Found {len(llm_citations)} citations via LLM")

            # Combine and deduplicate
            seen_ids = set()
            paper_citations = []

            # Add regex-found IDs first (high confidence)
            for arxiv_id in regex_ids:
                if arxiv_id and arxiv_id not in seen_ids and arxiv_id != paper_id:
                    seen_ids.add(arxiv_id)
                    paper_citations.append({
                        "arxiv_id": arxiv_id,
                        "source_paper_id": paper_id,
                        "source_paper_title": title,
                        "extraction_method": "regex",
                        "confidence": "high"
                    })
                    extraction_stats["arxiv_ids_found"] += 1

            # Add LLM-extracted citations
            for citation in llm_citations:
                arxiv_id = citation.get("arxiv_id")

                # If no arXiv ID, try to search for it
                if not arxiv_id:
                    extraction_stats["arxiv_ids_searched"] += 1
                    arxiv_id = _search_arxiv_for_citation(
                        title=citation.get("title", ""),
                        authors=citation.get("authors", "")
                    )
                    if arxiv_id:
                        extraction_stats["arxiv_ids_found"] += 1

                if arxiv_id and arxiv_id not in seen_ids and arxiv_id != paper_id:
                    seen_ids.add(arxiv_id)
                    paper_citations.append({
                        "arxiv_id": arxiv_id,
                        "title": citation.get("title"),
                        "authors": citation.get("authors"),
                        "relevance": citation.get("relevance"),
                        "year": citation.get("year"),
                        "source_paper_id": paper_id,
                        "source_paper_title": title,
                        "extraction_method": "llm",
                        "confidence": "medium"
                    })

            if paper_citations:
                extraction_stats["papers_with_citations"] += 1
                extraction_stats["total_citations_found"] += len(paper_citations)

                # Take top 3-5 citations per paper (per plan)
                top_citations = paper_citations[:5]
                all_citation_candidates.extend(top_citations)

                logger.info(f"✅ Found {len(top_citations)} citations for {paper_id or title[:30]}")

            # Remember extraction patterns
            citation_extractor_agent.remember(
                f"citation_patterns: {domains[0] if domains else 'general'} paper -> {len(paper_citations)} citations",
                importance=0.5
            )

        except Exception as e:
            logger.error(f"Failed to extract citations from '{title}': {e}")
            extraction_stats["errors"].append(f"{title}: {str(e)}")

    logger.info(f"✅ Citation extraction complete:")
    logger.info(f"   Papers with citations: {extraction_stats['papers_with_citations']}/{extraction_stats['total_papers']}")
    logger.info(f"   Total citations found: {extraction_stats['total_citations_found']}")
    logger.info(f"   ArXiv IDs found: {extraction_stats['arxiv_ids_found']}")

    # Remember session summary
    citation_extractor_agent.remember(
        f"session_complete: {extraction_stats['total_citations_found']} citations from {extraction_stats['papers_with_citations']} papers",
        importance=0.8
    )

    # Deduplicate across all papers
    unique_citations = {}
    for citation in all_citation_candidates:
        arxiv_id = citation.get("arxiv_id")
        if arxiv_id and arxiv_id not in unique_citations:
            unique_citations[arxiv_id] = citation

    final_candidates = list(unique_citations.values())
    logger.info(f"📦 Unique citation candidates for indexing: {len(final_candidates)}")

    # Broadcast for linked_paper_indexer
    broadcast_payload = {
        "type": "citations_extracted",
        "knowledge": {
            "citation_candidates": final_candidates,
            "source_papers": [p.get("arxiv_id") for p in summarized_papers if p.get("arxiv_id")],
            "original_query": query_context,
            "extraction_stats": extraction_stats,
            "processing_metadata": {
                "agent": "citation_extractor",
                "session_timestamp": datetime.now(timezone.utc).isoformat(),
                "memory_contexts_used": len(past_patterns)
            }
        }
    }

    logger.info(f"📡 BROADCASTING citations_extracted event")
    logger.info(f"   Citation candidates: {len(final_candidates)}")

    broadcast_result = broadcast(broadcast_payload)
    logger.info(f"✅ BROADCAST COMPLETE - Result: {broadcast_result}")
    logger.info("=" * 80)


# Agent metadata for system introspection
AGENT_METADATA = {
    "identity": "citation extraction specialist",
    "domain": "research",
    "capabilities": [
        "Reference parsing from PDF text",
        "ArXiv ID identification (regex + LLM)",
        "ArXiv search for unidentified citations",
        "Citation relevance ranking",
        "Cross-paper relationship mapping"
    ],
    "responds_to": ["papers_summarized"],
    "broadcasts": ["citations_extracted"],
    "memory_enabled": True,
    "learning_focus": "citation patterns and domain-specific reference extraction",
    "storage_integrations": ["MinIO (for reading extracted text)"]
}
