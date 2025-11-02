"""
Real ArXiv API client with Praval tool integration.

This module provides production-ready ArXiv search functionality that can be
used as tools by Praval agents. No mock data - all responses come from
the actual ArXiv API.
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
import time

# Import settings with proper path handling
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from agentic_research.core.config import get_settings
except ImportError:
    # Fallback to basic configuration if core config not available
    def get_settings():
        class BasicSettings:
            ARXIV_MAX_RESULTS = 50
            ARXIV_RATE_LIMIT = 3
        return BasicSettings()
    print("⚠️ Using basic ArXiv settings - core config not available")

logger = logging.getLogger(__name__)


class ArXivAPIError(Exception):
    """Exception raised for ArXiv API errors."""
    pass


class ArXivClient:
    """
    Production ArXiv API client with rate limiting and error handling.
    
    Provides real search functionality that integrates with Praval agents
    as tools. Handles the ArXiv API format, rate limiting, and error cases.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.ARXIV_BASE_URL
        self.max_results = self.settings.ARXIV_MAX_RESULTS
        self.rate_limit = self.settings.ARXIV_RATE_LIMIT
        self.delay_seconds = self.settings.ARXIV_DELAY_SECONDS
        self.last_request_time = 0
        
    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.delay_seconds:
            sleep_time = self.delay_seconds - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _build_search_query(self, query: str, domain: Optional[str] = None) -> str:
        """
        Build ArXiv API search query with domain filtering.
        
        Args:
            query: Search terms
            domain: Research domain for category filtering
            
        Returns:
            Formatted search query for ArXiv API
        """
        # Domain to ArXiv category mapping
        domain_categories = {
            "computer_science": "cs.*",
            "physics": "physics.*", 
            "mathematics": "math.*",
            "artificial_intelligence": "cs.AI",
            "machine_learning": "cs.LG OR cs.AI",
            "biology": "q-bio.*",
            "chemistry": "physics.chem-ph"
        }
        
        # Escape query for URL
        escaped_query = quote_plus(query)
        
        # Add category filter if domain specified
        if domain and domain.lower() in domain_categories:
            category = domain_categories[domain.lower()]
            search_query = f"all:{escaped_query} AND cat:{category}"
        else:
            search_query = f"all:{escaped_query}"
            
        return search_query
    
    def _parse_arxiv_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """
        Parse ArXiv API XML response into structured data.
        
        Args:
            xml_content: Raw XML response from ArXiv API
            
        Returns:
            List of paper dictionaries with standardized fields
        """
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            papers = []
            
            for entry in root.findall('atom:entry', namespaces):
                paper = {}
                
                # Title
                title_elem = entry.find('atom:title', namespaces)
                paper['title'] = title_elem.text.strip() if title_elem is not None else ""
                
                # Authors
                authors = []
                for author in entry.findall('atom:author', namespaces):
                    name_elem = author.find('atom:name', namespaces)
                    if name_elem is not None:
                        authors.append(name_elem.text.strip())
                paper['authors'] = authors
                
                # Abstract
                summary_elem = entry.find('atom:summary', namespaces)
                paper['abstract'] = summary_elem.text.strip() if summary_elem is not None else ""
                
                # ArXiv ID and URL
                id_elem = entry.find('atom:id', namespaces)
                if id_elem is not None:
                    arxiv_url = id_elem.text.strip()
                    paper['url'] = arxiv_url
                    # Extract ArXiv ID from URL
                    if '/abs/' in arxiv_url:
                        paper['arxiv_id'] = arxiv_url.split('/abs/')[-1]
                    else:
                        paper['arxiv_id'] = None
                else:
                    paper['url'] = None
                    paper['arxiv_id'] = None
                
                # Published date
                published_elem = entry.find('atom:published', namespaces)
                if published_elem is not None:
                    try:
                        # Parse ISO format datetime
                        pub_date = datetime.fromisoformat(published_elem.text.replace('Z', '+00:00'))
                        paper['published_date'] = pub_date.strftime('%Y-%m-%d')
                    except (ValueError, AttributeError):
                        paper['published_date'] = None
                else:
                    paper['published_date'] = None
                
                # Categories
                categories = []
                for category in entry.findall('arxiv:primary_category', namespaces):
                    term = category.get('term')
                    if term:
                        categories.append(term)
                
                # Also check regular categories
                for category in entry.findall('atom:category', namespaces):
                    term = category.get('term')
                    if term and term not in categories:
                        categories.append(term)
                        
                paper['categories'] = categories
                
                # Journal/venue info
                journal_elem = entry.find('arxiv:journal_ref', namespaces)
                paper['venue'] = journal_elem.text.strip() if journal_elem is not None else None
                
                papers.append(paper)
                
            return papers
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse ArXiv XML response: {e}")
            raise ArXivAPIError(f"Invalid XML response from ArXiv API: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing ArXiv response: {e}")
            raise ArXivAPIError(f"Failed to parse ArXiv response: {e}")
    
    async def search_papers(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        domain: Optional[str] = None,
        sort_by: str = "relevance"
    ) -> List[Dict[str, Any]]:
        """
        Search ArXiv for papers matching the query.
        
        Args:
            query: Search terms
            max_results: Maximum number of results (default from settings)
            domain: Research domain for filtering
            sort_by: Sort order (relevance, lastUpdatedDate, submittedDate)
            
        Returns:
            List of paper dictionaries from real ArXiv API
            
        Raises:
            ArXivAPIError: If API request fails or returns invalid data
        """
        if not query or len(query.strip()) < 3:
            raise ArXivAPIError("Query must be at least 3 characters long")
        
        # Use default max_results if not specified
        if max_results is None:
            max_results = self.max_results
        
        # Clamp max_results to ArXiv limits
        max_results = min(max_results, self.max_results)
        
        # Build search query
        search_query = self._build_search_query(query.strip(), domain)
        
        # Build API URL
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': sort_by,
            'sortOrder': 'descending'
        }
        
        # Create URL with parameters
        param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        url = f"{self.base_url}?{param_string}"
        
        logger.info(f"Searching ArXiv: {query} (domain: {domain}, max: {max_results})")
        
        # Enforce rate limiting
        await self._rate_limit()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"ArXiv API error {response.status}: {error_text}")
                        raise ArXivAPIError(f"ArXiv API returned status {response.status}")
                    
                    xml_content = await response.text()
                    
                    if not xml_content.strip():
                        logger.warning("Empty response from ArXiv API")
                        return []
                    
                    # Parse XML response
                    papers = self._parse_arxiv_response(xml_content)
                    
                    logger.info(f"Found {len(papers)} papers for query: {query}")
                    return papers
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error calling ArXiv API: {e}")
            raise ArXivAPIError(f"Network error: {e}")
        except asyncio.TimeoutError:
            logger.error("Timeout calling ArXiv API")
            raise ArXivAPIError("Request to ArXiv API timed out")
        except Exception as e:
            logger.error(f"Unexpected error calling ArXiv API: {e}")
            raise ArXivAPIError(f"Unexpected error: {e}")


# Global client instance
_arxiv_client = None

def get_arxiv_client() -> ArXivClient:
    """Get singleton ArXiv client instance."""
    global _arxiv_client
    if _arxiv_client is None:
        _arxiv_client = ArXivClient()
    return _arxiv_client


# Praval tool functions - these can be registered with agents
async def search_arxiv_papers(
    query: str, 
    max_results: int = 10, 
    domain: str = "computer_science"
) -> List[Dict[str, Any]]:
    """
    Praval tool: Search ArXiv for academic papers.
    
    Args:
        query: Search terms for papers
        max_results: Maximum number of papers to return (1-50)
        domain: Research domain for filtering
        
    Returns:
        List of paper dictionaries with title, authors, abstract, etc.
        
    Example:
        papers = await search_arxiv_papers("transformer neural networks", 5, "machine_learning")
    """
    client = get_arxiv_client()
    return await client.search_papers(query, max_results, domain)


async def get_paper_details(arxiv_id: str) -> Dict[str, Any]:
    """
    Praval tool: Get detailed information about a specific ArXiv paper.
    
    Args:
        arxiv_id: ArXiv paper ID (e.g., "2106.04560")
        
    Returns:
        Detailed paper information dictionary
    """
    client = get_arxiv_client()
    
    # Search by specific ID
    try:
        papers = await client.search_papers(f"id:{arxiv_id}", max_results=1)
        if papers:
            return papers[0]
        else:
            raise ArXivAPIError(f"Paper with ID {arxiv_id} not found")
    except Exception as e:
        raise ArXivAPIError(f"Failed to get paper details: {e}")


def calculate_paper_relevance(paper: Dict[str, Any], query: str) -> float:
    """
    Calculate relevance score for a paper based on query.
    
    Args:
        paper: Paper dictionary
        query: Original search query
        
    Returns:
        Relevance score between 0.0 and 1.0
    """
    query_lower = query.lower()
    score = 0.0
    
    # Title relevance (40% weight)
    title = paper.get('title', '').lower()
    title_matches = sum(1 for term in query_lower.split() if term in title)
    if query_lower.split():
        title_score = title_matches / len(query_lower.split())
        score += title_score * 0.4
    
    # Abstract relevance (50% weight)  
    abstract = paper.get('abstract', '').lower()
    abstract_matches = sum(1 for term in query_lower.split() if term in abstract)
    if query_lower.split():
        abstract_score = abstract_matches / len(query_lower.split())
        score += abstract_score * 0.5
    
    # Category relevance (10% weight)
    categories = [cat.lower() for cat in paper.get('categories', [])]
    category_matches = sum(1 for term in query_lower.split() 
                          if any(term in cat for cat in categories))
    if query_lower.split():
        category_score = category_matches / len(query_lower.split())
        score += category_score * 0.1
    
    return min(score, 1.0)