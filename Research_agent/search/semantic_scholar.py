"""
hsi_research_agent/search/semantic_scholar.py
Semantic Scholar Graph API search module with exponential backoff on HTTP 429.
"""

import logging
import time
from typing import Any, Dict, List, Optional
import requests

from hsi_research_agent.config import settings

logger = logging.getLogger("hsi_research_agent.search.semantic_scholar")
_last_request_time = 0.0
_is_disabled = False


def _throttle() -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < settings.DELAY_SEMANTIC_SCHOLAR:
        time.sleep(settings.DELAY_SEMANTIC_SCHOLAR - elapsed)
    _last_request_time = time.time()


def search_semantic_scholar(
    query: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    max_results: int = 50,
) -> List[Dict[str, Any]]:
    """
    Search Semantic Scholar graph API with rate limit handling and graceful backoff.
    """
    global _is_disabled
    if _is_disabled:
        return []

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"User-Agent": f"HSIResearchAgent/1.0 (mailto:{settings.CONTACT_EMAIL})"}
    if settings.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY

    params: Dict[str, Any] = {
        "query": query,
        "limit": min(max_results, 50),
        "fields": "title,abstract,year,venue,externalIds,citationCount,authors",
    }
    if start_year and end_year:
        params["year"] = f"{start_year}-{end_year}"
    elif start_year:
        params["year"] = f"{start_year}-"
    elif end_year:
        params["year"] = f"-{end_year}"

    backoff = 2.0
    for attempt in range(3):
        _throttle()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("data", [])
            elif resp.status_code == 429:
                if attempt < 2:
                    logger.info("Semantic Scholar 429 rate limit. Backing off for %.1fs...", backoff)
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                else:
                    _is_disabled = True
                    logger.warning("Semantic Scholar rate limit exceeded. Disabling for remainder of run.")
                    return []
            elif resp.status_code >= 500:
                time.sleep(1.5)
                continue
            else:
                logger.warning("Semantic Scholar returned status %d for '%s'", resp.status_code, query)
                return []
        except Exception as e:
            logger.warning("Semantic Scholar attempt %d failed: %s", attempt + 1, str(e))
            time.sleep(1.5)

    return []
