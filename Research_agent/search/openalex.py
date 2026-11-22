"""
hsi_research_agent/search/openalex.py
OpenAlex academic database search module (250M+ multidisciplinary papers, zero API key).
"""

import logging
import time
from typing import Any, Dict, List, Optional
import requests

from hsi_research_agent.config import settings

logger = logging.getLogger("hsi_research_agent.search.openalex")
_last_request_time = 0.0


def _throttle() -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < settings.DELAY_OPENALEX:
        time.sleep(settings.DELAY_OPENALEX - elapsed)
    _last_request_time = time.time()


def search_openalex(
    query: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    max_results: int = 50,
    email: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search OpenAlex works API with polite headers and date filtering.
    """
    _throttle()
    url = "https://api.openalex.org/works"
    contact = email or settings.CONTACT_EMAIL
    headers = {"User-Agent": f"HSIResearchAgent/1.0 (mailto:{contact})"}

    filter_parts = []
    if start_year:
        filter_parts.append(f"from_publication_date:{start_year}-01-01")
    if end_year:
        filter_parts.append(f"to_publication_date:{end_year}-12-31")

    params: Dict[str, Any] = {
        "search": query,
        "per-page": min(max_results, 50),
        "sort": "cited_by_count:desc",
    }
    if filter_parts:
        params["filter"] = ",".join(filter_parts)
    if contact and "example.com" not in contact:
        params["mailto"] = contact

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("results", [])
            elif resp.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                continue
            else:
                logger.warning("OpenAlex search returned status %d for '%s'", resp.status_code, query)
                return []
        except Exception as e:
            logger.warning("OpenAlex attempt %d failed: %s", attempt + 1, str(e))
            time.sleep(1.5)

    return []
