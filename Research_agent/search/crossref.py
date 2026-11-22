"""
hsi_research_agent/search/crossref.py
Crossref official DOI registry search module.
"""

import logging
import time
from typing import Any, Dict, List, Optional
import requests

from hsi_research_agent.config import settings

logger = logging.getLogger("hsi_research_agent.search.crossref")
_last_request_time = 0.0


def _throttle() -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < settings.DELAY_CROSSREF:
        time.sleep(settings.DELAY_CROSSREF - elapsed)
    _last_request_time = time.time()


def search_crossref(
    query: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    max_results: int = 50,
    email: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search Crossref REST API for published journal articles and conference proceedings.
    """
    _throttle()
    url = "https://api.crossref.org/works"
    contact = email or settings.CONTACT_EMAIL
    headers = {"User-Agent": f"HSIResearchAgent/1.0 (mailto:{contact})"}

    filter_tokens = ["type:journal-article", "type:proceedings-article"]
    if start_year:
        filter_tokens.append(f"from-pub-date:{start_year}-01-01")
    if end_year:
        filter_tokens.append(f"until-pub-date:{end_year}-12-31")

    params: Dict[str, Any] = {
        "query.bibliographic": query,
        "filter": ",".join(filter_tokens),
        "rows": min(max_results, 50),
        "select": "DOI,title,author,issued,container-title,is-referenced-by-count,abstract,type,URL,publisher",
    }
    if contact and "example.com" not in contact:
        params["mailto"] = contact

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("message", {}).get("items", [])
            elif resp.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                continue
            else:
                logger.warning("Crossref returned status %d for '%s'", resp.status_code, query)
                return []
        except Exception as e:
            logger.warning("Crossref attempt %d failed: %s", attempt + 1, str(e))
            time.sleep(1.5)

    return []
