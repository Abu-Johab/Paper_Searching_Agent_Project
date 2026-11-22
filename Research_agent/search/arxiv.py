"""
hsi_research_agent/search/arxiv.py
arXiv preprints search module using Atom XML feed.
"""

import logging
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
import requests

from hsi_research_agent.config import settings

logger = logging.getLogger("hsi_research_agent.search.arxiv")
_last_request_time = 0.0
_is_disabled = False
_consecutive_failures = 0


def _throttle() -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < settings.DELAY_ARXIV:
        time.sleep(settings.DELAY_ARXIV - elapsed)
    _last_request_time = time.time()


def search_arxiv(
    query: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    max_results: int = 50,
) -> List[Dict[str, Any]]:
    """
    Search arXiv Atom API for AI, CS, and remote sensing preprints.
    """
    global _is_disabled, _consecutive_failures
    if _is_disabled:
        return []

    _throttle()
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "max_results": min(max_results, 50),
        "sortBy": "relevance",
    }
    headers = {"User-Agent": f"HSIResearchAgent/1.0 (mailto:{settings.CONTACT_EMAIL})"}

    for attempt in range(2):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                _consecutive_failures = 0
                root = ET.fromstring(resp.content)
                ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
                entries = root.findall("atom:entry", ns)

                papers: List[Dict[str, Any]] = []
                for entry in entries:
                    published_el = entry.find("atom:published", ns)
                    pub_year: Optional[int] = None
                    if published_el is not None and published_el.text:
                        try:
                            pub_year = int(published_el.text[:4])
                        except ValueError:
                            pub_year = None

                    if pub_year is not None:
                        if start_year is not None and pub_year < start_year:
                            continue
                        if end_year is not None and pub_year > end_year:
                            continue

                    id_el = entry.find("atom:id", ns)
                    title_el = entry.find("atom:title", ns)
                    summary_el = entry.find("atom:summary", ns)
                    doi_el = entry.find("arxiv:doi", ns)
                    journal_el = entry.find("arxiv:journal_ref", ns)
                    authors_el = entry.findall("atom:author", ns)

                    authors = []
                    for a in authors_el:
                        name_el = a.find("atom:name", ns)
                        if name_el is not None and name_el.text:
                            authors.append(name_el.text.strip())

                    papers.append({
                        "id": id_el.text.strip() if id_el is not None and id_el.text else "",
                        "title": title_el.text.strip() if title_el is not None and title_el.text else "",
                        "summary": summary_el.text.strip() if summary_el is not None and summary_el.text else "",
                        "published_year": pub_year,
                        "doi": doi_el.text.strip() if doi_el is not None and doi_el.text else None,
                        "journal": journal_el.text.strip() if journal_el is not None and journal_el.text else "arXiv",
                        "authors": authors,
                    })

                return papers
            elif resp.status_code == 429:
                _is_disabled = True
                logger.warning("arXiv rate limit (429) hit. Disabling arXiv for remainder of run.")
                return []
            elif resp.status_code >= 500:
                time.sleep(2.0 * (attempt + 1))
                continue
            else:
                logger.warning("arXiv returned status %d for '%s'", resp.status_code, query)
                return []
        except Exception as e:
            logger.warning("arXiv attempt %d failed: %s", attempt + 1, str(e))
            _consecutive_failures += 1
            if _consecutive_failures >= 2:
                _is_disabled = True
                logger.warning("arXiv endpoint degraded/unresponsive. Disabling arXiv for remainder of run.")
                return []
            time.sleep(1.0)

    return []
