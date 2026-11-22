"""
hsi_research_agent/search/unpaywall.py
Unpaywall API client to resolve open-access status and legal full-text PDF links.
"""

import logging
import re
from typing import Any, Dict, Optional
import requests

from hsi_research_agent.config import settings

logger = logging.getLogger("hsi_research_agent.search.unpaywall")


def _clean_doi(raw_doi: Optional[str]) -> Optional[str]:
    if not raw_doi:
        return None
    d = str(raw_doi).strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    d = re.sub(r"^doi:\s*", "", d)
    return d.strip() if d else None


def resolve_open_access(doi: str, email: Optional[str] = None) -> Dict[str, Any]:
    """
    Query Unpaywall for a paper's Open Access status and legal full-text PDF URL.
    Never bypasses paywalls; uses legal institutional and publisher green/gold OA links.
    """
    clean_doi = _clean_doi(doi)
    if not clean_doi:
        return {"is_oa": False, "pdf_url": None, "oa_status": "Unknown"}

    contact = email or settings.UNPAYWALL_EMAIL
    url = f"https://api.unpaywall.org/v2/{clean_doi}"
    params = {"email": contact}

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            is_oa = bool(data.get("is_oa", False))
            oa_status = str(data.get("oa_status", "closed"))
            best_location = data.get("best_oa_location") or {}
            pdf_url = best_location.get("url_for_pdf") or best_location.get("url")

            return {
                "is_oa": is_oa,
                "oa_status": oa_status,
                "pdf_url": pdf_url,
                "landing_page_url": best_location.get("url_for_landing_page"),
                "version": best_location.get("version"),
            }
        else:
            return {"is_oa": False, "pdf_url": None, "oa_status": "Closed / Not indexed"}
    except Exception as e:
        logger.debug("Unpaywall lookup error for %s: %s", doi, str(e))
        return {"is_oa": False, "pdf_url": None, "oa_status": "Lookup Error"}
