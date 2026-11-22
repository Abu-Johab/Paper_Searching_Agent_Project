"""
hsi_research_agent/processing/normalizer.py
Standardizes bibliographic metadata from diverse academic APIs into canonical PaperRecord models.
"""

import html
import re
from typing import Any, Dict, List, Optional, Tuple

from hsi_research_agent.storage.models import PaperRecord


def clean_abstract(raw_text: Optional[str]) -> Optional[str]:
    """Strip JATS XML / HTML tags, decode entities, and normalize whitespace."""
    if not raw_text:
        return None
    text = re.sub(r"<[^>]+>", " ", str(raw_text))
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def clean_doi(raw_doi: Optional[str]) -> Optional[str]:
    """Normalize DOI: lowercase, strip prefixes and whitespace."""
    if not raw_doi:
        return None
    doi = str(raw_doi).strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip() if doi else None


def clean_arxiv_id(raw_id: Optional[str]) -> Optional[str]:
    """Extract and normalize canonical arXiv ID (e.g. 2301.12345)."""
    if not raw_id:
        return None
    s = str(raw_id).strip().lower()
    s = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", s)
    s = re.sub(r"^arxiv:\s*", "", s)
    s = re.sub(r"v\d+$", "", s)
    return s.strip() if s else None


def rebuild_abstract_from_inverted_index(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """Rebuild full abstract string from OpenAlex abstract_inverted_index."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return None
    indexed_tokens: List[Tuple[int, str]] = []
    for token, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                indexed_tokens.append((pos, token))
    if not indexed_tokens:
        return None
    indexed_tokens.sort(key=lambda x: x[0])
    return " ".join(token for _, token in indexed_tokens).strip()


def normalize_record(raw_record: Dict[str, Any], source: str, query_id: str) -> PaperRecord:
    """
    Transform raw API payload into a standardized canonical PaperRecord.
    """
    title = ""
    authors: List[str] = []
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    year: Optional[int] = None
    journal = ""
    publisher: Optional[str] = None
    citations = 0
    abstract: Optional[str] = None
    paper_url = ""
    pdf_url: Optional[str] = None
    open_access_status = "Unknown"

    if source == "openalex":
        title = re.sub(r"\s+", " ", str(raw_record.get("title") or "").strip())
        doi = clean_doi(raw_record.get("doi"))
        ids_dict = raw_record.get("ids") or {}
        arxiv_id = clean_arxiv_id(ids_dict.get("arxiv"))
        pmid = str(ids_dict.get("pmid") or "").strip() or None

        raw_year = raw_record.get("publication_year")
        try:
            year = int(raw_year) if raw_year is not None else None
        except (ValueError, TypeError):
            year = None

        primary_loc = raw_record.get("primary_location") or {}
        source_loc = primary_loc.get("source") or {}
        journal = str(source_loc.get("display_name") or "").strip()
        publisher = source_loc.get("host_organization_name")
        citations = int(raw_record.get("cited_by_count") or 0)

        raw_ab = raw_record.get("abstract")
        if not raw_ab and "abstract_inverted_index" in raw_record:
            raw_ab = rebuild_abstract_from_inverted_index(raw_record.get("abstract_inverted_index"))
        abstract = clean_abstract(raw_ab)

        oa_info = raw_record.get("open_access") or {}
        if oa_info.get("is_oa"):
            open_access_status = oa_info.get("oa_status") or "Open Access"
            pdf_url = oa_info.get("oa_url")

        for author_entry in raw_record.get("authorships", []):
            author_obj = author_entry.get("author", {})
            name = author_obj.get("display_name")
            if name:
                authors.append(name.strip())

        paper_url = f"https://doi.org/{doi}" if doi else (primary_loc.get("landing_page_url") or str(raw_record.get("id") or ""))

    elif source == "crossref":
        raw_title = raw_record.get("title", "")
        if isinstance(raw_title, list):
            title = str(raw_title[0]).strip() if raw_title else ""
        else:
            title = str(raw_title).strip()
        title = re.sub(r"\s+", " ", title)

        doi = clean_doi(raw_record.get("DOI"))
        abstract = clean_abstract(raw_record.get("abstract"))
        publisher = raw_record.get("publisher")

        date_parts = raw_record.get("issued", {}).get("date-parts", [[]])
        if date_parts and len(date_parts[0]) > 0 and date_parts[0][0] is not None:
            try:
                year = int(date_parts[0][0])
            except (ValueError, TypeError):
                year = None

        container = raw_record.get("container-title", [])
        if isinstance(container, list) and container:
            journal = str(container[0]).strip()
        elif isinstance(container, str):
            journal = container.strip()

        citations = int(raw_record.get("is-referenced-by-count") or 0)

        for a in raw_record.get("author", []):
            given = a.get("given", "").strip()
            family = a.get("family", "").strip()
            name = f"{given} {family}".strip()
            if name:
                authors.append(name)

        paper_url = f"https://doi.org/{doi}" if doi else str(raw_record.get("URL") or "")

    elif source == "arxiv":
        title = re.sub(r"\s+", " ", str(raw_record.get("title") or "").strip())
        raw_id = raw_record.get("id") or ""
        arxiv_id = clean_arxiv_id(raw_id)
        doi = clean_doi(raw_record.get("doi"))
        year = raw_record.get("published_year")
        journal = str(raw_record.get("journal") or "arXiv").strip()
        abstract = clean_abstract(raw_record.get("summary"))
        authors = raw_record.get("authors") or []
        open_access_status = "Open Access (Preprint)"
        paper_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else raw_id
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None

    elif source == "semanticscholar":
        title = re.sub(r"\s+", " ", str(raw_record.get("title") or "").strip())
        ext_ids = raw_record.get("externalIds") or {}
        doi = clean_doi(ext_ids.get("DOI"))
        arxiv_id = clean_arxiv_id(ext_ids.get("ArXiv"))
        raw_year = raw_record.get("year")
        try:
            year = int(raw_year) if raw_year is not None else None
        except (ValueError, TypeError):
            year = None

        journal = str(raw_record.get("venue") or "").strip()
        citations = int(raw_record.get("citationCount") or 0)
        abstract = clean_abstract(raw_record.get("abstract"))

        for a in raw_record.get("authors", []):
            name = a.get("name")
            if name:
                authors.append(name.strip())

        paper_id = raw_record.get("paperId")
        if doi:
            paper_url = f"https://doi.org/{doi}"
        elif arxiv_id:
            paper_url = f"https://arxiv.org/abs/{arxiv_id}"
        elif paper_id:
            paper_url = f"https://www.semanticscholar.org/paper/{paper_id}"

    # Generate stable unique paper ID
    if doi:
        paper_id = f"doi_{re.sub(r'[^a-zA-Z0-9]', '_', doi)}"
    elif arxiv_id:
        paper_id = f"arxiv_{re.sub(r'[^a-zA-Z0-9]', '_', arxiv_id)}"
    else:
        title_slug = re.sub(r"[^a-zA-Z0-9]", "_", title.lower())[:40]
        paper_id = f"{source}_{title_slug}_{year or 'unknown'}"

    return PaperRecord(
        paper_id=paper_id,
        title=title,
        authors=authors,
        publication_year=year,
        abstract=abstract,
        doi=doi,
        arxiv_id=arxiv_id,
        pmid=pmid,
        publisher=publisher,
        journal_or_conference=journal,
        paper_url=paper_url,
        pdf_url=pdf_url,
        source_database=source,
        retrieved_query_ids=[query_id] if query_id else [],
        citation_count=citations,
        open_access_status=open_access_status,
        full_text_status="Abstract only" if abstract else "Metadata only",
    )
