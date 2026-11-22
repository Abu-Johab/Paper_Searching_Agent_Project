"""
hsi_research_agent/processing/deduplicator.py
Multi-stage deterministic deduplication engine with metadata merging and query provenance preservation.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from hsi_research_agent.processing.normalizer import clean_arxiv_id, clean_doi
from hsi_research_agent.storage.models import PaperRecord

logger = logging.getLogger("hsi_research_agent.processing.deduplicator")


def _title_slug(title: Optional[str]) -> str:
    """Normalize title to alphanumeric characters only for fuzzy matching."""
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]", "", title.lower())


def deduplicate_papers(
    papers: List[PaperRecord],
) -> Tuple[List[PaperRecord], List[Dict[str, Any]]]:
    """
    Deduplicate paper records using:
    1. Normalized DOI matching
    2. Normalized arXiv ID matching
    3. Normalized PMID matching
    4. Fuzzy title slug matching (min length 10)

    When duplicates merge:
    - Merges retrieved_query_ids (provenance preserved).
    - Keeps record with non-empty abstract (or prefers longer abstract).
    - Keeps higher citation count.
    - Fills in missing DOI, arXiv ID, PMID, journal, publisher, or URLs.
    - Preserves audit trail of duplicate groups.
    """
    unique_list: List[PaperRecord] = []
    duplicate_groups: List[Dict[str, Any]] = []

    doi_map: Dict[str, int] = {}
    arxiv_map: Dict[str, int] = {}
    pmid_map: Dict[str, int] = {}
    title_map: Dict[str, int] = {}

    for paper in papers:
        doi = clean_doi(paper.doi) or ""
        arxiv_id = clean_arxiv_id(paper.arxiv_id) or ""
        pmid = (paper.pmid or "").strip()
        t_slug = _title_slug(paper.title)

        match_idx: Optional[int] = None
        match_reason: str = ""

        if doi and doi in doi_map:
            match_idx = doi_map[doi]
            match_reason = f"DOI Match: {doi}"
        elif arxiv_id and arxiv_id in arxiv_map:
            match_idx = arxiv_map[arxiv_id]
            match_reason = f"arXiv ID Match: {arxiv_id}"
        elif pmid and pmid in pmid_map:
            match_idx = pmid_map[pmid]
            match_reason = f"PMID Match: {pmid}"
        elif len(t_slug) >= 12 and t_slug in title_map:
            match_idx = title_map[t_slug]
            match_reason = f"Title Slug Match: {t_slug[:30]}..."

        if match_idx is not None:
            canonical = unique_list[match_idx]

            # Record duplicate group entry
            duplicate_groups.append({
                "canonical_paper_id": canonical.paper_id,
                "duplicate_paper_id": paper.paper_id,
                "matched_by": match_reason,
                "source_database": paper.source_database,
                "query_id": paper.retrieved_query_ids[0] if paper.retrieved_query_ids else "",
            })

            # 1. Merge retrieved query IDs
            merged_queries = list(
                dict.fromkeys(canonical.retrieved_query_ids + paper.retrieved_query_ids)
            )
            canonical.retrieved_query_ids = merged_queries

            # 2. Prefer non-empty or longer abstract
            if not canonical.abstract and paper.abstract:
                canonical.abstract = paper.abstract
            elif canonical.abstract and paper.abstract and len(paper.abstract) > len(canonical.abstract):
                canonical.abstract = paper.abstract

            # 3. Fill missing metadata fields
            if not canonical.doi and paper.doi:
                canonical.doi = paper.doi
            if not canonical.arxiv_id and paper.arxiv_id:
                canonical.arxiv_id = paper.arxiv_id
            if not canonical.pmid and paper.pmid:
                canonical.pmid = paper.pmid
            if not canonical.journal_or_conference and paper.journal_or_conference:
                canonical.journal_or_conference = paper.journal_or_conference
            if not canonical.publisher and paper.publisher:
                canonical.publisher = paper.publisher
            if not canonical.pdf_url and paper.pdf_url:
                canonical.pdf_url = paper.pdf_url
            if not canonical.paper_url and paper.paper_url:
                canonical.paper_url = paper.paper_url
            if not canonical.authors and paper.authors:
                canonical.authors = paper.authors

            # 4. Keep higher citation count
            canonical.citation_count = max(canonical.citation_count, paper.citation_count)

            # Update index maps with any newly filled identifiers
            if canonical.doi:
                doi_map[canonical.doi.lower()] = match_idx
            if canonical.arxiv_id:
                arxiv_map[canonical.arxiv_id.lower()] = match_idx
            if canonical.pmid:
                pmid_map[canonical.pmid] = match_idx

        else:
            # New unique record
            new_idx = len(unique_list)
            # Make a copy so we can mutate safely
            new_canonical = paper.model_copy(deep=True)
            unique_list.append(new_canonical)

            if doi:
                doi_map[doi] = new_idx
            if arxiv_id:
                arxiv_map[arxiv_id] = new_idx
            if pmid:
                pmid_map[pmid] = new_idx
            if len(t_slug) >= 12:
                title_map[t_slug] = new_idx

    logger.info(
        "Deduplication complete: %d raw records reduced to %d unique papers (%d duplicates merged).",
        len(papers),
        len(unique_list),
        len(duplicate_groups),
    )
    return unique_list, duplicate_groups
