"""
hsi_research_agent/processing/pdf_processor.py
Legal open-access full-text PDF retrieval and section parsing using PyMuPDF.
Never bypasses paywalls or unauthorized access restrictions.
"""

import io
import logging
import re
from typing import Dict, List, Optional
import requests
import pymupdf

from hsi_research_agent.config import settings
from hsi_research_agent.search.unpaywall import resolve_open_access
from hsi_research_agent.storage.models import PaperRecord

logger = logging.getLogger("hsi_research_agent.processing.pdf")


def extract_sections_from_pdf_stream(pdf_bytes: bytes) -> Dict[str, str]:
    """
    Extract text content from raw PDF bytes and partition into major academic sections:
    Abstract, Methodology, Experiments/Datasets, Limitations, and Conclusion.
    """
    sections = {
        "abstract": "",
        "methodology": "",
        "experiments": "",
        "limitations": "",
        "conclusion": "",
        "full_text": "",
    }

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            pages_text: List[str] = []
            for p in doc:
                pages_text.append(p.get_text())
            full_text = "\n\n".join(pages_text).strip()
            sections["full_text"] = full_text
        finally:
            doc.close()

        # Heuristic section partitioning
        lower_full = full_text.lower()

        # Limitations section
        limitations_match = re.search(
            r"(?:(?:\d+\.|\b)(?:limitations|threats to validity|future work|discussion and limitations)\b)([\s\S]{100,2500}?)(?:\n\s*(?:\d+\.|\b)(?:conclusion|references|acknowledgements)\b|$)",
            full_text,
            re.IGNORECASE,
        )
        if limitations_match:
            sections["limitations"] = limitations_match.group(1).strip()

        # Methodology / Proposed Architecture section
        methodology_match = re.search(
            r"(?:(?:\d+\.|\b)(?:methodology|proposed method|model architecture|method|framework)\b)([\s\S]{100,3500}?)(?:\n\s*(?:\d+\.|\b)(?:experiments|results|evaluation)\b|$)",
            full_text,
            re.IGNORECASE,
        )
        if methodology_match:
            sections["methodology"] = methodology_match.group(1).strip()

        # Experiments & Datasets section
        experiments_match = re.search(
            r"(?:(?:\d+\.|\b)(?:experiments|experimental setup|datasets and implementation|results)\b)([\s\S]{100,3500}?)(?:\n\s*(?:\d+\.|\b)(?:discussion|conclusion|limitations)\b|$)",
            full_text,
            re.IGNORECASE,
        )
        if experiments_match:
            sections["experiments"] = experiments_match.group(1).strip()

        # Conclusion section
        conclusion_match = re.search(
            r"(?:(?:\d+\.|\b)(?:conclusion|concluding remarks)\b)([\s\S]{50,2000}?)(?:\n\s*(?:\d+\.|\b)(?:references|acknowledgements)\b|$)",
            full_text,
            re.IGNORECASE,
        )
        if conclusion_match:
            sections["conclusion"] = conclusion_match.group(1).strip()

        return sections

    except Exception as e:
        logger.warning("Error parsing PDF with PyMuPDF: %s", str(e))
        return sections


def retrieve_paper_full_text(paper: PaperRecord) -> PaperRecord:
    """
    Attempt to retrieve legal open-access full-text PDF for a paper.
    Updates full_text and full_text_status attributes.
    """
    pdf_url = paper.pdf_url

    # If no PDF URL is known, check Unpaywall via DOI
    if not pdf_url and paper.doi:
        oa_info = resolve_open_access(paper.doi)
        if oa_info.get("is_oa") and oa_info.get("pdf_url"):
            pdf_url = oa_info.get("pdf_url")
            paper.pdf_url = pdf_url
            paper.open_access_status = oa_info.get("oa_status") or "Open Access"

    if not pdf_url:
        paper.full_text_status = "Full text unavailable" if paper.abstract else "Metadata only"
        return paper

    # Download PDF
    headers = {"User-Agent": f"HSIResearchAgent/1.0 (mailto:{settings.CONTACT_EMAIL})"}
    try:
        resp = requests.get(pdf_url, headers=headers, timeout=20, stream=True)
        if resp.status_code == 200:
            content_type = resp.headers.get("content-type", "").lower()
            if "pdf" in content_type or pdf_url.endswith(".pdf") or resp.content[:4] == b"%PDF":
                sections = extract_sections_from_pdf_stream(resp.content)
                if sections.get("full_text"):
                    paper.full_text = sections["full_text"][:30000]  # Cap to reasonable window
                    paper.full_text_status = "Full text retrieved"
                    logger.info("Successfully retrieved full text for: '%s'", paper.title[:40])
                    return paper

        paper.full_text_status = "Retrieval failed"
        return paper

    except Exception as e:
        logger.debug("PDF download failed for %s: %s", paper.paper_id, str(e))
        paper.full_text_status = "Retrieval failed"
        return paper
