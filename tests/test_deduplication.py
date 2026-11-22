"""
hsi_research_agent/tests/test_deduplication.py
Unit tests for multi-stage deduplication and query provenance tracking.
"""

import pytest
from hsi_research_agent.processing.deduplicator import deduplicate_papers, _title_slug
from hsi_research_agent.storage.models import PaperRecord


class TestDeduplicationEngine:
    """Tests for deduplication logic, key matching, and metadata merging."""

    def test_title_slug_normalization(self):
        slug1 = _title_slug("SpectralGPT: A Foundation Model for Spectral Remote Sensing!")
        slug2 = _title_slug("spectralgpt: a foundation model for spectral remote sensing")
        assert slug1 == slug2
        assert slug1.startswith("spectralgpt")

    def test_deduplicate_by_doi(self):
        p1 = PaperRecord(
            paper_id="p1",
            title="SpectralGPT",
            doi="10.1109/tgrs.2024.001",
            source_database="openalex",
            retrieved_query_ids=["Q01"],
            citation_count=10,
            abstract="Short abstract."
        )
        p2 = PaperRecord(
            paper_id="p2",
            title="SpectralGPT Alternate",
            doi="10.1109/TGRS.2024.001",  # Same DOI, uppercase
            source_database="crossref",
            retrieved_query_ids=["Q02"],
            citation_count=25,  # Higher citations
            abstract="A much longer and comprehensive abstract text."
        )

        unique, dups = deduplicate_papers([p1, p2])
        assert len(unique) == 1
        assert len(dups) == 1
        canonical = unique[0]
        assert canonical.citation_count == 25
        assert set(canonical.retrieved_query_ids) == {"Q01", "Q02"}
        assert "much longer" in canonical.abstract

    def test_deduplicate_by_arxiv_id(self):
        p1 = PaperRecord(
            paper_id="p1",
            title="SatMAE: Pretraining on Multi-Spectral and Hyperspectral Imagery",
            arxiv_id="2207.08051",
            source_database="arxiv",
            retrieved_query_ids=["Q01"]
        )
        p2 = PaperRecord(
            paper_id="p2",
            title="SatMAE Paper",
            arxiv_id="2207.08051v2",
            doi="10.5555/satmae",
            source_database="semanticscholar",
            retrieved_query_ids=["Q03"]
        )

        unique, dups = deduplicate_papers([p1, p2])
        assert len(unique) == 1
        assert unique[0].doi == "10.5555/satmae"
        assert set(unique[0].retrieved_query_ids) == {"Q01", "Q03"}

    def test_deduplicate_by_fuzzy_title_slug(self):
        p1 = PaperRecord(
            paper_id="p1",
            title="Deep Learning Framework for Airborne Hyperspectral Image Classification",
            source_database="openalex",
            retrieved_query_ids=["Q01"],
            abstract="Testing CNN on Pavia dataset."
        )
        p2 = PaperRecord(
            paper_id="p2",
            title="Deep Learning Framework for Airborne Hyperspectral Image Classification.",  # Identical slug
            source_database="crossref",
            retrieved_query_ids=["Q04"],
            doi="10.1016/j.rse.2023.01"
        )

        unique, dups = deduplicate_papers([p1, p2])
        assert len(unique) == 1
        assert unique[0].doi == "10.1016/j.rse.2023.01"
        assert set(unique[0].retrieved_query_ids) == {"Q01", "Q04"}
