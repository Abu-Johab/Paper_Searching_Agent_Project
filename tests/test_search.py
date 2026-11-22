"""
hsi_research_agent/tests/test_search.py
Unit tests for Phase 3:
- Multi-Source Academic Search (OpenAlex, Semantic Scholar, Crossref, arXiv)
- Unpaywall Open Access Resolver
- Metadata Normalizer
- Search Manager & Audit Logging
"""

from unittest.mock import MagicMock, patch
import pytest

from hsi_research_agent.processing.normalizer import (
    clean_abstract,
    clean_doi,
    clean_arxiv_id,
    normalize_record,
)
from hsi_research_agent.search.openalex import search_openalex
from hsi_research_agent.search.semantic_scholar import search_semantic_scholar
from hsi_research_agent.search.crossref import search_crossref
from hsi_research_agent.search.arxiv import search_arxiv
from hsi_research_agent.search.unpaywall import resolve_open_access
from hsi_research_agent.search.search_manager import SearchManager
from hsi_research_agent.storage.models import SearchQuery
from hsi_research_agent.storage.repository import ResearchRepository
from hsi_research_agent.storage.database import init_db


class TestPhase3AcademicSearch:
    """Tests for multi-database search modules and normalizers."""

    def test_clean_helpers(self):
        assert clean_doi("https://doi.org/10.1109/TGRS.2024.123") == "10.1109/tgrs.2024.123"
        assert clean_doi("doi: 10.1016/j.isprsjprs.2023.01") == "10.1016/j.isprsjprs.2023.01"
        assert clean_arxiv_id("https://arxiv.org/abs/2401.12345v2") == "2401.12345"
        assert clean_abstract("<jats:p>Hyperspectral &amp; spatial analysis.</jats:p>") == "Hyperspectral & spatial analysis."

    def test_normalize_record_openalex(self):
        raw = {
            "title": "SpectralGPT: A Foundation Model for Spectral Remote Sensing",
            "doi": "https://doi.org/10.1109/TGRS.2024.3355000",
            "publication_year": 2024,
            "cited_by_count": 89,
            "ids": {"arxiv": "https://arxiv.org/abs/2311.07113"},
            "primary_location": {
                "source": {
                    "display_name": "IEEE Transactions on Geoscience and Remote Sensing",
                    "host_organization_name": "IEEE"
                }
            },
            "abstract_inverted_index": {
                "We": [0], "present": [1], "SpectralGPT": [2], "for": [3], "remote": [4], "sensing": [5]
            },
            "authorships": [{"author": {"display_name": "Danfeng Hong"}}]
        }
        rec = normalize_record(raw, "openalex", "Q01")
        assert rec.title == "SpectralGPT: A Foundation Model for Spectral Remote Sensing"
        assert rec.doi == "10.1109/tgrs.2024.3355000"
        assert rec.arxiv_id == "2311.07113"
        assert rec.publication_year == 2024
        assert rec.citation_count == 89
        assert rec.journal_or_conference == "IEEE Transactions on Geoscience and Remote Sensing"
        assert rec.abstract == "We present SpectralGPT for remote sensing"
        assert rec.authors == ["Danfeng Hong"]
        assert "Q01" in rec.retrieved_query_ids

    def test_normalize_record_crossref(self):
        raw = {
            "title": ["Masked Autoencoders for Hyperspectral Imaging"],
            "DOI": "10.1016/j.compag.2024.108500",
            "issued": {"date-parts": [[2024, 5]]},
            "container-title": ["Computers and Electronics in Agriculture"],
            "is-referenced-by-count": 14,
            "publisher": "Elsevier",
            "abstract": "<jats:p>Self-supervised spatial-spectral pretraining.</jats:p>",
            "author": [{"given": "Jane", "family": "Doe"}]
        }
        rec = normalize_record(raw, "crossref", "Q02")
        assert rec.title == "Masked Autoencoders for Hyperspectral Imaging"
        assert rec.doi == "10.1016/j.compag.2024.108500"
        assert rec.publication_year == 2024
        assert rec.citation_count == 14
        assert rec.journal_or_conference == "Computers and Electronics in Agriculture"
        assert rec.publisher == "Elsevier"
        assert "Self-supervised" in rec.abstract
        assert rec.authors == ["Jane Doe"]

    def test_normalize_record_arxiv(self):
        raw = {
            "id": "http://arxiv.org/abs/2402.99999v1",
            "title": "SatMAE-HSI: Multi-Band Transformers",
            "summary": "Scaling self-supervised models to 200 bands.",
            "published_year": 2024,
            "doi": None,
            "journal": "arXiv",
            "authors": ["Alice Smith", "Bob Jones"]
        }
        rec = normalize_record(raw, "arxiv", "Q03")
        assert rec.title == "SatMAE-HSI: Multi-Band Transformers"
        assert rec.arxiv_id == "2402.99999"
        assert rec.publication_year == 2024
        assert rec.open_access_status == "Open Access (Preprint)"
        assert rec.pdf_url == "https://arxiv.org/pdf/2402.99999.pdf"

    @patch("requests.get")
    def test_search_openalex_request(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [{"title": "HSI Paper 1", "doi": "10.1000/1"}]
        }
        mock_get.return_value = mock_resp

        results = search_openalex("hyperspectral foundation model", start_year=2022, end_year=2026)
        assert len(results) == 1
        call_kwargs = mock_get.call_args[1]
        params = call_kwargs["params"]
        assert params["search"] == "hyperspectral foundation model"
        assert "from_publication_date:2022-01-01" in params["filter"]
        assert "to_publication_date:2026-12-31" in params["filter"]

    @patch("requests.get")
    def test_unpaywall_resolver(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "is_oa": True,
            "oa_status": "gold",
            "best_oa_location": {
                "url_for_pdf": "https://example.com/paper.pdf",
                "version": "publishedVersion"
            }
        }
        mock_get.return_value = mock_resp

        oa = resolve_open_access("10.1109/TGRS.2024.001")
        assert oa["is_oa"] is True
        assert oa["oa_status"] == "gold"
        assert oa["pdf_url"] == "https://example.com/paper.pdf"

    @patch("hsi_research_agent.search.search_manager.search_openalex")
    @patch("hsi_research_agent.search.search_manager.search_crossref")
    @patch("hsi_research_agent.search.search_manager.search_arxiv")
    @patch("hsi_research_agent.search.search_manager.search_semantic_scholar")
    def test_search_manager_execution_and_logging(
        self, mock_s2, mock_arxiv, mock_cr, mock_oa, tmp_path
    ):
        test_db = tmp_path / "test_search_log.db"
        init_db(test_db)
        repo = ResearchRepository(test_db)

        mock_oa.return_value = [{"title": "OpenAlex Paper", "doi": "10.1000/oa"}]
        mock_cr.return_value = [{"title": ["Crossref Paper"], "DOI": "10.1000/cr"}]
        mock_arxiv.return_value = [{"title": "arXiv Paper", "id": "2401.0001", "published_year": 2024}]
        mock_s2.return_value = [{"title": "S2 Paper", "externalIds": {"DOI": "10.1000/s2"}}]

        manager = SearchManager(
            repository=repo,
            sources=["openalex", "crossref", "arxiv", "semanticscholar"]
        )

        queries = [
            SearchQuery(
                query_id="Q01",
                query_text="spectralgpt",
                research_dimension="Architecture",
                search_objective="Test"
            ),
            SearchQuery(
                query_id="Q02",
                query_text="satmae",
                research_dimension="Pretraining",
                search_objective="Test"
            ),
        ]

        papers, logs = manager.execute_all_searches(queries)
        assert len(papers) == 8  # 2 queries * 4 sources = 8 papers
        assert len(logs) == 8    # 8 search log entries
        # Verify repository logged the searches
        saved_logs = repo.get_search_logs()
        assert len(saved_logs) == 8
        assert saved_logs[0].query_id == "Q01"
        assert saved_logs[0].status == "Success"
