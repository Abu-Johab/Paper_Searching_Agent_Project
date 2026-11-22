"""
hsi_research_agent/tests/test_screening.py
Unit tests for relevance scoring, title capping, screening decisions, and PDF processing.
"""

from unittest.mock import MagicMock, patch
import pymupdf
import pytest

from hsi_research_agent.agents.screening_agent import screen_paper, screen_all_papers, ScreeningOutput
from hsi_research_agent.processing.relevance_scorer import score_paper_relevance, RelevanceOutput
from hsi_research_agent.processing.pdf_processor import extract_sections_from_pdf_stream, retrieve_paper_full_text
from hsi_research_agent.storage.models import PaperRecord, ResearchTopic


class TestScreeningAndProcessing:
    """Tests for Phase 4 screening, relevance scoring, and full-text retrieval."""

    @patch("hsi_research_agent.llm.client.OllamaClient.generate_structured")
    def test_relevance_scorer_with_abstract(self, mock_structured):
        mock_structured.return_value = RelevanceOutput(
            score=9.5,
            reason="Directly proposes a foundation model for hyperspectral imagery."
        )

        paper = PaperRecord(
            paper_id="p1",
            title="SpectralGPT",
            abstract="A foundational vision transformer for spectral remote sensing.",
            source_database="openalex"
        )
        score, reason = score_paper_relevance(paper, "Foundation Models for Hyperspectral Imaging")
        assert score == 9.5
        assert "Directly proposes" in reason

    @patch("hsi_research_agent.llm.client.OllamaClient.generate_structured")
    def test_relevance_scorer_missing_abstract_capped_at_7(self, mock_structured):
        # Even if LLM returns 9.8, lack of abstract MUST cap score at 7.0
        mock_structured.return_value = RelevanceOutput(
            score=9.8,
            reason="Title implies high relevance."
        )

        paper = PaperRecord(
            paper_id="p2",
            title="SpectralGPT Architecture Survey",
            abstract=None,  # Missing
            source_database="crossref"
        )
        score, reason = score_paper_relevance(paper, "Foundation Models for Hyperspectral Imaging")
        assert score == 7.0

    @patch("hsi_research_agent.llm.client.OllamaClient.generate_structured")
    def test_screen_paper_include_and_exclude(self, mock_structured):
        mock_structured.side_effect = [
            ScreeningOutput(
                decision="Include",
                relevance_score=9.0,
                inclusion_reason="Proposes a self-supervised masked autoencoder for HSI cubes."
            ),
            ScreeningOutput(
                decision="Exclude",
                relevance_score=2.0,
                exclusion_reason="Clinical dentistry paper unrelated to hyperspectral remote sensing."
            )
        ]

        topic = ResearchTopic(
            topic_text="Foundation Models for Hyperspectral Imaging",
            target_domain="Remote Sensing"
        )
        p1 = PaperRecord(paper_id="p1", title="HSI MAE", abstract="Method for HSI", source_database="arxiv")
        p2 = PaperRecord(paper_id="p2", title="Dental X-ray imaging", abstract="Dental", source_database="pubmed")

        eligible, all_decisions = screen_all_papers([p1, p2], topic)
        assert len(eligible) == 1
        assert eligible[0].paper_id == "p1"
        assert len(all_decisions) == 2
        assert all_decisions[0].decision == "Include"
        assert all_decisions[1].decision == "Exclude"
        assert "dentistry" in all_decisions[1].exclusion_reason

    def test_extract_sections_from_pdf_stream(self):
        """Verify section partitioning from synthetic in-memory PDF."""
        doc = pymupdf.open()
        p = doc.new_page()
        pdf_text = (
            "Spectral Foundation Models\n\n"
            "Abstract\nWe present a foundation model for remote sensing.\n\n"
            "3. Methodology\nWe apply continuous 3D patch tokenization over 200 spectral bands.\n\n"
            "4. Experiments\nWe evaluate on Indian Pines and Pavia University benchmarks.\n\n"
            "5. Limitations\nThe model has high computational memory demands during pretraining.\n\n"
            "6. Conclusion\nThe method demonstrates state of the art results.\n"
        )
        p.insert_text((40, 40), pdf_text)
        pdf_bytes = doc.tobytes()

        sections = extract_sections_from_pdf_stream(pdf_bytes)
        assert "Spectral Foundation Models" in sections["full_text"]
        assert "Methodology" in sections["methodology"] or "patch tokenization" in sections["methodology"]
        assert "Indian Pines" in sections["experiments"] or "Experiments" in sections["experiments"]
        assert "computational memory demands" in sections["limitations"]
