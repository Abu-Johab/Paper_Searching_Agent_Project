"""
hsi_research_agent/tests/test_queries.py
Unit tests for Phase 2:
- Topic Analysis
- Research Question Formulation
- Exactly 10 Search Queries Generation
- Query Validation & Refinement
"""

from unittest.mock import MagicMock, patch
import pytest

from hsi_research_agent.agents.topic_analyzer import analyze_topic, TopicAnalysisOutput
from hsi_research_agent.agents.question_generator import generate_research_questions, QuestionListOutput
from hsi_research_agent.agents.query_generator import generate_search_queries
from hsi_research_agent.agents.query_validator import validate_all_queries, validate_query, SingleQueryEvaluation
from hsi_research_agent.storage.models import ResearchQuestion, ResearchTopic, SearchQuery, QueryList


class TestPhase2QueryGeneration:
    """Tests for Phase 2 query generation and validation pipeline."""

    @patch("hsi_research_agent.llm.client.OllamaClient.generate_structured")
    def test_topic_analysis(self, mock_structured):
        mock_structured.return_value = TopicAnalysisOutput(
            target_domain="Hyperspectral Remote Sensing",
            core_concepts=["Hyperspectral Imaging", "Foundation Models", "Vision Transformers"],
            related_terminology=["Masked Autoencoding", "Spectral Tokenizer"],
            research_dimensions=["Architectures", "Pretraining", "Cross-Sensor", "Classification"],
            relevant_datasets=["Indian Pines", "Pavia University", "Houston 2018"],
            important_methodologies=["SpectralGPT", "SatMAE"]
        )

        topic = analyze_topic("Foundation Models for Hyperspectral Imaging")
        assert topic.topic_text == "Foundation Models for Hyperspectral Imaging"
        assert "Hyperspectral Imaging" in topic.core_concepts
        assert len(topic.research_dimensions) >= 4
        assert "Indian Pines" in topic.relevant_datasets

    @patch("hsi_research_agent.llm.client.OllamaClient.generate_structured")
    def test_question_generation(self, mock_structured):
        mock_structured.return_value = QuestionListOutput(
            questions=[
                ResearchQuestion(
                    question_id="RQ1",
                    question_text="What architectures are used for spectral foundation models?",
                    framework="Domain-Specific",
                    research_aspect="Architectures",
                    rationale="Assesses backbone design."
                ),
                ResearchQuestion(
                    question_id="RQ2",
                    question_text="How well do foundation models generalize across hyperspectral sensors?",
                    framework="Domain-Specific",
                    research_aspect="Cross-Sensor Transfer",
                    rationale="Assesses band dimensionality transfer."
                ),
                ResearchQuestion(
                    question_id="RQ3",
                    question_text="What are standard downstream benchmarks for HSI foundation models?",
                    framework="Domain-Specific",
                    research_aspect="Evaluation Benchmarks",
                    rationale="Surveys benchmark accuracy."
                ),
            ]
        )

        topic = ResearchTopic(
            topic_text="Foundation Models for Hyperspectral Imaging",
            core_concepts=["HSI", "Transformers"],
            research_dimensions=["Architectures", "Transfer", "Benchmarks"]
        )
        questions = generate_research_questions(topic)
        assert len(questions) == 3
        assert questions[0].question_id == "RQ1"

    def test_generate_exactly_10_queries_with_fallback(self):
        """Verify that generate_search_queries guarantees EXACTLY 10 distinct queries."""
        topic = ResearchTopic(
            topic_text="Foundation Models for Hyperspectral Imaging",
            core_concepts=["HSI", "Foundation Models"],
            research_dimensions=["Backbones", "Pretraining", "Generalization"]
        )
        questions = [
            ResearchQuestion(
                question_id="RQ1",
                question_text="How do models scale?",
                framework="Domain-Specific",
                research_aspect="Scale"
            )
        ]

        # Test with mocked LLM raising error to test deterministic guarantee
        with patch("hsi_research_agent.llm.client.OllamaClient.generate_structured", side_effect=Exception("Timeout")):
            queries = generate_search_queries(topic, questions)
            assert len(queries) == 10
            # IDs Q01 to Q10
            for i, q in enumerate(queries, 1):
                assert q.query_id == f"Q{i:02d}"
                assert len(q.query_text) > 5
                assert q.research_dimension is not None
                assert q.search_objective is not None

    @patch("hsi_research_agent.llm.client.OllamaClient.generate_structured")
    def test_query_validation_and_refinement(self, mock_structured):
        # Query 1 passes, Query 2 is weak and gets refined
        mock_structured.side_effect = [
            SingleQueryEvaluation(
                score=92.0,
                is_valid=True,
                specificity_score=90.0,
                coverage_score=95.0,
                feedback="Excellent query."
            ),
            SingleQueryEvaluation(
                score=55.0,
                is_valid=False,
                specificity_score=50.0,
                coverage_score=60.0,
                feedback="Too generic.",
                suggested_replacement="hyperspectral foundation model self-supervised pretraining"
            )
        ]

        topic = ResearchTopic(topic_text="Foundation Models for Hyperspectral Imaging")
        queries = [
            SearchQuery(
                query_id="Q01",
                query_text="spectralgpt foundation model architecture",
                research_dimension="Architecture",
                search_objective="Backbones"
            ),
            SearchQuery(
                query_id="Q02",
                query_text="ai in farming",  # Weak
                research_dimension="Pretraining",
                search_objective="Self-supervision"
            )
        ]

        validated = validate_all_queries(queries, topic)
        assert len(validated) == 2
        assert validated[0].validation_score == 92.0
        assert validated[0].validation_status == "Validated"
        # Q02 should be refined
        assert validated[1].query_text == "hyperspectral foundation model self-supervised pretraining"
        assert validated[1].validation_score >= 70.0
        assert "Refined" in validated[1].validation_status
