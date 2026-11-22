"""
hsi_research_agent/tests/test_phase1.py
Unit and integration tests for Phase 1:
- Environment & Configuration
- SQLite Database & Repository CRUD
- Ollama Client & Structured Output with Pydantic
- LangGraph State & Checkpointer
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from pydantic import BaseModel, Field

from hsi_research_agent.config import settings
from hsi_research_agent.storage.database import init_db
from hsi_research_agent.storage.models import (
    NovelIdea,
    PaperRecord,
    ResearchGap,
    ResearchQuestion,
    ResearchTopic,
    SearchQuery,
)
from hsi_research_agent.storage.repository import ResearchRepository
from hsi_research_agent.llm.client import OllamaClient
from hsi_research_agent.graph.state import AgentState
from hsi_research_agent.graph.checkpoints import SQLiteWorkflowCheckpointer, get_default_checkpointer


class DummyAnalysis(BaseModel):
    domain: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    keywords: list[str]


class TestPhase1Foundation:
    """Test suite for Phase 1 components."""

    def test_configuration_and_directories(self, tmp_path):
        """Verify settings initialize and create necessary directories."""
        assert settings.OLLAMA_BASE_URL.startswith("http")
        assert "qwen" in settings.OLLAMA_MODEL.lower()
        assert settings.EXACT_QUERY_COUNT == 10
        assert settings.DATA_DIR.exists()
        assert settings.EXPORTS_DIR.exists()
        assert settings.LOGS_DIR.exists()

    def test_database_and_repository_crud(self, tmp_path):
        """Verify SQLite initialization and repository CRUD operations."""
        test_db = tmp_path / "test_research.db"
        init_db(test_db)
        assert test_db.exists()

        repo = ResearchRepository(test_db)

        # 1. Topic
        topic = ResearchTopic(
            topic_text="Foundation Models for Hyperspectral Imaging",
            target_domain="Remote Sensing & Computer Vision",
            start_year=2018,
            end_year=2026,
            core_concepts=["Hyperspectral Imaging", "Foundation Models", "Vision Transformers"],
            related_terminology=["Spectral-spatial feature extraction", "Self-supervised pretraining"],
            research_dimensions=["Model Architectures", "Downstream Tasks", "Cross-dataset generalization"],
            relevant_datasets=["Indian Pines", "Pavia University", "Houston 2018"],
            important_methodologies=["Masked Autoencoders", "Contrastive Learning"],
        )
        topic_id = repo.save_topic(topic)
        assert topic_id > 0

        # 2. Research Questions
        rq = ResearchQuestion(
            question_id="RQ1",
            question_text="How do vision foundation models adapt to high spectral dimensions?",
            framework="Domain-Specific",
            research_aspect="Architectural Adaptation",
            rationale="Explores spectral tokenization vs 3D convolutions."
        )
        repo.save_questions(topic_id, [rq])
        saved_questions = repo.get_questions(topic_id)
        assert len(saved_questions) == 1
        assert saved_questions[0].question_id == "RQ1"

        # 3. Exactly 10 Queries
        queries = [
            SearchQuery(
                query_id=f"Q{i:02d}",
                query_text=f"hyperspectral foundation model query {i}",
                research_dimension="Architecture",
                search_objective=f"Objective {i}",
                keywords=["hyperspectral", "foundation model"],
                expected_paper_types=["Journal Article"],
                priority=1,
                related_question_id="RQ1",
                validation_score=95.0,
                validation_status="Validated"
            )
            for i in range(1, 11)
        ]
        repo.save_queries(topic_id, queries)
        saved_queries = repo.get_queries(topic_id)
        assert len(saved_queries) == 10
        assert saved_queries[0].query_id == "Q01"

        # 4. Canonical Paper Upsert and Merge
        paper1 = PaperRecord(
            paper_id="paper_001",
            title="SpectralGPT: Spectral Foundation Model",
            authors=["Danfeng Hong", "Bing Zhang"],
            publication_year=2024,
            abstract="A novel foundation model for spectral remote sensing.",
            doi="10.1109/TGRS.2024.001",
            source_database="openalex",
            retrieved_query_ids=["Q01"],
            citation_count=45,
            relevance_score=9.2
        )
        repo.upsert_paper(paper1)

        # Upsert duplicate paper from crossref with different query id & higher citations
        paper2 = PaperRecord(
            paper_id="paper_002",
            title="SpectralGPT: Spectral Foundation Model (Crossref)",
            authors=["Danfeng Hong", "Bing Zhang"],
            publication_year=2024,
            abstract=None,
            doi="10.1109/TGRS.2024.001",  # Same DOI!
            source_database="crossref",
            retrieved_query_ids=["Q02"],
            citation_count=52,  # Higher!
            relevance_score=9.5
        )
        repo.upsert_paper(paper2)

        all_papers = repo.get_all_papers()
        assert len(all_papers) == 1  # Deduplicated and merged
        assert all_papers[0].citation_count == 52
        assert set(all_papers[0].retrieved_query_ids) == {"Q01", "Q02"}
        assert all_papers[0].abstract == "A novel foundation model for spectral remote sensing."

        # 5. Research Gaps & Novel Ideas
        gap = ResearchGap(
            gap_id="GAP-01",
            description="Lack of large-scale cross-sensor HSI pretraining datasets.",
            supporting_papers=["SpectralGPT: Spectral Foundation Model"],
            evidence="Studies train primarily on single sensor modalities (e.g. AVIRIS only).",
            existing_approaches="Sensor-specific fine-tuning",
            limitations_of_approaches="Fails to transfer across variable spectral band resolutions.",
            why_gap_matters="Limits real-world deployment on diverse aerial and satellite platforms."
        )
        repo.save_research_gaps(topic_id, [gap])
        saved_gaps = repo.get_research_gaps(topic_id)
        assert len(saved_gaps) == 1
        assert saved_gaps[0].gap_id == "GAP-01"

        idea = NovelIdea(
            idea_id="IDEA-01",
            proposed_title="Continuous Spectral Tokenizer for Cross-Sensor HSI Foundation Models",
            core_research_problem="Incompatible spectral channel dimensions across satellites",
            motivation="Enables one model to process any HSI sensor output zero-shot",
            research_gap_addressed="GAP-01",
            proposed_methodology="Wavelength-conditioned continuous neural implicit tokenization",
            experimental_design="Pretrain on AVIRIS, PRISMA, EnMAP; evaluate zero-shot on DESIS.",
            baseline_models=["SpectralGPT", "SatMAE"],
            evaluation_metrics=["Overall Accuracy", "Kappa Coefficient"],
            expected_contributions=["Zero-shot cross-sensor adaptation without re-tokenization"]
        )
        repo.save_novel_ideas(topic_id, [idea])
        saved_ideas = repo.get_novel_ideas(topic_id)
        assert len(saved_ideas) == 1
        assert saved_ideas[0].idea_id == "IDEA-01"

    def test_sqlite_checkpointer(self, tmp_path):
        """Verify workflow checkpoint persistence and restoration."""
        test_db = tmp_path / "test_checkpoints.db"
        init_db(test_db)

        checkpointer = SQLiteWorkflowCheckpointer(test_db)
        state_snapshot = {
            "research_topic": "Hyperspectral Vision Transformers",
            "current_stage": "QueryGeneration",
            "queries_generated": 10
        }
        checkpointer.save_checkpoint("run_001", "QueryGeneration", state_snapshot)

        loaded = checkpointer.load_checkpoint("run_001")
        assert loaded is not None
        assert loaded["research_topic"] == "Hyperspectral Vision Transformers"
        assert loaded["queries_generated"] == 10

        mem_checkpointer = get_default_checkpointer()
        assert mem_checkpointer is not None

    def test_ollama_client_connection(self):
        """Verify Ollama connection check function against running server."""
        client = OllamaClient()
        try:
            model_info = client.check_connection()
            assert model_info is not None
            assert "name" in model_info
        except RuntimeError as e:
            pytest.skip(f"Ollama server not active during test: {e}")

    @patch("requests.post")
    def test_ollama_client_structured_output_with_repair(self, mock_post):
        """
        Verify that OllamaClient handles structured output with Pydantic validation,
        and triggers a repair retry when the first attempt returns malformed data.
        """
        client = OllamaClient()

        # Attempt 1 returns invalid JSON structure
        mock_resp1 = MagicMock()
        mock_resp1.status_code = 200
        mock_resp1.json.return_value = {
            "message": {"content": '{"domain": "Computer Vision", "confidence": "INVALID_FLOAT"}'}
        }

        # Attempt 2 returns valid JSON conforming to DummyAnalysis schema
        mock_resp2 = MagicMock()
        mock_resp2.status_code = 200
        mock_resp2.json.return_value = {
            "message": {
                "content": '```json\n{"domain": "Computer Vision", "confidence": 0.95, "keywords": ["hsi", "transformer"]}\n```'
            }
        }

        mock_post.side_effect = [mock_resp1, mock_resp2]

        result = client.generate_structured(
            prompt="Analyze this domain",
            pydantic_class=DummyAnalysis,
            max_retries=2
        )

        assert isinstance(result, DummyAnalysis)
        assert result.domain == "Computer Vision"
        assert result.confidence == 0.95
        assert result.keywords == ["hsi", "transformer"]
        assert mock_post.call_count == 2  # Proves repair retry succeeded!

    def test_agent_state_typing(self):
        """Verify LangGraph AgentState conforms to TypedDict specification."""
        state: AgentState = {
            "research_topic": "Foundation Models for HSI",
            "start_year": 2018,
            "end_year": 2026,
            "target_domain": "Remote Sensing",
            "current_stage": "TopicAnalysis",
            "execution_errors": []
        }
        assert state["research_topic"] == "Foundation Models for HSI"
        assert state["current_stage"] == "TopicAnalysis"
