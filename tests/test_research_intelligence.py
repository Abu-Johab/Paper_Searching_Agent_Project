"""
Tests for Phase 6: Research Intelligence
(Research Gap Agent, Novel Idea Generator, Evidence Synthesis Agent)
"""

import pytest
from unittest.mock import MagicMock

from hsi_research_agent.storage.models import (
    ResearchTopic,
    PaperRecord,
    ExtractedInfo,
    ResearchGap,
    ResearchGapList,
    NovelIdea,
    NovelIdeaList,
    EvidenceSynthesis,
)
from hsi_research_agent.agents.research_gap_agent import ResearchGapAgent
from hsi_research_agent.agents.idea_generator import IdeaGeneratorAgent
from hsi_research_agent.agents.synthesis_agent import SynthesisAgent


@pytest.fixture
def sample_topic():
    return ResearchTopic(
        topic_id=1,
        topic_text="Hyperspectral Foundation Models in Agriculture",
        target_domain="Hyperspectral Remote Sensing",
    )


@pytest.fixture
def sample_papers():
    return [
        PaperRecord(
            paper_id="paper-01",
            title="SpectralFormer: Rethinking Hyperspectral Image Classification with Transformers",
            publication_year=2022,
            doi="10.1109/TGRS.2022.3144304",
            abstract="We present SpectralFormer for hyperspectral classification with spectral-spatial attention.",
            source_database="OpenAlex",
        ),
        PaperRecord(
            paper_id="paper-02",
            title="Masked Autoencoding for Hyperspectral Pretraining",
            publication_year=2023,
            doi="10.1109/TGRS.2023.3288111",
            abstract="Self-supervised pretraining using masked autoencoders over multi-sensor hyperspectral imagery.",
            source_database="arXiv",
        ),
    ]


@pytest.fixture
def sample_extracted():
    return [
        ExtractedInfo(
            paper_id="paper-01",
            title="SpectralFormer: Rethinking Hyperspectral Image Classification with Transformers",
            methodology="Spectral group transformer",
            model_architecture="SpectralFormer",
            dataset="Indian Pines, Pavia University",
            limitations="Quadratic computational complexity over spectral bands",
            main_findings="Outperformed 3D-CNNs by 2.4% overall accuracy",
        ),
        ExtractedInfo(
            paper_id="paper-02",
            title="Masked Autoencoding for Hyperspectral Pretraining",
            methodology="Masked spectral-spatial reconstruction",
            model_architecture="Masked ViT",
            dataset="Houston 2018",
            limitations="Struggles with domain adaptation across unseen sensor bands",
            main_findings="Achieved 94.2% top-1 accuracy with 10% labeled fine-tuning data",
        ),
    ]


def test_research_gap_agent_fallback(sample_topic, sample_papers):
    mock_llm = MagicMock()
    mock_llm.invoke_structured.side_effect = RuntimeError("Ollama offline")

    agent = ResearchGapAgent(llm_client=mock_llm)
    gaps = agent.identify_gaps(sample_topic, sample_papers)

    assert len(gaps) >= 1
    assert gaps[0].gap_id.startswith("GAP-")
    assert gaps[0].supporting_papers
    assert gaps[0].verification_status in ["Supported by Literature", "Unverified Hypothesis (Literature Needed)"]


def test_research_gap_agent_structured(sample_topic, sample_papers, sample_extracted):
    mock_llm = MagicMock()
    mock_llm.invoke_structured.return_value = ResearchGapList(
        gaps=[
            ResearchGap(
                gap_id="GAP-01",
                description="Cross-sensor spectral mismatch in self-supervised foundation models.",
                supporting_papers=["Masked Autoencoding for Hyperspectral Pretraining"],
                evidence="Reported domain adaptation drop when fine-tuning on unseen sensors.",
                existing_approaches="Full retraining per sensor",
                limitations_of_approaches="Requires large compute per new sensor geometry",
                why_gap_matters="Limits real-world deployment",
                confidence_level="High",
                gap_type="Generalization",
                verification_status="Supported by Literature",
            )
        ]
    )

    agent = ResearchGapAgent(llm_client=mock_llm)
    gaps = agent.identify_gaps(sample_topic, sample_papers, sample_extracted)

    assert len(gaps) == 1
    assert gaps[0].gap_id == "GAP-01"
    assert "Masked Autoencoding for Hyperspectral Pretraining" in gaps[0].supporting_papers


def test_idea_generator_fallback(sample_topic, sample_papers):
    mock_llm = MagicMock()
    mock_llm.invoke_structured.side_effect = RuntimeError("Ollama offline")

    agent = IdeaGeneratorAgent(llm_client=mock_llm)
    gaps = [
        ResearchGap(
            gap_id="GAP-01",
            description="Lack of sensor-agnostic tokenization",
            supporting_papers=["SpectralFormer"],
            evidence="Hardcoded band count in embedding layer",
            existing_approaches="Fixed linear projection",
            limitations_of_approaches="Fails with different band counts",
            why_gap_matters="Cannot reuse model across AVIRIS and EnMAP",
        )
    ]
    ideas = agent.generate_ideas(sample_topic, gaps, sample_papers)

    assert len(ideas) >= 1
    assert ideas[0].idea_id.startswith("IDEA-")
    assert 0.0 <= ideas[0].feasibility_score <= 10.0
    assert ideas[0].research_gap_addressed == "GAP-01"
    assert ideas[0].closest_existing_studies


def test_synthesis_agent_deterministic_and_llm(sample_topic, sample_papers, sample_extracted):
    mock_llm = MagicMock()
    mock_llm.invoke_structured.return_value = EvidenceSynthesis(
        publication_year_distribution={"2022": 1, "2023": 1},
        frequent_architectures=[{"name": "SpectralFormer", "count": 1}],
        frequent_datasets=[{"name": "Indian Pines", "count": 1}],
        dominant_methodologies=["Transformer attention"],
        common_limitations=["Quadratic complexity"],
        emerging_directions=["Foundation models"],
        thematic_summary="Literature indicates rapid transition to Transformer models.",
    )

    agent = SynthesisAgent(llm_client=mock_llm)
    synthesis = agent.synthesize(sample_topic, sample_papers, sample_extracted)

    assert synthesis.publication_year_distribution == {"2022": 1, "2023": 1}
    assert len(synthesis.dominant_methodologies) >= 1
    assert "SpectralFormer" in synthesis.frequent_architectures[0]["name"]
