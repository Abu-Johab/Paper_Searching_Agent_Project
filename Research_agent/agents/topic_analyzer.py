"""
hsi_research_agent/agents/topic_analyzer.py
Analyzes research topic, detects domain, extracts technical concepts, datasets, and dimensions.
"""

import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from hsi_research_agent.config import settings
from hsi_research_agent.llm.client import OllamaClient
from hsi_research_agent.storage.models import ResearchTopic

logger = logging.getLogger("hsi_research_agent.topic_analyzer")


class TopicAnalysisOutput(BaseModel):
    """Structured LLM output for topic analysis."""
    target_domain: str = "Machine Learning / Computer Vision"
    core_concepts: list[str] = Field(default_factory=list)
    related_terminology: list[str] = Field(default_factory=list)
    research_dimensions: list[str] = Field(default_factory=list)
    relevant_datasets: list[str] = Field(default_factory=list)
    important_methodologies: list[str] = Field(default_factory=list)


def analyze_topic(
    topic_text: str,
    target_domain: str = "Machine Learning / Computer Vision",
    start_year: int = 2018,
    end_year: Optional[int] = None,
    user_objectives: Optional[str] = None,
    client: Optional[OllamaClient] = None,
) -> ResearchTopic:
    """
    Analyze the research topic and establish research scope using Ollama Qwen 2.5 14B.
    """
    client = client or OllamaClient()
    logger.info("Analyzing research topic: '%s'", topic_text)

    prompt_path = settings.BASE_DIR / "prompts" / "topic_analysis.txt"
    if prompt_path.exists():
        template = prompt_path.read_text(encoding="utf-8")
    else:
        template = "Analyze the research topic: {topic_text} in {target_domain}."

    prompt = template.format(
        topic_text=topic_text,
        target_domain=target_domain,
        start_year=start_year,
        end_year=end_year or "Present",
        user_objectives=user_objectives or "Comprehensive academic literature discovery",
    )

    analysis: TopicAnalysisOutput = client.generate_structured(
        prompt=prompt,
        pydantic_class=TopicAnalysisOutput,
        system_prompt=(
            "You are an expert scientific researcher. Analyze the research topic and extract "
            "technical concepts, methodologies, datasets, and research dimensions in valid JSON."
        ),
        temperature=0.1,
    )

    # Ensure reasonable defaults if model returned sparse fields
    core_concepts = analysis.core_concepts or [topic_text]
    dimensions = analysis.research_dimensions or [
        "Model Architectures",
        "Pretraining Strategies",
        "Cross-Domain Generalization",
        "Evaluation Benchmarks",
        "Downstream Tasks"
    ]

    return ResearchTopic(
        topic_text=topic_text,
        target_domain=analysis.target_domain or target_domain,
        start_year=start_year,
        end_year=end_year,
        core_concepts=core_concepts,
        related_terminology=analysis.related_terminology,
        research_dimensions=dimensions,
        relevant_datasets=analysis.relevant_datasets,
        important_methodologies=analysis.important_methodologies,
        user_objectives=user_objectives,
    )
