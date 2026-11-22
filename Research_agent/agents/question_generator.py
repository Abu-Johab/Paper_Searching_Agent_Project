"""
hsi_research_agent/agents/question_generator.py
Generates 3 to 5 targeted research questions adapted to computer vision & machine learning.
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from hsi_research_agent.config import settings
from hsi_research_agent.llm.client import OllamaClient
from hsi_research_agent.storage.models import ResearchQuestion, ResearchTopic

logger = logging.getLogger("hsi_research_agent.question_generator")


class QuestionListOutput(BaseModel):
    """Structured LLM output for research questions."""
    questions: List[ResearchQuestion] = Field(..., min_length=3, max_length=5)


def generate_research_questions(
    topic: ResearchTopic,
    client: Optional[OllamaClient] = None,
) -> List[ResearchQuestion]:
    """
    Formulate 3 to 5 domain-adapted research questions based on the analyzed topic.
    """
    client = client or OllamaClient()
    logger.info("Generating research questions for topic: '%s'", topic.topic_text)

    prompt_path = settings.BASE_DIR / "prompts" / "question_generation.txt"
    if prompt_path.exists():
        template = prompt_path.read_text(encoding="utf-8")
    else:
        template = "Formulate 3 to 5 research questions for: {topic_text}."

    prompt = template.format(
        topic_text=topic.topic_text,
        target_domain=topic.target_domain,
        core_concepts=", ".join(topic.core_concepts[:6]),
        research_dimensions=", ".join(topic.research_dimensions[:6]),
    )

    try:
        output: QuestionListOutput = client.generate_structured(
            prompt=prompt,
            pydantic_class=QuestionListOutput,
            system_prompt=(
                "You are an academic advisor. Formulate 3 to 5 rigorous, answerable research questions "
                "in valid JSON matching the schema."
            ),
            temperature=0.2,
        )
        return output.questions
    except Exception as e:
        logger.warning("Structured question generation fallback triggered: %s", str(e))
        # Deterministic domain fallback
        return [
            ResearchQuestion(
                question_id="RQ1",
                question_text=f"What foundational architectures and tokenization strategies are currently employed in {topic.topic_text}?",
                framework="Domain-Specific",
                research_aspect="Architectural Design",
                rationale="Surveys backbone models and spatial-spectral embedding methods."
            ),
            ResearchQuestion(
                question_id="RQ2",
                question_text=f"How effectively do pretraining strategies transfer across disparate datasets and sensor modalities in {topic.topic_text}?",
                framework="Domain-Specific",
                research_aspect="Cross-Domain Generalization",
                rationale="Evaluates transferability across heterogeneous spectral band resolutions."
            ),
            ResearchQuestion(
                question_id="RQ3",
                question_text=f"What downstream tasks and benchmark datasets dominate the evaluation of {topic.topic_text}?",
                framework="Domain-Specific",
                research_aspect="Empirical Evaluation",
                rationale="Identifies common evaluation protocols and metric rigor."
            ),
        ]
