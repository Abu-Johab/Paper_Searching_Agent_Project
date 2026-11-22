"""
hsi_research_agent/processing/relevance_scorer.py
Relevance scoring engine powered by Ollama Qwen 2.5 14B with title-only capping on missing abstracts.
"""

import logging
from typing import Optional, Tuple
from pydantic import BaseModel, Field

from hsi_research_agent.llm.client import OllamaClient
from hsi_research_agent.storage.models import PaperRecord

logger = logging.getLogger("hsi_research_agent.processing.relevance_scorer")


class RelevanceOutput(BaseModel):
    """Structured LLM output for paper relevance evaluation."""
    score: float = Field(..., ge=0.0, le=10.0)
    reason: str = Field(..., min_length=5)


def score_paper_relevance(
    paper: PaperRecord,
    topic_text: str,
    client: Optional[OllamaClient] = None,
) -> Tuple[float, str]:
    """
    Score relevance of a research paper (0.0 to 10.0) to the research topic.
    If abstract is missing: score based on title alone and strictly cap score at 7.0.
    """
    client = client or OllamaClient()
    abstract_missing = not bool(paper.abstract and str(paper.abstract).strip())

    if abstract_missing:
        prompt = (
            f"You are an expert academic literature reviewer.\n"
            f"The abstract for this paper is MISSING.\n"
            f"Score the relevance of this paper to the research topic based on its title alone on a scale from 0.0 to 7.0.\n"
            f"(Note: Score MUST NOT exceed 7.0 because the abstract is missing).\n\n"
            f"Research Topic: {topic_text}\n"
            f"Paper Title: {paper.title}\n"
            f"Abstract: [MISSING]\n\n"
            f"Return strict JSON matching the schema."
        )
    else:
        prompt = (
            f"You are an expert academic literature reviewer.\n"
            f"Score the relevance of this paper to the research topic on a scale from 0.0 to 10.0 "
            f"(where 0 is completely irrelevant and 10 is central and foundational).\n\n"
            f"Research Topic: {topic_text}\n"
            f"Paper Title: {paper.title}\n"
            f"Abstract: {paper.abstract[:1500]}\n\n"
            f"Return strict JSON matching the schema."
        )

    try:
        output: RelevanceOutput = client.generate_structured(
            prompt=prompt,
            pydantic_class=RelevanceOutput,
            system_prompt="You are a scientific evaluator. Return valid JSON only.",
            temperature=0.0,
            max_retries=2,
        )
        score = output.score
        if abstract_missing:
            score = min(max(0.0, score), 7.0)
        else:
            score = min(max(0.0, score), 10.0)
        return round(score, 2), output.reason.strip().replace("\n", " ")

    except Exception as e:
        logger.warning("Relevance scoring fallback for '%s': %s", paper.title[:30], str(e))
        # Lexical relevance fallback
        score = 5.0
        topic_words = [w.lower() for w in topic_text.split() if len(w) > 3]
        title_lower = paper.title.lower()
        matches = sum(1 for w in topic_words if w in title_lower)
        score += min(matches * 1.5, 4.0)
        if abstract_missing:
            score = min(score, 7.0)
        return round(score, 2), "Heuristic scoring based on keyword overlap."
