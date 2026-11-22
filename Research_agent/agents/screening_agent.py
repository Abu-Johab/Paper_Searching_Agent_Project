"""
hsi_research_agent/agents/screening_agent.py
Title and abstract screening agent with documented inclusion and exclusion decisions.
"""

import logging
import re
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from hsi_research_agent.config import settings
from hsi_research_agent.llm.client import OllamaClient
from hsi_research_agent.processing.relevance_scorer import score_paper_relevance
from hsi_research_agent.storage.models import PaperRecord, ResearchTopic, ScreeningDecision

logger = logging.getLogger("hsi_research_agent.screening_agent")


class ScreeningOutput(BaseModel):
    """Structured LLM output for paper screening."""
    decision: str = Field(..., description="One of: 'Include', 'Exclude', 'Maybe'")
    relevance_score: float = Field(..., ge=0.0, le=10.0)
    inclusion_reason: Optional[str] = None
    exclusion_reason: Optional[str] = None


def screen_paper(
    paper: PaperRecord,
    topic: ResearchTopic,
    client: Optional[OllamaClient] = None,
) -> ScreeningDecision:
    """
    Screen a single paper against research topic and scope criteria.
    Explicitly documents inclusion or exclusion reason.
    """
    client = client or OllamaClient()

    prompt = (
        f"You are an academic literature screening specialist.\n"
        f"Evaluate whether to Include, Exclude, or mark as Maybe the following paper for research on:\n"
        f"Topic: \"{topic.topic_text}\"\n"
        f"Target Domain: {topic.target_domain}\n\n"
        f"Paper Details:\n"
        f"- Title: {paper.title}\n"
        f"- Year: {paper.publication_year or 'Unknown'}\n"
        f"- Journal/Venue: {paper.journal_or_conference or 'Unknown'}\n"
        f"- Abstract: {paper.abstract or '[No abstract available]'}\n\n"
        f"SCREENING RULES:\n"
        f"1. Classify as 'Include' if the paper discusses models, methods, benchmarks, or architectures directly relevant to the topic.\n"
        f"2. Classify as 'Exclude' if the paper is unrelated to the domain or focus (e.g. completely unrelated discipline, purely non-spectral biology without AI, etc.).\n"
        f"3. Classify as 'Maybe' if relevance is plausible from the title but abstract is missing or ambiguous.\n"
        f"4. If 'Include': provide a concise inclusion_reason explaining what the paper contributes.\n"
        f"5. If 'Exclude': provide a concise exclusion_reason explicitly stating why it was excluded.\n\n"
        f"Return strict JSON matching the schema."
    )

    try:
        output: ScreeningOutput = client.generate_structured(
            prompt=prompt,
            pydantic_class=ScreeningOutput,
            system_prompt="You are an academic screening specialist. Return valid JSON only.",
            temperature=0.0,
            max_retries=2,
        )

        decision_clean = output.decision.capitalize()
        if decision_clean not in ("Include", "Exclude", "Maybe"):
            decision_clean = "Include" if output.relevance_score >= 6.0 else "Exclude"

        paper.relevance_score = output.relevance_score
        paper.relevance_reason = output.inclusion_reason or output.exclusion_reason or ""

        return ScreeningDecision(
            paper_id=paper.paper_id,
            title=paper.title,
            decision=decision_clean,
            relevance_score=output.relevance_score,
            inclusion_reason=output.inclusion_reason,
            exclusion_reason=output.exclusion_reason,
            screening_stage="Title and Abstract Screening",
            reviewer=f"Automated Agent ({settings.OLLAMA_MODEL})",
        )

    except Exception as e:
        logger.warning("Screening LLM fallback for '%s': %s", paper.title[:30], str(e))
        score, reason = score_paper_relevance(paper, topic.topic_text, client=client)
        paper.relevance_score = score
        paper.relevance_reason = reason

        decision = "Include" if score >= 6.0 else ("Maybe" if score >= 4.5 else "Exclude")
        inc_reason = reason if decision in ("Include", "Maybe") else None
        exc_reason = reason if decision == "Exclude" else None

        return ScreeningDecision(
            paper_id=paper.paper_id,
            title=paper.title,
            decision=decision,
            relevance_score=score,
            inclusion_reason=inc_reason,
            exclusion_reason=exc_reason,
            screening_stage="Title and Abstract Screening",
            reviewer="Deterministic Heuristic Fallback",
        )


def screen_all_papers(
    papers: List[PaperRecord],
    topic: ResearchTopic,
    client: Optional[OllamaClient] = None,
    max_to_screen: Optional[int] = 30,
) -> Tuple[List[PaperRecord], List[ScreeningDecision]]:
    """
    Screen candidate papers against research scope.
    If papers count exceeds max_to_screen (default 30), prioritizes the top candidates
    via fast keyword relevance ranking for in-depth LLM screening with documented reasons,
    and documents heuristic exclusion for the remaining lower-ranked papers.
    """
    screening_decisions: List[ScreeningDecision] = []
    eligible_papers: List[PaperRecord] = []

    if not papers:
        return eligible_papers, screening_decisions

    limit = max_to_screen if (max_to_screen and max_to_screen > 0) else len(papers)

    if len(papers) <= limit:
        top_candidates = papers
        remaining_papers = []
    else:
        topic_words = [w.lower() for w in re.findall(r"\w+", topic.topic_text) if len(w) > 3]

        def rank_key(p: PaperRecord):
            text = f"{p.title} {p.abstract or ''}".lower()
            overlap = sum(1 for w in topic_words if w in text)
            has_abstract = 1 if p.abstract else 0
            return (p.relevance_score, overlap, has_abstract, p.citation_count)

        sorted_papers = sorted(papers, key=rank_key, reverse=True)
        top_candidates = sorted_papers[:limit]
        remaining_papers = sorted_papers[limit:]

    logger.info(
        "Screening %d papers with LLM (and %d via heuristic filter)...",
        len(top_candidates),
        len(remaining_papers),
    )

    # 1. In-depth LLM screening for top candidates
    for idx, paper in enumerate(top_candidates, 1):
        decision = screen_paper(paper, topic, client=client)
        screening_decisions.append(decision)

        if decision.decision in ("Include", "Maybe"):
            eligible_papers.append(paper)

    # 2. Documented exclusion for remaining lower-ranked papers
    for paper in remaining_papers:
        screening_decisions.append(
            ScreeningDecision(
                paper_id=paper.paper_id,
                title=paper.title,
                decision="Exclude",
                relevance_score=paper.relevance_score,
                exclusion_reason="Preliminary relevance score ranked outside top screening window",
                screening_stage="Title and Abstract Screening",
                reviewer="Deterministic Heuristic Filter",
            )
        )

    # Sort eligible papers by score descending
    eligible_papers.sort(key=lambda p: (p.relevance_score, p.citation_count), reverse=True)

    logger.info(
        "Screening complete: %d screened, %d eligible (Include/Maybe), %d excluded.",
        len(papers),
        len(eligible_papers),
        len(screening_decisions) - len(eligible_papers),
    )
    return eligible_papers, screening_decisions
