"""
Markdown Report Generator for Autonomous Literature Discovery and Analysis.
Produces an in-depth academic synthesis report summarizing questions, search queries,
literature landscape, evidence-supported research gaps, and novel research ideas.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import Counter

from hsi_research_agent.config import settings
from hsi_research_agent.storage.models import (
    ResearchTopic,
    ResearchQuestion,
    SearchQuery,
    PaperRecord,
    ScreeningDecision,
    ExtractedInfo,
    QualityAssessment,
    ResearchGap,
    NovelIdea,
    EvidenceSynthesis,
)


class ReportGenerator:
    """Generates structured Markdown reports from literature discovery results."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.EXPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        topic: ResearchTopic,
        papers: List[PaperRecord],
        queries: List[SearchQuery],
        questions: Optional[List[ResearchQuestion]] = None,
        screening_decisions: Optional[List[ScreeningDecision]] = None,
        extracted_info_list: Optional[List[ExtractedInfo]] = None,
        quality_assessments: Optional[List[QualityAssessment]] = None,
        research_gaps: Optional[List[ResearchGap]] = None,
        novel_ideas: Optional[List[NovelIdea]] = None,
        synthesis: Optional[EvidenceSynthesis] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """Construct comprehensive Markdown research report."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        if not filename:
            safe_topic = "".join(c if c.isalnum() else "_" for c in topic.topic_text[:30])
            filename = f"research_report_{safe_topic}_{date_str}.md"

        lines: List[str] = []

        # 1. Header
        lines.append(f"# Academic Literature Discovery & Research Gap Analysis Report")
        lines.append(f"**Research Topic:** {topic.topic_text}  ")
        lines.append(f"**Target Domain:** {topic.target_domain}  ")
        lines.append(f"**Generated On:** {datetime.now().strftime('%B %d, %Y')}  ")
        lines.append(f"**Autonomous Agent Orchestrator:** LangGraph + Ollama (`{settings.OLLAMA_MODEL}`)  ")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 2. Executive Summary
        lines.append("## 1. Executive Summary")
        lines.append("")
        included_count = sum(1 for s in (screening_decisions or []) if s.decision == "Include")
        avg_rel = sum(p.relevance_score for p in papers) / len(papers) if papers else 0.0
        avg_qa = sum(q.total_quality_score for q in (quality_assessments or [])) / len(quality_assessments) if quality_assessments else 0.0

        lines.append(
            f"This autonomous literature discovery investigated **{topic.topic_text}** across "
            f"multiple major academic databases (OpenAlex, Semantic Scholar, Crossref, arXiv). "
            f"A total of **{len(queries)}** multi-dimensional search queries were formulated and validated. "
            f"The system discovered **{len(papers)} unique academic papers** following automated multi-tiered deduplication. "
            f"Title and abstract screening identified **{included_count} highly relevant studies** with an average relevance "
            f"score of **{avg_rel:.2f}/10.0** and average methodological quality score of **{avg_qa:.1f}/90.0**."
        )
        lines.append("")
        if synthesis and synthesis.thematic_summary:
            lines.append(f"> **Thematic Finding:** {synthesis.thematic_summary}")
            lines.append("")

        # 3. Formulated Research Questions
        if questions:
            lines.append("## 2. Formulated Research Questions")
            lines.append("")
            for q in questions:
                lines.append(f"- **[{q.question_id}] {q.question_text}**")
                lines.append(f"  - *Aspect:* {q.research_aspect} | *Framework:* {q.framework}")
                if q.rationale:
                    lines.append(f"  - *Rationale:* {q.rationale}")
            lines.append("")

        # 4. Search Strategy & 10 Queries
        lines.append("## 3. Search Strategy & Formulated Queries")
        lines.append("")
        lines.append("| ID | Research Dimension | Search Query | Validation Score | Status |")
        lines.append("|---|---|---|---|---|")
        for q in queries:
            lines.append(f"| `{q.query_id}` | {q.research_dimension} | `{q.query_text}` | {q.validation_score:.1f}/100 | {q.validation_status} |")
        lines.append("")

        # 5. Database Distribution & Literature Landscape
        lines.append("## 4. Literature Landscape & Synthesis")
        lines.append("")
        db_counts = Counter(p.source_database for p in papers)
        lines.append("### Database Representation")
        total_p = len(papers)
        for db, count in db_counts.items():
            pct = (count / total_p * 100) if total_p > 0 else 0.0
            lines.append(f"- **{db}:** {count} papers ({pct:.1f}%)")
        lines.append("")

        if synthesis:
            if synthesis.publication_year_distribution:
                lines.append("### Publication Year Trajectory")
                for yr, cnt in sorted(synthesis.publication_year_distribution.items()):
                    lines.append(f"- **{yr}:** {cnt} papers")
                lines.append("")

            if synthesis.dominant_methodologies:
                lines.append("### Dominant Methodological Paradigms")
                for m in synthesis.dominant_methodologies:
                    lines.append(f"- {m}")
                lines.append("")

            if synthesis.common_limitations:
                lines.append("### Common Technical Bottlenecks in Current Literature")
                for lim in synthesis.common_limitations:
                    lines.append(f"- {lim}")
                lines.append("")

        # 6. Top Ranked Research Literature
        lines.append("## 5. Top Ranked Academic Papers")
        lines.append("")
        top_papers = sorted(papers, key=lambda p: (p.relevance_score, p.citation_count), reverse=True)[:10]
        lines.append("| # | Title | Year | Citations | Relevance | DOI / Link |")
        lines.append("|---|---|---|---|---|---|")
        for idx, p in enumerate(top_papers, 1):
            doi_link = f"[{p.doi}](https://doi.org/{p.doi})" if p.doi else (f"[Link]({p.paper_url})" if p.paper_url else "N/A")
            lines.append(f"| {idx} | **{p.title}** | {p.publication_year or 'n.d.'} | {p.citation_count} | {p.relevance_score:.2f} | {doi_link} |")
        lines.append("")

        # 7. Evidence-Based Research Gaps
        if research_gaps:
            lines.append("## 6. Evidence-Supported Research Gaps")
            lines.append("")
            for g in research_gaps:
                lines.append(f"### `{g.gap_id}`: {g.description}")
                lines.append(f"- **Confidence Level:** `{g.confidence_level}` | **Gap Type:** `{g.gap_type}` | **Verification:** `{g.verification_status}`")
                lines.append(f"- **Supporting Papers:** {', '.join(g.supporting_papers) if g.supporting_papers else 'Literature Needed'}")
                lines.append(f"- **Direct Evidence:** {g.evidence}")
                lines.append(f"- **Existing Approaches:** {g.existing_approaches}")
                lines.append(f"- **Technical Limitations:** {g.limitations_of_approaches}")
                lines.append(f"- **Why Gap Matters:** {g.why_gap_matters}")
                lines.append("")

        # 8. Novel Research Ideas
        if novel_ideas:
            lines.append("## 7. Proposed Novel Research Ideas")
            lines.append("")
            for idea in novel_ideas:
                lines.append(f"### `{idea.idea_id}`: {idea.proposed_title}")
                lines.append(f"- **Feasibility Score:** `{idea.feasibility_score:.1f}/10.0` ({idea.technical_feasibility} Feasibility)")
                lines.append(f"- **Addressed Gap:** {idea.research_gap_addressed}")
                lines.append(f"- **Core Problem:** {idea.core_research_problem}")
                lines.append(f"- **Proposed Methodology:** {idea.proposed_methodology}")
                lines.append(f"- **Proposed Architecture:** `{idea.possible_architecture}`")
                lines.append(f"- **Target Datasets:** {', '.join(idea.potential_datasets)}")
                lines.append(f"- **Baseline Models:** {', '.join(idea.baseline_models)}")
                lines.append(f"- **Evaluation Metrics:** {', '.join(idea.evaluation_metrics)}")
                if idea.expected_contributions:
                    lines.append("- **Expected Contributions:**")
                    for c in idea.expected_contributions:
                        lines.append(f"  - {c}")
                if idea.closest_existing_studies:
                    lines.append(f"- **Closest Prior Art:** {', '.join(idea.closest_existing_studies)}")
                if idea.main_risks:
                    lines.append(f"- **Key Technical Risks:** {idea.main_risks}")
                lines.append("")

        # 9. Conclusion & Next Steps
        lines.append("## 8. Conclusion & Actionable Next Steps")
        lines.append("")
        lines.append("1. **Experimentation:** Implement the proposed architecture outlined in `IDEA-01`, utilizing open benchmark datasets.")
        lines.append("2. **Comparative Benchmarking:** Evaluate against the identified baseline models across identical cross-sensor splits.")
        lines.append("3. **Ablation Studies:** Quantify the individual contribution of the continuous spectral tokenization mechanism.")
        lines.append("4. **Full Artifacts:** Review the accompanying `HSI_Literature_Discovery.xlsx` workbook and CSV files for complete metadata.")
        lines.append("")

        output_path = self.output_dir / filename
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path
