"""
CSV Exporter for Literature Discovery Data.
Exports papers, queries, research gaps, and novel ideas to standardized CSV files.
"""

from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd

from hsi_research_agent.config import settings
from hsi_research_agent.storage.models import (
    PaperRecord,
    SearchQuery,
    ResearchGap,
    NovelIdea,
)


class CSVExporter:
    """Exports structured data to CSV files."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.EXPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_papers(self, papers: List[PaperRecord], filename: str = "papers.csv") -> Path:
        """Export papers to CSV."""
        data = []
        for p in papers:
            data.append({
                "paper_id": p.paper_id,
                "title": p.title,
                "authors": "; ".join(p.authors),
                "year": p.publication_year or "",
                "doi": p.doi or "",
                "arxiv_id": p.arxiv_id or "",
                "journal_or_conference": p.journal_or_conference or "",
                "citation_count": p.citation_count,
                "relevance_score": round(p.relevance_score, 2),
                "open_access_status": p.open_access_status,
                "full_text_status": p.full_text_status,
                "source_database": p.source_database,
                "retrieved_queries": "; ".join(p.retrieved_query_ids),
                "paper_url": p.paper_url,
                "pdf_url": p.pdf_url or "",
            })
        df = pd.DataFrame(data)
        out_path = self.output_dir / filename
        df.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    def export_queries(self, queries: List[SearchQuery], filename: str = "queries.csv") -> Path:
        """Export search queries to CSV."""
        data = []
        for q in queries:
            data.append({
                "query_id": q.query_id,
                "query_text": q.query_text,
                "research_dimension": q.research_dimension,
                "search_objective": q.search_objective,
                "keywords": "; ".join(q.keywords),
                "priority": q.priority,
                "validation_score": round(q.validation_score, 1),
                "validation_status": q.validation_status,
                "validation_reason": q.validation_reason,
            })
        df = pd.DataFrame(data)
        out_path = self.output_dir / filename
        df.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    def export_gaps(self, gaps: List[ResearchGap], filename: str = "gaps.csv") -> Path:
        """Export research gaps to CSV."""
        data = []
        for g in gaps:
            data.append({
                "gap_id": g.gap_id,
                "description": g.description,
                "confidence_level": g.confidence_level,
                "gap_type": g.gap_type,
                "verification_status": g.verification_status,
                "supporting_papers": "; ".join(g.supporting_papers),
                "evidence": g.evidence,
                "existing_approaches": g.existing_approaches,
                "limitations_of_approaches": g.limitations_of_approaches,
                "why_gap_matters": g.why_gap_matters,
            })
        df = pd.DataFrame(data)
        out_path = self.output_dir / filename
        df.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    def export_ideas(self, ideas: List[NovelIdea], filename: str = "ideas.csv") -> Path:
        """Export novel research ideas to CSV."""
        data = []
        for idea in ideas:
            data.append({
                "idea_id": idea.idea_id,
                "proposed_title": idea.proposed_title,
                "feasibility_score": round(idea.feasibility_score, 1),
                "technical_feasibility": idea.technical_feasibility,
                "research_gap_addressed": idea.research_gap_addressed,
                "core_research_problem": idea.core_research_problem,
                "proposed_methodology": idea.proposed_methodology,
                "possible_architecture": idea.possible_architecture,
                "potential_datasets": "; ".join(idea.potential_datasets),
                "baseline_models": "; ".join(idea.baseline_models),
                "evaluation_metrics": "; ".join(idea.evaluation_metrics),
                "expected_contributions": " | ".join(idea.expected_contributions),
                "closest_existing_studies": "; ".join(idea.closest_existing_studies),
                "novelty_verification_status": idea.novelty_verification_status,
                "main_risks": idea.main_risks,
            })
        df = pd.DataFrame(data)
        out_path = self.output_dir / filename
        df.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    def export_all(
        self,
        papers: List[PaperRecord],
        queries: List[SearchQuery],
        gaps: Optional[List[ResearchGap]] = None,
        ideas: Optional[List[NovelIdea]] = None,
    ) -> Dict[str, Path]:
        """Export all entities to their respective CSV files."""
        results = {
            "papers": self.export_papers(papers),
            "queries": self.export_queries(queries),
        }
        if gaps:
            results["gaps"] = self.export_gaps(gaps)
        if ideas:
            results["ideas"] = self.export_ideas(ideas)
        return results
