"""
Excel Exporter for Literature Discovery and Research Gap Analysis.
Generates a multi-sheet, beautifully styled master Excel workbook (HSI_Literature_Discovery.xlsx)
using openpyxl with frozen headers, autofilters, clickable hyperlinks, and auto-adjusted column widths.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import Counter
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from hsi_research_agent.config import settings
from hsi_research_agent.storage.models import (
    ResearchTopic,
    ResearchQuestion,
    SearchQuery,
    SearchLog,
    PaperRecord,
    ScreeningDecision,
    ExtractedInfo,
    QualityAssessment,
    ResearchGap,
    NovelIdea,
    EvidenceSynthesis,
)


class ExcelExporter:
    """Creates formatted master Excel reports for academic literature discovery."""

    HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    DATA_FONT = Font(name="Calibri", size=10)
    LINK_FONT = Font(name="Calibri", size=10, color="0563C1", underline="single")
    BORDER_THIN = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.EXPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_workbook(
        self,
        topic: ResearchTopic,
        papers: List[PaperRecord],
        queries: List[SearchQuery],
        screening_decisions: Optional[List[ScreeningDecision]] = None,
        extracted_info_list: Optional[List[ExtractedInfo]] = None,
        quality_assessments: Optional[List[QualityAssessment]] = None,
        research_gaps: Optional[List[ResearchGap]] = None,
        novel_ideas: Optional[List[NovelIdea]] = None,
        search_logs: Optional[List[SearchLog]] = None,
        synthesis: Optional[EvidenceSynthesis] = None,
        filename: str = "HSI_Literature_Discovery.xlsx",
    ) -> Path:
        """Create and populate the complete 8-sheet Excel workbook."""
        wb = openpyxl.Workbook()
        # Remove default active sheet
        wb.remove(wb.active)

        # 1. Research_Papers
        self._add_papers_sheet(wb, papers)

        # 2. Research_Queries
        self._add_queries_sheet(wb, queries)

        # 3. Search_Logs
        self._add_logs_sheet(wb, search_logs or [])

        # 4. Screening_Results
        self._add_screening_sheet(wb, screening_decisions or [])

        # 5. Research_Gaps
        self._add_gaps_sheet(wb, research_gaps or [])

        # 6. Novel_Ideas
        self._add_ideas_sheet(wb, novel_ideas or [])

        # 7. Quality_Assessment
        self._add_quality_sheet(wb, quality_assessments or [])

        # 8. Statistics
        self._add_statistics_sheet(wb, topic, papers, queries, screening_decisions, quality_assessments, synthesis)

        output_path = self.output_dir / filename
        wb.save(output_path)
        return output_path

    def _style_header(self, ws, num_cols: int) -> None:
        """Apply header styling, borders, and freeze pane."""
        ws.freeze_panes = "A2"
        for col_idx in range(1, num_cols + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = self.HEADER_FILL
            cell.font = self.HEADER_FONT
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    def _auto_fit_columns(self, ws, min_width: int = 12, max_width: int = 50) -> None:
        """Adjust column widths based on maximum contents."""
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val = str(cell.value or "")
                if "\n" in val:
                    val = max(val.split("\n"), key=len)
                max_len = max(max_len, len(val))
            ws.column_dimensions[col_letter].width = max(min_width, min(max_len + 3, max_width))

    def _safe_set_hyperlink(self, cell, url: Optional[str]) -> None:
        """Safely attach hyperlink to cell without throwing on invalid URLs."""
        if not url:
            return
        clean_url = str(url).strip()
        if clean_url.startswith(("http://", "https://")):
            try:
                cell.hyperlink = clean_url
                cell.font = self.LINK_FONT
            except Exception:
                pass

    def _add_papers_sheet(self, wb: openpyxl.Workbook, papers: List[PaperRecord]) -> None:
        ws = wb.create_sheet(title="Research_Papers")
        headers = [
            "Paper ID", "Title", "Authors", "Year", "Journal / Conference",
            "DOI", "arXiv ID", "Citations", "Relevance Score", "Open Access",
            "Full Text Status", "Source Database", "Queries Retrieved", "Paper URL", "PDF URL"
        ]
        ws.append(headers)

        for p in papers:
            authors_str = ", ".join(p.authors[:5]) + (" et al." if len(p.authors) > 5 else "")
            queries_str = ", ".join(p.retrieved_query_ids)
            row = [
                p.paper_id,
                p.title,
                authors_str,
                p.publication_year or "",
                p.journal_or_conference or "",
                p.doi or "",
                p.arxiv_id or "",
                p.citation_count,
                round(p.relevance_score, 2),
                p.open_access_status,
                p.full_text_status,
                p.source_database,
                queries_str,
                p.paper_url or "",
                p.pdf_url or "",
            ]
            ws.append(row)

            # Hyperlinks for DOI and URLs
            curr_row = ws.max_row
            if p.doi:
                self._safe_set_hyperlink(ws.cell(row=curr_row, column=6), f"https://doi.org/{p.doi}")
            if p.paper_url:
                self._safe_set_hyperlink(ws.cell(row=curr_row, column=14), p.paper_url)
            if p.pdf_url:
                self._safe_set_hyperlink(ws.cell(row=curr_row, column=15), p.pdf_url)

        self._style_header(ws, len(headers))
        if ws.max_row > 1:
            ws.auto_filter.ref = ws.dimensions
        self._auto_fit_columns(ws, min_width=12, max_width=45)

    def _add_queries_sheet(self, wb: openpyxl.Workbook, queries: List[SearchQuery]) -> None:
        ws = wb.create_sheet(title="Research_Queries")
        headers = [
            "Query ID", "Search Query Text", "Research Dimension", "Search Objective",
            "Keywords", "Priority", "Validation Score", "Status", "Validation Feedback"
        ]
        ws.append(headers)

        for q in queries:
            ws.append([
                q.query_id,
                q.query_text,
                q.research_dimension,
                q.search_objective,
                ", ".join(q.keywords),
                q.priority,
                round(q.validation_score, 1),
                q.validation_status,
                q.validation_reason,
            ])

        self._style_header(ws, len(headers))
        if ws.max_row > 1:
            ws.auto_filter.ref = ws.dimensions
        self._auto_fit_columns(ws, min_width=12, max_width=50)

    def _add_logs_sheet(self, wb: openpyxl.Workbook, logs: List[SearchLog]) -> None:
        ws = wb.create_sheet(title="Search_Logs")
        headers = [
            "Log ID", "Query ID", "Query Text", "Source Database",
            "Timestamp", "Records Retrieved", "Status", "Error Details"
        ]
        ws.append(headers)

        for idx, log in enumerate(logs, 1):
            ws.append([
                log.search_id or idx,
                log.query_id,
                log.query_text,
                log.source_database,
                log.timestamp,
                log.records_retrieved,
                log.status,
                log.error_details or "",
            ])

        self._style_header(ws, len(headers))
        if ws.max_row > 1:
            ws.auto_filter.ref = ws.dimensions
        self._auto_fit_columns(ws, min_width=12, max_width=40)

    def _add_screening_sheet(self, wb: openpyxl.Workbook, decisions: List[ScreeningDecision]) -> None:
        ws = wb.create_sheet(title="Screening_Results")
        headers = [
            "Paper ID", "Title", "Decision", "Relevance Score",
            "Inclusion Reason", "Exclusion Reason", "Stage", "Reviewer"
        ]
        ws.append(headers)

        for d in decisions:
            ws.append([
                d.paper_id,
                d.title,
                d.decision,
                round(d.relevance_score, 2),
                d.inclusion_reason or "",
                d.exclusion_reason or "",
                d.screening_stage,
                d.reviewer,
            ])

        self._style_header(ws, len(headers))
        if ws.max_row > 1:
            ws.auto_filter.ref = ws.dimensions
        self._auto_fit_columns(ws, min_width=12, max_width=45)

    def _add_gaps_sheet(self, wb: openpyxl.Workbook, gaps: List[ResearchGap]) -> None:
        ws = wb.create_sheet(title="Research_Gaps")
        headers = [
            "Gap ID", "Gap Description", "Confidence Level", "Gap Type",
            "Verification Status", "Supporting Papers", "Evidence",
            "Existing Approaches", "Limitations of Approaches", "Why Gap Matters"
        ]
        ws.append(headers)

        for g in gaps:
            ws.append([
                g.gap_id,
                g.description,
                g.confidence_level,
                g.gap_type,
                g.verification_status,
                "\n".join(g.supporting_papers),
                g.evidence,
                g.existing_approaches,
                g.limitations_of_approaches,
                g.why_gap_matters,
            ])

        self._style_header(ws, len(headers))
        if ws.max_row > 1:
            ws.auto_filter.ref = ws.dimensions
        self._auto_fit_columns(ws, min_width=14, max_width=50)

    def _add_ideas_sheet(self, wb: openpyxl.Workbook, ideas: List[NovelIdea]) -> None:
        ws = wb.create_sheet(title="Novel_Ideas")
        headers = [
            "Idea ID", "Proposed Title", "Feasibility Score", "Technical Feasibility",
            "Addressed Gap", "Core Research Problem", "Proposed Methodology",
            "Architecture", "Potential Datasets", "Baselines", "Metrics",
            "Expected Contributions", "Closest Existing Studies", "Novelty Status", "Risks"
        ]
        ws.append(headers)

        for idea in ideas:
            ws.append([
                idea.idea_id,
                idea.proposed_title,
                round(idea.feasibility_score, 1),
                idea.technical_feasibility,
                idea.research_gap_addressed,
                idea.core_research_problem,
                idea.proposed_methodology,
                idea.possible_architecture,
                ", ".join(idea.potential_datasets),
                ", ".join(idea.baseline_models),
                ", ".join(idea.evaluation_metrics),
                "\n".join(f"• {c}" for c in idea.expected_contributions),
                ", ".join(idea.closest_existing_studies),
                idea.novelty_verification_status,
                idea.main_risks,
            ])

        self._style_header(ws, len(headers))
        if ws.max_row > 1:
            ws.auto_filter.ref = ws.dimensions
        self._auto_fit_columns(ws, min_width=14, max_width=50)

    def _add_quality_sheet(self, wb: openpyxl.Workbook, assessments: List[QualityAssessment]) -> None:
        ws = wb.create_sheet(title="Quality_Assessment")
        headers = [
            "Paper ID", "Objectives Clarity", "Dataset Transparency", "Method Clarity",
            "Reproducibility", "Baselines", "Appropriate Metrics", "Statistical Validation",
            "Limitations Discussed", "Code/Data Open", "Total Score (0-90)", "Justification"
        ]
        ws.append(headers)

        for qa in assessments:
            ws.append([
                qa.paper_id,
                qa.clarity_of_objectives,
                qa.dataset_transparency,
                qa.methodological_clarity,
                qa.experimental_reproducibility,
                qa.baseline_comparisons,
                qa.metrics_appropriateness,
                qa.statistical_validation,
                qa.limitations_discussion,
                qa.code_data_availability,
                round(qa.total_quality_score, 1),
                qa.justification,
            ])

        self._style_header(ws, len(headers))
        if ws.max_row > 1:
            ws.auto_filter.ref = ws.dimensions
        self._auto_fit_columns(ws, min_width=12, max_width=45)

    def _add_statistics_sheet(
        self,
        wb: openpyxl.Workbook,
        topic: ResearchTopic,
        papers: List[PaperRecord],
        queries: List[SearchQuery],
        screening: Optional[List[ScreeningDecision]],
        qa_list: Optional[List[QualityAssessment]],
        synthesis: Optional[EvidenceSynthesis],
    ) -> None:
        ws = wb.create_sheet(title="Statistics")
        headers = ["Metric / Parameter", "Value", "Description / Details"]
        ws.append(headers)

        included_count = sum(1 for s in (screening or []) if s.decision == "Include")
        excluded_count = sum(1 for s in (screening or []) if s.decision == "Exclude")
        maybe_count = sum(1 for s in (screening or []) if s.decision == "Maybe")

        avg_rel = sum(p.relevance_score for p in papers) / len(papers) if papers else 0.0
        avg_qa = sum(q.total_quality_score for q in (qa_list or [])) / len(qa_list) if qa_list else 0.0

        db_counts = Counter(p.source_database for p in papers)
        db_summary = ", ".join(f"{db}: {cnt}" for db, cnt in db_counts.items())

        stats_rows = [
            ["Research Topic", topic.topic_text, "Input research domain topic"],
            ["Research Domain", topic.target_domain, "Disciplinary focus area"],
            ["Search Queries Generated", len(queries), "Number of multi-dimensional search queries (target: 10)"],
            ["Total Unique Papers Retrieved", len(papers), "After cross-source deduplication"],
            ["Screening: Included", included_count, "Passed title/abstract screening threshold"],
            ["Screening: Excluded", excluded_count, "Excluded during relevance assessment"],
            ["Screening: Under Review (Maybe)", maybe_count, "Flagged for manual inspection"],
            ["Average Relevance Score (0-10)", f"{avg_rel:.2f}", "Mean semantic relevance across literature"],
            ["Average Quality Score (0-90)", f"{avg_qa:.1f}", "Methodological rigor and transparency"],
            ["Database Representation", db_summary or "N/A", "Papers distributed across academic sources"],
        ]

        if synthesis and synthesis.thematic_summary:
            stats_rows.append(["Executive Thematic Summary", synthesis.thematic_summary, "Key evidence synthesis narrative"])

        for r in stats_rows:
            ws.append(r)

        self._style_header(ws, len(headers))
        self._auto_fit_columns(ws, min_width=18, max_width=60)
