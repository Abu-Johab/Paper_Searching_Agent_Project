"""
hsi_research_agent/storage/repository.py
Repository pattern for clean database operations across topics, queries, papers, extractions, gaps, and ideas.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from hsi_research_agent.storage.database import get_db_connection
from hsi_research_agent.storage.models import (
    ExtractedInfo,
    NovelIdea,
    PaperRecord,
    QualityAssessment,
    ResearchGap,
    ResearchQuestion,
    ResearchTopic,
    ScreeningDecision,
    SearchLog,
    SearchQuery,
)

logger = logging.getLogger("hsi_research_agent.repository")


class ResearchRepository:
    """Data access repository for research literature entities."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path

    # =========================================================================
    # Research Topics & Questions
    # =========================================================================

    def save_topic(self, topic: ResearchTopic) -> int:
        """Save a research topic and return its generated ID."""
        sql = """
        INSERT INTO research_topics (
            topic_text, target_domain, start_year, end_year,
            core_concepts, related_terminology, research_dimensions,
            relevant_datasets, important_methodologies, user_objectives
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            topic.topic_text,
            topic.target_domain,
            topic.start_year,
            topic.end_year,
            json.dumps(topic.core_concepts),
            json.dumps(topic.related_terminology),
            json.dumps(topic.research_dimensions),
            json.dumps(topic.relevant_datasets),
            json.dumps(topic.important_methodologies),
            topic.user_objectives,
        )
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            topic_id = cursor.lastrowid
            return int(topic_id)

    def save_questions(self, topic_id: int, questions: List[ResearchQuestion]) -> None:
        """Save formulated research questions."""
        sql = """
        INSERT OR REPLACE INTO research_questions (
            question_id, topic_id, question_text, framework, research_aspect, rationale
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        with get_db_connection(self.db_path) as conn:
            for q in questions:
                conn.execute(sql, (
                    q.question_id, topic_id, q.question_text,
                    q.framework, q.research_aspect, q.rationale
                ))

    def get_questions(self, topic_id: int) -> List[ResearchQuestion]:
        """Retrieve all questions for a topic."""
        sql = "SELECT * FROM research_questions WHERE topic_id = ?"
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(sql, (topic_id,)).fetchall()
            return [
                ResearchQuestion(
                    question_id=r["question_id"],
                    question_text=r["question_text"],
                    framework=r["framework"],
                    research_aspect=r["research_aspect"],
                    rationale=r["rationale"] or ""
                )
                for r in rows
            ]


    # =========================================================================
    # Search Queries
    # =========================================================================

    def save_queries(self, topic_id: int, queries: List[SearchQuery]) -> None:
        """Save generated search queries."""
        sql = """
        INSERT OR REPLACE INTO search_queries (
            query_id, topic_id, query_text, research_dimension, search_objective,
            keywords, expected_paper_types, priority, related_question_id,
            validation_score, validation_status, validation_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with get_db_connection(self.db_path) as conn:
            for q in queries:
                conn.execute(sql, (
                    q.query_id,
                    topic_id,
                    q.query_text,
                    q.research_dimension,
                    q.search_objective,
                    json.dumps(q.keywords),
                    json.dumps(q.expected_paper_types),
                    q.priority,
                    q.related_question_id,
                    q.validation_score,
                    q.validation_status,
                    q.validation_reason,
                ))

    def update_query_validation(
        self, query_id: str, score: float, status: str, reason: str = ""
    ) -> None:
        """Update validation score and status for a query."""
        sql = """
        UPDATE search_queries
        SET validation_score = ?, validation_status = ?, validation_reason = ?
        WHERE query_id = ?
        """
        with get_db_connection(self.db_path) as conn:
            conn.execute(sql, (score, status, reason, query_id))

    def get_queries(self, topic_id: int) -> List[SearchQuery]:
        """Retrieve queries for a topic."""
        sql = "SELECT * FROM search_queries WHERE topic_id = ? ORDER BY query_id"
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(sql, (topic_id,)).fetchall()
            return [
                SearchQuery(
                    query_id=r["query_id"],
                    query_text=r["query_text"],
                    research_dimension=r["research_dimension"],
                    search_objective=r["search_objective"],
                    keywords=json.loads(r["keywords"] or "[]"),
                    expected_paper_types=json.loads(r["expected_paper_types"] or "[]"),
                    priority=r["priority"],
                    related_question_id=r["related_question_id"],
                    validation_score=r["validation_score"],
                    validation_status=r["validation_status"],
                    validation_reason=r["validation_reason"] or "",
                )
                for r in rows
            ]


    # =========================================================================
    # Search Logs
    # =========================================================================

    def log_search(self, log: SearchLog) -> None:
        """Record a search audit log."""
        sql = """
        INSERT INTO search_logs (
            query_id, query_text, source_database, timestamp, records_retrieved, status, error_details
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with get_db_connection(self.db_path) as conn:
            conn.execute(sql, (
                log.query_id, log.query_text, log.source_database,
                log.timestamp, log.records_retrieved, log.status, log.error_details
            ))

    def get_search_logs(self) -> List[SearchLog]:
        """Retrieve all search audit logs."""
        sql = "SELECT * FROM search_logs ORDER BY id"
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(sql).fetchall()
            return [
                SearchLog(
                    search_id=r["id"],
                    query_id=r["query_id"],
                    query_text=r["query_text"],
                    source_database=r["source_database"],
                    timestamp=r["timestamp"],
                    records_retrieved=r["records_retrieved"],
                    status=r["status"],
                    error_details=r["error_details"],
                )
                for r in rows
            ]


    # =========================================================================
    # Papers (Canonical & Deduplication Tracking)
    # =========================================================================

    def upsert_paper(self, paper: PaperRecord) -> None:
        """
        Upsert a paper record. If DOI matches, merge metadata preserving query associations
        and choosing higher citation counts.
        """
        with get_db_connection(self.db_path) as conn:
            existing = None
            if paper.doi:
                existing = conn.execute("SELECT * FROM papers WHERE doi = ?", (paper.doi,)).fetchone()
            if not existing and paper.paper_id:
                existing = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper.paper_id,)).fetchone()

            if existing:
                # Merge query IDs
                old_queries = set(json.loads(existing["retrieved_query_ids"] or "[]"))
                new_queries = set(paper.retrieved_query_ids)
                merged_queries = list(old_queries.union(new_queries))

                # Keep higher citation count and non-empty abstract
                citations = max(existing["citation_count"] or 0, paper.citation_count)
                abstract = paper.abstract or existing["abstract"]
                full_text = paper.full_text or existing["full_text"]
                full_text_status = (
                    "Full text retrieved"
                    if (paper.full_text_status == "Full text retrieved" or existing["full_text_status"] == "Full text retrieved")
                    else paper.full_text_status
                )

                update_sql = """
                UPDATE papers SET
                    title = ?, authors = ?, publication_year = ?, abstract = ?,
                    arxiv_id = COALESCE(?, arxiv_id), pmid = COALESCE(?, pmid),
                    publisher = COALESCE(?, publisher), journal_or_conference = COALESCE(?, journal_or_conference),
                    paper_url = COALESCE(?, paper_url), pdf_url = COALESCE(?, pdf_url),
                    retrieved_query_ids = ?, citation_count = ?,
                    full_text_status = ?, full_text = ?,
                    relevance_score = MAX(relevance_score, ?),
                    relevance_reason = COALESCE(?, relevance_reason)
                WHERE paper_id = ?
                """
                conn.execute(update_sql, (
                    paper.title,
                    json.dumps(paper.authors),
                    paper.publication_year,
                    abstract,
                    paper.arxiv_id,
                    paper.pmid,
                    paper.publisher,
                    paper.journal_or_conference,
                    paper.paper_url,
                    paper.pdf_url,
                    json.dumps(merged_queries),
                    citations,
                    full_text_status,
                    full_text,
                    paper.relevance_score,
                    paper.relevance_reason,
                    existing["paper_id"]
                ))
            else:
                insert_sql = """
                INSERT INTO papers (
                    paper_id, title, authors, publication_year, abstract,
                    doi, arxiv_id, pmid, publisher, journal_or_conference,
                    publication_type, publication_status, keywords, paper_url,
                    pdf_url, source_database, retrieved_query_ids, citation_count,
                    open_access_status, full_text_status, full_text,
                    relevance_score, relevance_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                conn.execute(insert_sql, (
                    paper.paper_id,
                    paper.title,
                    json.dumps(paper.authors),
                    paper.publication_year,
                    paper.abstract,
                    paper.doi,
                    paper.arxiv_id,
                    paper.pmid,
                    paper.publisher,
                    paper.journal_or_conference,
                    paper.publication_type,
                    paper.publication_status,
                    json.dumps(paper.keywords),
                    paper.paper_url,
                    paper.pdf_url,
                    paper.source_database,
                    json.dumps(paper.retrieved_query_ids),
                    paper.citation_count,
                    paper.open_access_status,
                    paper.full_text_status,
                    paper.full_text,
                    paper.relevance_score,
                    paper.relevance_reason,
                ))

    def save_paper(self, paper: PaperRecord) -> None:
        """Alias for upsert_paper."""
        self.upsert_paper(paper)

    def save_papers(self, papers: List[PaperRecord]) -> None:
        """Batch save / upsert papers."""
        for p in papers:
            self.upsert_paper(p)

    def get_all_papers(self) -> List[PaperRecord]:
        """Retrieve all papers ordered by relevance score descending."""
        sql = "SELECT * FROM papers ORDER BY relevance_score DESC, citation_count DESC"
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(sql).fetchall()
            return [self._row_to_paper(r) for r in rows]

    def get_paper_by_id(self, paper_id: str) -> Optional[PaperRecord]:
        """Fetch a specific paper by ID."""
        sql = "SELECT * FROM papers WHERE paper_id = ?"
        with get_db_connection(self.db_path) as conn:
            row = conn.execute(sql, (paper_id,)).fetchone()
            return self._row_to_paper(row) if row else None

    def get_paper_by_doi(self, doi: str) -> Optional[PaperRecord]:
        """Fetch a specific paper by DOI (case-insensitive)."""
        sql = "SELECT * FROM papers WHERE LOWER(doi) = LOWER(?)"
        with get_db_connection(self.db_path) as conn:
            row = conn.execute(sql, (doi.strip(),)).fetchone()
            return self._row_to_paper(row) if row else None

    def _row_to_paper(self, r: Any) -> PaperRecord:
        """Convert a database row into a PaperRecord."""
        return PaperRecord(
            paper_id=r["paper_id"],
            title=r["title"],
            authors=json.loads(r["authors"] or "[]"),
            publication_year=r["publication_year"],
            abstract=r["abstract"],
            doi=r["doi"],
            arxiv_id=r["arxiv_id"],
            pmid=r["pmid"],
            publisher=r["publisher"],
            journal_or_conference=r["journal_or_conference"] or "",
            publication_type=r["publication_type"] or "Journal/Conference",
            publication_status=r["publication_status"] or "Published",
            keywords=json.loads(r["keywords"] or "[]"),
            paper_url=r["paper_url"] or "",
            pdf_url=r["pdf_url"],
            source_database=r["source_database"],
            retrieved_query_ids=json.loads(r["retrieved_query_ids"] or "[]"),
            citation_count=r["citation_count"] or 0,
            open_access_status=r["open_access_status"] or "Unknown",
            full_text_status=r["full_text_status"] or "Abstract only",
            full_text=r["full_text"],
            relevance_score=r["relevance_score"] or 0.0,
            relevance_reason=r["relevance_reason"] or "",
        )

    # =========================================================================
    # Screening Decisions
    # =========================================================================

    def save_screening_decision(self, decision: ScreeningDecision) -> None:
        """Save a title and abstract screening decision."""
        sql = """
        INSERT OR REPLACE INTO screening_results (
            paper_id, title, decision, relevance_score, inclusion_reason,
            exclusion_reason, screening_stage, reviewer
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with get_db_connection(self.db_path) as conn:
            conn.execute(sql, (
                decision.paper_id, decision.title, decision.decision,
                decision.relevance_score, decision.inclusion_reason,
                decision.exclusion_reason, decision.screening_stage, decision.reviewer
            ))

    def save_screening_decisions(self, decisions: List[ScreeningDecision]) -> None:
        """Batch save screening decisions."""
        for d in decisions:
            self.save_screening_decision(d)

    def get_screening_decisions(self) -> List[ScreeningDecision]:
        """Retrieve all screening decisions."""
        sql = "SELECT * FROM screening_results ORDER BY created_at"
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(sql).fetchall()
            return [
                ScreeningDecision(
                    paper_id=r["paper_id"],
                    title=r["title"],
                    decision=r["decision"],
                    relevance_score=r["relevance_score"],
                    inclusion_reason=r["inclusion_reason"],
                    exclusion_reason=r["exclusion_reason"],
                    screening_stage=r["screening_stage"],
                    reviewer=r["reviewer"],
                )
                for r in rows
            ]


    # =========================================================================
    # Extracted Information & Quality Assessments
    # =========================================================================

    def save_extracted_info(self, info: ExtractedInfo) -> None:
        """Save extracted information for an included study."""
        sql = """
        INSERT OR REPLACE INTO extracted_info (
            paper_id, title, doi, research_problem, research_objectives,
            methodology, model_architecture, pretraining_strategy, fine_tuning_strategy,
            dataset, experimental_setup, evaluation_metrics, main_findings,
            limitations, future_directions, code_availability, dataset_availability,
            hsi_sensor_characteristics, hsi_spectral_bands, hsi_spatial_resolution,
            hsi_spectral_spatial_strategy, hsi_downstream_task, cross_scene_evaluation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with get_db_connection(self.db_path) as conn:
            conn.execute(sql, (
                info.paper_id, info.title, info.doi, info.research_problem,
                info.research_objectives, info.methodology, info.model_architecture,
                info.pretraining_strategy, info.fine_tuning_strategy, info.dataset,
                info.experimental_setup, info.evaluation_metrics, info.main_findings,
                info.limitations, info.future_directions, info.code_availability,
                info.dataset_availability, info.hsi_sensor_characteristics,
                info.hsi_spectral_bands, info.hsi_spatial_resolution,
                info.hsi_spectral_spatial_strategy, info.hsi_downstream_task,
                info.cross_scene_evaluation
            ))

    def save_quality_assessment(self, qa: QualityAssessment) -> None:
        """Save quality assessment for an included study."""
        sql = """
        INSERT OR REPLACE INTO quality_assessments (
            paper_id, clarity_of_objectives, dataset_transparency,
            methodological_clarity, experimental_reproducibility,
            baseline_comparisons, metrics_appropriateness,
            statistical_validation, limitations_discussion,
            code_data_availability, total_quality_score, justification
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with get_db_connection(self.db_path) as conn:
            conn.execute(sql, (
                qa.paper_id, qa.clarity_of_objectives, qa.dataset_transparency,
                qa.methodological_clarity, qa.experimental_reproducibility,
                qa.baseline_comparisons, qa.metrics_appropriateness,
                qa.statistical_validation, qa.limitations_discussion,
                qa.code_data_availability, qa.total_quality_score, qa.justification
            ))

    def get_all_extracted_info(self) -> List[ExtractedInfo]:
        """Retrieve all extracted study info."""
        sql = "SELECT * FROM extracted_info"
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(sql).fetchall()
            return [
                ExtractedInfo(
                    paper_id=r["paper_id"],
                    title=r["title"],
                    doi=r["doi"],
                    research_problem=r["research_problem"] or "Not reported",
                    research_objectives=r["research_objectives"] or "Not reported",
                    methodology=r["methodology"] or "Not reported",
                    model_architecture=r["model_architecture"] or "Not reported",
                    pretraining_strategy=r["pretraining_strategy"] or "Not reported",
                    fine_tuning_strategy=r["fine_tuning_strategy"] or "Not reported",
                    dataset=r["dataset"] or "Not reported",
                    experimental_setup=r["experimental_setup"] or "Not reported",
                    evaluation_metrics=r["evaluation_metrics"] or "Not reported",
                    main_findings=r["main_findings"] or "Not reported",
                    limitations=r["limitations"] or "Not reported",
                    future_directions=r["future_directions"] or "Not reported",
                    code_availability=r["code_availability"] or "Not reported",
                    dataset_availability=r["dataset_availability"] or "Not reported",
                    hsi_sensor_characteristics=r["hsi_sensor_characteristics"] or "Not reported",
                    hsi_spectral_bands=r["hsi_spectral_bands"] or "Not reported",
                    hsi_spatial_resolution=r["hsi_spatial_resolution"] or "Not reported",
                    hsi_spectral_spatial_strategy=r["hsi_spectral_spatial_strategy"] or "Not reported",
                    hsi_downstream_task=r["hsi_downstream_task"] or "Not reported",
                    cross_scene_evaluation=r["cross_scene_evaluation"] or "Not reported",
                )
                for r in rows
            ]


    def get_extracted_info_by_id(self, paper_id: str) -> Optional[ExtractedInfo]:
        """Fetch extracted info for a specific paper ID."""
        sql = "SELECT * FROM extracted_info WHERE paper_id = ?"
        with get_db_connection(self.db_path) as conn:
            r = conn.execute(sql, (paper_id,)).fetchone()
            return self._row_to_extracted_info(r) if r else None

    def get_extracted_info_by_doi(self, doi: str) -> Optional[ExtractedInfo]:
        """Fetch extracted info for a specific DOI (case-insensitive)."""
        sql = "SELECT * FROM extracted_info WHERE LOWER(doi) = LOWER(?)"
        with get_db_connection(self.db_path) as conn:
            r = conn.execute(sql, (doi.strip(),)).fetchone()
            return self._row_to_extracted_info(r) if r else None

    def _row_to_extracted_info(self, r: Any) -> ExtractedInfo:
        """Convert a database row into an ExtractedInfo object."""
        return ExtractedInfo(
            paper_id=r["paper_id"],
            title=r["title"],
            doi=r["doi"],
            research_problem=r["research_problem"] or "Not reported",
            research_objectives=r["research_objectives"] or "Not reported",
            methodology=r["methodology"] or "Not reported",
            model_architecture=r["model_architecture"] or "Not reported",
            pretraining_strategy=r["pretraining_strategy"] or "Not reported",
            fine_tuning_strategy=r["fine_tuning_strategy"] or "Not reported",
            dataset=r["dataset"] or "Not reported",
            experimental_setup=r["experimental_setup"] or "Not reported",
            evaluation_metrics=r["evaluation_metrics"] or "Not reported",
            main_findings=r["main_findings"] or "Not reported",
            limitations=r["limitations"] or "Not reported",
            future_directions=r["future_directions"] or "Not reported",
            code_availability=r["code_availability"] or "Not reported",
            dataset_availability=r["dataset_availability"] or "Not reported",
            hsi_sensor_characteristics=r["hsi_sensor_characteristics"] or "Not reported",
            hsi_spectral_bands=r["hsi_spectral_bands"] or "Not reported",
            hsi_spatial_resolution=r["hsi_spatial_resolution"] or "Not reported",
            hsi_spectral_spatial_strategy=r["hsi_spectral_spatial_strategy"] or "Not reported",
            hsi_downstream_task=r["hsi_downstream_task"] or "Not reported",
            cross_scene_evaluation=r["cross_scene_evaluation"] or "Not reported",
        )

    def get_all_quality_assessments(self) -> List[QualityAssessment]:
        """Retrieve all quality assessments."""
        sql = "SELECT * FROM quality_assessments"
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(sql).fetchall()
            return [
                QualityAssessment(
                    paper_id=r["paper_id"],
                    clarity_of_objectives=r["clarity_of_objectives"],
                    dataset_transparency=r["dataset_transparency"],
                    methodological_clarity=r["methodological_clarity"],
                    experimental_reproducibility=r["experimental_reproducibility"],
                    baseline_comparisons=r["baseline_comparisons"],
                    metrics_appropriateness=r["metrics_appropriateness"],
                    statistical_validation=r["statistical_validation"],
                    limitations_discussion=r["limitations_discussion"],
                    code_data_availability=r["code_data_availability"],
                    total_quality_score=r["total_quality_score"],
                    justification=r["justification"] or "",
                )
                for r in rows
            ]


    # =========================================================================
    # Research Gaps & Novel Ideas
    # =========================================================================

    def save_research_gaps(self, topic_id: int, gaps: List[ResearchGap]) -> None:
        """Save identified evidence-supported research gaps."""
        sql = """
        INSERT OR REPLACE INTO research_gaps (
            gap_id, topic_id, description, supporting_papers, evidence,
            existing_approaches, limitations_of_approaches, why_gap_matters,
            confidence_level, gap_type, verification_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with get_db_connection(self.db_path) as conn:
            for g in gaps:
                conn.execute(sql, (
                    g.gap_id, topic_id, g.description, json.dumps(g.supporting_papers),
                    g.evidence, g.existing_approaches, g.limitations_of_approaches,
                    g.why_gap_matters, g.confidence_level, g.gap_type, g.verification_status
                ))

    def get_research_gaps(self, topic_id: int) -> List[ResearchGap]:
        """Retrieve research gaps for a topic."""
        sql = "SELECT * FROM research_gaps WHERE topic_id = ?"
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(sql, (topic_id,)).fetchall()
            return [
                ResearchGap(
                    gap_id=r["gap_id"],
                    description=r["description"],
                    supporting_papers=json.loads(r["supporting_papers"] or "[]"),
                    evidence=r["evidence"],
                    existing_approaches=r["existing_approaches"],
                    limitations_of_approaches=r["limitations_of_approaches"],
                    why_gap_matters=r["why_gap_matters"],
                    confidence_level=r["confidence_level"],
                    gap_type=r["gap_type"],
                    verification_status=r["verification_status"],
                )
                for r in rows
            ]


    def save_novel_ideas(self, topic_id: int, ideas: List[NovelIdea]) -> None:
        """Save proposed novel research ideas."""
        sql = """
        INSERT OR REPLACE INTO novel_ideas (
            idea_id, topic_id, gap_id, proposed_title, core_research_problem,
            motivation, research_gap_addressed, proposed_methodology,
            possible_architecture, potential_datasets, experimental_design,
            baseline_models, evaluation_metrics, expected_contributions,
            technical_feasibility, feasibility_score, main_risks,
            closest_existing_studies, novelty_verification_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with get_db_connection(self.db_path) as conn:
            for idea in ideas:
                conn.execute(sql, (
                    idea.idea_id, topic_id, idea.research_gap_addressed,
                    idea.proposed_title, idea.core_research_problem,
                    idea.motivation, idea.research_gap_addressed,
                    idea.proposed_methodology, idea.possible_architecture,
                    json.dumps(idea.potential_datasets), idea.experimental_design,
                    json.dumps(idea.baseline_models), json.dumps(idea.evaluation_metrics),
                    json.dumps(idea.expected_contributions), idea.technical_feasibility,
                    idea.feasibility_score, idea.main_risks,
                    json.dumps(idea.closest_existing_studies), idea.novelty_verification_status
                ))

    def get_novel_ideas(self, topic_id: int) -> List[NovelIdea]:
        """Retrieve novel research ideas for a topic."""
        sql = "SELECT * FROM novel_ideas WHERE topic_id = ?"
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(sql, (topic_id,)).fetchall()
            return [
                NovelIdea(
                    idea_id=r["idea_id"],
                    proposed_title=r["proposed_title"],
                    core_research_problem=r["core_research_problem"],
                    motivation=r["motivation"],
                    research_gap_addressed=r["research_gap_addressed"],
                    proposed_methodology=r["proposed_methodology"],
                    possible_architecture=r["possible_architecture"],
                    potential_datasets=json.loads(r["potential_datasets"] or "[]"),
                    experimental_design=r["experimental_design"],
                    baseline_models=json.loads(r["baseline_models"] or "[]"),
                    evaluation_metrics=json.loads(r["evaluation_metrics"] or "[]"),
                    expected_contributions=json.loads(r["expected_contributions"] or "[]"),
                    technical_feasibility=r["technical_feasibility"] or "High",
                    feasibility_score=r["feasibility_score"] or 8.0,
                    main_risks=r["main_risks"] or "",
                    closest_existing_studies=json.loads(r["closest_existing_studies"] or "[]"),
                    novelty_verification_status=r["novelty_verification_status"] or "Preliminary Hypothesis",
                )
                for r in rows
            ]

