"""
hsi_research_agent/graph/nodes.py
LangGraph workflow nodes for all 12 stages of the autonomous literature discovery pipeline:
1. Topic Analyzer
2. Question Generator
3. 10 Query Generator
4. Query Validator
5. Search Manager
6. Deduplicator
7. Screening Agent
8. PDF & Full-Text Retriever
9. Information Extraction Agent
10. Quality Assessment Agent
11. Research Gap Agent
12. Evidence Synthesis Agent
13. Novel Idea Generator
14. Master Exporter (Excel, CSV, Report)
"""

import logging
from typing import Any, Dict, List, Optional

from hsi_research_agent.agents.extraction_agent import extract_all_eligible_papers
from hsi_research_agent.agents.idea_generator import IdeaGeneratorAgent
from hsi_research_agent.agents.quality_agent import assess_all_papers_quality
from hsi_research_agent.agents.query_generator import generate_search_queries
from hsi_research_agent.agents.query_validator import validate_all_queries
from hsi_research_agent.agents.question_generator import generate_research_questions
from hsi_research_agent.agents.research_gap_agent import ResearchGapAgent
from hsi_research_agent.agents.screening_agent import screen_all_papers
from hsi_research_agent.agents.synthesis_agent import SynthesisAgent
from hsi_research_agent.agents.topic_analyzer import analyze_topic
from hsi_research_agent.export.csv_exporter import CSVExporter
from hsi_research_agent.export.excel_exporter import ExcelExporter
from hsi_research_agent.export.report_generator import ReportGenerator
from hsi_research_agent.graph.state import AgentState
from hsi_research_agent.processing.deduplicator import deduplicate_papers
from hsi_research_agent.processing.pdf_processor import retrieve_paper_full_text
from hsi_research_agent.search.search_manager import SearchManager
from hsi_research_agent.storage.models import (
    ExtractedInfo,
    PaperRecord,
    QualityAssessment,
    ResearchGap,
    ResearchQuestion,
    ResearchTopic,
    ScreeningDecision,
    SearchLog,
    SearchQuery,
    NovelIdea,
    EvidenceSynthesis,
)
from hsi_research_agent.storage.repository import ResearchRepository

logger = logging.getLogger("hsi_research_agent.graph.nodes")


# =============================================================================
# Stage 1: Topic Analysis
# =============================================================================
def node_topic_analyzer(state: AgentState) -> Dict[str, Any]:
    """Node 1: Analyze research topic and extract core concepts & dimensions."""
    logger.info("[1/14] Executing Node: Topic Analyzer")
    topic_text = state["research_topic"]
    target_domain = state.get("target_domain", "Machine Learning / Computer Vision")
    start_year = state.get("start_year", 2018)
    end_year = state.get("end_year")
    user_objectives = state.get("user_objectives")

    topic_model = analyze_topic(
        topic_text=topic_text,
        target_domain=target_domain,
        start_year=start_year,
        end_year=end_year,
        user_objectives=user_objectives,
    )

    repo = ResearchRepository()
    topic_id = repo.save_topic(topic_model)

    return {
        "topic_id": topic_id,
        "research_scope": topic_model.model_dump(),
        "current_stage": "TopicAnalysisComplete",
    }


# =============================================================================
# Stage 2: Question Generation
# =============================================================================
def node_question_generator(state: AgentState) -> Dict[str, Any]:
    """Node 2: Formulate 3 to 5 domain-adapted research questions."""
    logger.info("[2/14] Executing Node: Question Generator")
    topic_model = ResearchTopic(**state["research_scope"])
    questions = generate_research_questions(topic_model)

    topic_id = state.get("topic_id")
    if topic_id:
        repo = ResearchRepository()
        repo.save_questions(topic_id, questions)

    return {
        "research_questions": [q.model_dump() for q in questions],
        "current_stage": "QuestionGenerationComplete",
    }


# =============================================================================
# Stage 3: Query Generation (10 Queries)
# =============================================================================
def node_query_generator(state: AgentState) -> Dict[str, Any]:
    """Node 3: Generate EXACTLY 10 academic search queries covering distinct dimensions."""
    logger.info("[3/14] Executing Node: Query Generator (10 Queries)")
    topic_model = ResearchTopic(**state["research_scope"])
    questions = [ResearchQuestion(**q) for q in state["research_questions"]]

    queries = generate_search_queries(topic_model, questions)

    topic_id = state.get("topic_id")
    if topic_id:
        repo = ResearchRepository()
        repo.save_queries(topic_id, queries)

    return {
        "generated_queries": [q.model_dump() for q in queries],
        "current_stage": "QueryGenerationComplete",
    }


# =============================================================================
# Stage 4: Query Validation
# =============================================================================
def node_query_validator(state: AgentState) -> Dict[str, Any]:
    """Node 4: Validate and refine queries to guarantee 10 high-precision queries."""
    logger.info("[4/14] Executing Node: Query Validator")
    topic_model = ResearchTopic(**state["research_scope"])
    raw_queries = [SearchQuery(**q) for q in state["generated_queries"]]

    validated_queries = validate_all_queries(raw_queries, topic_model)

    topic_id = state.get("topic_id")
    if topic_id:
        repo = ResearchRepository()
        for q in validated_queries:
            repo.update_query_validation(
                query_id=q.query_id,
                score=q.validation_score,
                status=q.validation_status,
                reason=q.validation_reason,
            )

    return {
        "validated_queries": [q.model_dump() for q in validated_queries],
        "current_stage": "QueryValidationComplete",
    }


# =============================================================================
# Stage 5: Multi-Source Search Execution
# =============================================================================
def node_search_manager(state: AgentState) -> Dict[str, Any]:
    """Node 5: Execute multi-source search across all 10 validated queries."""
    logger.info("[5/14] Executing Node: Multi-Source Search Manager")
    queries = [SearchQuery(**q) for q in state["validated_queries"]]
    start_year = state.get("start_year", 2018)
    end_year = state.get("end_year")
    sources = state.get("sources")

    repo = ResearchRepository()
    manager = SearchManager(repository=repo, sources=sources)
    papers, logs = manager.execute_all_searches(
        queries=queries,
        start_year=start_year,
        end_year=end_year,
    )

    return {
        "normalized_papers": [p.model_dump() for p in papers],
        "search_logs": [l.model_dump() for l in logs],
        "current_stage": "SearchComplete",
    }


# =============================================================================
# Stage 6: Deduplication
# =============================================================================
def node_deduplicator(state: AgentState) -> Dict[str, Any]:
    """Node 6: Merge duplicates across DOI, arXiv, PMID, and fuzzy title slugs."""
    logger.info("[6/14] Executing Node: Deduplicator")
    raw_papers = [PaperRecord(**p) for p in state.get("normalized_papers", [])]
    unique_papers, duplicate_groups = deduplicate_papers(raw_papers)

    repo = ResearchRepository()
    repo.save_papers(unique_papers)

    return {
        "unique_papers": [p.model_dump() for p in unique_papers],
        "duplicate_groups": duplicate_groups,
        "current_stage": "DeduplicationComplete",
    }


# =============================================================================
# Stage 7: Title & Abstract Screening
# =============================================================================
def node_screening(state: AgentState) -> Dict[str, Any]:
    """Node 7: Screen papers (Include/Exclude/Maybe) with documented reasons."""
    logger.info("[7/14] Executing Node: Title and Abstract Screening Agent")
    papers = [PaperRecord(**p) for p in state.get("unique_papers", [])]
    topic_model = ResearchTopic(**state["research_scope"])

    eligible_papers, screening_decisions = screen_all_papers(papers, topic_model)

    repo = ResearchRepository()
    repo.save_screening_decisions(screening_decisions)
    # Save updated paper scores
    repo.save_papers(papers)

    return {
        "screening_results": [d.model_dump() for d in screening_decisions],
        "unique_papers": [p.model_dump() for p in papers],
        "current_stage": "ScreeningComplete",
    }


# =============================================================================
# Stage 8: Full-Text Retrieval
# =============================================================================
def node_pdf_retrieval(state: AgentState) -> Dict[str, Any]:
    """Node 8: Retrieve open access full-text PDFs and extract clean section text."""
    logger.info("[8/14] Executing Node: Full-Text PDF Retrieval")
    papers = [PaperRecord(**p) for p in state.get("unique_papers", [])]
    screening_map = {
        d["paper_id"]: d["decision"]
        for d in state.get("screening_results", [])
    }

    # Focus retrieval on eligible studies (Include/Maybe) or top 15 by relevance
    eligible = [p for p in papers if screening_map.get(p.paper_id) in ("Include", "Maybe")]
    if not eligible:
        eligible = sorted(papers, key=lambda p: p.relevance_score, reverse=True)[:10]

    repo = ResearchRepository()
    retrieved_count = 0
    for p in eligible[:15]:
        p_updated = retrieve_paper_full_text(p)
        if p_updated.full_text_status == "Full text retrieved":
            retrieved_count += 1
            repo.save_paper(p_updated)

    logger.info("Retrieved full text for %d eligible papers.", retrieved_count)
    return {
        "unique_papers": [p.model_dump() for p in papers],
        "current_stage": "PDFRetrievalComplete",
    }


# =============================================================================
# Stage 9: Information Extraction
# =============================================================================
def node_extraction(state: AgentState) -> Dict[str, Any]:
    """Node 9: Extract methodology, architectures, datasets, findings, and limitations."""
    logger.info("[9/14] Executing Node: Information Extraction Agent")
    papers = [PaperRecord(**p) for p in state.get("unique_papers", [])]
    screening_map = {
        d["paper_id"]: d["decision"]
        for d in state.get("screening_results", [])
    }

    # Extract from included papers (or top 10 relevant)
    eligible = [p for p in papers if screening_map.get(p.paper_id) == "Include"]
    if not eligible:
        eligible = [p for p in papers if screening_map.get(p.paper_id) in ("Include", "Maybe")]
    if not eligible:
        eligible = sorted(papers, key=lambda p: p.relevance_score, reverse=True)[:10]

    extractions = extract_all_eligible_papers(eligible, max_to_extract=15)

    repo = ResearchRepository()
    for ext in extractions:
        repo.save_extracted_info(ext)

    return {
        "extracted_information": [e.model_dump() for e in extractions],
        "current_stage": "ExtractionComplete",
    }


# =============================================================================
# Stage 10: Methodological Quality Assessment
# =============================================================================
def node_quality_assessment(state: AgentState) -> Dict[str, Any]:
    """Node 10: Assess 9-criterion methodological transparency and rigor."""
    logger.info("[10/14] Executing Node: Quality Assessment Agent")
    papers = [PaperRecord(**p) for p in state.get("unique_papers", [])]
    extractions = [ExtractedInfo(**e) for e in state.get("extracted_information", [])]
    ext_ids = {e.paper_id for e in extractions}

    target_papers = [p for p in papers if p.paper_id in ext_ids]
    if not target_papers:
        target_papers = papers[:10]

    assessments = assess_all_papers_quality(target_papers, extractions)

    repo = ResearchRepository()
    for qa in assessments:
        repo.save_quality_assessment(qa)

    return {
        "quality_assessments": [qa.model_dump() for qa in assessments],
        "current_stage": "QualityAssessmentComplete",
    }


# =============================================================================
# Stage 11: Research Gap Identification
# =============================================================================
def node_research_gaps(state: AgentState) -> Dict[str, Any]:
    """Node 11: Formulate evidence-based research gaps supported by literature."""
    logger.info("[11/14] Executing Node: Research Gap Agent")
    topic_model = ResearchTopic(**state["research_scope"])
    papers = [PaperRecord(**p) for p in state.get("unique_papers", [])]
    extractions = [ExtractedInfo(**e) for e in state.get("extracted_information", [])]

    gap_agent = ResearchGapAgent()
    gaps = gap_agent.identify_gaps(topic_model, papers, extractions)

    topic_id = state.get("topic_id")
    if topic_id:
        repo = ResearchRepository()
        repo.save_research_gaps(topic_id, gaps)

    return {
        "research_gaps": [g.model_dump() for g in gaps],
        "current_stage": "ResearchGapsComplete",
    }


# =============================================================================
# Stage 12: Evidence Synthesis
# =============================================================================
def node_synthesis(state: AgentState) -> Dict[str, Any]:
    """Node 12: Synthesize trends, architectures, benchmarks, and narrative overview."""
    logger.info("[12/14] Executing Node: Evidence Synthesis Agent")
    topic_model = ResearchTopic(**state["research_scope"])
    papers = [PaperRecord(**p) for p in state.get("unique_papers", [])]
    extractions = [ExtractedInfo(**e) for e in state.get("extracted_information", [])]

    synthesis_agent = SynthesisAgent()
    synthesis = synthesis_agent.synthesize(topic_model, papers, extractions)

    return {
        "evidence_synthesis": synthesis.model_dump(),
        "current_stage": "SynthesisComplete",
    }


# =============================================================================
# Stage 13: Novel Idea Generation
# =============================================================================
def node_novel_ideas(state: AgentState) -> Dict[str, Any]:
    """Node 13: Propose feasible novel research ideas addressing the identified gaps."""
    logger.info("[13/14] Executing Node: Novel Idea Generator")
    topic_model = ResearchTopic(**state["research_scope"])
    papers = [PaperRecord(**p) for p in state.get("unique_papers", [])]
    gaps = [ResearchGap(**g) for g in state.get("research_gaps", [])]

    idea_agent = IdeaGeneratorAgent()
    ideas = idea_agent.generate_ideas(topic_model, gaps, papers)

    topic_id = state.get("topic_id")
    if topic_id:
        repo = ResearchRepository()
        repo.save_novel_ideas(topic_id, ideas)

    return {
        "novel_ideas": [i.model_dump() for i in ideas],
        "current_stage": "NovelIdeasComplete",
    }


# =============================================================================
# Stage 14: Export (Master Excel, CSVs, Markdown Report)
# =============================================================================
def node_export(state: AgentState) -> Dict[str, Any]:
    """Node 14: Export master Excel workbook, CSV files, and academic Markdown report."""
    logger.info("[14/14] Executing Node: Master Exporter")
    topic_model = ResearchTopic(**state["research_scope"])
    papers = [PaperRecord(**p) for p in state.get("unique_papers", [])]
    queries = [SearchQuery(**q) for q in state.get("validated_queries", [])]
    questions = [ResearchQuestion(**q) for q in state.get("research_questions", [])]
    screening = [ScreeningDecision(**s) for s in state.get("screening_results", [])]
    extractions = [ExtractedInfo(**e) for e in state.get("extracted_information", [])]
    quality = [QualityAssessment(**q) for q in state.get("quality_assessments", [])]
    gaps = [ResearchGap(**g) for g in state.get("research_gaps", [])]
    ideas = [NovelIdea(**i) for i in state.get("novel_ideas", [])]
    logs = [SearchLog(**l) for l in state.get("search_logs", [])]
    synthesis = EvidenceSynthesis(**state.get("evidence_synthesis", {})) if state.get("evidence_synthesis") else None

    # 1. Master Excel
    excel_exp = ExcelExporter()
    excel_path = excel_exp.export_workbook(
        topic=topic_model,
        papers=papers,
        queries=queries,
        screening_decisions=screening,
        extracted_info_list=extractions,
        quality_assessments=quality,
        research_gaps=gaps,
        novel_ideas=ideas,
        search_logs=logs,
        synthesis=synthesis,
    )

    # 2. CSV Exports
    csv_exp = CSVExporter()
    csv_paths = csv_exp.export_all(
        papers=papers,
        queries=queries,
        gaps=gaps,
        ideas=ideas,
    )

    # 3. Comprehensive Markdown Report
    rep_gen = ReportGenerator()
    rep_path = rep_gen.generate_report(
        topic=topic_model,
        papers=papers,
        queries=queries,
        questions=questions,
        screening_decisions=screening,
        extracted_info_list=extractions,
        quality_assessments=quality,
        research_gaps=gaps,
        novel_ideas=ideas,
        synthesis=synthesis,
    )

    export_map = {
        "excel": str(excel_path),
        "report": str(rep_path),
        **{k: str(v) for k, v in csv_paths.items()},
    }

    logger.info("Successfully exported all research artefacts: %s", list(export_map.keys()))
    return {
        "export_paths": export_map,
        "current_stage": "PipelineComplete",
    }
