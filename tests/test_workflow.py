"""
hsi_research_agent/tests/test_workflow.py
Integration test for the LangGraph stateful workflow connecting:
Topic Analysis -> Question Formulation -> 10 Queries -> Query Validation -> Search Execution.
"""

from unittest.mock import MagicMock, patch
import pytest

from hsi_research_agent.graph.workflow import build_research_graph
from hsi_research_agent.graph.state import AgentState
from hsi_research_agent.agents.topic_analyzer import TopicAnalysisOutput
from hsi_research_agent.agents.question_generator import QuestionListOutput
from hsi_research_agent.agents.query_validator import SingleQueryEvaluation
from hsi_research_agent.agents.screening_agent import ScreeningOutput
from hsi_research_agent.agents.extraction_agent import RawExtractionOutput
from hsi_research_agent.agents.quality_agent import RawQualityOutput
from hsi_research_agent.storage.models import (
    ResearchQuestion,
    SearchQuery,
    QueryList,
    ResearchGap,
    ResearchGapList,
    NovelIdea,
    NovelIdeaList,
    EvidenceSynthesis,
)


class TestLangGraphWorkflow:
    """Test suite for LangGraph StateGraph pipeline execution."""

    @patch("hsi_research_agent.search.search_manager.search_openalex")
    @patch("hsi_research_agent.search.search_manager.search_crossref")
    @patch("hsi_research_agent.search.search_manager.search_arxiv")
    @patch("hsi_research_agent.search.search_manager.search_semantic_scholar")
    @patch("hsi_research_agent.llm.client.OllamaClient.generate_structured")
    def test_workflow_execution_phases_1_to_3(
        self, mock_structured, mock_s2, mock_arxiv, mock_cr, mock_oa
    ):
        # 1. Mock Topic Analyzer LLM output
        topic_output = TopicAnalysisOutput(
            target_domain="Hyperspectral Remote Sensing",
            core_concepts=["Hyperspectral Imaging", "Foundation Models"],
            research_dimensions=["Backbones", "Generalization", "Benchmarks"],
            relevant_datasets=["Indian Pines", "Pavia"],
            important_methodologies=["SpectralGPT", "SatMAE"]
        )

        # 2. Mock Question Generator LLM output
        question_output = QuestionListOutput(
            questions=[
                ResearchQuestion(
                    question_id="RQ1",
                    question_text="What architectures are used for HSI foundation models?",
                    framework="Domain-Specific",
                    research_aspect="Architectures"
                ),
                ResearchQuestion(
                    question_id="RQ2",
                    question_text="How do models transfer across sensors?",
                    framework="Domain-Specific",
                    research_aspect="Generalization"
                ),
                ResearchQuestion(
                    question_id="RQ3",
                    question_text="What are key classification benchmarks?",
                    framework="Domain-Specific",
                    research_aspect="Benchmarks"
                ),
            ]
        )

        # 3. Mock Query Generator LLM output (exactly 10 queries)
        queries_output = QueryList(
            queries=[
                SearchQuery(
                    query_id=f"Q{i:02d}",
                    query_text=f"hyperspectral foundation model dimension {i}",
                    research_dimension=f"Dimension {i}",
                    search_objective=f"Objective {i}",
                    keywords=["hyperspectral", "foundation model"],
                    expected_paper_types=["Journal Article"],
                    priority=1,
                    related_question_id="RQ1"
                )
                for i in range(1, 11)
            ]
        )

        # 4. Mock Query Validation LLM output (scores 90+)
        validation_output = SingleQueryEvaluation(
            score=95.0,
            is_valid=True,
            specificity_score=95.0,
            coverage_score=95.0,
            feedback="Well framed query."
        )

        # Sequence of structured responses:
        # call 1: topic analyzer
        # call 2: question generator
        # call 3: query generator
        # call 4-13: 10 query validations
        mock_structured.side_effect = (
            [topic_output, question_output, queries_output]
            + [validation_output] * 10
        )

        # Mock API search responses
        mock_oa.return_value = [{"title": "OpenAlex HSI Paper", "doi": "10.1000/oa"}]
        mock_cr.return_value = [{"title": ["Crossref HSI Paper"], "DOI": "10.1000/cr"}]
        mock_arxiv.return_value = [{"title": "arXiv HSI Paper", "id": "2401.0001", "published_year": 2024}]
        mock_s2.return_value = [{"title": "S2 HSI Paper", "externalIds": {"DOI": "10.1000/s2"}}]

        # Build and invoke compiled graph for Phases 1-3
        graph = build_research_graph(full_pipeline=False)

        initial_state: AgentState = {
            "research_topic": "Foundation Models for Hyperspectral Imaging",
            "target_domain": "Hyperspectral Remote Sensing",
            "start_year": 2020,
            "end_year": 2026,
            "user_objectives": "Identify architectures and generalization limitations",
            "current_stage": "Initialized",
            "execution_errors": []
        }

        config = {"configurable": {"thread_id": "test_thread_001"}}
        final_state = graph.invoke(initial_state, config=config)

        # Verify all nodes executed in order and populated state correctly:
        assert final_state["current_stage"] == "SearchComplete"
        assert final_state["topic_id"] is not None
        assert len(final_state["research_questions"]) == 3
        assert len(final_state["generated_queries"]) == 10
        assert len(final_state["validated_queries"]) == 10
        assert all(q["validation_status"] == "Validated" for q in final_state["validated_queries"])
        assert len(final_state["search_logs"]) == 40  # 10 queries * 4 sources = 40 searches executed
        assert len(final_state["normalized_papers"]) == 40  # 40 papers collected

    def test_full_workflow_compilation(self):
        """Verify the full 14-stage StateGraph compiles with all nodes and checkpointing."""
        graph = build_research_graph(full_pipeline=True)
        assert graph is not None
        # Nodes registered in graph
        expected_nodes = {
            "topic_analyzer", "question_generator", "query_generator", "query_validator",
            "search_manager", "deduplicator", "screening", "pdf_retrieval", "extraction",
            "quality_assessment", "research_gaps", "synthesis", "novel_ideas", "export"
        }
        for node_name in expected_nodes:
            assert node_name in graph.nodes

    @patch("hsi_research_agent.processing.pdf_processor.requests.get")
    @patch("hsi_research_agent.search.search_manager.search_openalex")
    @patch("hsi_research_agent.search.search_manager.search_crossref")
    @patch("hsi_research_agent.search.search_manager.search_arxiv")
    @patch("hsi_research_agent.search.search_manager.search_semantic_scholar")
    @patch("hsi_research_agent.llm.client.OllamaClient.generate_structured")
    def test_full_pipeline_execution_all_14_nodes(
        self, mock_structured, mock_s2, mock_arxiv, mock_cr, mock_oa, mock_pdf_req
    ):
        """End-to-end integration test executing all 14 stages of the LangGraph workflow."""
        topic_output = TopicAnalysisOutput(
            target_domain="Hyperspectral Remote Sensing",
            core_concepts=["Hyperspectral Imaging", "Foundation Models"],
            research_dimensions=["Backbones", "Generalization", "Benchmarks"],
            relevant_datasets=["Houston 2018"],
            important_methodologies=["SpectralGPT"]
        )
        question_output = QuestionListOutput(
            questions=[
                ResearchQuestion(
                    question_id="RQ1",
                    question_text="What architectures are used for HSI foundation models?",
                    framework="Domain-Specific",
                    research_aspect="Architectures"
                ),
                ResearchQuestion(
                    question_id="RQ2",
                    question_text="How do foundation models transfer across sensors?",
                    framework="Domain-Specific",
                    research_aspect="Generalization"
                ),
                ResearchQuestion(
                    question_id="RQ3",
                    question_text="What are key classification benchmarks?",
                    framework="Domain-Specific",
                    research_aspect="Benchmarks"
                ),
            ]
        )
        queries_output = QueryList(
            queries=[
                SearchQuery(
                    query_id=f"Q{i:02d}",
                    query_text=f"hyperspectral foundation model query {i}",
                    research_dimension="Architectures",
                    search_objective=f"Objective {i}",
                    keywords=["hyperspectral", "foundation model"],
                    expected_paper_types=["Journal Article"],
                    priority=1,
                    related_question_id="RQ1"
                )
                for i in range(1, 11)
            ]
        )
        validation_output = SingleQueryEvaluation(
            score=95.0,
            is_valid=True,
            specificity_score=95.0,
            coverage_score=95.0,
            feedback="Well framed query."
        )

        def dynamic_side_effect(prompt, pydantic_class, *args, **kwargs):
            if pydantic_class == TopicAnalysisOutput:
                return topic_output
            elif pydantic_class == QuestionListOutput:
                return question_output
            elif pydantic_class == QueryList:
                return queries_output
            elif pydantic_class == SingleQueryEvaluation:
                return validation_output
            elif pydantic_class == ScreeningOutput:
                return ScreeningOutput(decision="Include", relevance_score=9.0, inclusion_reason="Direct match")
            elif pydantic_class == RawExtractionOutput:
                return RawExtractionOutput(
                    research_problem="Hyperspectral classification under domain shifts",
                    methodology="Spectral Transformer",
                    model_architecture="ViT-HSI",
                    dataset="Houston 2018",
                    limitations="High memory footprint",
                    main_findings="95% OA achieved",
                )
            elif pydantic_class == RawQualityOutput:
                return RawQualityOutput(clarity_of_objectives=9.0, dataset_transparency=8.5, methodological_clarity=9.0)
            elif pydantic_class == ResearchGapList:
                return ResearchGapList(gaps=[
                    ResearchGap(
                        gap_id="GAP-01",
                        description="Cross-sensor spectral alignment",
                        supporting_papers=["OpenAlex HSI Paper"],
                        evidence="Severe drop on unseen bands",
                        existing_approaches="Retraining",
                        limitations_of_approaches="Expensive",
                        why_gap_matters="Real-world deployment",
                    )
                ])
            elif pydantic_class == EvidenceSynthesis:
                return EvidenceSynthesis(
                    publication_year_distribution={"2024": 1},
                    thematic_summary="Growing focus on foundation models.",
                )
            elif pydantic_class == NovelIdeaList:
                return NovelIdeaList(ideas=[
                    NovelIdea(
                        idea_id="IDEA-01",
                        proposed_title="Continuous Wavelength Transformer",
                        core_research_problem="Dynamic band tokenization",
                        motivation="Solves GAP-01",
                        research_gap_addressed="GAP-01",
                        proposed_methodology="Wavelength encodings",
                        possible_architecture="Transformer",
                        potential_datasets=["Houston 2018"],
                        baseline_models=["SpectralFormer"],
                        evaluation_metrics=["OA"],
                        feasibility_score=8.5,
                    )
                ])
            return pydantic_class()

        mock_structured.side_effect = dynamic_side_effect

        # Mock API search responses
        mock_oa.return_value = [{"title": "OpenAlex HSI Paper", "doi": "10.1000/oa", "cited_by_count": 25}]
        mock_cr.return_value = [{"title": ["Crossref HSI Paper"], "DOI": "10.1000/cr"}]
        mock_arxiv.return_value = [{"title": "arXiv HSI Paper", "id": "2401.0001", "published_year": 2024}]
        mock_s2.return_value = [{"title": "S2 HSI Paper", "externalIds": {"DOI": "10.1000/s2"}}]

        # Mock PDF download response (404 graceful)
        mock_pdf_req.return_value.status_code = 404

        # Build and invoke full compiled graph
        graph = build_research_graph(full_pipeline=True)

        initial_state: AgentState = {
            "research_topic": "Foundation Models for Hyperspectral Imaging",
            "target_domain": "Hyperspectral Remote Sensing",
            "start_year": 2020,
            "end_year": 2026,
            "user_objectives": "Identify architectures and generalization limitations",
            "current_stage": "Initialized",
            "execution_errors": []
        }

        config = {"configurable": {"thread_id": "test_full_thread_001"}}
        final_state = graph.invoke(initial_state, config=config)

        # Assert full workflow traversed from node 1 to node 14:
        assert final_state["current_stage"] == "PipelineComplete"
        assert len(final_state["unique_papers"]) > 0
        assert len(final_state["screening_results"]) > 0
        assert len(final_state["extracted_information"]) > 0
        assert len(final_state["quality_assessments"]) > 0
        assert len(final_state["research_gaps"]) > 0
        assert len(final_state["novel_ideas"]) > 0
        assert "excel" in final_state["export_paths"]
        assert "report" in final_state["export_paths"]
        assert "papers" in final_state["export_paths"]


