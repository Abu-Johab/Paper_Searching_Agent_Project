"""
Tests for Phase 7: Export & Master Excel Workbook
(ExcelExporter, CSVExporter, ReportGenerator)
"""

import openpyxl
import pandas as pd
import pytest

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
from hsi_research_agent.export.excel_exporter import ExcelExporter
from hsi_research_agent.export.csv_exporter import CSVExporter
from hsi_research_agent.export.report_generator import ReportGenerator


@pytest.fixture
def sample_export_data():
    topic = ResearchTopic(
        topic_id=1,
        topic_text="Hyperspectral Foundation Models in Agriculture",
        target_domain="Hyperspectral Remote Sensing",
    )
    queries = [
        SearchQuery(
            query_id="Q01",
            query_text="hyperspectral foundation model agriculture",
            research_dimension="Core Foundation Models",
            search_objective="Discover baseline models",
            keywords=["hyperspectral", "foundation model"],
            priority=1,
            validation_score=95.0,
            validation_status="Validated",
        )
    ]
    papers = [
        PaperRecord(
            paper_id="paper-01",
            title="SpectralFormer: Rethinking Hyperspectral Image Classification with Transformers",
            publication_year=2022,
            doi="10.1109/TGRS.2022.3144304",
            source_database="OpenAlex",
            relevance_score=8.5,
            paper_url="https://ieeexplore.ieee.org/document/9672087",
            citation_count=120,
        )
    ]
    logs = [
        SearchLog(
            query_id="Q01",
            query_text="hyperspectral foundation model agriculture",
            source_database="OpenAlex",
            timestamp="2026-11-22T00:00:00",
            records_retrieved=15,
            status="Success",
        )
    ]
    screening = [
        ScreeningDecision(
            paper_id="paper-01",
            title="SpectralFormer: Rethinking Hyperspectral Image Classification with Transformers",
            decision="Include",
            relevance_score=8.5,
            inclusion_reason="Directly addresses Transformer architecture for HSI.",
        )
    ]
    quality = [
        QualityAssessment(
            paper_id="paper-01",
            clarity_of_objectives=9.0,
            dataset_transparency=8.5,
            methodological_clarity=9.0,
            experimental_reproducibility=8.0,
            baseline_comparisons=8.5,
            metrics_appropriateness=9.0,
            statistical_validation=7.5,
            limitations_discussion=8.0,
            code_data_availability=9.0,
            total_quality_score=76.5,
        )
    ]
    gaps = [
        ResearchGap(
            gap_id="GAP-01",
            description="Lack of sensor-agnostic tokenization in hyperspectral transformers.",
            supporting_papers=["SpectralFormer"],
            evidence="Fixed channel projection fails under varying band configurations.",
            existing_approaches="Fixed linear projection",
            limitations_of_approaches="Cannot transfer across sensors with different band counts.",
            why_gap_matters="Prevents unified foundation model deployment.",
            confidence_level="High",
            gap_type="Methodological",
            verification_status="Supported by Literature",
        )
    ]
    ideas = [
        NovelIdea(
            idea_id="IDEA-01",
            proposed_title="Continuous Wavelength-Aware Spectral Transformer (CW-SpecFormer)",
            core_research_problem="Dynamic spectral resolution adaptation across heterogeneous sensors.",
            motivation="Enables true foundation model transfer without architecture changes.",
            research_gap_addressed="GAP-01",
            proposed_methodology="Continuous wavelength positional encodings coupled with cross-attention.",
            possible_architecture="Transformer Encoder with Continuous Spectral Positional Embeddings",
            potential_datasets=["Houston 2018", "Indian Pines"],
            baseline_models=["SpectralFormer", "SSFTT"],
            evaluation_metrics=["Overall Accuracy", "Kappa"],
            expected_contributions=["Sensor-agnostic spectral tokenization"],
            feasibility_score=8.5,
        )
    ]
    synthesis = EvidenceSynthesis(
        publication_year_distribution={"2022": 1},
        frequent_architectures=[{"name": "SpectralFormer", "count": 1}],
        frequent_datasets=[{"name": "Indian Pines", "count": 1}],
        dominant_methodologies=["Transformer attention"],
        common_limitations=["Quadratic complexity"],
        emerging_directions=["Foundation models"],
        thematic_summary="Rapid convergence towards attention-based spectral-spatial representations.",
    )
    return {
        "topic": topic,
        "queries": queries,
        "papers": papers,
        "logs": logs,
        "screening": screening,
        "quality": quality,
        "gaps": gaps,
        "ideas": ideas,
        "synthesis": synthesis,
    }


def test_excel_exporter(tmp_path, sample_export_data):
    exporter = ExcelExporter(output_dir=tmp_path)
    file_path = exporter.export_workbook(
        topic=sample_export_data["topic"],
        papers=sample_export_data["papers"],
        queries=sample_export_data["queries"],
        screening_decisions=sample_export_data["screening"],
        quality_assessments=sample_export_data["quality"],
        research_gaps=sample_export_data["gaps"],
        novel_ideas=sample_export_data["ideas"],
        search_logs=sample_export_data["logs"],
        synthesis=sample_export_data["synthesis"],
        filename="test_discovery.xlsx",
    )

    assert file_path.exists()

    # Load with openpyxl to verify sheets & structure
    wb = openpyxl.load_workbook(file_path)
    expected_sheets = [
        "Research_Papers",
        "Research_Queries",
        "Search_Logs",
        "Screening_Results",
        "Research_Gaps",
        "Novel_Ideas",
        "Quality_Assessment",
        "Statistics",
    ]
    assert wb.sheetnames == expected_sheets

    # Verify freeze pane and auto filter on Research_Papers
    ws_papers = wb["Research_Papers"]
    assert ws_papers.freeze_panes == "A2"
    assert ws_papers.max_row == 2
    assert ws_papers.cell(row=2, column=1).value == "paper-01"


def test_csv_exporter(tmp_path, sample_export_data):
    exporter = CSVExporter(output_dir=tmp_path)
    out_dict = exporter.export_all(
        papers=sample_export_data["papers"],
        queries=sample_export_data["queries"],
        gaps=sample_export_data["gaps"],
        ideas=sample_export_data["ideas"],
    )

    assert "papers" in out_dict and out_dict["papers"].exists()
    assert "queries" in out_dict and out_dict["queries"].exists()
    assert "gaps" in out_dict and out_dict["gaps"].exists()
    assert "ideas" in out_dict and out_dict["ideas"].exists()

    df_papers = pd.read_csv(out_dict["papers"])
    assert len(df_papers) == 1
    assert df_papers.iloc[0]["paper_id"] == "paper-01"

    df_gaps = pd.read_csv(out_dict["gaps"])
    assert len(df_gaps) == 1
    assert df_gaps.iloc[0]["gap_id"] == "GAP-01"


def test_report_generator(tmp_path, sample_export_data):
    generator = ReportGenerator(output_dir=tmp_path)
    report_path = generator.generate_report(
        topic=sample_export_data["topic"],
        papers=sample_export_data["papers"],
        queries=sample_export_data["queries"],
        screening_decisions=sample_export_data["screening"],
        quality_assessments=sample_export_data["quality"],
        research_gaps=sample_export_data["gaps"],
        novel_ideas=sample_export_data["ideas"],
        synthesis=sample_export_data["synthesis"],
        filename="test_report.md",
    )

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "# Academic Literature Discovery & Research Gap Analysis Report" in content
    assert "GAP-01" in content
    assert "IDEA-01" in content
    assert "SpectralFormer" in content
