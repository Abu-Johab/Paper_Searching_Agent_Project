"""
hsi_research_agent/storage/models.py
Pydantic data models for literature discovery, metadata extraction, research gap discovery, and idea generation.
(PRISMA-free, clean academic data models).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ResearchTopic(BaseModel):
    """Structured representation of the analyzed research topic and scope."""
    topic_id: Optional[int] = None
    topic_text: str
    target_domain: str = "Machine Learning / Computer Vision"
    start_year: int = 2018
    end_year: Optional[int] = None
    core_concepts: List[str] = Field(default_factory=list)
    related_terminology: List[str] = Field(default_factory=list)
    research_dimensions: List[str] = Field(default_factory=list)
    relevant_datasets: List[str] = Field(default_factory=list)
    important_methodologies: List[str] = Field(default_factory=list)
    user_objectives: Optional[str] = None

    @property
    def title(self) -> str:
        return self.topic_text

    @property
    def domain(self) -> str:
        return self.target_domain


class ResearchQuestion(BaseModel):
    """Formulated research question guiding the literature search."""
    question_id: str
    question_text: str
    framework: str = "Domain-Specific"
    research_aspect: str
    rationale: str = ""


class SearchQuery(BaseModel):
    """Exactly one of the 10 academic search queries generated for literature discovery."""
    query_id: str
    query_text: str
    research_dimension: str
    search_objective: str
    keywords: List[str] = Field(default_factory=list)
    expected_paper_types: List[str] = Field(default_factory=list)
    priority: int = 1
    related_question_id: Optional[str] = None
    validation_score: float = 0.0
    validation_status: str = "Pending"  # Validated / Weak / Regenerated
    validation_reason: str = ""


class QueryList(BaseModel):
    """Container for exactly 10 generated search queries."""
    queries: List[SearchQuery] = Field(..., min_length=10, max_length=10)


class QueryValidationResult(BaseModel):
    """Result of validating a search query."""
    query_id: str
    score: float = Field(..., ge=0.0, le=100.0)
    is_valid: bool
    specificity_score: float = 0.0
    coverage_score: float = 0.0
    redundancy_penalty: float = 0.0
    feedback: str = ""
    suggested_replacement: Optional[str] = None


class SearchLog(BaseModel):
    """Audit log record for an academic database search execution."""
    search_id: Optional[int] = None
    query_id: str
    query_text: str
    source_database: str
    timestamp: str
    records_retrieved: int = 0
    status: str = "Success"  # Success / RateLimited / Error / Timeout
    error_details: Optional[str] = None


class PaperRecord(BaseModel):
    """Canonical normalized research paper record."""
    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    publication_year: Optional[int] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    publisher: Optional[str] = None
    journal_or_conference: str = ""
    publication_type: str = "Journal/Conference"
    publication_status: str = "Published"
    keywords: List[str] = Field(default_factory=list)
    paper_url: str = ""
    pdf_url: Optional[str] = None
    source_database: str
    retrieved_query_ids: List[str] = Field(default_factory=list)
    citation_count: int = 0
    open_access_status: str = "Unknown"
    full_text_status: str = "Abstract only"  # Full text retrieved / Abstract only / Full text unavailable / Retrieval failed
    full_text: Optional[str] = None
    relevance_score: float = 0.0
    relevance_reason: str = ""

    @property
    def year(self) -> Optional[int]:
        return self.publication_year


class ScreeningDecision(BaseModel):
    """Documented title and abstract screening decision."""
    paper_id: str
    title: str
    decision: str = "Include"  # Include / Exclude / Maybe
    relevance_score: float = 0.0
    inclusion_reason: Optional[str] = None
    exclusion_reason: Optional[str] = None
    screening_stage: str = "Title and Abstract Screening"
    reviewer: str = "Automated Agent (Qwen 2.5 14B)"


class ExtractedInfo(BaseModel):
    """Structured research extraction from paper content or abstract."""
    paper_id: str
    title: str
    doi: Optional[str] = None
    research_problem: str = "Not reported"
    research_objectives: str = "Not reported"
    methodology: str = "Not reported"
    model_architecture: str = "Not reported"
    pretraining_strategy: str = "Not reported"
    fine_tuning_strategy: str = "Not reported"
    dataset: str = "Not reported"
    experimental_setup: str = "Not reported"
    evaluation_metrics: str = "Not reported"
    main_findings: str = "Not reported"
    limitations: str = "Not reported"
    future_directions: str = "Not reported"
    code_availability: str = "Not reported"
    dataset_availability: str = "Not reported"
    # Domain specific for Hyperspectral Imaging
    hsi_sensor_characteristics: str = "Not reported"
    hsi_spectral_bands: str = "Not reported"
    hsi_spatial_resolution: str = "Not reported"
    hsi_spectral_spatial_strategy: str = "Not reported"
    hsi_downstream_task: str = "Not reported"
    cross_scene_evaluation: str = "Not reported"


class QualityAssessment(BaseModel):
    """Quality and methodological transparency assessment of an included study."""
    paper_id: str
    clarity_of_objectives: float = Field(default=0.0, ge=0.0, le=10.0)
    dataset_transparency: float = Field(default=0.0, ge=0.0, le=10.0)
    methodological_clarity: float = Field(default=0.0, ge=0.0, le=10.0)
    experimental_reproducibility: float = Field(default=0.0, ge=0.0, le=10.0)
    baseline_comparisons: float = Field(default=0.0, ge=0.0, le=10.0)
    metrics_appropriateness: float = Field(default=0.0, ge=0.0, le=10.0)
    statistical_validation: float = Field(default=0.0, ge=0.0, le=10.0)
    limitations_discussion: float = Field(default=0.0, ge=0.0, le=10.0)
    code_data_availability: float = Field(default=0.0, ge=0.0, le=10.0)
    total_quality_score: float = 0.0
    justification: str = ""


class ResearchGap(BaseModel):
    """Identified evidence-based research gap."""
    gap_id: str
    description: str
    supporting_papers: List[str] = Field(default_factory=list)  # List of paper titles or DOIs
    evidence: str
    existing_approaches: str
    limitations_of_approaches: str
    why_gap_matters: str
    confidence_level: str = "Medium"  # High / Medium / Low / Hypothesis
    gap_type: str = "Methodological"  # Methodological / Dataset / Architectural / Evaluation / Generalization
    verification_status: str = "Supported by Literature"


class NovelIdea(BaseModel):
    """Proposed novel research idea based on identified gaps and prior art."""
    idea_id: str
    proposed_title: str
    core_research_problem: str
    motivation: str
    research_gap_addressed: str
    proposed_methodology: str
    possible_architecture: str = "Domain-Specific Neural Architecture"
    potential_datasets: List[str] = Field(default_factory=list)
    experimental_design: str = "Standard benchmark evaluation with cross-validation."
    baseline_models: List[str] = Field(default_factory=list)
    evaluation_metrics: List[str] = Field(default_factory=list)
    expected_contributions: List[str] = Field(default_factory=list)
    technical_feasibility: str = "High"  # High / Medium / Low
    feasibility_score: float = Field(default=8.0, ge=0.0, le=10.0)
    main_risks: str = ""
    closest_existing_studies: List[str] = Field(default_factory=list)
    novelty_verification_status: str = "Preliminary Hypothesis (Prior Art Checked)"


class EvidenceSynthesis(BaseModel):
    """Comparative synthesis across included literature."""
    publication_year_distribution: Dict[str, int] = Field(default_factory=dict)
    frequent_architectures: List[Dict[str, Any]] = Field(default_factory=list)
    frequent_datasets: List[Dict[str, Any]] = Field(default_factory=list)
    dominant_methodologies: List[str] = Field(default_factory=list)
    common_limitations: List[str] = Field(default_factory=list)
    emerging_directions: List[str] = Field(default_factory=list)
    thematic_summary: str = ""


class ResearchGapList(BaseModel):
    """List of identified research gaps."""
    gaps: List[ResearchGap] = Field(default_factory=list)


class NovelIdeaList(BaseModel):
    """List of proposed novel research ideas."""
    ideas: List[NovelIdea] = Field(default_factory=list)

