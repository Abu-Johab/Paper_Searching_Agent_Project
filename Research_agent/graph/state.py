"""
hsi_research_agent/graph/state.py
Centralized typed state definition for the LangGraph Autonomous Research Agent.
(PRISMA-free, clean academic discovery state).
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    Centralized typed state dictionary carried through the LangGraph StateGraph.
    Tracks all inputs, intermediate artifacts, screened papers, extracted data,
    discovered gaps, and generated novel research ideas.
    """
    # 1. User inputs & Scope
    research_topic: str
    user_objectives: Optional[str]
    start_year: int
    end_year: Optional[int]
    target_domain: str
    topic_id: Optional[int]
    research_scope: Dict[str, Any]
    sources: Optional[List[str]]
    
    # 2. Formulated Research Questions
    research_questions: List[Dict[str, Any]]
    
    # 3. 10 Academic Search Queries & Validation
    generated_queries: List[Dict[str, Any]]
    validated_queries: List[Dict[str, Any]]
    
    # 4. Multi-Source Retrieval
    search_results: List[Dict[str, Any]]
    search_logs: List[Dict[str, Any]]
    
    # 5. Normalization & Deduplication
    normalized_papers: List[Dict[str, Any]]
    duplicate_groups: List[Dict[str, Any]]
    unique_papers: List[Dict[str, Any]]
    
    # 6. Screening
    screening_results: List[Dict[str, Any]]
    
    # 7. Full-Text Retrieval & Content Extraction
    retrieved_full_texts: List[Dict[str, Any]]
    extracted_information: List[Dict[str, Any]]
    
    # 8. Methodological Quality Assessment
    quality_assessments: List[Dict[str, Any]]
    
    # 9. Research Intelligence (Gaps & Novel Ideas)
    research_gaps: List[Dict[str, Any]]
    novel_ideas: List[Dict[str, Any]]
    evidence_synthesis: Dict[str, Any]
    
    # 10. Execution Tracing & Export Artefacts
    export_paths: Dict[str, str]
    execution_errors: List[str]
    current_stage: str
