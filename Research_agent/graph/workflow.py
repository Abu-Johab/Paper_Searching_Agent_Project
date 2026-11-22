"""
hsi_research_agent/graph/workflow.py
LangGraph StateGraph builder for the autonomous literature discovery pipeline.
Orchestrates all 14 stages from topic analysis to multi-source search, screening,
full-text retrieval, quality assessment, gap analysis, idea generation, and master export.
"""

import logging
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, START, END

from hsi_research_agent.graph.checkpoints import get_default_checkpointer
from hsi_research_agent.graph.nodes import (
    node_deduplicator,
    node_export,
    node_extraction,
    node_novel_ideas,
    node_pdf_retrieval,
    node_quality_assessment,
    node_query_generator,
    node_query_validator,
    node_question_generator,
    node_research_gaps,
    node_screening,
    node_search_manager,
    node_synthesis,
    node_topic_analyzer,
)
from hsi_research_agent.graph.state import AgentState
from hsi_research_agent.storage.database import init_db

logger = logging.getLogger("hsi_research_agent.graph.workflow")


def build_research_graph(
    checkpointer: Optional[Any] = None,
    full_pipeline: bool = True,
    end_node: Optional[str] = None,
) -> Any:
    """
    Construct the stateful LangGraph workflow for autonomous research discovery.
    
    If full_pipeline=True (default), builds all 14 stages:
    START -> topic_analyzer -> question_generator -> query_generator -> query_validator
          -> search_manager -> deduplicator -> screening -> pdf_retrieval -> extraction
          -> quality_assessment -> research_gaps -> synthesis -> novel_ideas -> export -> END
    
    If full_pipeline=False or end_node is specified, builds up to the designated end_node.
    """
    init_db()

    graph = StateGraph(AgentState)

    # 1. Register Nodes
    graph.add_node("topic_analyzer", node_topic_analyzer)
    graph.add_node("question_generator", node_question_generator)
    graph.add_node("query_generator", node_query_generator)
    graph.add_node("query_validator", node_query_validator)
    graph.add_node("search_manager", node_search_manager)

    if full_pipeline and end_node != "search_manager":
        graph.add_node("deduplicator", node_deduplicator)
        graph.add_node("screening", node_screening)
        graph.add_node("pdf_retrieval", node_pdf_retrieval)
        graph.add_node("extraction", node_extraction)
        graph.add_node("quality_assessment", node_quality_assessment)
        graph.add_node("research_gaps", node_research_gaps)
        graph.add_node("synthesis", node_synthesis)
        graph.add_node("novel_ideas", node_novel_ideas)
        graph.add_node("export", node_export)

    # 2. Add Edges
    graph.add_edge(START, "topic_analyzer")
    graph.add_edge("topic_analyzer", "question_generator")
    graph.add_edge("question_generator", "query_generator")
    graph.add_edge("query_generator", "query_validator")
    graph.add_edge("query_validator", "search_manager")

    if not full_pipeline or end_node == "search_manager":
        graph.add_edge("search_manager", END)
    else:
        graph.add_edge("search_manager", "deduplicator")
        graph.add_edge("deduplicator", "screening")
        graph.add_edge("screening", "pdf_retrieval")
        graph.add_edge("pdf_retrieval", "extraction")
        graph.add_edge("extraction", "quality_assessment")
        graph.add_edge("quality_assessment", "research_gaps")
        graph.add_edge("research_gaps", "synthesis")
        graph.add_edge("synthesis", "novel_ideas")
        graph.add_edge("novel_ideas", "export")
        graph.add_edge("export", END)

    # 3. Compile Workflow with Checkpointing
    chk = checkpointer or get_default_checkpointer()
    compiled = graph.compile(checkpointer=chk)
    logger.info("Compiled LangGraph research discovery workflow successfully (full_pipeline=%s).", full_pipeline)
    return compiled


def run_research_pipeline(
    topic: str,
    target_domain: str = "Machine Learning / Computer Vision",
    start_year: int = 2018,
    end_year: Optional[int] = None,
    user_objectives: Optional[str] = None,
    sources: Optional[List[str]] = None,
    thread_id: str = "research_run_001",
    full_pipeline: bool = True,
) -> AgentState:
    """
    Execute the research discovery workflow for a given topic.
    """
    compiled = build_research_graph(full_pipeline=full_pipeline)

    initial_state: AgentState = {
        "research_topic": topic,
        "target_domain": target_domain,
        "start_year": start_year,
        "end_year": end_year,
        "user_objectives": user_objectives,
        "sources": sources,
        "current_stage": "Initialized",
        "execution_errors": [],
    }

    config = {"configurable": {"thread_id": thread_id}}
    result_state = compiled.invoke(initial_state, config=config)
    return result_state
