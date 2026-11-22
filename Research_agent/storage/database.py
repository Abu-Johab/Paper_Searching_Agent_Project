"""
hsi_research_agent/storage/database.py
SQLite database management, schema migration, and transaction management.
"""

import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from hsi_research_agent.config import settings

logger = logging.getLogger("hsi_research_agent.database")

SCHEMA_SQL = """
-- 1. Research Topics
CREATE TABLE IF NOT EXISTS research_topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_text TEXT NOT NULL,
    target_domain TEXT NOT NULL,
    start_year INTEGER NOT NULL,
    end_year INTEGER,
    core_concepts TEXT,
    related_terminology TEXT,
    research_dimensions TEXT,
    relevant_datasets TEXT,
    important_methodologies TEXT,
    user_objectives TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Research Questions
CREATE TABLE IF NOT EXISTS research_questions (
    question_id TEXT PRIMARY KEY,
    topic_id INTEGER,
    question_text TEXT NOT NULL,
    framework TEXT NOT NULL,
    research_aspect TEXT NOT NULL,
    rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(topic_id) REFERENCES research_topics(id)
);

-- 3. Exactly 10 Search Queries
CREATE TABLE IF NOT EXISTS search_queries (
    query_id TEXT PRIMARY KEY,
    topic_id INTEGER,
    query_text TEXT NOT NULL,
    research_dimension TEXT NOT NULL,
    search_objective TEXT NOT NULL,
    keywords TEXT,
    expected_paper_types TEXT,
    priority INTEGER DEFAULT 1,
    related_question_id TEXT,
    validation_score REAL DEFAULT 0.0,
    validation_status TEXT DEFAULT 'Pending',
    validation_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(topic_id) REFERENCES research_topics(id)
);

-- 4. Search Audit Logs
CREATE TABLE IF NOT EXISTS search_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    source_database TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    records_retrieved INTEGER DEFAULT 0,
    status TEXT NOT NULL,
    error_details TEXT
);

-- 5. Canonical Research Papers
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT,
    publication_year INTEGER,
    abstract TEXT,
    doi TEXT UNIQUE,
    arxiv_id TEXT,
    pmid TEXT,
    publisher TEXT,
    journal_or_conference TEXT,
    publication_type TEXT,
    publication_status TEXT,
    keywords TEXT,
    paper_url TEXT,
    pdf_url TEXT,
    source_database TEXT NOT NULL,
    retrieved_query_ids TEXT,
    citation_count INTEGER DEFAULT 0,
    open_access_status TEXT DEFAULT 'Unknown',
    full_text_status TEXT DEFAULT 'Abstract only',
    full_text TEXT,
    relevance_score REAL DEFAULT 0.0,
    relevance_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexing for rapid queries
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(publication_year);
CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source_database);

-- 6. Screening Results
CREATE TABLE IF NOT EXISTS screening_results (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    decision TEXT NOT NULL,
    relevance_score REAL DEFAULT 0.0,
    inclusion_reason TEXT,
    exclusion_reason TEXT,
    screening_stage TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
);

-- 7. Structured Information Extraction
CREATE TABLE IF NOT EXISTS extracted_info (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    doi TEXT,
    research_problem TEXT,
    research_objectives TEXT,
    methodology TEXT,
    model_architecture TEXT,
    pretraining_strategy TEXT,
    fine_tuning_strategy TEXT,
    dataset TEXT,
    experimental_setup TEXT,
    evaluation_metrics TEXT,
    main_findings TEXT,
    limitations TEXT,
    future_directions TEXT,
    code_availability TEXT,
    dataset_availability TEXT,
    hsi_sensor_characteristics TEXT,
    hsi_spectral_bands TEXT,
    hsi_spatial_resolution TEXT,
    hsi_spectral_spatial_strategy TEXT,
    hsi_downstream_task TEXT,
    cross_scene_evaluation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
);

-- 8. Quality Assessments
CREATE TABLE IF NOT EXISTS quality_assessments (
    paper_id TEXT PRIMARY KEY,
    clarity_of_objectives REAL,
    dataset_transparency REAL,
    methodological_clarity REAL,
    experimental_reproducibility REAL,
    baseline_comparisons REAL,
    metrics_appropriateness REAL,
    statistical_validation REAL,
    limitations_discussion REAL,
    code_data_availability REAL,
    total_quality_score REAL NOT NULL,
    justification TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
);

-- 9. Evidence-Based Research Gaps
CREATE TABLE IF NOT EXISTS research_gaps (
    gap_id TEXT PRIMARY KEY,
    topic_id INTEGER,
    description TEXT NOT NULL,
    supporting_papers TEXT,
    evidence TEXT NOT NULL,
    existing_approaches TEXT,
    limitations_of_approaches TEXT,
    why_gap_matters TEXT NOT NULL,
    confidence_level TEXT DEFAULT 'Medium',
    gap_type TEXT DEFAULT 'Methodological',
    verification_status TEXT DEFAULT 'Supported by Literature',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(topic_id) REFERENCES research_topics(id)
);

-- 10. Novel Research Ideas
CREATE TABLE IF NOT EXISTS novel_ideas (
    idea_id TEXT PRIMARY KEY,
    topic_id INTEGER,
    gap_id TEXT,
    proposed_title TEXT NOT NULL,
    core_research_problem TEXT NOT NULL,
    motivation TEXT NOT NULL,
    research_gap_addressed TEXT NOT NULL,
    proposed_methodology TEXT NOT NULL,
    possible_architecture TEXT,
    potential_datasets TEXT,
    experimental_design TEXT,
    baseline_models TEXT,
    evaluation_metrics TEXT,
    expected_contributions TEXT,
    technical_feasibility TEXT,
    feasibility_score REAL DEFAULT 8.0,
    main_risks TEXT,
    closest_existing_studies TEXT,
    novelty_verification_status TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(topic_id) REFERENCES research_topics(id)
);

-- 11. Workflow Checkpoints
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    stage_name TEXT NOT NULL,
    state_json TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize the SQLite database schema with indexes and pragmas."""
    target_path = Path(db_path or settings.DB_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    with sqlite3.connect(str(target_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    logger.info("Initialized database successfully at: %s", target_path)


@contextmanager
def get_db_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Provide a transactional SQLite connection with Row factory."""
    target_path = Path(db_path or settings.DB_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(target_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("Database transaction rolled back due to error: %s", str(e))
        raise
    finally:
        conn.close()
