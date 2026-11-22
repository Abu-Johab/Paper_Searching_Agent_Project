"""
hsi_research_agent/search/search_manager.py
Multi-source academic search coordinator executing all 10 queries across literature databases.
"""

import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

from hsi_research_agent.processing.normalizer import normalize_record
from hsi_research_agent.search.arxiv import search_arxiv
from hsi_research_agent.search.crossref import search_crossref
from hsi_research_agent.search.openalex import search_openalex
from hsi_research_agent.search.semantic_scholar import search_semantic_scholar
from hsi_research_agent.storage.models import PaperRecord, SearchLog, SearchQuery
from hsi_research_agent.storage.repository import ResearchRepository

logger = logging.getLogger("hsi_research_agent.search.manager")


class SearchManager:
    """Orchestrates multi-database literature discovery across all 10 validated queries."""

    def __init__(
        self,
        repository: Optional[ResearchRepository] = None,
        sources: Optional[List[str]] = None,
    ):
        self.repository = repository
        self.sources = sources or ["openalex", "crossref", "arxiv", "semanticscholar"]

    def execute_all_searches(
        self,
        queries: List[SearchQuery],
        start_year: Optional[int] = 2018,
        end_year: Optional[int] = None,
        max_results_per_query: int = 50,
    ) -> Tuple[List[PaperRecord], List[SearchLog]]:
        """
        Execute all 10 queries independently across enabled academic sources.
        Records audit logs and returns normalized paper records.
        """
        all_papers: List[PaperRecord] = []
        all_logs: List[SearchLog] = []

        logger.info(
            "Executing %d search queries across sources: %s",
            len(queries),
            ", ".join(self.sources),
        )

        for q in queries:
            query_id = q.query_id
            query_text = q.query_text
            logger.info("Executing [%s]: '%s'", query_id, query_text)

            # 1. OpenAlex
            if "openalex" in self.sources:
                t_start = datetime.datetime.now().isoformat()
                records: List[Dict[str, Any]] = []
                status = "Success"
                error_msg = None
                try:
                    records = search_openalex(
                        query=query_text,
                        start_year=start_year,
                        end_year=end_year,
                        max_results=max_results_per_query,
                    )
                except Exception as e:
                    status = "Error"
                    error_msg = str(e)
                    logger.warning("OpenAlex search error on %s: %s", query_id, error_msg)

                for r in records:
                    all_papers.append(normalize_record(r, "openalex", query_id))

                log = SearchLog(
                    query_id=query_id,
                    query_text=query_text,
                    source_database="openalex",
                    timestamp=t_start,
                    records_retrieved=len(records),
                    status=status,
                    error_details=error_msg,
                )
                all_logs.append(log)
                if self.repository:
                    self.repository.log_search(log)

            # 2. Crossref
            if "crossref" in self.sources:
                t_start = datetime.datetime.now().isoformat()
                records = []
                status = "Success"
                error_msg = None
                try:
                    records = search_crossref(
                        query=query_text,
                        start_year=start_year,
                        end_year=end_year,
                        max_results=max_results_per_query,
                    )
                except Exception as e:
                    status = "Error"
                    error_msg = str(e)
                    logger.warning("Crossref search error on %s: %s", query_id, error_msg)

                for r in records:
                    all_papers.append(normalize_record(r, "crossref", query_id))

                log = SearchLog(
                    query_id=query_id,
                    query_text=query_text,
                    source_database="crossref",
                    timestamp=t_start,
                    records_retrieved=len(records),
                    status=status,
                    error_details=error_msg,
                )
                all_logs.append(log)
                if self.repository:
                    self.repository.log_search(log)

            # 3. arXiv
            if "arxiv" in self.sources:
                t_start = datetime.datetime.now().isoformat()
                records = []
                status = "Success"
                error_msg = None
                try:
                    records = search_arxiv(
                        query=query_text,
                        start_year=start_year,
                        end_year=end_year,
                        max_results=max_results_per_query,
                    )
                except Exception as e:
                    status = "Error"
                    error_msg = str(e)
                    logger.warning("arXiv search error on %s: %s", query_id, error_msg)

                for r in records:
                    all_papers.append(normalize_record(r, "arxiv", query_id))

                log = SearchLog(
                    query_id=query_id,
                    query_text=query_text,
                    source_database="arxiv",
                    timestamp=t_start,
                    records_retrieved=len(records),
                    status=status,
                    error_details=error_msg,
                )
                all_logs.append(log)
                if self.repository:
                    self.repository.log_search(log)

            # 4. Semantic Scholar
            if "semanticscholar" in self.sources:
                t_start = datetime.datetime.now().isoformat()
                records = []
                status = "Success"
                error_msg = None
                try:
                    records = search_semantic_scholar(
                        query=query_text,
                        start_year=start_year,
                        end_year=end_year,
                        max_results=max_results_per_query,
                    )
                except Exception as e:
                    status = "Error"
                    error_msg = str(e)
                    logger.warning("Semantic Scholar search error on %s: %s", query_id, error_msg)

                for r in records:
                    all_papers.append(normalize_record(r, "semanticscholar", query_id))

                log = SearchLog(
                    query_id=query_id,
                    query_text=query_text,
                    source_database="semanticscholar",
                    timestamp=t_start,
                    records_retrieved=len(records),
                    status=status,
                    error_details=error_msg,
                )
                all_logs.append(log)
                if self.repository:
                    self.repository.log_search(log)

        logger.info(
            "Search complete across all %d queries. Total raw records gathered: %d",
            len(queries),
            len(all_papers),
        )
        return all_papers, all_logs
