"""
hsi_research_agent/main.py
Command-Line Interface and Interactive Research Discovery Agent.
Orchestrated by LangGraph with local Ollama Qwen 2.5 14B inference.
Supports autonomous pipeline execution and interactive conversational literature queries.
"""

import argparse
from datetime import datetime
import logging
from pathlib import Path
import re
import sys
from typing import Optional

from hsi_research_agent.config import settings
from hsi_research_agent.graph.workflow import run_research_pipeline
from hsi_research_agent.llm.client import OllamaClient
from hsi_research_agent.storage.database import init_db
from hsi_research_agent.storage.models import PaperRecord, ExtractedInfo
from hsi_research_agent.storage.repository import ResearchRepository


def setup_cli_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - [%(levelname)s] - %(message)s"
    logging.basicConfig(level=level, format=format_str, datefmt="%H:%M:%S")


def print_banner() -> None:
    print("=" * 80)
    print("   AUTONOMOUS ACADEMIC RESEARCH DISCOVERY & LITERATURE AGENT")
    print("   Orchestrator: LangGraph | Local LLM: Ollama (Qwen 2.5 14B)")
    print("   Multi-Source Search: OpenAlex | Semantic Scholar | Crossref | arXiv")
    print("   Master Export: Excel Workbook (8 sheets) | CSVs | Academic Report")
    print("=" * 80)
    print()


def interactive_chat_loop(topic_id: Optional[int] = None) -> None:
    """Interactive conversational interface to query discovered literature using Qwen."""
    repo = ResearchRepository()
    client = OllamaClient()

    print("\n" + "=" * 80)
    print("                INTERACTIVE LITERATURE CHAT & ANALYSIS")
    print("  Ask questions about collected papers, DOIs, methodologies, or gaps.")
    print("  Examples:")
    print("    - 'What were the objectives and findings for DOI: 10.1109/...'")
    print("    - 'Which papers used Vision Transformers for hyperspectral imaging?'")
    print("    - 'Summarize the top research gaps discovered.'")
    print("    - Type 'new topic' to run a fresh literature search.")
    print("    - Type 'exit' or 'quit' to end the session.")
    print("=" * 80 + "\n")

    while True:
        try:
            user_input = input("ResearchAgent> ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting research session. All literature and exports are saved.")
                break

            if user_input.lower() in ("new topic", "search new topic"):
                new_topic = input("Enter new research topic: ").strip()
                if new_topic:
                    run_pipeline_interactive(new_topic)
                continue

            # Check if user mentioned a specific DOI
            doi_match = re.search(r"10\.\d{4,9}/[^\s,\'\"]+", user_input, re.IGNORECASE)
            specific_doi = doi_match.group(0).rstrip(".:;,?") if doi_match else None

            papers = repo.get_all_papers()
            extracted_list = repo.get_all_extracted_info()
            ext_by_id = {e.paper_id: e for e in extracted_list}
            ext_by_doi = {e.doi.lower(): e for e in extracted_list if e.doi}

            relevant_context = []

            # If user mentioned a specific DOI, prioritize its full extracted record
            if specific_doi:
                target_paper = repo.get_paper_by_doi(specific_doi)
                target_ext = repo.get_extracted_info_by_doi(specific_doi) or (
                    ext_by_id.get(target_paper.paper_id) if target_paper else None
                )
                if target_paper or target_ext:
                    title_str = target_ext.title if target_ext else target_paper.title
                    year_str = target_paper.publication_year if target_paper else "N/A"
                    cites_str = target_paper.citation_count if target_paper else "N/A"
                    method_str = target_ext.methodology if target_ext else "Abstract review only"
                    arch_str = target_ext.model_architecture if target_ext else "Not reported"
                    data_str = target_ext.dataset if target_ext else "Not reported"
                    obj_str = target_ext.research_objectives if target_ext else "Not reported"
                    findings_str = target_ext.main_findings if target_ext else "Not reported"
                    limits_str = target_ext.limitations if target_ext else "Not reported"
                    abstract_str = target_paper.abstract if target_paper else ""

                    specific_block = (
                        f"*** TARGET PAPER REQUESTED BY USER ***\n"
                        f"Title: {title_str}\n"
                        f"DOI: {specific_doi} | Year: {year_str} | Citations: {cites_str}\n"
                        f"Research Objectives: {obj_str}\n"
                        f"Methodology: {method_str}\n"
                        f"Model Architecture: {arch_str}\n"
                        f"Dataset: {data_str}\n"
                        f"Main Findings: {findings_str}\n"
                        f"Limitations: {limits_str}\n"
                        f"Abstract Snippet: {abstract_str[:500]}\n"
                    )
                    relevant_context.append(specific_block)

            # Add context for other top literature
            for p in papers[:20]:
                if specific_doi and p.doi and p.doi.lower() == specific_doi.lower():
                    continue  # already added as target paper
                info = ext_by_id.get(p.paper_id) or (ext_by_doi.get(p.doi.lower()) if p.doi else None)
                if info:
                    relevant_context.append(
                        f"Paper: {p.title}\n"
                        f"DOI: {p.doi or 'N/A'} | Year: {p.publication_year or 'N/A'}\n"
                        f"Objectives: {info.research_objectives}\n"
                        f"Methodology: {info.methodology}\n"
                        f"Architecture: {info.model_architecture}\n"
                        f"Dataset: {info.dataset}\n"
                        f"Limitations: {info.limitations}\n"
                        f"Findings: {info.main_findings}\n"
                    )
                else:
                    relevant_context.append(
                        f"Paper: {p.title}\n"
                        f"DOI: {p.doi or 'N/A'} | Year: {p.publication_year or 'N/A'}\n"
                        f"Abstract: {(p.abstract or 'No abstract')[:300]}\n"
                    )

            gaps = repo.get_research_gaps(topic_id or 1)
            gaps_text = "\n".join(f"- {g.gap_id}: {g.description} (Evidence: {g.evidence[:150]})" for g in gaps[:5])

            system_context = (
                "You are an expert AI Research Assistant with access to the collected literature below.\n"
                "Answer the user's question accurately using ONLY this evidence. If a paper's details are requested, "
                "cite its exact title and DOI. Never fabricate findings.\n\n"
                f"COLLECTED LITERATURE CONTEXT:\n{'---'.join(relevant_context[:10])}\n\n"
                f"DISCOVERED RESEARCH GAPS:\n{gaps_text}\n"
            )

            prompt = f"{system_context}\n\nUser Question: {user_input}\n\nAnswer:"
            print("\nAnalyzing literature...")
            response = client.generate_text(prompt, temperature=0.3)
            print(f"\n{response}\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"Error processing query: {e}")


def run_pipeline_interactive(topic_text: str) -> None:
    """Helper to run pipeline from interactive prompt."""
    print(f"\nStarting autonomous literature discovery for: '{topic_text}'...")
    state = run_research_pipeline(
        topic=topic_text,
        target_domain="Hyperspectral Remote Sensing / Machine Learning",
        start_year=2018,
        thread_id=f"run_{int(datetime.now().timestamp())}",
    )
    print_results_summary(state)


def print_results_summary(state: dict) -> None:
    """Print clean summary table of the pipeline execution."""
    print("\n" + "=" * 80)
    print("                     LITERATURE DISCOVERY RESULTS")
    print("=" * 80)
    scope = state.get("research_scope", {})
    papers = state.get("unique_papers", [])
    queries = state.get("validated_queries", [])
    screening = state.get("screening_results", [])
    gaps = state.get("research_gaps", [])
    ideas = state.get("novel_ideas", [])
    exports = state.get("export_paths", {})

    print(f"Topic:                 {state.get('research_topic')}")
    print(f"Target Domain:         {scope.get('target_domain', 'N/A')}")
    print(f"Search Queries:        {len(queries)} multi-dimensional queries (all validated)")
    print(f"Unique Papers:         {len(papers)} papers retrieved & deduplicated")
    
    included = sum(1 for s in screening if s.get("decision") == "Include")
    maybe = sum(1 for s in screening if s.get("decision") == "Maybe")
    excluded = sum(1 for s in screening if s.get("decision") == "Exclude")
    print(f"Screening Decisions:   {included} Included | {maybe} Under Review | {excluded} Excluded")
    print(f"Identified Gaps:       {len(gaps)} evidence-based gaps formulated")
    print(f"Novel Research Ideas:  {len(ideas)} feasible ideas proposed")

    print("\n--- Identified Evidence-Based Research Gaps ---")
    for g in gaps[:4]:
        print(f"  [{g.get('gap_id')}] {g.get('description')}")
        print(f"         Confidence: {g.get('confidence_level')} | Verification: {g.get('verification_status')}")

    print("\n--- Proposed Novel Research Ideas ---")
    for idea in ideas[:3]:
        print(f"  [{idea.get('idea_id')}] {idea.get('proposed_title')}")
        print(f"         Feasibility: {idea.get('feasibility_score')}/10.0 | Architecture: {idea.get('possible_architecture')}")

    print("\n--- Exported Artefacts ---")
    for key, path in exports.items():
        print(f"  {key.capitalize():10s}: {path}")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Academic Research Discovery & Literature Agent (LangGraph + Ollama Qwen 2.5)"
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default="Hyperspectral Foundation Models in Agriculture",
        help="Research topic or search problem statement",
    )
    parser.add_argument(
        "--domain", "-d",
        type=str,
        default="Hyperspectral Remote Sensing / Computer Vision",
        help="Academic target domain",
    )
    parser.add_argument(
        "--start-year", "-s",
        type=int,
        default=2018,
        help="Start year filter for literature discovery (e.g. 2018)",
    )
    parser.add_argument(
        "--end-year", "-e",
        type=int,
        default=None,
        help="Optional end year filter (e.g. 2026)",
    )
    parser.add_argument(
        "--objectives", "-o",
        type=str,
        default=None,
        help="User-defined research objectives or specific focus areas",
    )
    parser.add_argument(
        "--chat", "-c",
        action="store_true",
        help="Launch directly into interactive chat mode to query collected literature",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive conversational session after literature discovery completes",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="openalex,crossref,arxiv,semanticscholar",
        help="Comma-separated academic databases (openalex, crossref, arxiv, semanticscholar)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable detailed debug logging",
    )

    args = parser.parse_args()
    setup_cli_logging(args.verbose)
    print_banner()

    # Initialize database
    init_db()

    if args.chat:
        interactive_chat_loop()
        return

    # Check local Ollama connectivity
    print(f"Connecting to Ollama at {settings.OLLAMA_BASE_URL} (Model: {settings.OLLAMA_MODEL})...")
    client = OllamaClient()
    try:
        client.check_connection()
        print("✓ Connected to Ollama server.")
    except Exception as e:
        print(f"! Notice: {e}")
        print("  Workflow will utilize robust heuristic fallbacks if LLM inference is unreachable.")

    # Interactive Wizard if run from terminal with no CLI flags
    topic = args.topic
    start_year = args.start_year
    end_year = args.end_year
    sources_str = args.sources
    interactive_mode = args.interactive

    if len(sys.argv) == 1:
        print("\n" + "=" * 80)
        print("                 RESEARCH AGENT INTERACTIVE SETUP")
        print("=" * 80)
        print("How would you like to start?")
        print("  [1] Search academic literature & enter chat (Recommended)")
        print("  [2] Chat with already collected papers (Q&A by DOI / Topic)")
        print("  [3] Search & Export only (no chat)")
        choice = input("\nSelect an option [1-3, default 1]: ").strip()

        if choice == "2":
            interactive_chat_loop()
            return
        elif choice == "3":
            interactive_mode = False
        else:
            interactive_mode = True

        print("\n--- Search Parameters ---")
        user_topic = input(f"1. Enter research topic [press Enter for '{topic}']: ").strip()
        if user_topic:
            topic = user_topic

        user_start = input(f"2. Enter start year [default {start_year}]: ").strip()
        if user_start:
            try:
                start_year = int(user_start)
            except ValueError:
                pass

        user_end = input(f"3. Enter end year [default {end_year or 2026}]: ").strip()
        if user_end:
            try:
                end_year = int(user_end)
            except ValueError:
                pass
        else:
            end_year = 2026

        user_sources = input(f"4. Enter databases [openalex, crossref, arxiv] (press Enter for fast openalex,crossref): ").strip()
        if user_sources:
            sources_str = user_sources
        else:
            sources_str = "openalex,crossref"

    # Parse sources
    enabled_sources = [s.strip().lower() for s in sources_str.split(",") if s.strip()]

    # Execute Autonomous Research Pipeline
    thread_id = f"research_run_{int(datetime.now().timestamp())}"
    print(f"\nLaunching Autonomous Research Workflow...")
    print(f"Topic:      {topic}")
    print(f"Domain:     {args.domain}")
    print(f"Years:      {start_year} - {end_year or 'Present'}")
    print(f"Sources:    {', '.join(enabled_sources)}")
    print(f"Thread ID:  {thread_id}\n")

    final_state = run_research_pipeline(
        topic=topic,
        target_domain=args.domain,
        start_year=start_year,
        end_year=end_year,
        user_objectives=args.objectives,
        sources=enabled_sources,
        thread_id=thread_id,
        full_pipeline=True,
    )

    print_results_summary(final_state)

    if interactive_mode:
        topic_id = final_state.get("topic_id")
        interactive_chat_loop(topic_id)


if __name__ == "__main__":
    main()
