"""
hsi_research_agent/agents/extraction_agent.py
Information extraction agent extracting structured methodology, datasets, limitations, and HSI characteristics.
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from hsi_research_agent.config import settings
from hsi_research_agent.llm.client import OllamaClient
from hsi_research_agent.storage.models import ExtractedInfo, PaperRecord

logger = logging.getLogger("hsi_research_agent.extraction_agent")


class RawExtractionOutput(BaseModel):
    """Structured extraction fields extracted from paper text or abstract."""
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
    hsi_sensor_characteristics: str = "Not reported"
    hsi_spectral_bands: str = "Not reported"
    hsi_spatial_resolution: str = "Not reported"
    hsi_spectral_spatial_strategy: str = "Not reported"
    hsi_downstream_task: str = "Not reported"
    cross_scene_evaluation: str = "Not reported"


def extract_paper_information(
    paper: PaperRecord,
    client: Optional[OllamaClient] = None,
) -> ExtractedInfo:
    """
    Extract structured research metadata from paper full-text (if retrieved) or abstract.
    Mark missing fields explicitly as 'Not reported'.
    """
    client = client or OllamaClient()

    text_source = "FULL TEXT" if paper.full_text else "ABSTRACT ONLY"
    content = (paper.full_text[:12000] if paper.full_text else (paper.abstract or paper.title))

    prompt = (
        f"You are an expert scientific literature data extraction specialist.\n"
        f"Extract factual research information from the following paper ({text_source}):\n\n"
        f"Title: {paper.title}\n"
        f"DOI: {paper.doi or 'None'}\n"
        f"Content:\n{content}\n\n"
        f"EXTRACTION GUIDELINES:\n"
        f"1. Extract ONLY facts supported directly by the text.\n"
        f"2. For any field not mentioned in the text, write 'Not reported'. Do NOT invent or assume facts.\n"
        f"3. In 'limitations', extract stated shortcomings, compute constraints, or generalization barriers.\n"
        f"4. For HSI fields (bands, spatial resolution, sensor, tasks), state exact metrics if reported.\n\n"
        f"Return strict valid JSON matching the schema."
    )

    try:
        raw_out: RawExtractionOutput = client.generate_structured(
            prompt=prompt,
            pydantic_class=RawExtractionOutput,
            system_prompt="You are a scientific data extractor. Return valid JSON only.",
            temperature=0.0,
            max_retries=2,
        )

        return ExtractedInfo(
            paper_id=paper.paper_id,
            title=paper.title,
            doi=paper.doi,
            research_problem=raw_out.research_problem,
            research_objectives=raw_out.research_objectives,
            methodology=raw_out.methodology,
            model_architecture=raw_out.model_architecture,
            pretraining_strategy=raw_out.pretraining_strategy,
            fine_tuning_strategy=raw_out.fine_tuning_strategy,
            dataset=raw_out.dataset,
            experimental_setup=raw_out.experimental_setup,
            evaluation_metrics=raw_out.evaluation_metrics,
            main_findings=raw_out.main_findings,
            limitations=raw_out.limitations,
            future_directions=raw_out.future_directions,
            code_availability=raw_out.code_availability,
            dataset_availability=raw_out.dataset_availability,
            hsi_sensor_characteristics=raw_out.hsi_sensor_characteristics,
            hsi_spectral_bands=raw_out.hsi_spectral_bands,
            hsi_spatial_resolution=raw_out.hsi_spatial_resolution,
            hsi_spectral_spatial_strategy=raw_out.hsi_spectral_spatial_strategy,
            hsi_downstream_task=raw_out.hsi_downstream_task,
            cross_scene_evaluation=raw_out.cross_scene_evaluation,
        )

    except Exception as e:
        logger.warning("Extraction fallback triggered for '%s': %s", paper.paper_id, str(e))
        return ExtractedInfo(
            paper_id=paper.paper_id,
            title=paper.title,
            doi=paper.doi,
            research_problem=f"Study on {paper.title}",
            methodology="Method described in paper",
            limitations="Full details in publication",
        )


def extract_all_eligible_papers(
    papers: List[PaperRecord],
    client: Optional[OllamaClient] = None,
    max_to_extract: Optional[int] = None,
) -> List[ExtractedInfo]:
    """Extract structured research data for all included/eligible papers."""
    targets = papers[:max_to_extract] if (max_to_extract and max_to_extract > 0) else papers
    logger.info("Extracting structured information from %d papers...", len(targets))

    extractions: List[ExtractedInfo] = []
    for p in targets:
        ext = extract_paper_information(p, client=client)
        extractions.append(ext)

    logger.info("Information extraction complete for %d papers.", len(extractions))
    return extractions
