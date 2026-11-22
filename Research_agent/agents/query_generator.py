"""
hsi_research_agent/agents/query_generator.py
Generates EXACTLY 10 diverse, high-precision academic search queries spanning distinct research dimensions.
"""

import json
import logging
import re
from typing import List, Optional

from hsi_research_agent.config import settings
from hsi_research_agent.llm.client import OllamaClient
from hsi_research_agent.storage.models import (
    QueryList,
    ResearchQuestion,
    ResearchTopic,
    SearchQuery,
)

logger = logging.getLogger("hsi_research_agent.query_generator")

STANDARD_HSI_DIMENSIONS = [
    ("Core Architectures & Spatial-Spectral Backbones", "Identify primary vision backbone architectures and spatial-spectral tokenizers."),
    ("Self-Supervised Pretraining & Masked Autoencoders", "Investigate self-supervised objectives (MAE, contrastive learning) on spectral cubes."),
    ("Cross-Sensor & Cross-Platform Generalization", "Assess adaptation across diverse sensors (AVIRIS, PRISMA, EnMAP, Hyperion)."),
    ("Few-Shot, Zero-Shot & Small-Sample Learning", "Examine model behavior under limited labeled spectral ground truth."),
    ("Spatial-Spectral Attention & Continuous Tokenizers", "Analyze wavelength-conditioned and continuous tokenization mechanisms."),
    ("Multimodal Fusion (HSI + LiDAR / SAR / MSI)", "Explore cross-modal fusion architectures integrating complementary remote sensing data."),
    ("Downstream Land-Cover Classification Benchmarks", "Survey classification accuracy and benchmarks (Indian Pines, Pavia, Houston)."),
    ("Downstream Spectral Unmixing & Abundance Estimation", "Investigate sub-pixel spectral unmixing and physical constraint integration."),
    ("Parameter-Efficient Fine-Tuning (PEFT / LoRA)", "Analyze lightweight adaptation methods for spectral foundation models."),
    ("Computational Efficiency & Edge Deployment Constraints", "Examine inference latency, model compression, and airborne/satellite deployment."),
]


def generate_search_queries(
    topic: ResearchTopic,
    questions: List[ResearchQuestion],
    client: Optional[OllamaClient] = None,
) -> List[SearchQuery]:
    """
    Generate EXACTLY 10 diverse academic search queries covering distinct research dimensions.
    """
    client = client or OllamaClient()
    logger.info("Generating exactly 10 academic search queries for '%s'", topic.topic_text)

    prompt_path = settings.BASE_DIR / "prompts" / "query_generation.txt"
    if prompt_path.exists():
        template = prompt_path.read_text(encoding="utf-8")
    else:
        template = "Generate exactly 10 search queries for: {topic_text}."

    questions_summary = "\n".join(
        [f"- {q.question_id}: {q.question_text}" for q in questions]
    )

    prompt = template.format(
        topic_text=topic.topic_text,
        target_domain=topic.target_domain,
        core_concepts=", ".join(topic.core_concepts[:8]),
        research_questions=questions_summary,
        research_dimensions=", ".join(topic.research_dimensions[:10]),
    )

    queries: List[SearchQuery] = []
    try:
        output: QueryList = client.generate_structured(
            prompt=prompt,
            pydantic_class=QueryList,
            system_prompt=(
                "You are an expert scientific search specialist. Return valid JSON containing "
                "EXACTLY 10 search queries (Q01 to Q10) matching the schema."
            ),
            temperature=0.1,
            max_retries=3,
        )
        queries = output.queries
    except Exception as e:
        logger.warning("LLM query generation fallback triggered: %s", str(e))

    # Guarantee exactly 10 queries even if model returned a different count
    cleaned_queries: List[SearchQuery] = []
    seen_texts = set()

    for idx, q in enumerate(queries, 1):
        q_text = q.query_text.strip()
        if q_text and q_text.lower() not in seen_texts and len(cleaned_queries) < 10:
            seen_texts.add(q_text.lower())
            cleaned_queries.append(
                SearchQuery(
                    query_id=f"Q{len(cleaned_queries)+1:02d}",
                    query_text=q_text,
                    research_dimension=q.research_dimension or STANDARD_HSI_DIMENSIONS[len(cleaned_queries)][0],
                    search_objective=q.search_objective or STANDARD_HSI_DIMENSIONS[len(cleaned_queries)][1],
                    keywords=q.keywords or [topic.topic_text],
                    expected_paper_types=q.expected_paper_types or ["Journal Article", "Conference Paper"],
                    priority=q.priority or 1,
                    related_question_id=q.related_question_id or (questions[0].question_id if questions else "RQ1"),
                )
            )

    # Supplement if fewer than 10
    while len(cleaned_queries) < 10:
        slot = len(cleaned_queries)
        dim_title, dim_obj = STANDARD_HSI_DIMENSIONS[slot]
        q_id = f"Q{slot+1:02d}"
        
        # Craft dimension-targeted query text
        if "pretrain" in dim_title.lower():
            q_text = f"{topic.topic_text} masked autoencoder self-supervised pretraining"
        elif "cross-sensor" in dim_title.lower():
            q_text = f"{topic.topic_text} cross-sensor generalization transfer"
        elif "few-shot" in dim_title.lower():
            q_text = f"{topic.topic_text} few-shot zero-shot classification"
        elif "multimodal" in dim_title.lower():
            q_text = f"{topic.topic_text} multimodal fusion LiDAR SAR"
        elif "unmixing" in dim_title.lower():
            q_text = f"{topic.topic_text} spectral unmixing abundance estimation"
        elif "parameter-efficient" in dim_title.lower() or "lora" in dim_title.lower():
            q_text = f"{topic.topic_text} parameter-efficient fine-tuning LoRA adapter"
        elif "efficiency" in dim_title.lower():
            q_text = f"{topic.topic_text} computational efficiency lightweight deployment"
        elif "attention" in dim_title.lower():
            q_text = f"{topic.topic_text} spatial spectral attention mechanism"
        elif "classification" in dim_title.lower():
            q_text = f"{topic.topic_text} land cover classification benchmark dataset"
        else:
            q_text = f"{topic.topic_text} foundation model architecture survey"

        cleaned_queries.append(
            SearchQuery(
                query_id=q_id,
                query_text=q_text,
                research_dimension=dim_title,
                search_objective=dim_obj,
                keywords=[topic.topic_text, dim_title.split()[0]],
                expected_paper_types=["Journal Article", "Conference Proceedings"],
                priority=1,
                related_question_id=questions[slot % len(questions)].question_id if questions else "RQ1",
            )
        )

    logger.info("Successfully assembled exactly 10 search queries (Q01 to Q10).")
    return cleaned_queries[:10]
