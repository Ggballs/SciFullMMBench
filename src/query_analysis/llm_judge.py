"""LLM-as-a-judge metrics for academic search queries."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


SEMANTIC_CONSTRAINT_TYPES = {
    "task_or_setting",
    "method_or_model",
    "dataset_or_benchmark",
    "comparison_or_result",
    "scope_or_exclusion",
    "visual_or_multimodal",
    "other",
}


def load_llm_config(config_path: Path) -> Dict[str, str]:
    """Load OpenAI-compatible LLM settings from config.yaml."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    llm_config = config.get("llm", {})
    result = {
        "base_url": llm_config.get("base_url", ""),
        "api_token": llm_config.get("api_token", ""),
        "model": llm_config.get("model", "gpt-4o-mini"),
    }
    if not result["base_url"] or not result["api_token"]:
        raise ValueError(
            f"Missing llm.base_url or llm.api_token in {config_path}; "
            "LLM-as-a-judge metrics require a configured API."
        )
    return result


def build_query_judge_prompt(queries: List[str]) -> str:
    """Build a batch prompt for qualitative and semantic constraint scoring."""
    numbered_queries = "\n".join(
        f"{idx}. {query}" for idx, query in enumerate(queries, start=1)
    )
    return f"""
You are an expert judge for academic paper-search queries.

Evaluate each query as a researcher-facing search query. Score conservatively.

Definitions:
- specificity_score: 0.0 to 1.0. Higher means the query has enough technical detail
  to retrieve a focused set of papers without being vague.
- naturalness_score: 0.0 to 1.0. Higher means the query sounds like something a real
  researcher would type into an academic search engine.
- academic_tone_score: 0.0 to 1.0. Higher means the query is precise and research-like
  without being overly formal or generated.
- semantic_constraint_count: count distinct retrieval-narrowing conditions
  semantically, not by keywords. Use 0, 1, 2, or 3 where 3 means three or more.

Examples of semantic constraints:
- task or setting constraints
- method/model constraints
- dataset/benchmark constraints
- comparison/result constraints
- scope/exclusion constraints
- visual/multimodal constraints

Allowed constraint_types:
- task_or_setting
- method_or_model
- dataset_or_benchmark
- comparison_or_result
- scope_or_exclusion
- visual_or_multimodal
- other

Queries:
{numbered_queries}

Return valid JSON only in this exact shape:
{{
  "results": [
    {{
      "index": 1,
      "query": "...",
      "specificity_score": 0.82,
      "naturalness_score": 0.91,
      "academic_tone_score": 0.88,
      "has_constraint": true,
      "semantic_constraint_count": 2,
      "constraint_types": ["task_or_setting", "method_or_model"],
      "rationale": "short reason"
    }}
  ]
}}
""".strip()


def _clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(parsed, 1.0))


def normalize_judge_results(
    queries: List[str],
    raw_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Normalize judge output into one result per query."""
    by_index: Dict[int, Dict[str, Any]] = {}
    for item in raw_results.get("results", []):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        by_index[index] = item

    normalized = []
    for index, query in enumerate(queries, start=1):
        item = by_index.get(index, {})
        try:
            count = int(item.get("semantic_constraint_count", 0))
        except (TypeError, ValueError):
            count = 0
        count = max(0, min(count, 3))

        raw_types = item.get("constraint_types", [])
        if not isinstance(raw_types, list):
            raw_types = []
        constraint_types = [
            constraint_type
            for constraint_type in raw_types
            if constraint_type in SEMANTIC_CONSTRAINT_TYPES
        ]

        has_constraint = bool(item.get("has_constraint", count > 0))
        if count == 0:
            has_constraint = False
            constraint_types = []
        elif has_constraint and not constraint_types:
            constraint_types = ["other"]

        normalized.append(
            {
                "index": index,
                "query": query,
                "specificity_score": _clamp_float(item.get("specificity_score")),
                "naturalness_score": _clamp_float(item.get("naturalness_score")),
                "academic_tone_score": _clamp_float(item.get("academic_tone_score")),
                "has_constraint": has_constraint,
                "semantic_constraint_count": count,
                "constraint_types": constraint_types,
                "rationale": str(item.get("rationale", "")).strip(),
            }
        )

    return normalized


def _summary_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def summarize_judge_results(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize per-query LLM judge results into existing metric shapes."""
    if not per_query:
        empty_constraints = {
            "method": "llm_as_judge",
            "constraints_per_query": {},
            "queries_with_constraints": 0,
            "constraint_ratio": 0,
            "constraint_type_distribution": {},
        }
        return {
            "qualitative_metrics": {},
            "constraint_count": empty_constraints,
            "semantic_constraint_analysis": {
                **empty_constraints,
                "semantic_constraints_per_query": {},
                "per_query": [],
            },
            "llm_judge": {
                "method": "llm_as_judge",
                "per_query": [],
            },
        }

    counts = [int(item["semantic_constraint_count"]) for item in per_query]
    type_distribution: Counter[str] = Counter()
    for item in per_query:
        type_distribution.update(item.get("constraint_types", []))

    queries_with_constraints = sum(1 for item in per_query if item["has_constraint"])
    constraint_ratio = queries_with_constraints / len(per_query)
    constraint_stats = {
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts)),
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "median": float(np.median(counts)),
    }
    constraint_count = {
        "method": "llm_as_judge",
        "scale": "0=no constraint, 1=one main constraint, 2=two constraints, 3=three or more",
        "constraints_per_query": constraint_stats,
        "queries_with_constraints": queries_with_constraints,
        "constraint_ratio": constraint_ratio,
        "constraint_type_distribution": dict(type_distribution),
    }

    qualitative_metrics = {
        "method": "llm_as_judge",
        "specificity": _summary_stats([item["specificity_score"] for item in per_query]),
        "naturalness": _summary_stats([item["naturalness_score"] for item in per_query]),
        "academic_tone": _summary_stats([item["academic_tone_score"] for item in per_query]),
    }

    return {
        "qualitative_metrics": qualitative_metrics,
        "constraint_count": constraint_count,
        "semantic_constraint_analysis": {
            "method": "llm_as_judge",
            "scale": constraint_count["scale"],
            "semantic_constraints_per_query": constraint_stats,
            "queries_with_constraints": queries_with_constraints,
            "constraint_ratio": constraint_ratio,
            "constraint_type_distribution": dict(type_distribution),
            "per_query": per_query,
        },
        "llm_judge": {
            "method": "llm_as_judge",
            "per_query": per_query,
        },
    }


def compute_llm_judged_metrics(
    queries: List[str],
    base_url: str,
    api_token: str,
    model: str,
    batch_size: int = 25,
) -> Dict[str, Any]:
    """Run LLM-as-a-judge scoring and return metric dictionaries."""
    from openreview_pipeline.llm import OpenAICompatibleBackend

    backend = OpenAICompatibleBackend(
        base_url=base_url,
        api_token=api_token,
        model=model,
        temperature=0.0,
    )
    per_query = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        logger.info(
            "LLM judge scoring queries %s-%s/%s",
            start + 1,
            start + len(batch),
            len(queries),
        )
        raw_results = backend.generate_json(build_query_judge_prompt(batch))
        batch_results = normalize_judge_results(batch, raw_results)
        for item in batch_results:
            item["index"] = start + item["index"]
        per_query.extend(batch_results)

    return summarize_judge_results(per_query)
