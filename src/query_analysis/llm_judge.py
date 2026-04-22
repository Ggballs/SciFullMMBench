"""LLM-as-a-judge style metrics for academic search queries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts" / "query_analysis"


SPECIFICITY_CALIBRATION_SCALE = (
    "1=severely under-specified, 2=under-specified, 3=well-calibrated (ideal), "
    "4=over-specified, 5=pathologically over-specified"
)
LEXICAL_NATURALISM_SCALE = (
    "1=keyword dump / fragmented, 2=stilted or awkward, 3=natural researcher register (ideal), "
    "4=over-formalized / essay-register, 5=synthetic fluent / LLM-polished prose"
)
FIT_SCALE = "0-1 closeness to ideal centered score of 3, computed as 1 - abs(score - 3) / 2"


@lru_cache(maxsize=None)
def _load_prompt_template(filename: str) -> str:
    path = PROMPT_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _render_prompt_template(filename: str, **replacements: str) -> str:
    prompt = _load_prompt_template(filename)
    for key, value in replacements.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", value)
    return prompt


def load_llm_config(config_path: Path) -> Dict[str, Any]:
    """Load OpenAI-compatible LLM settings from config.yaml."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    llm_config = config.get("llm", {})
    result = {
        "base_url": llm_config.get("base_url", ""),
        "api_token": llm_config.get("api_token", ""),
        "model": llm_config.get("model", "gpt-4o-mini"),
        "seed": llm_config.get("seed"),
    }
    if not result["base_url"] or not result["api_token"]:
        raise ValueError(
            f"Missing llm.base_url or llm.api_token in {config_path}; "
            "LLM-as-a-judge metrics require a configured API."
        )
    return result


def _normalize_judge_mode(judge_mode: str) -> str:
    normalized = str(judge_mode).strip().lower()
    if normalized not in {"batch", "single_query"}:
        raise ValueError("judge_mode must be 'batch' or 'single_query'")
    return normalized


def _build_batches(queries: List[str], batch_size: int, judge_mode: str) -> List[tuple[int, List[str]]]:
    effective_batch_size = 1 if _normalize_judge_mode(judge_mode) == "single_query" else max(1, batch_size)
    return [
        (start, queries[start:start + effective_batch_size])
        for start in range(0, len(queries), effective_batch_size)
    ]


def _score_batches(
    queries: List[str],
    base_url: str,
    api_token: str,
    model: str,
    batch_size: int,
    judge_mode: str,
    max_concurrency: int,
    seed: Optional[int],
    prompt_builder: Any,
    result_normalizer: Any,
    log_label: str,
) -> List[Dict[str, Any]]:
    from openreview_pipeline.llm import OpenAICompatibleBackend

    judge_mode = _normalize_judge_mode(judge_mode)
    max_concurrency = max(1, int(max_concurrency))
    batches = _build_batches(queries, batch_size=batch_size, judge_mode=judge_mode)

    def _score_single_batch(start: int, batch: List[str]) -> List[Dict[str, Any]]:
        backend = OpenAICompatibleBackend(
            base_url=base_url,
            api_token=api_token,
            model=model,
            temperature=0.0,
            seed=seed,
        )
        logger.info(
            "LLM %s scoring queries %s-%s/%s (%s)",
            log_label,
            start + 1,
            start + len(batch),
            len(queries),
            judge_mode,
        )
        raw_results = backend.generate_json(prompt_builder(batch))
        batch_results = result_normalizer(batch, raw_results)
        for item in batch_results:
            item["index"] = start + item["index"]
        return batch_results

    if max_concurrency == 1 or len(batches) <= 1:
        per_query: List[Dict[str, Any]] = []
        for start, batch in batches:
            per_query.extend(_score_single_batch(start, batch))
        return per_query

    per_query = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(_score_single_batch, start, batch): start
            for start, batch in batches
        }
        for future in as_completed(futures):
            per_query.extend(future.result())

    per_query.sort(key=lambda item: item.get("index", 0))
    return per_query


def build_query_judge_prompt(queries: List[str]) -> str:
    """Build a batch prompt for style-only academic query scoring."""
    numbered_queries = "\n".join(
        f"{idx}. {query}" for idx, query in enumerate(queries, start=1)
    )
    return _render_prompt_template("query_judge.txt", numbered_queries=numbered_queries).strip()


def build_semantic_constraint_prompt(queries: List[str]) -> str:
    """Build a batch prompt for LLM semantic constraint counting."""
    numbered_queries = "\n".join(
        f"{idx}. {query}" for idx, query in enumerate(queries, start=1)
    )
    return _render_prompt_template(
        "semantic_constraint_count.txt",
        numbered_queries=numbered_queries,
    ).strip()


def _parse_score_1_to_5(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= parsed <= 5:
        return parsed
    return None


def _centered_fit(score: Optional[int], ideal: int = 3) -> Optional[float]:
    if score is None:
        return None
    return max(0.0, 1.0 - (abs(score - ideal) / 2.0))


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


def normalize_judge_results(
    queries: List[str],
    raw_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Normalize judge output into one result per query."""
    raw_items = raw_results.get("results", [])
    if not isinstance(raw_items, list):
        raw_items = []

    normalized = []
    for index, query in enumerate(queries, start=1):
        item = raw_items[index - 1] if index - 1 < len(raw_items) and isinstance(raw_items[index - 1], dict) else {}

        specificity_obj = item.get("metric_1_specificity_calibration", {})
        if not isinstance(specificity_obj, dict):
            specificity_obj = {}

        lexical_obj = item.get("metric_2_lexical_naturalism", {})
        if not isinstance(lexical_obj, dict):
            lexical_obj = {}

        specificity_score = _parse_score_1_to_5(specificity_obj.get("score"))
        lexical_score = _parse_score_1_to_5(lexical_obj.get("score"))

        normalized.append(
            {
                "index": index,
                "query": query,
                "specificity_calibration_score": specificity_score,
                "specificity_calibration_rationale": str(
                    specificity_obj.get("rationale", "")
                ).strip(),
                "specificity_calibration_fit": _centered_fit(specificity_score),
                "lexical_naturalism_score": lexical_score,
                "lexical_naturalism_rationale": str(
                    lexical_obj.get("rationale", "")
                ).strip(),
                "lexical_naturalism_fit": _centered_fit(lexical_score),
            }
        )

    return normalized


def summarize_judge_results(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize per-query LLM judge results into style metric shapes."""
    if not per_query:
        return {
            "qualitative_metrics": {
                "method": "llm_as_judge",
            },
            "llm_judge": {
                "method": "llm_as_judge",
                "per_query": [],
            },
        }

    specificity_scores = [
        float(item["specificity_calibration_score"])
        for item in per_query
        if item.get("specificity_calibration_score") is not None
    ]
    specificity_fits = [
        float(item["specificity_calibration_fit"])
        for item in per_query
        if item.get("specificity_calibration_fit") is not None
    ]
    lexical_scores = [
        float(item["lexical_naturalism_score"])
        for item in per_query
        if item.get("lexical_naturalism_score") is not None
    ]
    lexical_fits = [
        float(item["lexical_naturalism_fit"])
        for item in per_query
        if item.get("lexical_naturalism_fit") is not None
    ]

    qualitative_metrics = {
        "method": "llm_as_judge",
        "specificity_calibration": {
            "scale": SPECIFICITY_CALIBRATION_SCALE,
            "ideal_score": 3,
            **_summary_stats(specificity_scores),
        },
        "specificity_calibration_fit": {
            "scale": FIT_SCALE,
            "ideal_score": 1.0,
            **_summary_stats(specificity_fits),
        },
        "lexical_naturalism": {
            "scale": LEXICAL_NATURALISM_SCALE,
            "ideal_score": 3,
            **_summary_stats(lexical_scores),
        },
        "lexical_naturalism_fit": {
            "scale": FIT_SCALE,
            "ideal_score": 1.0,
            **_summary_stats(lexical_fits),
        },
    }

    return {
        "qualitative_metrics": qualitative_metrics,
        "llm_judge": {
            "method": "llm_as_judge",
            "per_query": per_query,
        },
    }


def normalize_semantic_constraint_results(
    queries: List[str],
    raw_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Normalize semantic constraint output into one result per query."""
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
        count = max(0, count)

        normalized.append(
            {
                "index": index,
                "query": query,
                "has_constraint": count > 0,
                "semantic_constraint_count": count,
            }
        )

    return normalized


def summarize_semantic_constraints(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize per-query semantic constraint judgments."""
    if not per_query:
        empty_constraint_stats = {
            "method": "llm_as_judge",
            "scale": "exact non-negative integer semantic constraint count",
            "constraints_per_query": {},
            "queries_with_constraints": 0,
            "constraint_ratio": 0,
        }
        return {
            "constraint_count": empty_constraint_stats,
            "semantic_constraint_analysis": {
                "method": "llm_as_judge",
                "scale": empty_constraint_stats["scale"],
                "semantic_constraints_per_query": {},
                "queries_with_constraints": 0,
                "constraint_ratio": 0,
                "per_query": [],
            },
        }

    counts = [item["semantic_constraint_count"] for item in per_query]
    queries_with_constraints = sum(1 for item in per_query if item["has_constraint"])
    stats = {
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts)),
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "median": float(np.median(counts)),
    }
    scale = "exact non-negative integer semantic constraint count"

    return {
        "constraint_count": {
            "method": "llm_as_judge",
            "scale": scale,
            "constraints_per_query": stats,
            "queries_with_constraints": queries_with_constraints,
            "constraint_ratio": queries_with_constraints / len(per_query),
        },
        "semantic_constraint_analysis": {
            "method": "llm_as_judge",
            "scale": scale,
            "semantic_constraints_per_query": stats,
            "queries_with_constraints": queries_with_constraints,
            "constraint_ratio": queries_with_constraints / len(per_query),
            "per_query": per_query,
        },
    }


def compute_llm_judged_metrics(
    queries: List[str],
    base_url: str,
    api_token: str,
    model: str,
    batch_size: int = 25,
    judge_mode: str = "batch",
    max_concurrency: int = 1,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run LLM-as-a-judge scoring and return metric dictionaries."""
    per_query = _score_batches(
        queries=queries,
        base_url=base_url,
        api_token=api_token,
        model=model,
        batch_size=batch_size,
        judge_mode=judge_mode,
        max_concurrency=max_concurrency,
        seed=seed,
        prompt_builder=build_query_judge_prompt,
        result_normalizer=normalize_judge_results,
        log_label="judge",
    )
    return summarize_judge_results(per_query)


def compute_llm_semantic_constraint_metrics(
    queries: List[str],
    base_url: str,
    api_token: str,
    model: str,
    batch_size: int = 25,
    judge_mode: str = "batch",
    max_concurrency: int = 1,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run LLM semantic constraint counting and return metric dictionaries."""
    per_query = _score_batches(
        queries=queries,
        base_url=base_url,
        api_token=api_token,
        model=model,
        batch_size=batch_size,
        judge_mode=judge_mode,
        max_concurrency=max_concurrency,
        seed=seed,
        prompt_builder=build_semantic_constraint_prompt,
        result_normalizer=normalize_semantic_constraint_results,
        log_label="semantic constraint",
    )
    return summarize_semantic_constraints(per_query)
