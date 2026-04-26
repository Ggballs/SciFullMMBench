"""LLM-as-a-judge style metrics for academic search queries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from openreview_pipeline.llm.base import create_openai_compatible_backend, load_llm_config

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parents[4] / "prompts" / "query_analysis"
SPECIFICITY_CALIBRATION_SCALE = "1=severely under-specified, 2=under-specified, 3=well-calibrated (ideal), 4=over-specified, 5=pathologically over-specified"
LEXICAL_NATURALISM_SCALE = "1=keyword dump / fragmented, 2=stilted or awkward, 3=natural researcher register (ideal), 4=over-formalized / essay-register, 5=synthetic fluent / LLM-polished prose"
FIT_SCALE = "0-1 closeness to ideal centered score of 3, computed as 1 - abs(score - 3) / 2"
SEMANTIC_SCALE = "exact non-negative integer semantic constraint count"


@lru_cache(maxsize=None)
def _load_prompt_template(filename: str) -> str:
    return (PROMPT_DIR / filename).read_text(encoding="utf-8")


def _render_prompt_template(filename: str, **replacements: str) -> str:
    prompt = _load_prompt_template(filename)
    for key, value in replacements.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", value)
    return prompt


def _normalize_judge_mode(judge_mode: str) -> str:
    normalized = str(judge_mode).strip().lower()
    if normalized not in {"batch", "single_query"}:
        raise ValueError("judge_mode must be 'batch' or 'single_query'")
    return normalized


def _build_batches(queries: List[str], batch_size: int, judge_mode: str) -> List[tuple[int, List[str]]]:
    effective_batch_size = 1 if _normalize_judge_mode(judge_mode) == "single_query" else max(1, batch_size)
    return [(start, queries[start:start + effective_batch_size]) for start in range(0, len(queries), effective_batch_size)]


def build_style_analysis_prompt(queries: List[str]) -> str:
    numbered_queries = "\n".join(f"{idx}. {query}" for idx, query in enumerate(queries, start=1))
    return _render_prompt_template("style_analysis.txt", numbered_queries=numbered_queries).strip()


def _parse_score_1_to_5(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if 1 <= parsed <= 5 else None


def _parse_non_negative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _centered_fit(score: Optional[int], ideal: int = 3) -> Optional[float]:
    if score is None:
        return None
    return max(0.0, 1.0 - (abs(score - ideal) / 2.0))


def _summary_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values)), "median": float(np.median(values))}


def normalize_judge_results(queries: List[str], raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_items = raw_results.get("results", [])
    if not isinstance(raw_items, list):
        raw_items = []
    normalized = []
    for index, query in enumerate(queries, start=1):
        item = raw_items[index - 1] if index - 1 < len(raw_items) and isinstance(raw_items[index - 1], dict) else {}
        specificity_obj = item.get("metric_1_specificity_calibration", {}) if isinstance(item.get("metric_1_specificity_calibration", {}), dict) else {}
        lexical_obj = item.get("metric_2_lexical_naturalism", {}) if isinstance(item.get("metric_2_lexical_naturalism", {}), dict) else {}
        semantic_obj = item.get("metric_3_semantic_constraint_count", {}) if isinstance(item.get("metric_3_semantic_constraint_count", {}), dict) else {}
        specificity_score = _parse_score_1_to_5(specificity_obj.get("score"))
        lexical_score = _parse_score_1_to_5(lexical_obj.get("score"))
        semantic_count = _parse_non_negative_int(semantic_obj.get("count"))
        normalized.append({
            "index": index,
            "query": query,
            "specificity_calibration_score": specificity_score,
            "specificity_calibration_rationale": str(specificity_obj.get("rationale", "")).strip(),
            "specificity_calibration_fit": _centered_fit(specificity_score),
            "lexical_naturalism_score": lexical_score,
            "lexical_naturalism_rationale": str(lexical_obj.get("rationale", "")).strip(),
            "lexical_naturalism_fit": _centered_fit(lexical_score),
            "semantic_constraint_count": semantic_count,
            "semantic_constraint_rationale": str(semantic_obj.get("rationale", "")).strip(),
            "has_constraint": semantic_count > 0,
        })
    return normalized


def _score_batches(
    queries: List[str],
    base_url: str,
    api_token: str,
    model: str,
    batch_size: int,
    judge_mode: str,
    max_concurrency: int,
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    judge_mode = _normalize_judge_mode(judge_mode)
    batches = _build_batches(queries, batch_size=batch_size, judge_mode=judge_mode)
    max_concurrency = max(1, int(max_concurrency))

    def _score_single_batch(start: int, batch: List[str]) -> List[Dict[str, Any]]:
        backend = create_openai_compatible_backend(
            base_url=base_url,
            api_token=api_token,
            model=model,
            temperature=0.0,
            seed=seed,
        )
        logger.info("LLM judge scoring queries %s-%s/%s (%s)", start + 1, start + len(batch), len(queries), judge_mode)
        raw_results = backend.generate_json(build_style_analysis_prompt(batch))
        batch_results = normalize_judge_results(batch, raw_results)
        for item in batch_results:
            item["index"] = start + item["index"]
        return batch_results

    if max_concurrency == 1 or len(batches) <= 1:
        per_query: List[Dict[str, Any]] = []
        for start, batch in batches:
            per_query.extend(_score_single_batch(start, batch))
        return per_query

    per_query: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(_score_single_batch, start, batch): start for start, batch in batches}
        for future in as_completed(futures):
            per_query.extend(future.result())
    per_query.sort(key=lambda item: item.get("index", 0))
    return per_query


def summarize_judge_results(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    specificity_scores = [float(item["specificity_calibration_score"]) for item in per_query if item.get("specificity_calibration_score") is not None]
    specificity_fits = [float(item["specificity_calibration_fit"]) for item in per_query if item.get("specificity_calibration_fit") is not None]
    lexical_scores = [float(item["lexical_naturalism_score"]) for item in per_query if item.get("lexical_naturalism_score") is not None]
    lexical_fits = [float(item["lexical_naturalism_fit"]) for item in per_query if item.get("lexical_naturalism_fit") is not None]
    return {
        "qualitative_metrics": {
            "method": "llm_as_judge",
            "specificity_calibration": {"scale": SPECIFICITY_CALIBRATION_SCALE, "ideal_score": 3, **_summary_stats(specificity_scores)},
            "specificity_calibration_fit": {"scale": FIT_SCALE, "ideal_score": 1.0, **_summary_stats(specificity_fits)},
            "lexical_naturalism": {"scale": LEXICAL_NATURALISM_SCALE, "ideal_score": 3, **_summary_stats(lexical_scores)},
            "lexical_naturalism_fit": {"scale": FIT_SCALE, "ideal_score": 1.0, **_summary_stats(lexical_fits)},
        },
        "llm_judge": {"method": "llm_as_judge", "per_query": [
            {
                "index": item["index"],
                "query": item["query"],
                "specificity_calibration_score": item.get("specificity_calibration_score"),
                "specificity_calibration_rationale": item.get("specificity_calibration_rationale", ""),
                "specificity_calibration_fit": item.get("specificity_calibration_fit"),
                "lexical_naturalism_score": item.get("lexical_naturalism_score"),
                "lexical_naturalism_rationale": item.get("lexical_naturalism_rationale", ""),
                "lexical_naturalism_fit": item.get("lexical_naturalism_fit"),
            }
            for item in per_query
        ]},
    }


def summarize_semantic_constraints(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = [int(item.get("semantic_constraint_count", 0)) for item in per_query]
    queries_with_constraints = sum(1 for item in per_query if item.get("has_constraint"))
    stats = _summary_stats(counts)
    return {
        "constraint_count": {"method": "llm_as_judge", "scale": SEMANTIC_SCALE, "constraints_per_query": stats, "queries_with_constraints": queries_with_constraints, "constraint_ratio": queries_with_constraints / len(per_query) if per_query else 0.0},
        "semantic_constraint_analysis": {"method": "llm_as_judge", "scale": SEMANTIC_SCALE, "semantic_constraints_per_query": stats, "queries_with_constraints": queries_with_constraints, "constraint_ratio": queries_with_constraints / len(per_query) if per_query else 0.0, "per_query": [
            {
                "index": item["index"],
                "query": item["query"],
                "has_constraint": item.get("has_constraint", False),
                "semantic_constraint_count": int(item.get("semantic_constraint_count", 0)),
                "semantic_constraint_rationale": item.get("semantic_constraint_rationale", ""),
            }
            for item in per_query
        ]},
    }


def compute_llm_judged_metrics(queries: List[str], base_url: str, api_token: str, model: str, batch_size: int = 25, judge_mode: str = "batch", max_concurrency: int = 1, seed: Optional[int] = None) -> Dict[str, Any]:
    return summarize_judge_results(_score_batches(queries, base_url, api_token, model, batch_size, judge_mode, max_concurrency, seed))


def compute_llm_semantic_constraint_metrics(queries: List[str], base_url: str, api_token: str, model: str, batch_size: int = 25, judge_mode: str = "batch", max_concurrency: int = 1, seed: Optional[int] = None) -> Dict[str, Any]:
    return summarize_semantic_constraints(_score_batches(queries, base_url, api_token, model, batch_size, judge_mode, max_concurrency, seed))


def analyze_query(query: str, base_url: str, api_token: str, model: str, judge_mode: str = "single_query", max_concurrency: int = 1, seed: Optional[int] = None) -> Dict[str, Any]:
    combined = analyze_queries([query], base_url=base_url, api_token=api_token, model=model, batch_size=1, judge_mode=judge_mode, max_concurrency=max_concurrency, seed=seed)
    return {
        "llm_judge": combined.get("llm_judge", {}).get("per_query", [{}])[0] if combined.get("llm_judge", {}).get("per_query") else {},
        "semantic_constraint_analysis": combined.get("semantic_constraint_analysis", {}).get("per_query", [{}])[0] if combined.get("semantic_constraint_analysis", {}).get("per_query") else {},
    }


def analyze_queries(queries: List[str], base_url: str, api_token: str, model: str, batch_size: int = 25, judge_mode: str = "batch", max_concurrency: int = 1, seed: Optional[int] = None) -> Dict[str, Any]:
    per_query = _score_batches(queries, base_url, api_token, model, batch_size, judge_mode, max_concurrency, seed)
    return {**summarize_judge_results(per_query), **summarize_semantic_constraints(per_query)}
