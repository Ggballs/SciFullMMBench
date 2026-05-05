from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from openreview_pipeline.llm.base import create_openai_compatible_backend, load_llm_config
from . import llm_judge, rule_judge

FIT_SCALE = "0-1 closeness to ideal centered score of 3, computed as 1 - abs(score - 3) / 2"


@dataclass
class SummaryMetrics:
    query_count: int
    avg_chars: float
    avg_tokens: float
    constraints_per_query: float
    specificity_score: float
    specificity_fit: float
    lexical_score: float
    lexical_fit: float
    unmatched_ratio: float


def load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def slugify_dataset_name(dataset_name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in dataset_name.strip())
    return slug.strip("_") or "generated_queries"


def extract_queries(generated_data: Dict[str, Any]) -> List[str]:
    queries: List[str] = []
    for paper_queries in generated_data.get("papers_queries", []):
        if not isinstance(paper_queries, dict):
            continue
        for query in paper_queries.get("queries_by_view", []):
            if isinstance(query, dict):
                query_text = str(query.get("query_text", "")).strip()
                if query_text:
                    queries.append(query_text)
    return queries


def _empty_llm_metrics() -> Dict[str, Any]:
    return {
        "qualitative_metrics": {
            "method": "llm_as_judge",
            "specificity_calibration": {"scale": llm_judge.SPECIFICITY_CALIBRATION_SCALE, "ideal_score": 3},
            "specificity_calibration_fit": {"scale": FIT_SCALE, "ideal_score": 1.0},
            "lexical_naturalism": {"scale": llm_judge.LEXICAL_NATURALISM_SCALE, "ideal_score": 3},
            "lexical_naturalism_fit": {"scale": FIT_SCALE, "ideal_score": 1.0},
        },
        "llm_judge": {"method": "llm_as_judge", "per_query": []},
        "semantic_constraint_analysis": {"method": "llm_as_judge", "scale": llm_judge.SEMANTIC_SCALE, "semantic_constraints_per_query": {}, "queries_with_constraints": 0, "constraint_ratio": 0.0, "per_query": []},
        "constraint_count": {"method": "llm_as_judge", "scale": llm_judge.SEMANTIC_SCALE, "constraints_per_query": {}, "queries_with_constraints": 0, "constraint_ratio": 0.0},
    }


def build_analysis(dataset_name: str, queries: List[str], llm_judge_config: Optional[Dict[str, Any]] = None, llm_seed: Optional[int] = None) -> Dict[str, Any]:
    rule_results = rule_judge.analyze_queries(queries)
    combined_metrics = {"length_stats": rule_results["length_stats"], "question_templates": rule_results["question_templates"], **_empty_llm_metrics()}
    if llm_judge_config and queries:
        backend = create_openai_compatible_backend(
            base_url=str(llm_judge_config["base_url"]),
            api_tokens=[str(token) for token in llm_judge_config["api_tokens"]],
            model=str(llm_judge_config["model"]),
            max_tokens=int(llm_judge_config.get("max_tokens", 4096)),
            temperature=float(llm_judge_config.get("temperature", 0.0)),
            seed=llm_seed,
            per_key_request_interval_seconds=float(llm_judge_config.get("per_key_request_interval_seconds", 0.0)),
            per_key_max_concurrent_requests=int(llm_judge_config.get("per_key_max_concurrent_requests", 1)),
            max_retries=int(llm_judge_config.get("max_retries", 3)),
            retry_backoff_seconds=float(llm_judge_config.get("retry_backoff_seconds", 8.0)),
        )
        combined_metrics.update(llm_judge.analyze_queries(queries=queries, llm=backend))
    return {
        "dataset": dataset_name,
        "dataset_overview": {"generated_queries": {"total_queries": len(queries), "source": "generated query JSON"}, "combined": {"total_queries": len(queries)}},
        "metrics": {"combined": combined_metrics},
        "representative_examples": rule_results["representative_examples"],
        "query_examples": queries,
        "question_template_distribution": {
            "combined": {
                "templates": {
                    template: {
                        "count": count,
                        "ratio": combined_metrics["question_templates"]["template_ratios"].get(template, 0.0),
                        "example": combined_metrics["question_templates"]["template_examples"].get(template, [None])[0],
                    }
                    for template, count in combined_metrics["question_templates"]["template_distribution"].items()
                    if count >= 4
                },
                "total_queries": len(queries),
                "templates_with_5plus": sum(1 for count in combined_metrics["question_templates"]["template_distribution"].values() if count >= 4),
            }
        },
    }


def render_style_analysis_markdown(analysis: Dict[str, Any]) -> str:
    combined = analysis["metrics"]["combined"]
    length_stats = combined["length_stats"]
    question_templates = combined["question_templates"]
    qualitative = combined["qualitative_metrics"]
    semantic = combined["semantic_constraint_analysis"]
    lines = ["# Query Style Analysis", "", "## Overview", "", f"- Dataset: `{analysis.get('dataset', 'generated_queries')}`", f"- Total queries: {analysis['dataset_overview']['combined']['total_queries']}", "", "## Core Metrics", "", f"- Avg characters: {length_stats.get('char_length', {}).get('mean', 0.0):.1f}", f"- Avg tokens: {length_stats.get('token_length', {}).get('mean', 0.0):.1f}", f"- Other / unmatched template share: {question_templates.get('unmatched_ratio', 0.0) * 100:.1f}%"]
    semantic_mean = semantic.get("semantic_constraints_per_query", {}).get("mean")
    if semantic_mean is not None:
        lines.append(f"- Semantic constraints per query: {semantic_mean:.2f}")
    spec_mean = qualitative.get("specificity_calibration", {}).get("mean")
    lex_mean = qualitative.get("lexical_naturalism", {}).get("mean")
    if spec_mean is not None:
        lines.append(f"- Specificity score (3 ideal): {spec_mean:.3f}")
    if lex_mean is not None:
        lines.append(f"- Lexical score (3 ideal): {lex_mean:.3f}")
    lines.extend(["", "## Representative Examples", ""])
    for example in analysis.get("representative_examples", [])[:10]:
        lines.append(f"- [{example['template']}] {example['query']}")
    return "\n".join(lines) + "\n"


def extract_summary_metrics(data: Dict[str, Any]) -> SummaryMetrics:
    combined = data["metrics"]["combined"]
    length_stats = combined["length_stats"]
    qualitative = combined["qualitative_metrics"]
    question_templates = combined["question_templates"]
    constraint_stats = combined.get("semantic_constraint_analysis") or combined.get("constraint_count", {})
    return SummaryMetrics(
        query_count=int(length_stats.get("total_queries", 0)),
        avg_chars=float(length_stats.get("char_length", {}).get("mean", 0.0)),
        avg_tokens=float(length_stats.get("token_length", {}).get("mean", 0.0)),
        constraints_per_query=float(constraint_stats.get("semantic_constraints_per_query", {}).get("mean", constraint_stats.get("constraints_per_query", {}).get("mean", 0.0))),
        specificity_score=float(qualitative.get("specificity_calibration", {}).get("mean", 0.0)),
        specificity_fit=float(qualitative.get("specificity_calibration_fit", {}).get("mean", 0.0)),
        lexical_score=float(qualitative.get("lexical_naturalism", {}).get("mean", 0.0)),
        lexical_fit=float(qualitative.get("lexical_naturalism_fit", {}).get("mean", 0.0)),
        unmatched_ratio=float(question_templates.get("unmatched_ratio", 0.0)),
    )


def extract_per_query_frame(data: Dict[str, Any]) -> pd.DataFrame:
    combined = data["metrics"]["combined"]
    semantic_rows = combined.get("semantic_constraint_analysis", {}).get("per_query", [])
    judge_rows = combined.get("llm_judge", {}).get("per_query", [])
    if not semantic_rows or not judge_rows:
        raise ValueError("Per-query semantic or LLM-judge results are missing from style analysis JSON.")
    semantic_df = pd.DataFrame([{"index": int(row["index"]), "constraints_per_query": row["semantic_constraint_count"]} for row in semantic_rows if "index" in row and "semantic_constraint_count" in row])
    judge_df = pd.DataFrame([{"index": int(row["index"]), "specificity": row["specificity_calibration_score"], "naturalness": row["lexical_naturalism_score"]} for row in judge_rows if "index" in row and row.get("specificity_calibration_score") is not None and row.get("lexical_naturalism_score") is not None])
    merged = semantic_df.merge(judge_df, on="index", how="inner")
    if merged.empty:
        raise ValueError("No overlapping per-query rows found for semantic and judge outputs.")
    return merged


def compute_distribution_based_human_closeness(human: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, float]:
    required_columns = ["constraints_per_query", "specificity", "naturalness"]
    for name, frame in [("human", human), ("synthetic", synthetic)]:
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
    human_df = human[required_columns].apply(pd.to_numeric, errors="coerce").dropna()
    synthetic_df = synthetic[required_columns].apply(pd.to_numeric, errors="coerce").dropna()
    if human_df.empty or synthetic_df.empty:
        raise ValueError("Human or synthetic per-query frame has no valid numeric rows.")
    w_c = wasserstein_distance(human_df["constraints_per_query"], synthetic_df["constraints_per_query"])
    sigma_c = float(np.std(human_df["constraints_per_query"]))
    hc_constraints = float(np.exp(-w_c / (sigma_c + 1e-8)))
    w_s = wasserstein_distance(human_df["specificity"], synthetic_df["specificity"])
    hc_specificity = float(np.clip(1 - w_s / 4, 0, 1))
    w_n = wasserstein_distance(human_df["naturalness"], synthetic_df["naturalness"])
    hc_naturalness = float(np.clip(1 - w_n / 4, 0, 1))
    return {"W_constraints": float(w_c), "HC_constraints": hc_constraints, "W_specificity": float(w_s), "HC_specificity": hc_specificity, "W_naturalness": float(w_n), "HC_naturalness": hc_naturalness, "HC_overall": float(np.mean([hc_constraints, hc_specificity, hc_naturalness]))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a canonical style analysis for generated query JSON.")
    parser.add_argument("--generated-queries", required=True, help="Path to generated queries JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for style_analysis outputs.")
    parser.add_argument("--dataset-name", default=None, help="Optional dataset name.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml with LLM settings.")
    parser.add_argument("--llm-seed", type=int, default=None)
    parser.add_argument("--no-llm-judge", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    generated_queries_path = Path(args.generated_queries).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_data = load_json(generated_queries_path)
    queries = extract_queries(generated_data)
    dataset_name = args.dataset_name or generated_queries_path.stem
    llm_config = None
    llm_seed = args.llm_seed
    if not args.no_llm_judge:
        llm_config = load_llm_config(Path(args.config).expanduser().resolve())
        if llm_seed is None:
            llm_seed = llm_config.pop("seed", None)
        else:
            llm_config.pop("seed", None)
    analysis = build_analysis(dataset_name=dataset_name, queries=queries, llm_judge_config=llm_config, llm_seed=llm_seed)
    json_path = output_dir / "style_analysis.json"
    md_path = output_dir / "style_analysis.md"
    write_text(json_path, json.dumps(analysis, indent=2, ensure_ascii=False))
    write_text(md_path, render_style_analysis_markdown(analysis))
    print(f"Saved style analysis JSON to {json_path}")
    print(f"Saved style analysis markdown to {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
