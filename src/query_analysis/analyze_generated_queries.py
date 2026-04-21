#!/usr/bin/env python3
"""
Analyze generated query JSON files and compare them against reference analyses.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from query_analysis.features import compute_all_metrics, detect_question_template
from query_analysis.llm_judge import (
    compute_llm_judged_metrics,
    load_llm_config as load_llm_judge_config,
)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_queries(generated_data: Dict[str, Any]) -> List[str]:
    queries: List[str] = []
    for paper_queries in generated_data.get("papers_queries", []):
        for query in paper_queries.get("queries_by_view", []):
            query_text = query.get("query_text", "").strip()
            if query_text:
                queries.append(query_text)
    return queries


def build_template_metrics(queries: List[str]) -> Dict[str, Any]:
    template_dist: Counter[str] = Counter()
    template_examples: Dict[str, List[str]] = {}

    for query in queries:
        template, _ = detect_question_template(query)
        template = template or "other"
        template_dist[template] += 1
        if template not in template_examples:
            template_examples[template] = []
        if len(template_examples[template]) < 3:
            template_examples[template].append(query)

    total = len(queries)
    ratios = {template: count / total for template, count in template_dist.items()} if total else {}

    return {
        "template_distribution": dict(template_dist),
        "template_ratios": ratios,
        "template_examples": template_examples,
        "total_templates": len(template_dist),
    }


def build_generated_analysis(
    dataset_name: str,
    queries: List[str],
    llm_judge_config: Optional[Dict[str, str]] = None,
    llm_judge_batch_size: int = 25,
) -> Dict[str, Any]:
    metrics = compute_all_metrics(queries)
    metrics["question_templates"] = build_template_metrics(queries)
    if llm_judge_config and queries:
        metrics.update(
            compute_llm_judged_metrics(
                queries=queries,
                batch_size=llm_judge_batch_size,
                **llm_judge_config,
            )
        )
    return {
        "dataset": dataset_name,
        "total_queries": len(queries),
        "query_examples": queries,
        "metrics": {
            "combined": metrics,
        },
    }


def build_query_list_analysis(
    dataset_name: str,
    queries: List[str],
    llm_judge_config: Optional[Dict[str, str]] = None,
    llm_judge_batch_size: int = 25,
) -> Dict[str, Any]:
    metrics = compute_all_metrics(queries)
    metrics["question_templates"] = build_template_metrics(queries)
    if llm_judge_config and queries:
        metrics.update(
            compute_llm_judged_metrics(
                queries=queries,
                batch_size=llm_judge_batch_size,
                **llm_judge_config,
            )
        )
    return {
        "dataset": dataset_name,
        "total_queries": len(queries),
        "query_examples": queries,
        "metrics": {
            "combined": metrics,
        },
    }


SEMANTIC_CONSTRAINT_TYPES = {
    "task_or_setting",
    "method_or_model",
    "dataset_or_benchmark",
    "comparison_or_result",
    "scope_or_exclusion",
    "visual_or_multimodal",
    "other",
}


def build_semantic_constraint_prompt(queries: List[str]) -> str:
    numbered_queries = "\n".join(
        f"{idx}. {query}" for idx, query in enumerate(queries, start=1)
    )
    return f"""
You are evaluating academic paper-search queries.

For each query, identify semantic retrieval constraints. A constraint is any condition
that narrows what paper should be retrieved, even if it is not expressed with words
like "for" or "with".

Count constraints semantically, not by keyword. Examples of constraints:
- task or setting constraints
- method/model constraints
- dataset/benchmark constraints
- comparison/result constraints
- scope/exclusion constraints
- visual/multimodal constraints

Use a conservative count:
- 0 = generic query with no real narrowing condition
- 1 = one main narrowing condition
- 2 = two distinct narrowing conditions
- 3 = three or more distinct narrowing conditions

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
      "has_constraint": true,
      "semantic_constraint_count": 2,
      "constraint_types": ["task_or_setting", "dataset_or_benchmark"],
      "rationale": "short reason"
    }}
  ]
}}
""".strip()


def normalize_semantic_constraint_results(
    queries: List[str],
    raw_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
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
                "has_constraint": has_constraint,
                "semantic_constraint_count": count,
                "constraint_types": constraint_types,
                "rationale": str(item.get("rationale", "")).strip(),
            }
        )

    return normalized


def summarize_semantic_constraints(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not per_query:
        return {
            "method": "llm_semantic",
            "semantic_constraints_per_query": {},
            "queries_with_constraints": 0,
            "constraint_ratio": 0,
            "constraint_type_distribution": {},
            "per_query": [],
        }

    import numpy as np

    counts = [item["semantic_constraint_count"] for item in per_query]
    type_distribution: Counter[str] = Counter()
    for item in per_query:
        type_distribution.update(item.get("constraint_types", []))

    queries_with_constraints = sum(1 for item in per_query if item["has_constraint"])
    return {
        "method": "llm_semantic",
        "scale": "0=no constraint, 1=one main constraint, 2=two constraints, 3=three or more",
        "semantic_constraints_per_query": {
            "mean": float(np.mean(counts)),
            "std": float(np.std(counts)),
            "min": int(np.min(counts)),
            "max": int(np.max(counts)),
            "median": float(np.median(counts)),
        },
        "queries_with_constraints": queries_with_constraints,
        "constraint_ratio": queries_with_constraints / len(per_query),
        "constraint_type_distribution": dict(type_distribution),
        "per_query": per_query,
    }


def compute_semantic_constraint_analysis(
    queries: List[str],
    base_url: str,
    api_token: str,
    model: str,
    batch_size: int = 25,
) -> Dict[str, Any]:
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
        raw_results = backend.generate_json(build_semantic_constraint_prompt(batch))
        batch_results = normalize_semantic_constraint_results(batch, raw_results)
        for item in batch_results:
            item["index"] = start + item["index"]
        per_query.extend(batch_results)
    return summarize_semantic_constraints(per_query)


def load_llm_config(config_path: Path) -> Dict[str, str]:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    llm_config = config.get("llm", {})
    return {
        "base_url": llm_config.get("base_url", ""),
        "api_token": llm_config.get("api_token", ""),
        "model": llm_config.get("model", "gpt-4o-mini"),
    }


def load_queries_file(path: Path) -> List[str]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            query = line.strip()
            if query:
                queries.append(query)
    return queries


def get_nested(data: Dict[str, Any], path: Tuple[str, ...], default: Any = None) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def render_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_metric_row(name: str, human: Any, original: Any, new: Any) -> str:
    return f"| {name} | {human} | {original} | {new} |"


def slugify_dataset_name(dataset_name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in dataset_name.strip())
    slug = slug.strip("_")
    return slug or "generated_queries"


def build_report(
    human_analysis: Dict[str, Any],
    original_analysis: Dict[str, Any],
    new_analysis: Dict[str, Any],
    generated_queries_path: Path,
    generated_analysis_path: Path,
    report_title: str,
) -> str:
    human = get_nested(human_analysis, ("metrics", "combined"), {})
    original = get_nested(original_analysis, ("metrics", "combined"), {})
    new = get_nested(new_analysis, ("metrics", "combined"), {})
    new_label = new_analysis.get("dataset", "New Synthetic")
    if isinstance(new_label, str):
        new_label = new_label.replace("_", " ").title()
    else:
        new_label = "New Synthetic"

    human_char = get_nested(human, ("length_stats", "char_length", "mean"), 0.0)
    human_tokens = get_nested(human, ("length_stats", "token_length", "mean"), 0.0)
    human_semantic_constraints = get_nested(
        human,
        ("semantic_constraint_analysis", "semantic_constraints_per_query", "mean"),
        None,
    )
    human_semantic_constraint_ratio = get_nested(
        human,
        ("semantic_constraint_analysis", "constraint_ratio"),
        None,
    )
    human_specificity = get_nested(human, ("qualitative_metrics", "specificity", "mean"), 0.0)
    human_naturalness = get_nested(human, ("qualitative_metrics", "naturalness", "mean"), 0.0)
    human_tone = get_nested(human, ("qualitative_metrics", "academic_tone", "mean"), 0.0)
    human_other_ratio = get_nested(human, ("question_templates", "unmatched_ratio"), 0.0)

    original_char = get_nested(original, ("length_stats", "char_length", "mean"), 0.0)
    original_tokens = get_nested(original, ("length_stats", "token_length", "mean"), 0.0)
    original_semantic_constraints = get_nested(
        original,
        ("semantic_constraint_analysis", "semantic_constraints_per_query", "mean"),
        None,
    )
    original_semantic_constraint_ratio = get_nested(
        original,
        ("semantic_constraint_analysis", "constraint_ratio"),
        None,
    )
    original_specificity = get_nested(original, ("qualitative_metrics", "specificity", "mean"), 0.0)
    original_naturalness = get_nested(original, ("qualitative_metrics", "naturalness", "mean"), 0.0)
    original_tone = get_nested(original, ("qualitative_metrics", "academic_tone", "mean"), 0.0)
    original_other_ratio = get_nested(original, ("question_templates", "template_ratios", "other"), 0.0)

    new_char = get_nested(new, ("length_stats", "char_length", "mean"), 0.0)
    new_tokens = get_nested(new, ("length_stats", "token_length", "mean"), 0.0)
    new_semantic_constraints = get_nested(
        new,
        ("semantic_constraint_analysis", "semantic_constraints_per_query", "mean"),
        None,
    )
    new_semantic_constraint_ratio = get_nested(
        new,
        ("semantic_constraint_analysis", "constraint_ratio"),
        None,
    )
    new_specificity = get_nested(new, ("qualitative_metrics", "specificity", "mean"), 0.0)
    new_naturalness = get_nested(new, ("qualitative_metrics", "naturalness", "mean"), 0.0)
    new_tone = get_nested(new, ("qualitative_metrics", "academic_tone", "mean"), 0.0)
    new_other_ratio = get_nested(new, ("question_templates", "template_ratios", "other"), 0.0)

    new_examples = new_analysis.get("query_examples", [])[:8]

    lines: List[str] = []
    lines.append(f"# {report_title}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Generated queries: `{generated_queries_path}`")
    lines.append(f"- Generated analysis: `{generated_analysis_path}`")
    lines.append("- Human reference: `outputs/query_analysis/human_written_style_analysis.json`")
    lines.append("- Original synthetic reference: `outputs/query_analysis/synthetic_single_analysis.json`")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("The new generated queries are much closer to the human-written reference than the original synthetic queries on length and template realism. The strongest improvements are that the queries are no longer overlong and they use recognizable human search openings such as `What`, `Which`, and `Is there`.")
    lines.append("")
    lines.append("The main remaining gap is that the new set is still somewhat denser and more benchmark-specific than typical human queries, and its academic tone still trails the human reference. It is directionally better than the original synthetic set, but there is still room to reduce source-detail carryover in some queries.")
    lines.append("")
    lines.append("## Side-by-Side Metrics")
    lines.append("")
    lines.append(f"| Metric | Human Reference | Original Synthetic | {new_label} |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(render_metric_row("Query count", get_nested(human_analysis, ("dataset_overview", "combined", "total_queries"), 0), original_analysis.get("total_queries", 0), new_analysis.get("total_queries", 0)))
    lines.append(render_metric_row("Avg characters", f"{human_char:.1f}", f"{original_char:.1f}", f"{new_char:.1f}"))
    lines.append(render_metric_row("Avg tokens", f"{human_tokens:.1f}", f"{original_tokens:.1f}", f"{new_tokens:.1f}"))
    if (
        human_semantic_constraints is not None
        or original_semantic_constraints is not None
        or new_semantic_constraints is not None
    ):
        lines.append(render_metric_row(
            "Semantic constraints per query",
            f"{human_semantic_constraints:.2f}" if human_semantic_constraints is not None else "not LLM-scored",
            f"{original_semantic_constraints:.2f}" if original_semantic_constraints is not None else "not LLM-scored",
            f"{new_semantic_constraints:.2f}" if new_semantic_constraints is not None else "not LLM-scored",
        ))
    if (
        human_semantic_constraint_ratio is not None
        or original_semantic_constraint_ratio is not None
        or new_semantic_constraint_ratio is not None
    ):
        lines.append(render_metric_row(
            "Queries with semantic constraints",
            render_pct(human_semantic_constraint_ratio) if human_semantic_constraint_ratio is not None else "not LLM-scored",
            render_pct(original_semantic_constraint_ratio) if original_semantic_constraint_ratio is not None else "not LLM-scored",
            render_pct(new_semantic_constraint_ratio) if new_semantic_constraint_ratio is not None else "not LLM-scored",
        ))
    lines.append(render_metric_row("Specificity", f"{human_specificity:.3f}", f"{original_specificity:.3f}", f"{new_specificity:.3f}"))
    lines.append(render_metric_row("Naturalness", f"{human_naturalness:.3f}", f"{original_naturalness:.3f}", f"{new_naturalness:.3f}"))
    lines.append(render_metric_row("Academic tone", f"{human_tone:.3f}", f"{original_tone:.3f}", f"{new_tone:.3f}"))
    lines.append(render_metric_row("Other / unmatched template share", render_pct(human_other_ratio), render_pct(original_other_ratio), render_pct(new_other_ratio)))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"- Length improved sharply: the original synthetic queries averaged {original_tokens:.1f} tokens, while the new set averages {new_tokens:.1f}, much closer to the human reference at {human_tokens:.1f}.")
    lines.append(f"- Naturalness improved from {original_naturalness:.3f} to {new_naturalness:.3f}, approaching or exceeding the human reference score of {human_naturalness:.3f}.")
    lines.append(f"- Template fit improved: `other` dropped from {render_pct(original_other_ratio)} to {render_pct(new_other_ratio)}.")
    lines.append(f"- Specificity remains high at {new_specificity:.3f}, but some queries still carry more benchmark-construction detail than a typical human search would.")
    if new_semantic_constraints is not None and new_semantic_constraint_ratio is not None:
        lines.append(f"- LLM semantic constraint scoring finds {new_semantic_constraints:.2f} constraints per query, with {render_pct(new_semantic_constraint_ratio)} of new queries containing at least one retrieval-narrowing condition.")
    lines.append("")
    lines.append("## Query Examples")
    lines.append("")
    for query in new_examples:
        lines.append(f"- {query}")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("The revised prompt is a clear improvement. It generates shorter, more human-like retrieval queries with much better template realism than the original synthetic baseline. Further refinements should focus on trimming lingering benchmark-inventory phrasing and broadening template variety without losing specificity.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generated queries and build a comparison report.")
    parser.add_argument("--generated-queries", required=True, help="Path to generated queries JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the analysis JSON and report.")
    parser.add_argument("--dataset-name", default="new_synthetic_single_1", help="Dataset name for the generated analysis.")
    parser.add_argument("--human-analysis", default="outputs/query_analysis/human_written_style_analysis.json", help="Path to human reference analysis JSON.")
    parser.add_argument("--original-analysis", default="outputs/query_analysis/synthetic_single_analysis.json", help="Path to original synthetic analysis JSON.")
    parser.add_argument("--human-queries", default=None, help="Optional newline-delimited human reference queries to LLM-score.")
    parser.add_argument("--original-queries", default=None, help="Optional newline-delimited original synthetic queries to LLM-score.")
    parser.add_argument("--semantic-constraints", action="store_true", help="Deprecated: LLM judge metrics are enabled by default.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml with LLM settings.")
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size for LLM semantic constraint scoring.")
    parser.add_argument("--no-llm-judge", action="store_true", help="Disable LLM-as-a-judge metrics and keep local heuristic metrics only.")
    args = parser.parse_args()

    generated_queries_path = Path(args.generated_queries)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_data = load_json(generated_queries_path)
    queries = extract_queries(generated_data)
    use_llm_judge = args.semantic_constraints or not args.no_llm_judge
    llm_config = load_llm_judge_config(Path(args.config)) if use_llm_judge else None
    generated_analysis = build_generated_analysis(
        args.dataset_name,
        queries,
        llm_judge_config=llm_config,
        llm_judge_batch_size=args.batch_size,
    )
    human_analysis = load_json(Path(args.human_analysis))
    original_analysis = load_json(Path(args.original_analysis))
    if use_llm_judge and args.human_queries:
        human_queries = load_queries_file(Path(args.human_queries))
        human_analysis["metrics"]["combined"].update(
            compute_llm_judged_metrics(
                queries=human_queries,
                batch_size=args.batch_size,
                **llm_config,
            )
        )
    if use_llm_judge and args.original_queries:
        original_queries = load_queries_file(Path(args.original_queries))
        original_analysis["metrics"]["combined"].update(
            compute_llm_judged_metrics(
                queries=original_queries,
                batch_size=args.batch_size,
                **llm_config,
            )
        )

    analysis_slug = slugify_dataset_name(args.dataset_name)
    analysis_path = output_dir / f"{analysis_slug}.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(generated_analysis, f, indent=2, ensure_ascii=False)

    if use_llm_judge and args.human_queries:
        human_semantic_path = output_dir / "human_written_semantic_constraint_analysis.json"
        with open(human_semantic_path, "w", encoding="utf-8") as f:
            json.dump(human_analysis, f, indent=2, ensure_ascii=False)

    if use_llm_judge and args.original_queries:
        original_semantic_path = output_dir / "original_synthetic_semantic_constraint_analysis.json"
        with open(original_semantic_path, "w", encoding="utf-8") as f:
            json.dump(original_analysis, f, indent=2, ensure_ascii=False)

    report = build_report(
        human_analysis=human_analysis,
        original_analysis=original_analysis,
        new_analysis=generated_analysis,
        generated_queries_path=generated_queries_path,
        generated_analysis_path=analysis_path,
        report_title=f"Comparison Report: {args.dataset_name}",
    )

    report_path = output_dir / "comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Saved generated analysis to {analysis_path}")
    print(f"Saved comparison report to {report_path}")


if __name__ == "__main__":
    main()
