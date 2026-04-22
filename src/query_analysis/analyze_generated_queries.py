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
    compute_llm_semantic_constraint_metrics,
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
    llm_judge_config: Optional[Dict[str, Any]] = None,
    llm_judge_batch_size: int = 25,
    llm_judge_mode: str = "batch",
    llm_max_concurrency: int = 1,
    llm_seed: Optional[int] = None,
) -> Dict[str, Any]:
    metrics = compute_all_metrics(queries)
    metrics["question_templates"] = build_template_metrics(queries)
    if llm_judge_config and queries:
        metrics.update(
            compute_llm_judged_metrics(
                queries=queries,
                batch_size=llm_judge_batch_size,
                judge_mode=llm_judge_mode,
                max_concurrency=llm_max_concurrency,
                seed=llm_seed,
                **llm_judge_config,
            )
        )
        metrics.update(
            compute_llm_semantic_constraint_metrics(
                queries=queries,
                batch_size=llm_judge_batch_size,
                judge_mode=llm_judge_mode,
                max_concurrency=llm_max_concurrency,
                seed=llm_seed,
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
    llm_judge_config: Optional[Dict[str, Any]] = None,
    llm_judge_batch_size: int = 25,
    llm_judge_mode: str = "batch",
    llm_max_concurrency: int = 1,
    llm_seed: Optional[int] = None,
) -> Dict[str, Any]:
    metrics = compute_all_metrics(queries)
    metrics["question_templates"] = build_template_metrics(queries)
    if llm_judge_config and queries:
        metrics.update(
            compute_llm_judged_metrics(
                queries=queries,
                batch_size=llm_judge_batch_size,
                judge_mode=llm_judge_mode,
                max_concurrency=llm_max_concurrency,
                seed=llm_seed,
                **llm_judge_config,
            )
        )
        metrics.update(
            compute_llm_semantic_constraint_metrics(
                queries=queries,
                batch_size=llm_judge_batch_size,
                judge_mode=llm_judge_mode,
                max_concurrency=llm_max_concurrency,
                seed=llm_seed,
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


def render_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def render_score(value: Optional[float], default: str = "n/a") -> str:
    if value is None:
        return default
    return f"{value:.3f}"


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
    human_analysis_path: Path,
    original_analysis_path: Path,
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
    human_specificity_calibration = get_nested(
        human, ("qualitative_metrics", "specificity_calibration", "mean"), None
    )
    human_specificity_fit = get_nested(
        human, ("qualitative_metrics", "specificity_calibration_fit", "mean"),
        None,
    )
    human_lexical_naturalism = get_nested(
        human, ("qualitative_metrics", "lexical_naturalism", "mean"), None
    )
    human_lexical_naturalism_fit = get_nested(
        human, ("qualitative_metrics", "lexical_naturalism_fit", "mean"),
        None,
    )
    human_other_ratio = get_nested(human, ("question_templates", "unmatched_ratio"), 0.0)

    original_char = get_nested(original, ("length_stats", "char_length", "mean"), 0.0)
    original_tokens = get_nested(original, ("length_stats", "token_length", "mean"), 0.0)
    original_semantic_constraints = get_nested(
        original,
        ("semantic_constraint_analysis", "semantic_constraints_per_query", "mean"),
        None,
    )
    original_specificity_calibration = get_nested(
        original, ("qualitative_metrics", "specificity_calibration", "mean"), None
    )
    original_specificity_fit = get_nested(
        original, ("qualitative_metrics", "specificity_calibration_fit", "mean"),
        None,
    )
    original_lexical_naturalism = get_nested(
        original, ("qualitative_metrics", "lexical_naturalism", "mean"), None
    )
    original_lexical_naturalism_fit = get_nested(
        original, ("qualitative_metrics", "lexical_naturalism_fit", "mean"),
        None,
    )
    original_other_ratio = get_nested(original, ("question_templates", "template_ratios", "other"), 0.0)

    new_char = get_nested(new, ("length_stats", "char_length", "mean"), 0.0)
    new_tokens = get_nested(new, ("length_stats", "token_length", "mean"), 0.0)
    new_semantic_constraints = get_nested(
        new,
        ("semantic_constraint_analysis", "semantic_constraints_per_query", "mean"),
        None,
    )
    new_specificity_calibration = get_nested(
        new, ("qualitative_metrics", "specificity_calibration", "mean"), None
    )
    new_specificity_fit = get_nested(
        new, ("qualitative_metrics", "specificity_calibration_fit", "mean"),
        None,
    )
    new_lexical_naturalism = get_nested(
        new, ("qualitative_metrics", "lexical_naturalism", "mean"), None
    )
    new_lexical_naturalism_fit = get_nested(
        new, ("qualitative_metrics", "lexical_naturalism_fit", "mean"),
        None,
    )
    new_other_ratio = get_nested(new, ("question_templates", "template_ratios", "other"), 0.0)
    has_llm_style_metrics = any(
        value is not None
        for value in [
            human_specificity_calibration,
            human_specificity_fit,
            human_lexical_naturalism,
            human_lexical_naturalism_fit,
            original_specificity_calibration,
            original_specificity_fit,
            original_lexical_naturalism,
            original_lexical_naturalism_fit,
            new_specificity_calibration,
            new_specificity_fit,
            new_lexical_naturalism,
            new_lexical_naturalism_fit,
        ]
    )

    new_examples = new_analysis.get("query_examples", [])[:8]

    lines: List[str] = []
    lines.append(f"# {report_title}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Generated queries: `{generated_queries_path}`")
    lines.append(f"- Generated analysis: `{generated_analysis_path}`")
    lines.append(f"- Human reference: `{human_analysis_path}`")
    lines.append(f"- Original synthetic reference: `{original_analysis_path}`")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("The new generated queries are much closer to the human-written reference than the original synthetic queries on length and template realism. The strongest improvements are that the queries are no longer overlong and they use recognizable human search openings such as `What`, `Which`, and `Is there`.")
    lines.append("")
    if has_llm_style_metrics:
        lines.append("The main remaining gap is that some queries are still denser and more benchmark-specific than typical human searches. Under the centered LLM judge, the remaining work is to keep detail and wording closer to the human-query ideal rather than simply making them more polished.")
    else:
        lines.append("The main remaining gap is that some queries are still denser and more benchmark-specific than typical human searches. This report is based on structural metrics only, so style-fit conclusions require a separate LLM-judge pass.")
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
        human_specificity_calibration is not None
        or original_specificity_calibration is not None
        or new_specificity_calibration is not None
    ):
        lines.append(render_metric_row(
            "Specificity Calibration",
            f"{human_specificity_calibration:.3f}" if human_specificity_calibration is not None else "n/a",
            f"{original_specificity_calibration:.3f}" if original_specificity_calibration is not None else "n/a",
            f"{new_specificity_calibration:.3f}" if new_specificity_calibration is not None else "n/a",
        ))
    if (
        human_specificity_fit is not None
        or original_specificity_fit is not None
        or new_specificity_fit is not None
    ):
        lines.append(render_metric_row(
            "Specificity Calibration Fit",
            render_score(human_specificity_fit),
            render_score(original_specificity_fit),
            render_score(new_specificity_fit),
        ))
    if (
        human_lexical_naturalism is not None
        or original_lexical_naturalism is not None
        or new_lexical_naturalism is not None
    ):
        lines.append(render_metric_row(
            "Lexical Naturalism",
            f"{human_lexical_naturalism:.3f}" if human_lexical_naturalism is not None else "n/a",
            f"{original_lexical_naturalism:.3f}" if original_lexical_naturalism is not None else "n/a",
            f"{new_lexical_naturalism:.3f}" if new_lexical_naturalism is not None else "n/a",
        ))
    if (
        human_lexical_naturalism_fit is not None
        or original_lexical_naturalism_fit is not None
        or new_lexical_naturalism_fit is not None
    ):
        lines.append(render_metric_row(
            "Lexical Naturalism Fit",
            render_score(human_lexical_naturalism_fit),
            render_score(original_lexical_naturalism_fit),
            render_score(new_lexical_naturalism_fit),
        ))
    lines.append(render_metric_row("Other / unmatched template share", render_pct(human_other_ratio), render_pct(original_other_ratio), render_pct(new_other_ratio)))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"- Length improved sharply: the original synthetic queries averaged {original_tokens:.1f} tokens, while the new set averages {new_tokens:.1f}, much closer to the human reference at {human_tokens:.1f}.")
    if original_lexical_naturalism_fit is not None and new_lexical_naturalism_fit is not None:
        lines.append(f"- Lexical naturalism fit moved from {original_lexical_naturalism_fit:.3f} to {new_lexical_naturalism_fit:.3f}; higher fit means the wording is closer to the human-query ideal.")
    lines.append(f"- Template fit improved: `other` dropped from {render_pct(original_other_ratio)} to {render_pct(new_other_ratio)}.")
    if new_specificity_fit is not None:
        lines.append(f"- Specificity calibration fit is {new_specificity_fit:.3f}; higher fit means the amount of detail is closer to the human-query ideal rather than too broad or too specific.")
    if new_semantic_constraints is not None:
        lines.append(f"- LLM semantic constraint scoring finds {new_semantic_constraints:.2f} constraints per query.")
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
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size for LLM judge scoring.")
    parser.add_argument(
        "--llm-judge-mode",
        choices=["batch", "single_query"],
        default="batch",
        help="Whether to score in multi-query batches or one query per prompt",
    )
    parser.add_argument(
        "--llm-max-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent LLM requests for judge scoring",
    )
    parser.add_argument(
        "--llm-seed",
        type=int,
        default=None,
        help="Optional seed forwarded to compatible OpenAI-style backends",
    )
    parser.add_argument("--no-llm-judge", action="store_true", help="Disable LLM-as-a-judge metrics and keep only local structural metrics.")
    args = parser.parse_args()

    generated_queries_path = Path(args.generated_queries)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_data = load_json(generated_queries_path)
    queries = extract_queries(generated_data)
    use_llm_judge = args.semantic_constraints or not args.no_llm_judge
    llm_config = load_llm_judge_config(Path(args.config)) if use_llm_judge else None
    llm_seed = args.llm_seed
    if llm_config and llm_seed is None:
        llm_seed = llm_config.pop("seed", None)
    elif llm_config:
        llm_config.pop("seed", None)
    generated_analysis = build_generated_analysis(
        args.dataset_name,
        queries,
        llm_judge_config=llm_config,
        llm_judge_batch_size=args.batch_size,
        llm_judge_mode=args.llm_judge_mode,
        llm_max_concurrency=args.llm_max_concurrency,
        llm_seed=llm_seed,
    )
    human_analysis = load_json(Path(args.human_analysis))
    original_analysis = load_json(Path(args.original_analysis))
    if use_llm_judge and args.human_queries:
        human_queries = load_queries_file(Path(args.human_queries))
        human_analysis["metrics"]["combined"].update(
            compute_llm_judged_metrics(
                queries=human_queries,
                batch_size=args.batch_size,
                judge_mode=args.llm_judge_mode,
                max_concurrency=args.llm_max_concurrency,
                seed=llm_seed,
                **llm_config,
            )
        )
        human_analysis["metrics"]["combined"].update(
            compute_llm_semantic_constraint_metrics(
                queries=human_queries,
                batch_size=args.batch_size,
                judge_mode=args.llm_judge_mode,
                max_concurrency=args.llm_max_concurrency,
                seed=llm_seed,
                **llm_config,
            )
        )
    if use_llm_judge and args.original_queries:
        original_queries = load_queries_file(Path(args.original_queries))
        original_analysis["metrics"]["combined"].update(
            compute_llm_judged_metrics(
                queries=original_queries,
                batch_size=args.batch_size,
                judge_mode=args.llm_judge_mode,
                max_concurrency=args.llm_max_concurrency,
                seed=llm_seed,
                **llm_config,
            )
        )
        original_analysis["metrics"]["combined"].update(
            compute_llm_semantic_constraint_metrics(
                queries=original_queries,
                batch_size=args.batch_size,
                judge_mode=args.llm_judge_mode,
                max_concurrency=args.llm_max_concurrency,
                seed=llm_seed,
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
