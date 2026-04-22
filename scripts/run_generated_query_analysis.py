#!/usr/bin/env python3
"""Analyze generated queries against human and synthetic references."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from query_analysis.analyze_generated_queries import (
    build_generated_analysis,
    build_report,
    extract_queries,
    load_json,
    load_queries_file,
    slugify_dataset_name,
)
from query_analysis.llm_judge import (
    compute_llm_judged_metrics,
    compute_llm_semantic_constraint_metrics,
    load_llm_config,
)


def parse_env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def pick_existing_path(*candidates: Path) -> Optional[Path]:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def default_human_analysis() -> Optional[Path]:
    return pick_existing_path(
        REPO_ROOT / "outputs/query_analysis_new/03_litsearch_human/style_analysis.json",
        REPO_ROOT / "outputs/query_aalysis_new_5_size/03_litsearch_human/style_analysis.json",
        REPO_ROOT / "outputs/query_analysis/human_written_style_analysis.json",
    )


def default_original_analysis() -> Optional[Path]:
    return pick_existing_path(
        REPO_ROOT / "outputs/query_analysis_new/01_original_synthetic/style_analysis.json",
        REPO_ROOT / "outputs/query_aalysis_new_5_size/01_original_synthetic/style_analysis.json",
        REPO_ROOT / "outputs/query_analysis/synthetic_single_analysis.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run generated-query comparison analysis and produce a report.",
    )
    parser.add_argument(
        "generated_queries_json",
        help="Path to generated queries JSON",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Optional output directory",
    )
    parser.add_argument(
        "dataset_name",
        nargs="?",
        help="Optional dataset label used in the report",
    )
    parser.add_argument(
        "--human-analysis",
        default=os.environ.get("HUMAN_ANALYSIS"),
        help="Path to human reference analysis JSON",
    )
    parser.add_argument(
        "--original-analysis",
        default=os.environ.get("ORIGINAL_ANALYSIS"),
        help="Path to original synthetic reference analysis JSON",
    )
    parser.add_argument(
        "--human-queries",
        default=os.environ.get("HUMAN_QUERIES"),
        help="Optional newline-delimited human queries for rescoring",
    )
    parser.add_argument(
        "--original-queries",
        default=os.environ.get("ORIGINAL_QUERIES"),
        help="Optional newline-delimited original synthetic queries for rescoring",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("CONFIG_PATH", str(REPO_ROOT / "config.yaml")),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "5")),
        help="Batch size for LLM judge scoring",
    )
    parser.add_argument(
        "--llm-judge-mode",
        choices=["batch", "single_query"],
        default=os.environ.get("LLM_JUDGE_MODE", "batch"),
        help="Judge mode for LLM scoring",
    )
    parser.add_argument(
        "--llm-max-concurrency",
        type=int,
        default=int(os.environ.get("LLM_MAX_CONCURRENCY", "1")),
        help="Maximum concurrent LLM judge requests",
    )
    parser.add_argument(
        "--llm-seed",
        type=int,
        default=int(os.environ["LLM_SEED"]) if os.environ.get("LLM_SEED") else None,
        help="Optional seed forwarded to the LLM backend",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        default=parse_env_flag("NO_LLM_JUDGE", False),
        help="Disable LLM judge and keep only local structural metrics",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    generated_queries_path = Path(args.generated_queries_json).expanduser().resolve()
    if not generated_queries_path.is_file():
        parser.error(f"Generated queries file not found: {generated_queries_path}")

    dataset_name = args.dataset_name or generated_queries_path.stem
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (REPO_ROOT / "outputs/query_analysis_runs" / dataset_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    human_analysis_path = (
        Path(args.human_analysis).expanduser().resolve()
        if args.human_analysis
        else default_human_analysis()
    )
    original_analysis_path = (
        Path(args.original_analysis).expanduser().resolve()
        if args.original_analysis
        else default_original_analysis()
    )

    if human_analysis_path is None or not human_analysis_path.is_file():
        parser.error("Human reference analysis JSON not found. Pass --human-analysis explicitly.")
    if original_analysis_path is None or not original_analysis_path.is_file():
        parser.error("Original synthetic analysis JSON not found. Pass --original-analysis explicitly.")

    use_llm_judge = not args.no_llm_judge
    llm_config = load_llm_config(Path(args.config).expanduser().resolve()) if use_llm_judge else None
    llm_seed = args.llm_seed
    if llm_config and llm_seed is None:
        llm_seed = llm_config.pop("seed", None)
    elif llm_config:
        llm_config.pop("seed", None)

    generated_data = load_json(generated_queries_path)
    queries = extract_queries(generated_data)
    generated_analysis = build_generated_analysis(
        dataset_name=dataset_name,
        queries=queries,
        llm_judge_config=llm_config,
        llm_judge_batch_size=args.batch_size,
        llm_judge_mode=args.llm_judge_mode,
        llm_max_concurrency=args.llm_max_concurrency,
        llm_seed=llm_seed,
    )

    human_analysis = load_json(human_analysis_path)
    original_analysis = load_json(original_analysis_path)

    if use_llm_judge and args.human_queries:
        human_queries = load_queries_file(Path(args.human_queries).expanduser().resolve())
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
        original_queries = load_queries_file(Path(args.original_queries).expanduser().resolve())
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

    analysis_path = output_dir / f"{slugify_dataset_name(dataset_name)}.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(generated_analysis, f, indent=2, ensure_ascii=False)

    report = build_report(
        human_analysis=human_analysis,
        original_analysis=original_analysis,
        new_analysis=generated_analysis,
        generated_queries_path=generated_queries_path,
        generated_analysis_path=analysis_path,
        human_analysis_path=human_analysis_path,
        original_analysis_path=original_analysis_path,
        report_title=f"Comparison Report: {dataset_name}",
    )
    report_path = output_dir / "comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Generated analysis saved to: {analysis_path}")
    print(f"Comparison report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
