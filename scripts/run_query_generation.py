#!/usr/bin/env python3
"""Run a selected slice of the OpenReview pipeline from Python."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openreview_pipeline.runner import parse_stage_spec, run_selected_stages


def infer_queries_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if "02_summarized" in stem:
        stem = stem.replace("02_summarized", "03_queries")
    else:
        stem = f"{stem}_queries"
    return input_path.with_name(f"{stem}.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a contiguous slice of the OpenReview pipeline.",
    )
    parser.add_argument(
        "pipeline_input",
        nargs="?",
        help="Input for the first selected stage that needs one (e.g. filtered JSON for stage 2, summarized JSON for stage 3).",
    )
    parser.add_argument(
        "generated_queries_json",
        nargs="?",
        help="Backward-compatible alias for the stage-3 generated queries output path.",
    )
    parser.add_argument(
        "--stages",
        default="3",
        help="Contiguous stage slice to run, e.g. 0-3, 2-3, 3, 3-4.",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for default stage outputs.",
    )
    parser.add_argument("--download-output", help="Optional output path for stage 0.")
    parser.add_argument("--filtered-output", help="Optional output path for stage 1.")
    parser.add_argument("--summarized-output", help="Optional output path for stage 2.")
    parser.add_argument("--queries-output", help="Optional output path for stage 3.")
    parser.add_argument("--filtered-queries-output", help="Optional output path for stage 4.")
    parser.add_argument("--hard-negatives-output", help="Optional output path for stage 5.")
    parser.add_argument("--venue", default="ICLR", help="Venue used by stage 0 download.")
    parser.add_argument("--year", type=int, default=None, help="Year used by stage 0 download.")
    parser.add_argument("--download-limit", type=int, default=None, help="Optional paper limit for stage 0.")
    parser.add_argument("--filter-limit", type=int, default=None, help="Optional paper limit for stage 1.")
    parser.add_argument("--llm-limit", type=int, default=None, help="Optional LLM paper limit for stage 2.")
    parser.add_argument("--rules-config", default=None, help="Optional rules config for stage 1.")
    parser.add_argument("--base-url", default=None, help="LLM API base URL override.")
    parser.add_argument("--api-token", default=None, help="LLM API token override.")
    parser.add_argument("--model", default=None, help="LLM model override.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional threshold forwarded to stage 4.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        stages = parse_stage_spec(args.stages)
    except ValueError as exc:
        parser.error(str(exc))

    input_path = None
    if args.pipeline_input:
        input_path = Path(args.pipeline_input).expanduser().resolve()
        if not input_path.is_file():
            parser.error(f"Input file not found: {input_path}")
    elif stages[0] != "download":
        parser.error(f"Stage selection '{args.stages}' requires an input file.")

    config_path = Path(args.config).expanduser().resolve()

    queries_output = args.queries_output or args.generated_queries_json
    if "generate_queries" not in stages and queries_output:
        parser.error("A queries output path was provided, but stage 3 is not selected.")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir is None and input_path is not None:
        output_dir = input_path.parent

    queries_output_path = None
    if queries_output:
        queries_output_path = Path(queries_output).expanduser().resolve()
    elif stages == ["generate_queries"] and input_path is not None:
        queries_output_path = infer_queries_output_path(input_path)

    try:
        paths = run_selected_stages(
            stages,
            input_path=input_path,
            output_dir=output_dir,
            downloaded_path=args.download_output,
            filtered_path=args.filtered_output,
            summarized_path=args.summarized_output,
            queries_path=queries_output_path,
            filtered_queries_path=args.filtered_queries_output,
            hard_negatives_path=args.hard_negatives_output,
            venue=args.venue,
            year=args.year,
            download_limit=args.download_limit,
            filter_limit=args.filter_limit,
            llm_limit=args.llm_limit,
            rules_config_path=args.rules_config,
            config_path=config_path,
            base_url=args.base_url,
            api_token=args.api_token,
            model=args.model,
            threshold=args.threshold,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    print(f"Completed stages: {', '.join(stages)}")
    if "download" in stages:
        print(f"Downloaded dataset: {paths.downloaded_path}")
    if "filter" in stages:
        print(f"Filtered papers: {paths.filtered_path}")
    if "summarize" in stages:
        print(f"Summaries: {paths.summarized_path}")
    if "generate_queries" in stages:
        print(f"Generated queries: {paths.queries_path}")
    if "filter_queries" in stages:
        print(f"Filtered queries: {paths.filtered_queries_path}")
    if "hard_negative_mining" in stages:
        print(f"Hard negatives: {paths.hard_negatives_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
