#!/usr/bin/env python3
"""Generate queries, then analyze them in one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openreview_pipeline.runner import parse_stage_spec


def infer_dataset_name(input_path: Path | None) -> str:
    if input_path is None:
        return "generated"
    stem = input_path.stem
    if "02_summarized" in stem:
        stem = stem.replace("02_summarized", "generated")
    return stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run query generation followed by generated-query analysis.",
    )
    parser.add_argument(
        "pipeline_input",
        nargs="?",
        help="Input for the first selected generation stage that needs one.",
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Optional output run directory",
    )
    parser.add_argument(
        "dataset_name",
        nargs="?",
        help="Optional dataset label used in analysis outputs",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config.yaml"),
        help="Path to config.yaml forwarded to analysis",
    )
    parser.add_argument(
        "--stages",
        default="3",
        help="Contiguous stage slice forwarded to run_query_generation.py. Must include stage 3.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    pipeline_input = None
    if args.pipeline_input:
        pipeline_input = Path(args.pipeline_input).expanduser().resolve()
        if not pipeline_input.is_file():
            parser.error(f"Generation input file not found: {pipeline_input}")

    try:
        stages = parse_stage_spec(args.stages)
    except ValueError as exc:
        parser.error(str(exc))
    if "generate_queries" not in stages:
        parser.error("The combined script requires stage 3 (generate queries) to be included.")

    dataset_name = args.dataset_name or infer_dataset_name(pipeline_input)
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else (REPO_ROOT / "outputs/query_pipeline_runs" / dataset_name)
    )
    generated_output = run_dir / "generated_queries.json"
    analysis_dir = run_dir / "analysis"
    pipeline_dir = run_dir / "pipeline"

    run_dir.mkdir(parents=True, exist_ok=True)

    generation_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "run_query_generation.py"),
        "--stages",
        args.stages,
        "--queries-output",
        str(generated_output),
        "--output-dir",
        str(pipeline_dir),
        "--config",
        str(Path(args.config).expanduser().resolve()),
    ]
    if pipeline_input is not None:
        generation_cmd.insert(2, str(pipeline_input))
    subprocess.run(generation_cmd, check=True, cwd=REPO_ROOT)

    analysis_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "run_generated_query_analysis.py"),
        str(generated_output),
        str(analysis_dir),
        dataset_name,
        "--config",
        str(Path(args.config).expanduser().resolve()),
    ]
    subprocess.run(analysis_cmd, check=True, cwd=REPO_ROOT)

    print(f"Generated queries: {generated_output}")
    print(f"Analysis output: {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
