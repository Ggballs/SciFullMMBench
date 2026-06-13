#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))  # needed for per_view_retrieval's own imports
_spec = importlib.util.spec_from_file_location(
    "per_view_retrieval",
    str(_SRC / "openreview_pipeline" / "evaluations" / "per_view_retrieval.py"),
)
_per_view = importlib.util.module_from_spec(_spec)
sys.modules["per_view_retrieval"] = _per_view
_spec.loader.exec_module(_per_view)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run per-view retrieval evaluation.")
    parser.add_argument(
        "--config",
        default="configs/per_view_retrieval_maple_final.yaml",
        help="Path to evaluation YAML config.",
    )
    args = parser.parse_args()

    config = _per_view.load_config(Path(args.config).expanduser().resolve())
    result = _per_view.run_per_view_retrieval_evaluation(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
