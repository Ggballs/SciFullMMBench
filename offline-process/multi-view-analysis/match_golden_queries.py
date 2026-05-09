from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openreview_pipeline.llm.base import create_openai_compatible_backend, load_llm_config

VALID_LABELS = {"motivation", "method", "experiment", "unclear"}


@dataclass(frozen=True)
class GoldenQuery:
    query: str
    source_path: str
    row_index: int


@dataclass(frozen=True)
class ViewMatch:
    query: str
    view: str
    confidence: float
    rationale: str
    source_path: str
    row_index: int


def _extract_query_from_row(row: dict[str, Any], query_column: str) -> Optional[str]:
    value = row.get(query_column)
    if value is None and query_column != "query":
        value = row.get("query")
    if value is None:
        return None
    query = str(value).strip()
    return query or None


def load_golden_queries_from_path(path: Path, query_column: str = "query") -> list[GoldenQuery]:
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]] = []

    if suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            rows = [item for item in value if isinstance(item, dict)]
        elif isinstance(value, dict):
            for candidate_key in ("queries", "data", "items", "examples"):
                candidate = value.get(candidate_key)
                if isinstance(candidate, list):
                    rows = [item for item in candidate if isinstance(item, dict)]
                    break
            if not rows:
                rows = [value]
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported golden-query file extension: {path.suffix}")

    queries: list[GoldenQuery] = []
    for index, row in enumerate(rows, 1):
        query = _extract_query_from_row(row, query_column)
        if query:
            queries.append(GoldenQuery(query=query, source_path=str(path), row_index=index))
    return queries


def load_golden_queries(paths: Iterable[Path], query_column: str = "query") -> list[GoldenQuery]:
    queries: list[GoldenQuery] = []
    for path in paths:
        queries.extend(load_golden_queries_from_path(path, query_column=query_column))
    return queries


def build_match_prompt(queries: list[GoldenQuery]) -> str:
    numbered = "\n".join(f"{idx}. {item.query}" for idx, item in enumerate(queries, 1))
    return f"""You are categorizing real researcher search queries for a scientific paper retrieval system.

Assign each query to exactly one view:
- motivation: the query asks about a research problem, need, gap, goal, hypothesis, or why the work matters
- method: the query asks about a proposed approach, model, algorithm, system, dataset construction process, or implementation design
- experiment: the query asks about evaluation setup, benchmarks, test datasets, metrics, baselines, ablations, empirical findings, comparisons, or observed limitations
- unclear: the query cannot be assigned to one view with reasonable confidence

Return valid JSON only with this shape:
{{
  "results": [
    {{
      "index": 1,
      "view": "motivation|method|experiment|unclear",
      "confidence": 0.0,
      "rationale": "short reason"
    }}
  ]
}}

Queries:
{numbered}
"""


def _normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if label in {"result", "results", "experiments", "outcome", "outcomes", "finding", "findings"}:
        return "experiment"
    if label in {"dataset", "method_dataset", "approach", "algorithm"}:
        return "method"
    if label in {"contribution", "problem", "gap", "novelty"}:
        return "motivation"
    return label if label in VALID_LABELS else "unclear"


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))


def normalize_match_results(queries: list[GoldenQuery], raw_response: dict[str, Any]) -> list[ViewMatch]:
    raw_items = raw_response.get("results", [])
    if not isinstance(raw_items, list):
        raw_items = []

    by_index: dict[int, dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        by_index[index] = item

    matches: list[ViewMatch] = []
    for index, query in enumerate(queries, 1):
        item = by_index.get(index, {})
        matches.append(
            ViewMatch(
                query=query.query,
                view=_normalize_label(item.get("view")),
                confidence=_normalize_confidence(item.get("confidence")),
                rationale=str(item.get("rationale", "")).strip(),
                source_path=query.source_path,
                row_index=query.row_index,
            )
        )
    return matches


def batched(items: list[GoldenQuery], batch_size: int) -> Iterable[list[GoldenQuery]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def match_queries(queries: list[GoldenQuery], backend: Any, batch_size: int) -> list[ViewMatch]:
    matches: list[ViewMatch] = []
    for batch in batched(queries, max(1, batch_size)):
        raw_response = backend.generate_json(build_match_prompt(batch))
        matches.extend(normalize_match_results(batch, raw_response))
    return matches


def distribution(matches: list[ViewMatch]) -> dict[str, Any]:
    counts = Counter(match.view for match in matches)
    total = len(matches)
    return {
        "total_queries": total,
        "labels": {
            label: {
                "count": int(counts.get(label, 0)),
                "ratio": (counts.get(label, 0) / total) if total else 0.0,
            }
            for label in sorted(VALID_LABELS)
        },
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_backend(config_path: Path):
    llm_config = load_llm_config(config_path)
    return create_openai_compatible_backend(
        base_url=str(llm_config["base_url"]),
        api_tokens=[str(token) for token in llm_config["api_tokens"]],
        model=str(llm_config["model"]),
        max_tokens=int(llm_config.get("max_tokens", 4096)),
        temperature=float(llm_config.get("temperature", 0.0)),
        seed=llm_config.get("seed"),
        embedding_model=llm_config.get("embedding_model"),
        per_key_request_interval_seconds=float(llm_config.get("per_key_request_interval_seconds", 0.0)),
        per_key_max_concurrent_requests=int(llm_config.get("per_key_max_concurrent_requests", 1)),
        max_retries=int(llm_config.get("max_retries", 3)),
        retry_backoff_seconds=float(llm_config.get("retry_backoff_seconds", 8.0)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Match golden human queries to multi-view labels.")
    parser.add_argument("--golden-queries", nargs="+", required=True, help="JSONL/JSON/CSV golden query files.")
    parser.add_argument("--output-dir", required=True, help="Directory for match outputs.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--query-column", default="query", help="Column/key containing query text.")
    parser.add_argument("--batch-size", type=int, default=25, help="LLM judge batch size.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [Path(path).expanduser().resolve() for path in args.golden_queries]
    queries = load_golden_queries(paths, query_column=args.query_column)
    backend = build_backend(Path(args.config).expanduser().resolve())
    matches = match_queries(queries, backend, max(1, int(args.batch_size)))
    match_rows = [asdict(match) for match in matches]

    write_json(output_dir / "golden_view_matches.json", {"matches": match_rows})
    write_csv(output_dir / "golden_view_matches.csv", match_rows)
    write_json(output_dir / "golden_view_distribution.json", distribution(matches))
    print(f"Saved golden-query view matches to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
