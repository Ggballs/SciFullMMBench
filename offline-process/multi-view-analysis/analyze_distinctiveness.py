from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openreview_pipeline.llm.base import create_openai_compatible_backend, load_llm_config

VALID_VIEWS = ("motivation", "method", "experiment")


@dataclass(frozen=True)
class QueryRecord:
    paper_id: str
    paper_title: str
    source_view: str
    query_text: str


def load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def extract_generated_queries(data: dict[str, Any]) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    for paper in data.get("papers_queries", []):
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("paper_id", "")).strip()
        paper_title = str(paper.get("paper_title", "")).strip()
        for query in paper.get("queries_by_view", []):
            if not isinstance(query, dict):
                continue
            query_text = str(query.get("query_text", "")).strip()
            source_view = str(query.get("source_view", "")).strip()
            if query_text and source_view:
                records.append(
                    QueryRecord(
                        paper_id=paper_id,
                        paper_title=paper_title,
                        source_view=source_view,
                        query_text=query_text,
                    )
                )
    return records


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_values(embeddings: np.ndarray, left_indices: list[int], right_indices: list[int]) -> list[float]:
    if not left_indices or not right_indices:
        return []
    normalized = l2_normalize(embeddings)
    values: list[float] = []
    for left in left_indices:
        for right in right_indices:
            values.append(float(np.dot(normalized[left], normalized[right])))
    return values


def within_view_values(embeddings: np.ndarray, indices: list[int]) -> list[float]:
    if len(indices) < 2:
        return []
    normalized = l2_normalize(embeddings)
    return [
        float(np.dot(normalized[left], normalized[right]))
        for left, right in combinations(indices, 2)
    ]


def summary_stats(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {
            "available": False,
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "available": True,
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _safe_distinctiveness(cross_view_values: list[float]) -> Optional[float]:
    if not cross_view_values:
        return None
    return float(1.0 - np.mean(cross_view_values))


def compute_distinctiveness(records: list[QueryRecord], embeddings: list[list[float]]) -> dict[str, Any]:
    if len(records) != len(embeddings):
        raise ValueError("records and embeddings must have the same length")
    if not records:
        return {
            "total_queries": 0,
            "total_papers": 0,
            "per_paper": [],
            "aggregate": {},
        }

    matrix = np.asarray(embeddings, dtype=float)
    by_paper: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_paper.setdefault(record.paper_id, []).append(index)

    per_paper: list[dict[str, Any]] = []
    aggregate_within: dict[str, list[float]] = {}
    aggregate_cross: dict[str, list[float]] = {}
    aggregate_distinctiveness: list[float] = []

    for paper_id, paper_indices in by_paper.items():
        paper_title = records[paper_indices[0]].paper_title
        indices_by_view: dict[str, list[int]] = {}
        for index in paper_indices:
            indices_by_view.setdefault(records[index].source_view, []).append(index)

        within: dict[str, dict[str, Any]] = {}
        for view_name, indices in sorted(indices_by_view.items()):
            values = within_view_values(matrix, indices)
            within[view_name] = summary_stats(values)
            aggregate_within.setdefault(view_name, []).extend(values)

        cross: dict[str, dict[str, Any]] = {}
        paper_cross_values: list[float] = []
        for left_view, right_view in combinations(sorted(indices_by_view), 2):
            pair_key = f"{left_view}__{right_view}"
            values = cosine_values(matrix, indices_by_view[left_view], indices_by_view[right_view])
            cross[pair_key] = summary_stats(values)
            paper_cross_values.extend(values)
            aggregate_cross.setdefault(pair_key, []).extend(values)

        distinctiveness = _safe_distinctiveness(paper_cross_values)
        if distinctiveness is not None:
            aggregate_distinctiveness.append(distinctiveness)

        per_paper.append(
            {
                "paper_id": paper_id,
                "paper_title": paper_title,
                "query_count": len(paper_indices),
                "views": {view: len(indices) for view, indices in sorted(indices_by_view.items())},
                "within_view_similarity": within,
                "cross_view_similarity": cross,
                "distinctiveness": distinctiveness,
            }
        )

    return {
        "total_queries": len(records),
        "total_papers": len(by_paper),
        "per_paper": per_paper,
        "aggregate": {
            "within_view_similarity": {
                view: summary_stats(values)
                for view, values in sorted(aggregate_within.items())
            },
            "cross_view_similarity": {
                pair: summary_stats(values)
                for pair, values in sorted(aggregate_cross.items())
            },
            "distinctiveness": summary_stats(aggregate_distinctiveness),
        },
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate = analysis.get("aggregate", {})
    for view, stats in aggregate.get("within_view_similarity", {}).items():
        rows.append({"metric": "within_view_similarity", "group": view, **stats})
    for pair, stats in aggregate.get("cross_view_similarity", {}).items():
        rows.append({"metric": "cross_view_similarity", "group": pair, **stats})
    rows.append({"metric": "distinctiveness", "group": "overall", **aggregate.get("distinctiveness", {})})
    return rows


def by_paper_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in analysis.get("per_paper", []):
        base = {
            "paper_id": item.get("paper_id"),
            "paper_title": item.get("paper_title"),
            "query_count": item.get("query_count"),
            "distinctiveness": item.get("distinctiveness"),
        }
        for view, count in item.get("views", {}).items():
            base[f"view_count_{view}"] = count
        rows.append(base)
    return rows


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def embed_records(records: list[QueryRecord], backend: Any, embedding_model: str, batch_size: int) -> list[list[float]]:
    embeddings: list[list[float]] = []
    texts = [record.query_text for record in records]
    for batch in batched(texts, batch_size):
        embeddings.extend(backend.embed_texts(batch, model=embedding_model))
    return embeddings


def build_backend(config_path: Path, embedding_model: Optional[str]):
    llm_config = load_llm_config(config_path)
    return create_openai_compatible_backend(
        base_url=str(llm_config["base_url"]),
        api_tokens=[str(token) for token in llm_config["api_tokens"]],
        model=str(llm_config["model"]),
        max_tokens=int(llm_config.get("max_tokens", 4096)),
        temperature=float(llm_config.get("temperature", 0.0)),
        seed=llm_config.get("seed"),
        embedding_model=embedding_model or llm_config.get("embedding_model"),
        per_key_request_interval_seconds=float(llm_config.get("per_key_request_interval_seconds", 0.0)),
        per_key_max_concurrent_requests=int(llm_config.get("per_key_max_concurrent_requests", 1)),
        max_retries=int(llm_config.get("max_retries", 3)),
        retry_backoff_seconds=float(llm_config.get("retry_backoff_seconds", 8.0)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze generated query distinctiveness by multi-view.")
    parser.add_argument("--generated-queries", required=True, help="Path to stage-3 generated queries JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for analysis outputs.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--embedding-model", default=None, help="Embedding model name; overrides llm.embedding_model.")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    generated_queries_path = Path(args.generated_queries).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = extract_generated_queries(load_json_object(generated_queries_path))
    backend = build_backend(Path(args.config).expanduser().resolve(), args.embedding_model)
    embedding_model = args.embedding_model or getattr(backend, "embedding_model", None)
    if not embedding_model:
        raise ValueError("Embedding model is required via --embedding-model or llm.embedding_model.")

    embeddings = embed_records(records, backend, str(embedding_model), max(1, int(args.batch_size)))
    analysis = compute_distinctiveness(records, embeddings)

    write_json(output_dir / "generated_view_distinctiveness.json", analysis)
    write_csv(output_dir / "generated_view_distinctiveness.csv", aggregate_rows(analysis))
    write_csv(output_dir / "generated_view_distinctiveness_by_paper.csv", by_paper_rows(analysis))
    print(f"Saved distinctiveness analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
