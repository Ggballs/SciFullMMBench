from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openreview_pipeline.runner import resolve_generate_query_settings
from openreview_pipeline.utils.db.golden_query_embeddings import (
    GoldenQueryEmbeddingRow,
    ensure_schema,
    get_engine,
    upsert_golden_query_embeddings,
)
from openreview_pipeline.utils.embeddings import TextEmbedder, build_text_embedder
from openreview_pipeline.utils.golden_retrieval_icl import DEFAULT_OUTPUT_PATH


def _load_embedding_rows(path: Path, embedder: TextEmbedder) -> list[GoldenQueryEmbeddingRow]:
    import json

    raw_rows = json.loads(path.read_text(encoding="utf-8"))
    eligible = [row for row in raw_rows if str(row.get("indexing_content") or "").strip()]

    vectors = embedder.embed_texts([str(row.get("indexing_content") or "") for row in eligible])
    rows: list[GoldenQueryEmbeddingRow] = []
    for row, embedding in zip(eligible, vectors):
        rows.append(
            GoldenQueryEmbeddingRow(
                example_id=str(row.get("example_id") or ""),
                query_id=str(row.get("query_id") or ""),
                query_type=str(row.get("query_type") or "").strip().upper(),
                view_label=str(row.get("view_label") or "").strip(),
                query_text=str(row.get("query") or "").strip(),
                target_papers=[str(title) for title in row.get("target_papers", [])],
                answer_original_content=str(row.get("answer_original_content") or ""),
                answer_tldr=str(row.get("answer_tldr") or ""),
                human_view_note=str(row.get("human_view_note") or ""),
                indexing_content=str(row.get("indexing_content") or "").strip(),
                retrieval_content=str(row.get("retrieval_content") or "").strip(),
                embedding=embedding,
            )
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import golden query embeddings into pgvector.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--db-url", default=None, help="PostgreSQL SQLAlchemy URL.")
    parser.add_argument(
        "--golden-classifications-path",
        default=None,
        help="Normalized retrieval-ICL examples JSON.",
    )
    parser.add_argument("--bge-model-path", default=None, help="BGE-M3 model path.")
    parser.add_argument("--bge-device", default=None, help="BGE device, e.g. cuda:2.")
    parser.add_argument(
        "--embedding-service-url",
        default=None,
        help="HTTP embedding endpoint. If set, this is used instead of loading BGE locally.",
    )
    parser.add_argument(
        "--embedding-service-timeout",
        type=float,
        default=120.0,
        help="HTTP embedding service timeout in seconds.",
    )
    parser.add_argument("--embedding-dimension", type=int, default=1024, help="pgvector dimension.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = resolve_generate_query_settings(args.config)
    db_url = args.db_url or settings["golden_embedding_db_url"]
    golden_path = Path(
        args.golden_classifications_path or settings.get("golden_classifications_path") or DEFAULT_OUTPUT_PATH
    ).expanduser().resolve()
    embedder = build_text_embedder(
        model_path=str(args.bge_model_path or settings["bge_model_path"]),
        device=str(args.bge_device or settings["bge_device"]),
        service_url=str(
            args.embedding_service_url or settings.get("embedding_service_url") or ""
        ).strip()
        or None,
        timeout_seconds=float(args.embedding_service_timeout),
    )
    engine = get_engine(str(db_url))
    ensure_schema(engine, embedding_dimension=int(args.embedding_dimension))
    rows = _load_embedding_rows(golden_path, embedder)
    count = upsert_golden_query_embeddings(engine, rows)
    print(f"Imported {count} golden query embedding rows into PostgreSQL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
