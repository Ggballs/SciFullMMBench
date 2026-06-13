from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional


GOLDEN_QUERY_EMBEDDINGS_TABLE = "golden_query_embeddings"
VIEW_LABELS = {"motivation", "method", "experiment/result"}
QUERY_TYPES = {"IR", "QA"}


@dataclass(frozen=True)
class GoldenQueryEmbeddingRow:
    example_id: str
    query_id: str
    query_type: str
    view_label: str
    query_text: str
    target_papers: list[str]
    answer_original_content: str
    answer_tldr: str
    human_view_note: str
    indexing_content: str
    retrieval_content: str
    embedding: list[float]


@dataclass(frozen=True)
class GoldenQueryExample:
    example_id: str
    query_id: str
    query_type: str
    view_label: str
    query_text: str
    target_papers: list[str]
    answer_original_content: str
    answer_tldr: str
    human_view_note: str
    indexing_content: str
    retrieval_content: str
    distance: Optional[float] = None


def get_engine(db_url: Optional[str] = None):
    from sqlalchemy import create_engine

    url = db_url or os.getenv("GOLDEN_EMBEDDING_DB_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError(
            "Set stages.generate_queries.golden_embedding_db_url, "
            "GOLDEN_EMBEDDING_DB_URL, or DATABASE_URL to use golden query embeddings."
        )
    return create_engine(url, pool_pre_ping=True)


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.10g}" for value in vector) + "]"


def _json_list_from_db(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = value.split("|")
    if not isinstance(value, list):
        value = [value]
    return [str(label).strip() for label in value if str(label).strip()]


def ensure_schema(engine, embedding_dimension: int = 1024) -> None:
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {GOLDEN_QUERY_EMBEDDINGS_TABLE} (
                  example_id TEXT PRIMARY KEY,
                  query_id TEXT NOT NULL,
                  query_type TEXT NOT NULL,
                  view_label TEXT NOT NULL,
                  query_text TEXT NOT NULL,
                  target_papers JSONB NOT NULL DEFAULT '[]'::jsonb,
                  answer_original_content TEXT NOT NULL DEFAULT '',
                  answer_tldr TEXT NOT NULL DEFAULT '',
                  human_view_note TEXT NOT NULL DEFAULT '',
                  indexing_content TEXT NOT NULL DEFAULT '',
                  retrieval_content TEXT NOT NULL DEFAULT '',
                  embedding vector({int(embedding_dimension)}) NOT NULL,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  CONSTRAINT chk_golden_query_type
                    CHECK (query_type IN ('IR', 'QA')),
                  CONSTRAINT chk_golden_view_label
                    CHECK (view_label IN ('motivation', 'method', 'experiment/result'))
                )
                """
            )
        )
        conn.execute(text(f"ALTER TABLE {GOLDEN_QUERY_EMBEDDINGS_TABLE} ADD COLUMN IF NOT EXISTS example_id TEXT"))
        conn.execute(text(f"ALTER TABLE {GOLDEN_QUERY_EMBEDDINGS_TABLE} ADD COLUMN IF NOT EXISTS target_papers JSONB NOT NULL DEFAULT '[]'::jsonb"))
        conn.execute(text(f"ALTER TABLE {GOLDEN_QUERY_EMBEDDINGS_TABLE} ADD COLUMN IF NOT EXISTS answer_original_content TEXT NOT NULL DEFAULT ''"))
        conn.execute(text(f"ALTER TABLE {GOLDEN_QUERY_EMBEDDINGS_TABLE} ADD COLUMN IF NOT EXISTS answer_tldr TEXT NOT NULL DEFAULT ''"))
        conn.execute(text(f"ALTER TABLE {GOLDEN_QUERY_EMBEDDINGS_TABLE} ADD COLUMN IF NOT EXISTS human_view_note TEXT NOT NULL DEFAULT ''"))
        conn.execute(text(f"ALTER TABLE {GOLDEN_QUERY_EMBEDDINGS_TABLE} ADD COLUMN IF NOT EXISTS indexing_content TEXT NOT NULL DEFAULT ''"))
        conn.execute(text(f"ALTER TABLE {GOLDEN_QUERY_EMBEDDINGS_TABLE} ADD COLUMN IF NOT EXISTS retrieval_content TEXT NOT NULL DEFAULT ''"))
        conn.execute(text(f"ALTER TABLE {GOLDEN_QUERY_EMBEDDINGS_TABLE} ADD COLUMN IF NOT EXISTS specific INTEGER NOT NULL DEFAULT 1"))
        conn.execute(text(f"ALTER TABLE {GOLDEN_QUERY_EMBEDDINGS_TABLE} DROP CONSTRAINT IF EXISTS chk_golden_specific"))
        conn.execute(
            text(
                f"""
                ALTER TABLE {GOLDEN_QUERY_EMBEDDINGS_TABLE}
                ADD CONSTRAINT chk_golden_specific
                CHECK (specific IN (0, 1))
                """
            )
        )
        conn.execute(text(f"ALTER TABLE {GOLDEN_QUERY_EMBEDDINGS_TABLE} DROP CONSTRAINT IF EXISTS chk_golden_view_label"))
        conn.execute(
            text(
                f"""
                UPDATE {GOLDEN_QUERY_EMBEDDINGS_TABLE}
                SET view_label = 'experiment/result'
                WHERE view_label = 'experiment'
                """
            )
        )
        conn.execute(
            text(
                f"""
                ALTER TABLE {GOLDEN_QUERY_EMBEDDINGS_TABLE}
                ADD CONSTRAINT chk_golden_view_label
                CHECK (view_label IN ('motivation', 'method', 'experiment/result'))
                """
            )
        )
        conn.execute(
            text(
                f"""
                UPDATE {GOLDEN_QUERY_EMBEDDINGS_TABLE}
                SET example_id = query_id || '__' || replace(view_label, '/', '_')
                WHERE example_id IS NULL OR example_id = ''
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_golden_query_embeddings_example_id
                ON {GOLDEN_QUERY_EMBEDDINGS_TABLE} (example_id)
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS idx_golden_query_embeddings_type_view
                ON {GOLDEN_QUERY_EMBEDDINGS_TABLE} (query_type, view_label)
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS idx_golden_query_embeddings_embedding_cosine
                ON {GOLDEN_QUERY_EMBEDDINGS_TABLE}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE OR REPLACE FUNCTION set_golden_query_embeddings_updated_at()
                RETURNS TRIGGER AS $$
                BEGIN
                  NEW.updated_at = CURRENT_TIMESTAMP;
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """
            )
        )
        conn.execute(
            text(
                f"""
                DROP TRIGGER IF EXISTS trg_golden_query_embeddings_updated_at
                ON {GOLDEN_QUERY_EMBEDDINGS_TABLE}
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE TRIGGER trg_golden_query_embeddings_updated_at
                BEFORE UPDATE ON {GOLDEN_QUERY_EMBEDDINGS_TABLE}
                FOR EACH ROW
                EXECUTE FUNCTION set_golden_query_embeddings_updated_at()
                """
            )
        )


def upsert_golden_query_embeddings(engine, rows: Iterable[GoldenQueryEmbeddingRow]) -> int:
    from sqlalchemy import text

    rows = list(rows)
    if not rows:
        return 0

    stmt = text(
        f"""
        INSERT INTO {GOLDEN_QUERY_EMBEDDINGS_TABLE}
          (example_id, query_id, query_type, view_label, query_text, target_papers,
           answer_original_content, answer_tldr, human_view_note, indexing_content,
           retrieval_content, embedding)
        VALUES
          (:example_id, :query_id, :query_type, :view_label, :query_text,
           CAST(:target_papers AS jsonb), :answer_original_content, :answer_tldr,
           :human_view_note, :indexing_content, :retrieval_content, CAST(:embedding AS vector))
        ON CONFLICT (example_id)
        DO UPDATE SET
          query_id = EXCLUDED.query_id,
          query_type = EXCLUDED.query_type,
          view_label = EXCLUDED.view_label,
          query_text = EXCLUDED.query_text,
          target_papers = EXCLUDED.target_papers,
          answer_original_content = EXCLUDED.answer_original_content,
          answer_tldr = EXCLUDED.answer_tldr,
          human_view_note = EXCLUDED.human_view_note,
          indexing_content = EXCLUDED.indexing_content,
          retrieval_content = EXCLUDED.retrieval_content,
          embedding = EXCLUDED.embedding
        """
    )
    payload = [
        {
            "example_id": row.example_id,
            "query_id": row.query_id,
            "query_type": row.query_type,
            "view_label": row.view_label,
            "query_text": row.query_text,
            "target_papers": json.dumps(row.target_papers),
            "answer_original_content": row.answer_original_content,
            "answer_tldr": row.answer_tldr,
            "human_view_note": row.human_view_note,
            "indexing_content": row.indexing_content,
            "retrieval_content": row.retrieval_content,
            "embedding": _vector_literal(row.embedding),
        }
        for row in rows
    ]
    with engine.begin() as conn:
        conn.execute(stmt, payload)
    return len(payload)


def retrieve_golden_query_examples(
    engine,
    *,
    query_type: str,
    view_label: str,
    embedding: list[float],
    limit: int,
    exclude_litsearch: bool = False,
) -> list[GoldenQueryExample]:
    from sqlalchemy import text

    source_filter = (
        "AND (query_id ILIKE '%pasa%' OR example_id ILIKE '%pasa%')"
        if exclude_litsearch
        else ""
    )
    stmt = text(
        f"""
        SELECT
          example_id,
          query_id,
          query_type,
          view_label,
          query_text,
          target_papers,
          answer_original_content,
          answer_tldr,
          human_view_note,
          indexing_content,
          retrieval_content,
          embedding <=> CAST(:embedding AS vector) AS distance
        FROM {GOLDEN_QUERY_EMBEDDINGS_TABLE}
        WHERE query_type = :query_type
          AND view_label = :view_label
          AND specific = 1
          {source_filter}
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(
            stmt,
            {
                "query_type": query_type,
                "view_label": view_label,
                "embedding": _vector_literal(embedding),
                "limit": int(limit),
            },
        ).mappings()
        return [
            GoldenQueryExample(
                example_id=str(row["example_id"]),
                query_id=str(row["query_id"]),
                query_type=str(row["query_type"]),
                view_label=str(row["view_label"]),
                query_text=str(row["query_text"]),
                target_papers=_json_list_from_db(row["target_papers"]),
                answer_original_content=str(row["answer_original_content"] or ""),
                answer_tldr=str(row["answer_tldr"] or ""),
                human_view_note=str(row["human_view_note"] or ""),
                indexing_content=str(row["indexing_content"] or ""),
                retrieval_content=str(row["retrieval_content"] or ""),
                distance=float(row["distance"]) if row["distance"] is not None else None,
            )
            for row in rows
        ]
