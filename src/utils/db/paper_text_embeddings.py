from __future__ import annotations

import os

from sqlalchemy import text


PAPER_TEXT_EMBEDDINGS_TABLE = "paper_text_embeddings"
_DEFAULT_DB_URL = "postgresql+psycopg://scifull:westlakenlp@localhost:5433/scifullmmbench"


def get_engine(db_url: str | None = None):
    from sqlalchemy import create_engine

    url = (
        db_url
        or os.getenv("SCIFULL_GOLDEN_EMBEDDING_DB_URL")
        or os.getenv("DATABASE_URL")
        or _DEFAULT_DB_URL
    )
    return create_engine(url, pool_pre_ping=True)


def ensure_table(engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {PAPER_TEXT_EMBEDDINGS_TABLE} (
                    paper_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    task TEXT NOT NULL DEFAULT 'legacy',
                    markdown_chars INTEGER NOT NULL DEFAULT 0,
                    embedding vector NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (paper_id, model_name, task)
                )
                """
            )
        )
        conn.commit()


def existing_embeddings(engine, task: str | None = None) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    with engine.connect() as conn:
        if task:
            rows = conn.execute(
                text(f"SELECT paper_id, model_name FROM {PAPER_TEXT_EMBEDDINGS_TABLE} WHERE task = :task"),
                {"task": task},
            )
        else:
            rows = conn.execute(text(f"SELECT paper_id, model_name FROM {PAPER_TEXT_EMBEDDINGS_TABLE}"))
        for row in rows:
            pairs.add((str(row[0]), str(row[1])))
    return pairs


def _vector_str(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.10g}" for v in vec) + "]"


def insert_embeddings(
    engine,
    paper_ids: list[str],
    model_name: str,
    embeddings: list[list[float]],
    markdown_chars: dict[str, int],
    task: str = "legacy",
) -> int:
    insert_sql = text(
        f"""
        INSERT INTO {PAPER_TEXT_EMBEDDINGS_TABLE} (paper_id, model_name, task, markdown_chars, embedding)
        VALUES (:paper_id, :model_name, :task, :markdown_chars, CAST(:embedding AS vector))
        ON CONFLICT (paper_id, model_name, task) DO NOTHING
        """
    )
    inserted = 0
    with engine.connect() as conn:
        for paper_id, embedding in zip(paper_ids, embeddings):
            result = conn.execute(
                insert_sql,
                {
                    "paper_id": paper_id,
                    "model_name": model_name,
                    "task": task,
                    "markdown_chars": markdown_chars.get(paper_id, 0),
                    "embedding": _vector_str(embedding),
                },
            )
            if result.rowcount:
                inserted += 1
        conn.commit()
    return inserted
