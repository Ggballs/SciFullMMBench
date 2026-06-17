from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import (
    BigInteger,
    Column,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    TIMESTAMP,
    UniqueConstraint,
    create_engine,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, insert as postgres_insert
from sqlalchemy.engine import Engine

STAGE5_CANDIDATE_QUEUE_TABLE = "stage5_candidate_queue"
STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW = "stage5_candidate_queue_canonical"

metadata = MetaData()

_CANONICAL_RANKED_CTE_SQL = f"""
WITH ranked AS (
    SELECT
        q.*,
        ROW_NUMBER() OVER (
            PARTITION BY q.query_key, q.retrieval_rank
            ORDER BY
                CASE q.review_status
                    WHEN 'completed' THEN 0
                    WHEN 'processing' THEN 1
                    WHEN 'pending' THEN 2
                    WHEN 'failed' THEN 3
                    ELSE 4
                END,
                CASE q.parse_status
                    WHEN 'parsed' THEN 0
                    WHEN 'metadata_only' THEN 1
                    WHEN 'no_pdf' THEN 2
                    WHEN 'pending' THEN 3
                    ELSE 4
                END,
                CASE WHEN q.review_payload IS NOT NULL THEN 0 ELSE 1 END,
                q.updated_at DESC,
                q.id DESC
        ) AS canonical_rank
    FROM {STAGE5_CANDIDATE_QUEUE_TABLE} AS q
)
"""

stage5_candidate_queue = Table(
    STAGE5_CANDIDATE_QUEUE_TABLE,
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("candidate_key", String(64), nullable=False),
    Column("query_key", String(64), nullable=False),
    Column("paper_id", String(128), nullable=False),
    Column("paper_title", Text, nullable=False),
    Column("query_text", Text, nullable=False),
    Column("query_type", String(32), nullable=False),
    Column("source_view", String(64), nullable=False),
    Column("is_multimodal", String(8), nullable=False, server_default=text("'false'")),
    Column("related_bullet_indice", Integer, nullable=True),
    Column("related_bullet_justification", Text, nullable=True),
    Column("multimodal_rationale", Text, nullable=True),
    Column("keywords_extracted", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("search_query", Text, nullable=False),
    Column("retrieval_rank", Integer, nullable=False),
    Column("candidate_title", Text, nullable=False),
    Column("candidate_arxiv_id", String(64), nullable=True),
    Column("candidate_url", Text, nullable=True),
    Column("candidate_pdf_url", Text, nullable=True),
    Column("candidate_venue", String(256), nullable=True),
    Column("candidate_year", Integer, nullable=True),
    Column("candidate_authors", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("candidate_abstract", Text, nullable=True),
    Column("candidate_citations", Integer, nullable=True),
    Column("parse_status", String(32), nullable=False, server_default=text("'pending'")),
    Column("review_status", String(32), nullable=False, server_default=text("'pending'")),
    Column("review_label", String(32), nullable=True),
    Column("review_reason", Text, nullable=True),
    Column("error_message", Text, nullable=True),
    Column("review_payload", JSONB, nullable=True),
    Column(
        "created_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column(
        "updated_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP"),
    ),
    UniqueConstraint("candidate_key", name="uq_stage5_candidate_queue_candidate_key"),
    Index("idx_stage5_candidate_queue_query_key", "query_key"),
    Index("idx_stage5_candidate_queue_review_status", "review_status"),
    Index("idx_stage5_candidate_queue_parse_status", "parse_status"),
)


def get_engine(db_url: Optional[str] = None) -> Engine:
    url = (
        db_url
        or os.getenv("SCIFULL_STAGE5_DB_URL")
        or os.getenv("GOLDEN_EMBEDDING_DB_URL")
        or os.getenv("DATABASE_URL")
    )
    if not url:
        raise ValueError(
            "Set SCIFULL_STAGE5_DB_URL, GOLDEN_EMBEDDING_DB_URL, or DATABASE_URL to use Stage5 queue storage."
        )
    return create_engine(url, pool_pre_ping=True)


def ensure_schema(*, engine: Optional[Engine] = None, db_url: Optional[str] = None) -> Engine:
    db = engine or get_engine(db_url)
    metadata.create_all(db, tables=[stage5_candidate_queue])
    with db.begin() as conn:
        conn.execute(
            text(
                f"""
                CREATE OR REPLACE VIEW {STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW} AS
                {_CANONICAL_RANKED_CTE_SQL}
                SELECT *
                FROM ranked
                WHERE canonical_rank = 1
                """
            )
        )
    return db


def build_query_key(
    paper_id: str,
    query_text: str,
    source_view: str,
    query_type: str,
) -> str:
    raw = f"{paper_id}\n{query_text}\n{source_view}\n{query_type}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_candidate_key(
    query_key: str,
    candidate_title: str,
    candidate_arxiv_id: Optional[str],
    candidate_url: Optional[str],
) -> str:
    raw = f"{query_key}\n{candidate_title}\n{candidate_arxiv_id or ''}\n{candidate_url or ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def upsert_candidates(
    rows: Iterable[Dict[str, Any]],
    *,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> int:
    prepared_rows = list(rows)
    if not prepared_rows:
        return 0

    db = ensure_schema(engine=engine, db_url=db_url)
    stmt = postgres_insert(stage5_candidate_queue).values(prepared_rows)
    # Retrieval reruns should refresh candidate metadata, but must not clobber
    # review progress already written by the review loop.
    protected_columns = {
        "id",
        "candidate_key",
        "created_at",
        "updated_at",
        "parse_status",
        "review_status",
        "review_label",
        "review_reason",
        "error_message",
        "review_payload",
    }
    update_columns = {
        col.name: stmt.excluded[col.name]
        for col in stage5_candidate_queue.columns
        if col.name not in protected_columns
    }
    update_columns["updated_at"] = text("CURRENT_TIMESTAMP")
    stmt = stmt.on_conflict_do_update(
        index_elements=["candidate_key"],
        set_=update_columns,
    )
    with db.begin() as conn:
        conn.execute(stmt)
    return len(prepared_rows)


def claim_pending_candidates(
    limit: int,
    *,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    db = ensure_schema(engine=engine, db_url=db_url)
    claim_sql = text(
        f"""
        WITH
        claimable AS (
            SELECT q.id
            FROM {STAGE5_CANDIDATE_QUEUE_TABLE} AS q
            JOIN {STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW} AS c
              ON c.id = q.id
            WHERE q.review_status = 'pending'
            ORDER BY q.query_key, q.retrieval_rank, q.id
            FOR UPDATE OF q SKIP LOCKED
            LIMIT :limit
        )
        UPDATE {STAGE5_CANDIDATE_QUEUE_TABLE} AS target
        SET review_status = 'processing',
            updated_at = CURRENT_TIMESTAMP
        FROM claimable
        WHERE target.id = claimable.id
        RETURNING target.*
        """
    )
    with db.begin() as conn:
        rows = conn.execute(claim_sql, {"limit": int(limit)}).mappings().all()
    return [dict(row) for row in rows]


def update_candidate_result(
    candidate_id: int,
    *,
    parse_status: str,
    review_status: str,
    review_label: Optional[str] = None,
    review_reason: Optional[str] = None,
    error_message: Optional[str] = None,
    review_payload: Optional[Dict[str, Any]] = None,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> None:
    db = ensure_schema(engine=engine, db_url=db_url)
    update_sql = text(
        f"""
        UPDATE {STAGE5_CANDIDATE_QUEUE_TABLE}
        SET parse_status = :parse_status,
            review_status = :review_status,
            review_label = :review_label,
            review_reason = :review_reason,
            error_message = :error_message,
            review_payload = CAST(:review_payload AS jsonb),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = :candidate_id
        """
    )
    with db.begin() as conn:
        conn.execute(
            update_sql,
            {
                "candidate_id": int(candidate_id),
                "parse_status": str(parse_status),
                "review_status": str(review_status),
                "review_label": str(review_label) if review_label is not None else None,
                "review_reason": str(review_reason) if review_reason is not None else None,
                "error_message": str(error_message) if error_message is not None else None,
                "review_payload": None if review_payload is None else json.dumps(review_payload),
            },
        )


def reset_stale_processing_candidates(
    *,
    stale_after_seconds: int,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> int:
    db = ensure_schema(engine=engine, db_url=db_url)
    reset_sql = text(
        f"""
        WITH stale_ids AS (
            SELECT q.id
            FROM {STAGE5_CANDIDATE_QUEUE_TABLE} AS q
            JOIN {STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW} AS c
              ON c.id = q.id
            WHERE q.review_status = 'processing'
              AND q.updated_at < (CURRENT_TIMESTAMP - (:stale_after_seconds * INTERVAL '1 second'))
        )
        UPDATE {STAGE5_CANDIDATE_QUEUE_TABLE} AS target
        SET review_status = 'pending',
            updated_at = CURRENT_TIMESTAMP
        FROM stale_ids
        WHERE target.id = stale_ids.id
        """
    )
    with db.begin() as conn:
        result = conn.execute(reset_sql, {"stale_after_seconds": max(1, int(stale_after_seconds))})
    return int(result.rowcount or 0)


def reset_retryable_candidates(
    *,
    include_completed_metadata_only: bool = True,
    include_failed: bool = True,
    retry_failed_error_messages: Optional[Iterable[str]] = None,
    exclude_failed_error_messages: Optional[Iterable[str]] = None,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> int:
    db = ensure_schema(engine=engine, db_url=db_url)

    failed_filters: list[str] = []
    params: dict[str, Any] = {}

    if include_failed:
        failed_filters.append("q.review_status = 'failed'")
        retry_errors = [str(item) for item in (retry_failed_error_messages or []) if str(item)]
        exclude_errors = [str(item) for item in (exclude_failed_error_messages or []) if str(item)]
        if retry_errors:
            params["retry_failed_error_messages"] = retry_errors
            failed_filters[-1] += " AND COALESCE(q.error_message, '') = ANY(:retry_failed_error_messages)"
        if exclude_errors:
            params["exclude_failed_error_messages"] = exclude_errors
            failed_filters[-1] += " AND COALESCE(q.error_message, '') <> ALL(:exclude_failed_error_messages)"

    target_predicates: list[str] = []
    if failed_filters:
        target_predicates.append("(" + " OR ".join(failed_filters) + ")")
    if include_completed_metadata_only:
        target_predicates.append("(q.review_status = 'completed' AND q.parse_status = 'metadata_only')")

    if not target_predicates:
        return 0

    reset_sql = text(
        f"""
        WITH target_ids AS (
            SELECT q.id
            FROM {STAGE5_CANDIDATE_QUEUE_TABLE} AS q
            JOIN {STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW} AS c
              ON c.id = q.id
            WHERE {" OR ".join(target_predicates)}
        )
        UPDATE {STAGE5_CANDIDATE_QUEUE_TABLE} AS target
        SET review_status = 'pending',
            parse_status = 'pending',
            review_label = NULL,
            review_reason = NULL,
            error_message = NULL,
            review_payload = NULL,
            updated_at = CURRENT_TIMESTAMP
        FROM target_ids
        WHERE target.id = target_ids.id
        """
    )
    with db.begin() as conn:
        result = conn.execute(reset_sql, params)
    return int(result.rowcount or 0)


def load_status_summary(
    *,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
    canonical: bool = True,
) -> Dict[str, int]:
    db = ensure_schema(engine=engine, db_url=db_url)
    source = STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW if canonical else STAGE5_CANDIDATE_QUEUE_TABLE
    stmt = text(
        f"""
        SELECT review_status, COUNT(*) AS count
        FROM {source}
        GROUP BY review_status
        ORDER BY review_status
        """
    )
    with db.begin() as conn:
        rows = conn.execute(stmt).mappings().all()
    return {str(row["review_status"]): int(row["count"]) for row in rows}


def load_label_summary(
    *,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
    canonical: bool = True,
) -> Dict[str, int]:
    db = ensure_schema(engine=engine, db_url=db_url)
    source = STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW if canonical else STAGE5_CANDIDATE_QUEUE_TABLE
    stmt = text(
        f"""
        SELECT review_label, COUNT(*) AS count
        FROM {source}
        GROUP BY review_label
        """
    )
    with db.begin() as conn:
        rows = conn.execute(stmt).mappings().all()
    return {str(row["review_label"]): int(row["count"]) for row in rows if row["review_label"] is not None}


def load_queue_snapshot(
    *,
    latest_limit: int = 10,
    stale_processing_seconds: int = 600,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> Dict[str, Any]:
    db = ensure_schema(engine=engine, db_url=db_url)
    with db.begin() as conn:
        raw_total = int(conn.execute(text(f"SELECT COUNT(*) FROM {STAGE5_CANDIDATE_QUEUE_TABLE}")).scalar() or 0)
        canonical_total = int(
            conn.execute(text(f"SELECT COUNT(*) FROM {STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW}")).scalar() or 0
        )
        distinct_query_key = int(
            conn.execute(text(f"SELECT COUNT(DISTINCT query_key) FROM {STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW}")).scalar()
            or 0
        )
        raw_status_rows = conn.execute(
            text(
                f"""
                SELECT review_status, COUNT(*) AS count
                FROM {STAGE5_CANDIDATE_QUEUE_TABLE}
                GROUP BY review_status
                """
            )
        ).mappings().all()
        canonical_status_rows = conn.execute(
            text(
                f"""
                SELECT review_status, COUNT(*) AS count
                FROM {STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW}
                GROUP BY review_status
                """
            )
        ).mappings().all()
        canonical_label_rows = conn.execute(
            text(
                f"""
                SELECT review_label, COUNT(*) AS count
                FROM {STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW}
                GROUP BY review_label
                """
            )
        ).mappings().all()
        latest_rows = conn.execute(
            text(
                f"""
                SELECT id, retrieval_rank, query_text, review_label, candidate_title, updated_at, parse_status
                FROM {STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW}
                WHERE review_status = :status
                ORDER BY updated_at DESC
                LIMIT :limit
                """
            ),
            {"status": "completed", "limit": int(latest_limit)},
        ).mappings().all()
        completed_parse_rows = conn.execute(
            text(
                f"""
                SELECT parse_status, COUNT(*) AS count
                FROM {STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW}
                WHERE review_status = :status
                GROUP BY parse_status
                """
            ),
            {"status": "completed"},
        ).mappings().all()
        stale_processing_count = int(
            conn.execute(
                text(
                    f"""
                    SELECT COUNT(*)
                    FROM {STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW}
                    WHERE review_status = 'processing'
                      AND updated_at < (CURRENT_TIMESTAMP - (:stale_after_seconds * INTERVAL '1 second'))
                    """
                ),
                {"stale_after_seconds": max(1, int(stale_processing_seconds))},
            ).scalar()
            or 0
        )

    raw_counts = {str(row["review_status"]): int(row["count"]) for row in raw_status_rows}
    canonical_counts = {str(row["review_status"]): int(row["count"]) for row in canonical_status_rows}
    canonical_labels = {
        str(row["review_label"]): int(row["count"])
        for row in canonical_label_rows
        if row["review_label"] is not None
    }
    completed_parse_counts = {
        str(row["parse_status"]): int(row["count"])
        for row in completed_parse_rows
        if row["parse_status"] is not None
    }
    latest_completed_parse_counts: Dict[str, int] = {}
    for row in latest_rows:
        parse_status = row["parse_status"]
        if parse_status is None:
            continue
        key = str(parse_status)
        latest_completed_parse_counts[key] = latest_completed_parse_counts.get(key, 0) + 1
    for key in ("pending", "processing", "completed", "failed"):
        raw_counts.setdefault(key, 0)
        canonical_counts.setdefault(key, 0)

    return {
        "raw_total": raw_total,
        "canonical_total": canonical_total,
        "distinct_query_key": distinct_query_key,
        "raw_counts": raw_counts,
        "canonical_counts": canonical_counts,
        "canonical_labels": canonical_labels,
        "completed_parse_counts": completed_parse_counts,
        "latest_completed_parse_counts": latest_completed_parse_counts,
        "stale_processing_count": stale_processing_count,
        "latest_completed": latest_rows,
    }


def load_candidates_for_query(
    query_key: str,
    *,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
    canonical: bool = False,
) -> List[Dict[str, Any]]:
    db = ensure_schema(engine=engine, db_url=db_url)
    if canonical:
        stmt = text(
            f"""
            SELECT *
            FROM {STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW}
            WHERE query_key = :query_key
            ORDER BY retrieval_rank, id
            """
        )
        with db.begin() as conn:
            rows = conn.execute(stmt, {"query_key": str(query_key)}).mappings().all()
    else:
        stmt = (
            select(stage5_candidate_queue)
            .where(stage5_candidate_queue.c.query_key == str(query_key))
            .order_by(stage5_candidate_queue.c.retrieval_rank, stage5_candidate_queue.c.id)
        )
        with db.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
    return [dict(row) for row in rows]
