from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import BigInteger, Column, Index, Integer, MetaData, String, Table, Text, TIMESTAMP, UniqueConstraint, create_engine, select, text
from sqlalchemy.dialects.postgresql import JSONB, insert as postgres_insert
from sqlalchemy.engine import Engine

STAGE5_CANDIDATE_QUEUE_TABLE = "stage5_candidate_queue"
STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW = "stage5_candidate_queue_canonical"
STAGE5_QUEUE_TABLE_ENV = "SCIFULL_STAGE5_QUEUE_TABLE"

_TABLE_CACHE: dict[str, tuple[MetaData, Table, str]] = {}


def _normalized_table_name(table_name: Optional[str] = None) -> str:
    raw = str(table_name or os.getenv(STAGE5_QUEUE_TABLE_ENV) or STAGE5_CANDIDATE_QUEUE_TABLE).strip()
    if not raw:
        return STAGE5_CANDIDATE_QUEUE_TABLE
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if raw[0].isdigit() or any(ch not in allowed for ch in raw):
        raise ValueError(f"Invalid Stage5 queue table name: {raw!r}")
    return raw


def resolve_queue_storage_names(table_name: Optional[str] = None) -> tuple[str, str]:
    resolved = _normalized_table_name(table_name)
    if resolved == STAGE5_CANDIDATE_QUEUE_TABLE:
        return resolved, STAGE5_CANDIDATE_QUEUE_CANONICAL_VIEW
    canonical = f"{resolved}_canonical"
    if len(canonical) <= 63:
        return resolved, canonical
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:8]
    return resolved, f"{resolved[:63 - len('_canonical_') - len(digest)]}_canonical_{digest}"


def _sql_ident(prefix: str, table_name: str) -> str:
    base = f"{prefix}_{table_name}"
    if len(base) <= 63:
        return base
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    return f"{base[:63 - len(digest) - 1]}_{digest}"


def _canonical_ranked_cte_sql(table_name: str) -> str:
    return f"""
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
    FROM {table_name} AS q
)
"""


def _get_queue_table(table_name: Optional[str] = None) -> tuple[MetaData, Table, str, str]:
    resolved_table_name, canonical_view_name = resolve_queue_storage_names(table_name)
    cached = _TABLE_CACHE.get(resolved_table_name)
    if cached is not None:
        metadata, table, cached_view_name = cached
        return metadata, table, resolved_table_name, cached_view_name

    metadata = MetaData()
    table = Table(
        resolved_table_name,
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
        Column("task", String(128), nullable=True),
        Column("created_at", TIMESTAMP(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
        Column(
            "updated_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
            server_onupdate=text("CURRENT_TIMESTAMP"),
        ),
        UniqueConstraint("candidate_key", name=_sql_ident("uq_stage5_candidate_queue_candidate_key", resolved_table_name)),
        Index(_sql_ident("idx_stage5_candidate_queue_query_key", resolved_table_name), "query_key"),
        Index(_sql_ident("idx_stage5_candidate_queue_review_status", resolved_table_name), "review_status"),
        Index(_sql_ident("idx_stage5_candidate_queue_parse_status", resolved_table_name), "parse_status"),
        Index(_sql_ident("idx_stage5_candidate_queue_task", resolved_table_name), "task"),
    )
    _TABLE_CACHE[resolved_table_name] = (metadata, table, canonical_view_name)
    return metadata, table, resolved_table_name, canonical_view_name


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
    pool_size = int(os.environ.get("SCIFULL_DB_POOL_SIZE", "5"))
    max_overflow = int(os.environ.get("SCIFULL_DB_POOL_OVERFLOW", "5"))
    for attempt in range(1, 11):
        try:
            engine = create_engine(url, pool_pre_ping=True, pool_size=pool_size, max_overflow=max_overflow)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except Exception as e:
            if "too many clients" in str(e) and attempt < 10:
                import time
                time.sleep(attempt * 3)
                continue
            raise


def ensure_schema(*, engine: Optional[Engine] = None, db_url: Optional[str] = None) -> Engine:
    db = engine or get_engine(db_url)
    metadata, queue_table, queue_table_name, canonical_view_name = _get_queue_table()
    metadata.create_all(db, tables=[queue_table])
    with db.begin() as conn:
        conn.execute(
            text(
                f"""
                CREATE OR REPLACE VIEW {canonical_view_name} AS
                {_canonical_ranked_cte_sql(queue_table_name)}
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
    _, queue_table, _, _ = _get_queue_table()
    stmt = postgres_insert(queue_table).values(prepared_rows)
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
        for col in queue_table.columns
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
    task_filter: Optional[str] = None,
    parse_status_filter: Optional[str] = None,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    db = ensure_schema(engine=engine, db_url=db_url)
    _, _, queue_table_name, canonical_view_name = _get_queue_table()
    clauses = []
    if task_filter:
        clauses.append("AND q.task = :task_filter")
    if parse_status_filter:
        clauses.append("AND q.parse_status = :parse_status_filter")
    extra_where = " ".join(clauses)
    claim_sql = text(
        f"""
        WITH
        query_progress AS (
            SELECT
                query_key,
                COUNT(*) FILTER (WHERE review_status = 'completed') AS completed_reviews,
                COUNT(*) FILTER (WHERE review_status = 'completed' AND review_label = 'hard_negative') AS completed_hard_negatives
            FROM {queue_table_name}
            GROUP BY query_key
        ),
        claimable AS (
            SELECT q.id
            FROM {queue_table_name} AS q
            LEFT JOIN query_progress AS qp
              ON qp.query_key = q.query_key
            WHERE q.review_status = 'pending'
              AND COALESCE(qp.completed_hard_negatives, 0) < 10
              AND COALESCE(qp.completed_reviews, 0) < 30
              {extra_where}
            ORDER BY q.query_key, q.retrieval_rank, q.id
            FOR UPDATE OF q SKIP LOCKED
            LIMIT :limit
        )
        UPDATE {queue_table_name} AS target
        SET review_status = 'processing',
            updated_at = CURRENT_TIMESTAMP
        FROM claimable
        WHERE target.id = claimable.id
        RETURNING target.*
        """
    )
    params: dict[str, Any] = {"limit": int(limit)}
    if task_filter:
        params["task_filter"] = task_filter
    if parse_status_filter:
        params["parse_status_filter"] = parse_status_filter
    with db.begin() as conn:
        rows = conn.execute(claim_sql, params).mappings().all()
    return [dict(row) for row in rows]


def claim_parse_candidates(
    limit: int,
    *,
    min_query_candidates: int = 25,
    task_filter: Optional[str] = None,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    db = engine if engine else ensure_schema(engine=engine, db_url=db_url)
    _, _, queue_table_name, _ = _get_queue_table()
    task_clause = "AND q.task = :task_filter" if task_filter else ""
    claim_sql = text(
        f"""
        WITH
        eligible_queries AS (
            SELECT query_key
            FROM {queue_table_name}
            GROUP BY query_key
            HAVING COUNT(*) >= :min_query_candidates
        ),
        claimable AS (
            SELECT q.id
            FROM {queue_table_name} AS q
            JOIN eligible_queries AS eq
              ON eq.query_key = q.query_key
            WHERE q.review_status = 'pending'
              AND q.parse_status = 'downloaded'
              {task_clause}
            ORDER BY q.query_key, q.retrieval_rank, q.id
            FOR UPDATE OF q SKIP LOCKED
            LIMIT :limit
        )
        UPDATE {queue_table_name} AS target
        SET parse_status = :processing_status,
            updated_at = CURRENT_TIMESTAMP
        FROM claimable
        WHERE target.id = claimable.id
        RETURNING target.*
        """
    )
    import os
    parser_id = os.environ.get("SCIFULL_PARSER_ID", "A")
    processing_status = f"processing_parser_{parser_id}"
    params: dict[str, Any] = {
        "limit": int(limit),
        "min_query_candidates": max(1, int(min_query_candidates)),
        "processing_status": processing_status,
    }
    if task_filter:
        params["task_filter"] = task_filter
    with db.begin() as conn:
        rows = conn.execute(claim_sql, params).mappings().all()
    return [dict(row) for row in rows]


def claim_download_candidates(
    limit: int,
    *,
    task_filter: str,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Claim rows that need PDF download: pending/downloaded parse_status, no cached PDF on disk."""
    db = ensure_schema(engine=engine, db_url=db_url)
    _, _, queue_table_name, _ = _get_queue_table()
    claim_sql = text(
        f"""
        WITH claimable AS (
            SELECT q.id
            FROM {queue_table_name} AS q
            WHERE q.task = :task_filter
              AND q.review_status = 'pending'
              AND q.parse_status IN ('pending', 'downloaded')
              AND q.candidate_pdf_url IS NOT NULL
              AND (
                  q.review_payload IS NULL
                  OR q.review_payload->>'candidate_pdf_path' IS NULL
              )
            ORDER BY q.query_key, q.retrieval_rank, q.id
            FOR UPDATE OF q SKIP LOCKED
            LIMIT :limit
        )
        UPDATE {queue_table_name} AS target
        SET parse_status = 'downloading',
            updated_at = CURRENT_TIMESTAMP
        FROM claimable
        WHERE target.id = claimable.id
        RETURNING target.*
        """
    )
    with db.begin() as conn:
        rows = conn.execute(
            claim_sql,
            {"limit": int(limit), "task_filter": str(task_filter)},
        ).mappings().all()
    return [dict(row) for row in rows]


def batch_update_download_results(
    results: list[dict[str, Any]],  # list of {"candidate_id": int, "pdf_path": str, "parse_status": str}
    *,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> int:
    """Batch update download results for multiple rows."""
    if not results:
        return 0
    db = ensure_schema(engine=engine, db_url=db_url)
    _, _, queue_table_name, _ = _get_queue_table()
    updated = 0
    with db.begin() as conn:
        for r in results:
            pdf_path_value = str(r["pdf_path"]).strip() if r.get("pdf_path") else None
            conn.execute(
                text(
                    f"""
                    UPDATE {queue_table_name}
                    SET parse_status = :parse_status,
                        review_payload = jsonb_build_object('candidate_pdf_path', CAST(:pdf_path AS text)),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :candidate_id
                    """
                ),
                {
                    "candidate_id": int(r["candidate_id"]),
                    "parse_status": str(r.get("parse_status", "downloaded")),
                    "pdf_path": pdf_path_value,
                },
            )
            updated += 1
    return updated


def claim_pending_query(
    task_filter: str,
    *,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Claim all pending review rows for a single query_key (query-level concurrency).

    Returns all rows for the claimed query, ordered by retrieval_rank.
    Returns None if no pending query is available.
    """
    db = ensure_schema(engine=engine, db_url=db_url)
    _, _, queue_table_name, _ = _get_queue_table()
    with db.begin() as conn:
        claim_sql = text(
            f"""
            WITH picked AS (
                SELECT q.query_key
                FROM {queue_table_name} AS q
                WHERE q.review_status = 'pending'
                  AND q.task = :task_filter
                ORDER BY q.query_key, q.retrieval_rank
                LIMIT 1
                FOR UPDATE OF q SKIP LOCKED
            )
            UPDATE {queue_table_name} AS target
            SET review_status = 'processing',
                updated_at = CURRENT_TIMESTAMP
            FROM picked
            WHERE target.query_key = picked.query_key
              AND target.review_status = 'pending'
            RETURNING target.*
            """
        )
        rows = conn.execute(claim_sql, {"task_filter": str(task_filter)}).mappings().all()
    if not rows:
        return None
    return sorted([dict(row) for row in rows], key=lambda r: int(r.get("retrieval_rank", 0) or 0))


def close_query_remaining(
    query_key: str,
    *,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> int:
    """Mark remaining pending rows for a query as skipped after review budget met."""
    db = ensure_schema(engine=engine, db_url=db_url)
    _, _, queue_table_name, _ = _get_queue_table()
    with db.begin() as conn:
        result = conn.execute(
            text(
                f"""
                UPDATE {queue_table_name}
                SET review_status = 'completed',
                    review_label = 'ignored',
                    review_reason = 'review_budget_exhausted',
                    updated_at = CURRENT_TIMESTAMP
                WHERE query_key = :query_key
                  AND review_status = 'pending'
                """
            ),
            {"query_key": str(query_key)},
        )
    return int(result.rowcount or 0)


def update_download_result(
    candidate_id: int,
    *,
    pdf_path: str,
    parse_status: str = "downloaded",
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> None:
    """Store downloaded PDF path and set parse_status."""
    db = ensure_schema(engine=engine, db_url=db_url)
    _, _, queue_table_name, _ = _get_queue_table()
    pdf_path_value = str(pdf_path).strip() if pdf_path else None
    update_sql = text(
        f"""
        UPDATE {queue_table_name}
        SET parse_status = :parse_status,
            review_payload = jsonb_build_object('candidate_pdf_path', CAST(:pdf_path AS text)),
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
                "pdf_path": pdf_path_value,
            },
        )


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
    _, _, queue_table_name, _ = _get_queue_table()
    update_sql = text(
        f"""
        UPDATE {queue_table_name}
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


def close_satisfied_query_pending_candidates(
    query_key: str,
    *,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> int:
    db = ensure_schema(engine=engine, db_url=db_url)
    _, _, queue_table_name, _ = _get_queue_table()
    close_sql = text(
        f"""
        WITH query_progress AS (
            SELECT
                query_key,
                COUNT(*) FILTER (WHERE review_status = 'completed') AS completed_reviews,
                COUNT(*) FILTER (WHERE review_status = 'completed' AND review_label = 'hard_negative') AS completed_hard_negatives
            FROM {queue_table_name}
            WHERE query_key = :query_key
            GROUP BY query_key
        )
        UPDATE {queue_table_name} AS q
        SET parse_status = 'skipped',
            review_status = 'completed',
            review_label = 'ignored',
            review_reason = 'review_budget_exhausted',
            error_message = NULL,
            review_payload = jsonb_build_object(
                'label', 'ignored',
                'reason', 'review_budget_exhausted'
            ),
            updated_at = CURRENT_TIMESTAMP
        FROM query_progress AS qp
        WHERE q.query_key = :query_key
          AND q.review_status = 'pending'
          AND (
                COALESCE(qp.completed_hard_negatives, 0) >= 10
                OR COALESCE(qp.completed_reviews, 0) >= 30
          )
        """
    )
    with db.begin() as conn:
        result = conn.execute(close_sql, {"query_key": str(query_key)})
    return int(result.rowcount or 0)


def reset_stale_processing_candidates(
    *,
    stale_after_seconds: int,
    engine: Optional[Engine] = None,
    db_url: Optional[str] = None,
) -> int:
    db = ensure_schema(engine=engine, db_url=db_url)
    _, _, queue_table_name, _ = _get_queue_table()
    reset_sql = text(
        f"""
        WITH stale_ids AS (
            SELECT q.id
            FROM {queue_table_name} AS q
            WHERE q.review_status = 'processing'
              AND q.updated_at < (CURRENT_TIMESTAMP - (:stale_after_seconds * INTERVAL '1 second'))
        )
        UPDATE {queue_table_name} AS target
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
    _, _, queue_table_name, canonical_view_name = _get_queue_table()

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
            FROM {queue_table_name} AS q
            JOIN {canonical_view_name} AS c
              ON c.id = q.id
            WHERE {" OR ".join(target_predicates)}
        )
        UPDATE {queue_table_name} AS target
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
    _, _, queue_table_name, canonical_view_name = _get_queue_table()
    source = canonical_view_name if canonical else queue_table_name
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
    _, _, queue_table_name, canonical_view_name = _get_queue_table()
    source = canonical_view_name if canonical else queue_table_name
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
    _, queue_table, queue_table_name, canonical_view_name = _get_queue_table()
    with db.begin() as conn:
        raw_total = int(conn.execute(text(f"SELECT COUNT(*) FROM {queue_table_name}")).scalar() or 0)
        canonical_total = int(conn.execute(text(f"SELECT COUNT(*) FROM {canonical_view_name}")).scalar() or 0)
        distinct_query_key = int(
            conn.execute(text(f"SELECT COUNT(DISTINCT query_key) FROM {canonical_view_name}")).scalar()
            or 0
        )
        raw_status_rows = conn.execute(
            text(
                f"""
                SELECT review_status, COUNT(*) AS count
                FROM {queue_table_name}
                GROUP BY review_status
                """
            )
        ).mappings().all()
        canonical_status_rows = conn.execute(
            text(
                f"""
                SELECT review_status, COUNT(*) AS count
                FROM {canonical_view_name}
                GROUP BY review_status
                """
            )
        ).mappings().all()
        canonical_label_rows = conn.execute(
            text(
                f"""
                SELECT review_label, COUNT(*) AS count
                FROM {canonical_view_name}
                GROUP BY review_label
                """
            )
        ).mappings().all()
        latest_rows = conn.execute(
            text(
                f"""
                SELECT id, retrieval_rank, query_text, review_label, candidate_title, updated_at, parse_status
                FROM {canonical_view_name}
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
                FROM {canonical_view_name}
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
                    FROM {canonical_view_name}
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
    _, queue_table, _, canonical_view_name = _get_queue_table()
    if canonical:
        stmt = text(
            f"""
            SELECT *
            FROM {canonical_view_name}
            WHERE query_key = :query_key
            ORDER BY retrieval_rank, id
            """
        )
        with db.begin() as conn:
            rows = conn.execute(stmt, {"query_key": str(query_key)}).mappings().all()
    else:
        stmt = (
            select(queue_table)
            .where(queue_table.c.query_key == str(query_key))
            .order_by(queue_table.c.retrieval_rank, queue_table.c.id)
        )
        with db.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
    return [dict(row) for row in rows]
