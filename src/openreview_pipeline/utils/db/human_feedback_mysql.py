from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from openreview_pipeline.app_logging import resolve_log_dir
from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    MetaData,
    String,
    TIMESTAMP,
    Table,
    Text,
    create_engine,
    delete,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.mysql import insert as mysql_insert


HUMAN_FEEDBACK_TABLE = "human_feedback"
logger = logging.getLogger(__name__)

metadata = MetaData()

human_feedback = Table(
    HUMAN_FEEDBACK_TABLE,
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("paper_forum_id", String(128), nullable=False),
    Column("query_id", String(64), nullable=False),
    Column("query_text", Text, nullable=False),
    Column("feedback_item_id", String(64), nullable=False),
    Column("reviewer_username", String(64), nullable=False),
    Column("judgement", String(32), nullable=False),
    Column("selection_type", JSON, nullable=True),
    Column("reason_note", Text, nullable=True),
    Column("feedback_raw_json", JSON, nullable=True),
    Column("created_at", TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    Column(
        "updated_at",
        TIMESTAMP,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP"),
    ),
)


def configure_sql_access_logging(log_dir: Optional[Path] = None) -> None:
    """Write SQL access logs to the project logs directory."""
    target_dir = Path(log_dir or os.getenv("HUMAN_FEEDBACK_LOG_DIR") or resolve_log_dir())
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / "human_feedback_mysql.log"

    if any(
        isinstance(handler, RotatingFileHandler)
        and Path(getattr(handler, "baseFilename", "")) == log_path
        for handler in logger.handlers
    ):
        return

    handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = True


def get_engine(db_url: Optional[str] = None) -> Engine:
    """Create a SQLAlchemy engine from an explicit URL or environment."""
    url = db_url or os.getenv("HUMAN_FEEDBACK_DB_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("Set HUMAN_FEEDBACK_DB_URL or DATABASE_URL to use MySQL feedback CRUD.")
    return create_engine(url, pool_pre_ping=True)


def stable_field_id(value: Any) -> str:
    """Mirror the Gradio app's stable field id helper."""
    text = str(value if value is not None else "")
    total = sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))
    return str(total % 1_000_000_007)


def derive_paper_forum_id(context: Mapping[str, Any]) -> str:
    paper_forum_id = (
        context.get("paper_forum_id")
        or context.get("forum_id")
        or context.get("paper_id")
        or context.get("openreview_forum_id")
    )
    if not paper_forum_id:
        raise ValueError("Missing paper forum id; expected paper_forum_id, forum_id, or paper_id.")
    return str(paper_forum_id)


def derive_query_id(context: Mapping[str, Any]) -> str:
    query_id = context.get("query_id") or context.get("query_key")
    if query_id:
        return str(query_id)

    paper_forum_id = derive_paper_forum_id(context)
    source_view = context.get("source_view") or "unknown_view"
    query_text = context.get("query_text")
    if query_text is None:
        raise ValueError("Missing query id; expected query_id, query_key, or query_text.")
    return f"{paper_forum_id}::{source_view}::{stable_field_id(query_text)}"


def _require_reviewer(reviewer_username: str) -> str:
    reviewer = str(reviewer_username or "").strip()
    if not reviewer:
        raise ValueError("reviewer_username is required.")
    return reviewer


def _as_selection(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value or None
    if isinstance(value, tuple):
        return list(value) or None
    return [value]


def _base_row(
    context: Mapping[str, Any],
    reviewer_username: str,
    feedback_item_id: str,
    judgement: Any,
    *,
    selection_type: Any = None,
    reason_note: Any = None,
    feedback_raw_json: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    judgement_text = str(judgement or "").strip()
    if not judgement_text:
        raise ValueError(f"Missing judgement for feedback item {feedback_item_id!r}.")

    return {
        "paper_forum_id": derive_paper_forum_id(context),
        "query_id": derive_query_id(context),
        "query_text": str(context.get("query_text") or ""),
        "feedback_item_id": feedback_item_id,
        "reviewer_username": _require_reviewer(reviewer_username),
        "judgement": judgement_text,
        "selection_type": _as_selection(selection_type),
        "reason_note": str(reason_note) if reason_note not in (None, "") else None,
        "feedback_raw_json": dict(feedback_raw_json or {}),
    }


def feedback_payload_to_rows(
    context: Mapping[str, Any],
    feedback_payload: Mapping[str, Any],
    reviewer_username: str,
) -> List[Dict[str, Any]]:
    """Convert the current Gradio feedback payload into human_feedback rows."""
    rows: List[Dict[str, Any]] = []

    query_relevance = feedback_payload.get("query_paper_relevance") or {}
    if query_relevance.get("relevance"):
        rows.append(
            _base_row(
                context,
                reviewer_username,
                "query_relevance",
                query_relevance.get("relevance"),
                reason_note=query_relevance.get("notes"),
                feedback_raw_json=query_relevance,
            )
        )

    human_like = feedback_payload.get("human_like_search") or {}
    if human_like.get("real_researcher_search"):
        rows.append(
            _base_row(
                context,
                reviewer_username,
                "human_like",
                human_like.get("real_researcher_search"),
                selection_type=human_like.get("non_human_like_type"),
                reason_note=human_like.get("notes"),
                feedback_raw_json=human_like,
            )
        )

    candidate_checks = feedback_payload.get("candidate_checks") or {}
    for feedback_item_id, candidate in candidate_checks.items():
        if not isinstance(candidate, Mapping) or not candidate.get("label_correct"):
            continue
        rows.append(
            _base_row(
                context,
                reviewer_username,
                str(feedback_item_id),
                candidate.get("label_correct"),
                selection_type=candidate.get("wrong_label_type"),
                reason_note=candidate.get("notes"),
                feedback_raw_json=candidate,
            )
        )

    return rows


def rows_to_feedback_payload(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Reconstruct the nested Gradio feedback payload from human_feedback rows."""
    payload: Dict[str, Any] = {
        "query_paper_relevance": {},
        "human_like_search": {},
        "candidate_checks": {},
    }

    for row in rows:
        item_id = str(row.get("feedback_item_id") or "")
        raw = row.get("feedback_raw_json") or {}
        if not isinstance(raw, Mapping):
            raw = {}

        if item_id == "query_relevance":
            payload["query_paper_relevance"] = dict(raw) or {
                "relevance": row.get("judgement"),
                "notes": row.get("reason_note") or "",
            }
        elif item_id == "human_like":
            payload["human_like_search"] = dict(raw) or {
                "real_researcher_search": row.get("judgement"),
                "non_human_like_type": row.get("selection_type") or [],
                "non_human_like_other": "",
                "notes": row.get("reason_note") or "",
            }
        elif item_id:
            payload["candidate_checks"][item_id] = dict(raw) or {
                "label_correct": row.get("judgement"),
                "wrong_label_type": row.get("selection_type") or [],
                "notes": row.get("reason_note") or "",
            }

    return payload


def save_query_feedback(
    context: Mapping[str, Any],
    feedback_payload: Mapping[str, Any],
    reviewer_username: str,
    *,
    engine: Optional[Engine] = None,
) -> None:
    rows = feedback_payload_to_rows(context, feedback_payload, reviewer_username)
    if not rows:
        logger.info(
            "save skipped reviewer=%s paper=%s query=%s rows=0 reason=no_feedback_rows",
            reviewer_username,
            context.get("paper_id") or context.get("paper_forum_id"),
            context.get("query_key") or context.get("query_id"),
        )
        return

    db = engine or get_engine()
    reviewer = _require_reviewer(reviewer_username)
    paper_forum_id = derive_paper_forum_id(context)
    query_id = derive_query_id(context)
    insert_stmt = mysql_insert(human_feedback).values(rows)
    update_columns = {
        column.name: insert_stmt.inserted[column.name]
        for column in human_feedback.columns
        if column.name not in {"id", "created_at", "updated_at"}
    }
    stmt = insert_stmt.on_duplicate_key_update(**update_columns)
    logger.info(
        "save start reviewer=%s paper=%s query=%s rows=%s",
        reviewer,
        paper_forum_id,
        query_id,
        len(rows),
    )
    try:
        with db.begin() as conn:
            result = conn.execute(stmt)
        logger.info(
            "save success reviewer=%s paper=%s query=%s rows=%s affected_rows=%s",
            reviewer,
            paper_forum_id,
            query_id,
            len(rows),
            result.rowcount,
        )
    except Exception:
        logger.exception(
            "save failed reviewer=%s paper=%s query=%s rows=%s",
            reviewer,
            paper_forum_id,
            query_id,
            len(rows),
        )
        raise


def load_query_feedback(
    context: Mapping[str, Any],
    reviewer_username: str,
    *,
    engine: Optional[Engine] = None,
) -> Dict[str, Any]:
    logger.info(
        "load start reviewer=%s paper=%s query=%s",
        reviewer_username,
        context.get("paper_id") or context.get("paper_forum_id"),
        context.get("query_key") or context.get("query_id"),
    )
    return list_feedback_for_query(
        derive_paper_forum_id(context),
        derive_query_id(context),
        reviewer_username=reviewer_username,
        engine=engine,
    )


def delete_query_feedback(
    context: Mapping[str, Any],
    reviewer_username: str,
    *,
    engine: Optional[Engine] = None,
) -> int:
    db = engine or get_engine()
    reviewer = _require_reviewer(reviewer_username)
    paper_forum_id = derive_paper_forum_id(context)
    query_id = derive_query_id(context)
    stmt = (
        delete(human_feedback)
        .where(human_feedback.c.paper_forum_id == paper_forum_id)
        .where(human_feedback.c.query_id == query_id)
        .where(human_feedback.c.reviewer_username == reviewer)
    )
    logger.info("delete start reviewer=%s paper=%s query=%s", reviewer, paper_forum_id, query_id)
    try:
        with db.begin() as conn:
            result = conn.execute(stmt)
        row_count = int(result.rowcount or 0)
        logger.info(
            "delete success reviewer=%s paper=%s query=%s rows=%s",
            reviewer,
            paper_forum_id,
            query_id,
            row_count,
        )
        return row_count
    except Exception:
        logger.exception(
            "delete failed reviewer=%s paper=%s query=%s",
            reviewer,
            paper_forum_id,
            query_id,
        )
        raise


def list_feedback_for_query(
    paper_forum_id: str,
    query_id: str,
    reviewer_username: Optional[str] = None,
    *,
    engine: Optional[Engine] = None,
) -> Dict[str, Any]:
    db = engine or get_engine()
    reviewer = _require_reviewer(reviewer_username) if reviewer_username is not None else None
    stmt = (
        select(human_feedback)
        .where(human_feedback.c.paper_forum_id == str(paper_forum_id))
        .where(human_feedback.c.query_id == str(query_id))
        .order_by(
            human_feedback.c.feedback_item_id,
            human_feedback.c.updated_at,
            human_feedback.c.id,
        )
    )
    operation = "load" if reviewer is not None else "list"
    logger.info(
        "%s query start reviewer=%s paper=%s query=%s",
        operation,
        reviewer,
        paper_forum_id,
        query_id,
    )
    try:
        with db.begin() as conn:
            rows = [dict(row) for row in conn.execute(stmt).mappings()]
        logger.info(
            "%s query success reviewer=%s paper=%s query=%s rows=%s",
            operation,
            reviewer,
            paper_forum_id,
            query_id,
            len(rows),
        )
    except Exception:
        logger.exception(
            "%s query failed reviewer=%s paper=%s query=%s",
            operation,
            reviewer,
            paper_forum_id,
            query_id,
        )
        raise

    if reviewer_username is not None:
        return rows_to_feedback_payload(rows)

    by_reviewer: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_reviewer.setdefault(str(row.get("reviewer_username") or ""), []).append(row)
    latest_row = max(
        rows,
        key=lambda row: (str(row.get("updated_at") or ""), int(row.get("id") or 0)),
        default={},
    )
    return {
        "latest_reviewer_username": str(latest_row.get("reviewer_username") or ""),
        "latest_update": str(latest_row.get("updated_at") or ""),
        "reviewers": {
            reviewer: rows_to_feedback_payload(reviewer_rows)
            for reviewer, reviewer_rows in by_reviewer.items()
        }
    }
