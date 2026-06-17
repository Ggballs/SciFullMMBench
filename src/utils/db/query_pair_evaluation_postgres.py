from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger,
    Column,
    Index,
    Integer,
    MetaData,
    String,
    TIMESTAMP,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    select,
    text,
    func,
)
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.postgresql import JSONB, insert as postgres_insert

QUERY_PAIR_EVALUATION_TABLE = "query_pair_evaluation"
logger = logging.getLogger(__name__)

metadata = MetaData()

query_pair_evaluation = Table(
    QUERY_PAIR_EVALUATION_TABLE,
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("pair_id", String(128), nullable=False),
    Column("mode", String(32), nullable=False),
    Column("paper_id", String(128), nullable=False),
    Column("reviewer_username", String(64), nullable=False),
    Column("choice", String(32), nullable=False),
    Column("confidence", String(16), nullable=True),
    Column("note", Text, nullable=True),
    Column("ordering_seed", Integer, nullable=False),
    Column("pair_snapshot", JSONB, nullable=True),
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
    UniqueConstraint(
        "pair_id",
        "reviewer_username",
        name="uq_query_pair_evaluation",
    ),
    Index("idx_query_pair_eval_pair_id", "pair_id"),
    Index("idx_query_pair_eval_reviewer_username", "reviewer_username"),
    Index("idx_query_pair_eval_mode", "mode"),
)


def get_engine(db_url: Optional[str] = None) -> Engine:
    url = db_url or os.getenv("HUMAN_FEEDBACK_DB_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError(
            "Set HUMAN_FEEDBACK_DB_URL or DATABASE_URL to use PostgreSQL evaluation CRUD."
        )
    return create_engine(url, pool_pre_ping=True)


def save_evaluation(
    pair_id: str,
    mode: str,
    paper_id: str,
    reviewer_username: str,
    choice: str,
    ordering_seed: int,
    pair_snapshot: Dict[str, Any],
    *,
    confidence: Optional[str] = None,
    note: Optional[str] = None,
    engine: Optional[Engine] = None,
) -> None:
    db = engine or get_engine()

    row = {
        "pair_id": str(pair_id),
        "mode": str(mode),
        "paper_id": str(paper_id),
        "reviewer_username": str(reviewer_username).strip(),
        "choice": str(choice),
        "confidence": str(confidence) if confidence else None,
        "note": str(note) if note else None,
        "ordering_seed": int(ordering_seed),
        "pair_snapshot": pair_snapshot,
    }

    stmt = postgres_insert(query_pair_evaluation).values(row)
    update_columns = {
        col.name: stmt.excluded[col.name]
        for col in query_pair_evaluation.columns
        if col.name not in {"id", "created_at", "updated_at"}
    }
    update_columns["updated_at"] = text("CURRENT_TIMESTAMP")
    stmt = stmt.on_conflict_do_update(
        index_elements=["pair_id", "reviewer_username"],
        set_=update_columns,
    )

    logger.info(
        "save_evaluation pair=%s reviewer=%s choice=%s",
        pair_id,
        reviewer_username,
        choice,
    )
    with db.begin() as conn:
        conn.execute(stmt)


def load_evaluation(
    pair_id: str,
    reviewer_username: str,
    *,
    engine: Optional[Engine] = None,
) -> Optional[Dict[str, Any]]:
    db = engine or get_engine()
    stmt = (
        select(query_pair_evaluation)
        .where(query_pair_evaluation.c.pair_id == str(pair_id))
        .where(query_pair_evaluation.c.reviewer_username == str(reviewer_username).strip())
    )
    with db.begin() as conn:
        row = conn.execute(stmt).mappings().first()
    return dict(row) if row else None


def load_evaluations_for_reviewer(
    reviewer_username: str,
    *,
    mode_filter: Optional[str] = None,
    engine: Optional[Engine] = None,
) -> List[Dict[str, Any]]:
    db = engine or get_engine()
    stmt = select(query_pair_evaluation).where(
        query_pair_evaluation.c.reviewer_username == str(reviewer_username).strip()
    )
    if mode_filter and mode_filter != "All":
        stmt = stmt.where(query_pair_evaluation.c.mode == str(mode_filter))
    stmt = stmt.order_by(query_pair_evaluation.c.pair_id)
    with db.begin() as conn:
        rows = conn.execute(stmt).mappings()
        return [dict(row) for row in rows]


def load_review_counts(
    *,
    mode_filter: Optional[str] = None,
    engine: Optional[Engine] = None,
) -> Dict[str, int]:
    db = engine or get_engine()
    stmt = select(
        query_pair_evaluation.c.pair_id,
        func.count(query_pair_evaluation.c.id).label("cnt"),
    ).group_by(query_pair_evaluation.c.pair_id)

    if mode_filter and mode_filter != "All":
        stmt = stmt.where(query_pair_evaluation.c.mode == str(mode_filter))

    with db.begin() as conn:
        rows = conn.execute(stmt).mappings()
        return {str(row["pair_id"]): int(row["cnt"]) for row in rows}
