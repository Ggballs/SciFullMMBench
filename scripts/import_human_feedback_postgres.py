from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as postgres_insert


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openreview_pipeline.utils.db.human_feedback_postgres import human_feedback  # noqa: E402


DEFAULT_DB_URL = "postgresql+psycopg://scifull:westlakenlp@127.0.0.1:5432/scifullmmbench"


def resolve_db_url(raw_db_url: Optional[str]) -> str:
    return (
        raw_db_url
        or os.getenv("HUMAN_FEEDBACK_DB_URL")
        or os.getenv("DATABASE_URL")
        or DEFAULT_DB_URL
    )


def parse_timestamp(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return value


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def load_export_rows(export_path: Path) -> List[Dict[str, Any]]:
    with export_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError(f"Expected top-level rows list in {export_path}")
    return [normalize_row(row) for row in rows if isinstance(row, Mapping)]


def normalize_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    allowed_fields = {
        "paper_forum_id",
        "query_id",
        "query_text",
        "feedback_item_id",
        "reviewer_username",
        "judgement",
        "selection_type",
        "reason_note",
        "feedback_raw_json",
        "created_at",
        "updated_at",
    }
    normalized = {key: row.get(key) for key in allowed_fields if key in row}
    normalized["selection_type"] = normalize_json_value(normalized.get("selection_type"))
    normalized["feedback_raw_json"] = normalize_json_value(normalized.get("feedback_raw_json"))
    normalized["created_at"] = parse_timestamp(normalized.get("created_at"))
    normalized["updated_at"] = parse_timestamp(normalized.get("updated_at"))
    return normalized


def import_rows(engine, rows: Iterable[Mapping[str, Any]]) -> int:
    clean_rows = [dict(row) for row in rows]
    if not clean_rows:
        return 0

    insert_stmt = postgres_insert(human_feedback).values(clean_rows)
    update_columns = {
        column.name: insert_stmt.excluded[column.name]
        for column in human_feedback.columns
        if column.name not in {"id", "created_at"}
    }
    stmt = insert_stmt.on_conflict_do_update(
        index_elements=[
            human_feedback.c.paper_forum_id,
            human_feedback.c.query_id,
            human_feedback.c.feedback_item_id,
            human_feedback.c.reviewer_username,
        ],
        set_=update_columns,
    )
    with engine.begin() as conn:
        result = conn.execute(stmt)
    return int(result.rowcount or 0)


def print_validation(engine) -> None:
    query = text(
        """
        SELECT
          COUNT(*) AS total_rows,
          COUNT(DISTINCT reviewer_username) AS total_reviewers,
          COUNT(DISTINCT query_id) AS total_queries,
          MAX(updated_at) AS latest_update
        FROM human_feedback
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query).mappings().one()
    print(
        "PostgreSQL human_feedback summary: "
        f"rows={row['total_rows']} reviewers={row['total_reviewers']} "
        f"queries={row['total_queries']} latest_update={row['latest_update']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import exported human_feedback rows into PostgreSQL."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a human feedback export JSON file.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help=(
            "SQLAlchemy PostgreSQL URL. Defaults to HUMAN_FEEDBACK_DB_URL, DATABASE_URL, "
            "or the Docker Compose local default."
        ),
    )
    args = parser.parse_args()

    export_path = Path(args.input).expanduser().resolve()
    engine = create_engine(resolve_db_url(args.db_url), pool_pre_ping=True)
    rows = load_export_rows(export_path)
    affected = import_rows(engine, rows)
    print(f"Imported feedback rows from {export_path}: affected_rows={affected}")
    print_validation(engine)


if __name__ == "__main__":
    main()
