from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from sqlalchemy import create_engine, inspect, text


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_URL = "mysql+pymysql://scifull:westlakenlp@127.0.0.1:3306/scifullmmbench?charset=utf8mb4"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "human_feedback_mysql_full_export.json"
DEFAULT_TABLE = "human_feedback"


def resolve_db_url(raw_db_url: Optional[str]) -> str:
    return (
        raw_db_url
        or os.getenv("MYSQL_HUMAN_FEEDBACK_DB_URL")
        or os.getenv("MYSQL_DATABASE_URL")
        or DEFAULT_DB_URL
    )


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    return str(value)


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def normalize_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    for key, value in list(normalized.items()):
        if isinstance(value, (datetime, date)):
            normalized[key] = value.isoformat()
        elif isinstance(value, Decimal):
            normalized[key] = str(value)
        elif isinstance(value, (bytes, bytearray)):
            normalized[key] = value.hex()
        elif key.endswith("_json") or key in {"selection_type", "feedback_raw_json"}:
            normalized[key] = normalize_json_value(value)
    return normalized


def quote_identifier(identifier: str) -> str:
    return "`" + identifier.replace("`", "``") + "`"


def table_columns(engine, table_name: str) -> List[str]:
    columns = [column["name"] for column in inspect(engine).get_columns(table_name)]
    if not columns:
        raise ValueError(f"No columns found for table {table_name!r}.")
    return columns


def fetch_table_rows(
    engine,
    table_name: str,
    *,
    reviewer: Optional[str] = None,
    order_by: Optional[str] = None,
) -> tuple[List[str], List[Dict[str, Any]]]:
    columns = table_columns(engine, table_name)
    quoted_table = quote_identifier(table_name)
    query = f"SELECT * FROM {quoted_table}"
    params: Dict[str, Any] = {}

    if reviewer:
        if "reviewer_username" not in columns:
            raise ValueError(f"Table {table_name!r} has no reviewer_username column.")
        query += " WHERE reviewer_username = :reviewer"
        params["reviewer"] = reviewer

    if order_by:
        order_columns = [column.strip() for column in order_by.split(",") if column.strip()]
    else:
        preferred_order = [
            "paper_forum_id",
            "query_id",
            "reviewer_username",
            "feedback_item_id",
            "id",
        ]
        order_columns = [column for column in preferred_order if column in columns]

    if order_columns:
        missing = [column for column in order_columns if column not in columns]
        if missing:
            raise ValueError(f"Unknown order column(s) for {table_name!r}: {', '.join(missing)}")
        query += " ORDER BY " + ", ".join(quote_identifier(column) for column in order_columns)

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()
    return columns, [normalize_row(row) for row in rows]


def count_distinct(rows: Iterable[Mapping[str, Any]], key: str) -> int:
    return len({str(row.get(key) or "") for row in rows if row.get(key) not in (None, "")})


def latest_value(rows: Iterable[Mapping[str, Any]], key: str) -> Optional[str]:
    latest = None
    for row in rows:
        value = row.get(key)
        if value is not None and (latest is None or str(value) > latest):
            latest = str(value)
    return latest


def build_export(table_name: str, columns: List[str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    export: Dict[str, Any] = {
        "artifact_type": "human_feedback_mysql_full_export",
        "generated_at": datetime.now().isoformat(),
        "source": {
            "database": "mysql",
            "table": table_name,
        },
        "columns": columns,
        "total_rows": len(rows),
        "rows": rows,
    }

    if "paper_forum_id" in columns:
        export["total_papers"] = count_distinct(rows, "paper_forum_id")
    if "query_id" in columns:
        export["total_queries"] = count_distinct(rows, "query_id")
    if "reviewer_username" in columns:
        export["total_reviewers"] = count_distinct(rows, "reviewer_username")
    if "updated_at" in columns:
        export["latest_update"] = latest_value(rows, "updated_at")

    return export


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export full MySQL table rows to JSON for PostgreSQL cutover."
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help=(
            "SQLAlchemy MySQL URL. Defaults to MYSQL_HUMAN_FEEDBACK_DB_URL, "
            "MYSQL_DATABASE_URL, or the legacy local default."
        ),
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"MySQL table to export. Defaults to {DEFAULT_TABLE}.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path for the exported full-table JSON.",
    )
    parser.add_argument(
        "--reviewer",
        default=None,
        help="Optional reviewer_username filter. Only valid for tables with that column.",
    )
    parser.add_argument(
        "--order-by",
        default=None,
        help="Optional comma-separated column list for deterministic export ordering.",
    )
    args = parser.parse_args()

    export_path = Path(args.output).expanduser().resolve()
    engine = create_engine(resolve_db_url(args.db_url), pool_pre_ping=True)
    columns, rows = fetch_table_rows(
        engine,
        args.table,
        reviewer=args.reviewer,
        order_by=args.order_by,
    )
    export = build_export(args.table, columns, rows)
    write_json(export_path, export)
    print(f"Wrote full MySQL export: {export_path}")
    print(f"Exported rows={export['total_rows']} columns={len(columns)} table={args.table}")


if __name__ == "__main__":
    main()
