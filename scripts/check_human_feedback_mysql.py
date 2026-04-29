from __future__ import annotations

import argparse
import os
from typing import Optional

from sqlalchemy import create_engine, text


DEFAULT_DB_URL = "mysql+pymysql://scifull:westlakenlp@127.0.0.1:3306/scifullmmbench?charset=utf8mb4"


def resolve_db_url(raw_db_url: Optional[str]) -> str:
    return (
        raw_db_url
        or os.getenv("HUMAN_FEEDBACK_DB_URL")
        or os.getenv("DATABASE_URL")
        or DEFAULT_DB_URL
    )


def print_recent_rows(engine, limit: int) -> None:
    query = text(
        """
        SELECT
          id,
          reviewer_username,
          paper_forum_id,
          query_id,
          query_text,
          feedback_item_id,
          judgement,
          updated_at
        FROM human_feedback
        ORDER BY updated_at DESC
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"limit": limit}).mappings().all()

    print(f"\nRecent feedback rows (limit {limit})")
    print("-" * 120)
    if not rows:
        print("(no rows)")
        return
    for row in rows:
        query_id = str(row["query_id"])
        if len(query_id) > 48:
            query_id = query_id[:45] + "..."
        print(
            f"id={row['id']} user={row['reviewer_username']} paper={row['paper_forum_id']} "
            f"query={query_id}  query_text={row['query_text']} item={row['feedback_item_id']} judgement={row['judgement']} "
            f"updated={row['updated_at']}"
        )


def print_user_counts(engine) -> None:
    query = text(
        """
        SELECT
          reviewer_username,
          COUNT(*) AS feedback_rows,
          COUNT(DISTINCT query_id) AS judged_queries,
          COUNT(DISTINCT paper_forum_id) AS papers,
          MAX(updated_at) AS latest_update
        FROM human_feedback
        GROUP BY reviewer_username
        ORDER BY feedback_rows DESC, reviewer_username ASC
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    print("\nSubmit counts by user")
    print("-" * 120)
    if not rows:
        print("(no rows)")
        return
    for row in rows:
        print(
            f"user={row['reviewer_username']} feedback_rows={row['feedback_rows']} "
            f"judged_queries={row['judged_queries']} papers={row['papers']} "
            f"latest_update={row['latest_update']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect human_feedback rows in MySQL.")
    parser.add_argument(
        "--db-url",
        default=None,
        help=(
            "SQLAlchemy MySQL URL. Defaults to HUMAN_FEEDBACK_DB_URL, DATABASE_URL, "
            "or the Docker Compose local default."
        ),
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of recent rows to print.")
    args = parser.parse_args()

    db_url = resolve_db_url(args.db_url)
    engine = create_engine(db_url, pool_pre_ping=True)
    print_recent_rows(engine, args.limit)
    print_user_counts(engine)


if __name__ == "__main__":
    main()
