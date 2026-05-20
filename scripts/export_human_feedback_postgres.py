from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from sqlalchemy import create_engine, text


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openreview_pipeline.utils.db.human_feedback_postgres import rows_to_feedback_payload  # noqa: E402


DEFAULT_DB_URL = "postgresql+psycopg://scifull:westlakenlp@127.0.0.1:5432/scifullmmbench"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "human_feedback_postgres_export.json"


def resolve_db_url(raw_db_url: Optional[str]) -> str:
    return (
        raw_db_url
        or os.getenv("HUMAN_FEEDBACK_DB_URL")
        or os.getenv("DATABASE_URL")
        or DEFAULT_DB_URL
    )


def json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
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
    normalized["selection_type"] = normalize_json_value(normalized.get("selection_type"))
    normalized["feedback_raw_json"] = normalize_json_value(normalized.get("feedback_raw_json"))
    for key in ("created_at", "updated_at"):
        if isinstance(normalized.get(key), (datetime, date)):
            normalized[key] = normalized[key].isoformat()
    return normalized


def fetch_feedback_rows(engine, reviewer: Optional[str] = None) -> List[Dict[str, Any]]:
    query = """
        SELECT
          id,
          paper_forum_id,
          query_id,
          query_text,
          feedback_item_id,
          reviewer_username,
          judgement,
          selection_type,
          reason_note,
          feedback_raw_json,
          created_at,
          updated_at
        FROM human_feedback
    """
    params: Dict[str, Any] = {}
    if reviewer:
        query += " WHERE reviewer_username = :reviewer"
        params["reviewer"] = reviewer
    query += (
        " ORDER BY paper_forum_id ASC, query_id ASC, "
        "reviewer_username ASC, feedback_item_id ASC"
    )

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()
    return [normalize_row(row) for row in rows]


def latest_timestamp(rows: Iterable[Mapping[str, Any]]) -> Optional[str]:
    latest = None
    for row in rows:
        value = row.get("updated_at")
        if value is not None and (latest is None or str(value) > latest):
            latest = str(value)
    return latest


def build_export(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    reviewers = set()
    for row in rows:
        paper_id = str(row.get("paper_forum_id") or "")
        query_id = str(row.get("query_id") or "")
        grouped[paper_id][query_id].append(row)
        reviewers.add(str(row.get("reviewer_username") or ""))

    papers = []
    for paper_id, query_groups in grouped.items():
        queries = []
        for query_id, query_rows in query_groups.items():
            by_reviewer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in query_rows:
                by_reviewer[str(row.get("reviewer_username") or "")].append(row)

            feedback_by_reviewer = []
            reviewers_payload = {}
            for reviewer, reviewer_rows in sorted(by_reviewer.items()):
                payload = rows_to_feedback_payload(reviewer_rows)
                reviewers_payload[reviewer] = payload
                feedback_by_reviewer.append(
                    {
                        "reviewer_username": reviewer,
                        "row_count": len(reviewer_rows),
                        "latest_update": latest_timestamp(reviewer_rows),
                        "payload": payload,
                    }
                )

            queries.append(
                {
                    "query_id": query_id,
                    "query_text": str(query_rows[0].get("query_text") or ""),
                    "row_count": len(query_rows),
                    "latest_update": latest_timestamp(query_rows),
                    "reviewers": reviewers_payload,
                    "feedback_by_reviewer": feedback_by_reviewer,
                    "rows": query_rows,
                }
            )

        papers.append(
            {
                "paper_forum_id": paper_id,
                "row_count": sum(len(query_rows) for query_rows in query_groups.values()),
                "query_count": len(query_groups),
                "queries": queries,
            }
        )

    return {
        "artifact_type": "human_feedback_postgres_export",
        "generated_at": datetime.now().isoformat(),
        "total_rows": len(rows),
        "total_papers": len(grouped),
        "total_queries": sum(len(query_groups) for query_groups in grouped.values()),
        "total_reviewers": len({reviewer for reviewer in reviewers if reviewer}),
        "papers": papers,
        "rows": rows,
    }


def build_feedback_index(export: Mapping[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index = {}
    for paper in export.get("papers", []):
        if not isinstance(paper, Mapping):
            continue
        paper_id = str(paper.get("paper_forum_id") or "")
        for query in paper.get("queries", []):
            if not isinstance(query, Mapping):
                continue
            query_text = str(query.get("query_text") or "")
            index[(paper_id, query_text)] = dict(query)
    return index


def combine_with_final_output(
    final_output_path: Path,
    export: Mapping[str, Any],
    export_path: Path,
) -> Dict[str, Any]:
    with final_output_path.open("r", encoding="utf-8") as handle:
        final_output = json.load(handle)
    if not isinstance(final_output, dict):
        raise ValueError(f"Expected top-level JSON object in {final_output_path}")

    feedback_index = build_feedback_index(export)
    matched_keys = set()
    for paper in final_output.get("papers", []):
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("paper_id") or "")
        for query in paper.get("queries", []):
            if not isinstance(query, dict):
                continue
            key = (paper_id, str(query.get("query_text") or ""))
            feedback = feedback_index.get(key)
            if feedback is None:
                continue
            query["human_feedback"] = {
                "query_id": feedback.get("query_id"),
                "latest_update": feedback.get("latest_update"),
                "row_count": feedback.get("row_count"),
                "reviewers": feedback.get("reviewers", {}),
                "feedback_by_reviewer": feedback.get("feedback_by_reviewer", []),
                "rows": feedback.get("rows", []),
            }
            matched_keys.add(key)

    unmatched = [
        query
        for key, query in feedback_index.items()
        if key not in matched_keys
    ]
    paths = final_output.setdefault("paths", {})
    if isinstance(paths, dict):
        paths["human_feedback_postgres_export"] = str(export_path.resolve())
    overview = final_output.setdefault("dataset_overview", {})
    if isinstance(overview, dict):
        overview["human_feedback_total_rows"] = export.get("total_rows")
        overview["human_feedback_total_papers"] = export.get("total_papers")
        overview["human_feedback_total_queries"] = export.get("total_queries")
        overview["human_feedback_total_reviewers"] = export.get("total_reviewers")
        overview["human_feedback_matched_queries"] = len(matched_keys)
        overview["human_feedback_unmatched_queries"] = len(unmatched)
    final_output["human_feedback_summary"] = {
        "export_path": str(export_path.resolve()),
        "total_rows": export.get("total_rows"),
        "total_papers": export.get("total_papers"),
        "total_queries": export.get("total_queries"),
        "total_reviewers": export.get("total_reviewers"),
        "matched_queries": len(matched_keys),
        "unmatched_queries": len(unmatched),
    }
    final_output["human_feedback_unmatched"] = unmatched
    return final_output


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")


def default_combined_output_path(final_output_path: Path) -> Path:
    return final_output_path.with_name(f"{final_output_path.stem}_with_human_feedback.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export PostgreSQL human_feedback rows to JSON and optionally merge them "
            "into final_pipeline_output.json."
        )
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help=(
            "SQLAlchemy PostgreSQL URL. Defaults to HUMAN_FEEDBACK_DB_URL, DATABASE_URL, "
            "or the Docker Compose local default."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path for the exported feedback JSON.",
    )
    parser.add_argument(
        "--final-output",
        default=None,
        help="Optional final_pipeline_output.json to enrich with human_feedback fields.",
    )
    parser.add_argument(
        "--combined-output",
        default=None,
        help="Path for the enriched final output. Defaults beside --final-output.",
    )
    parser.add_argument(
        "--reviewer",
        default=None,
        help="Optional reviewer_username filter.",
    )
    args = parser.parse_args()

    export_path = Path(args.output).expanduser().resolve()
    db_url = resolve_db_url(args.db_url)
    engine = create_engine(db_url, pool_pre_ping=True)
    rows = fetch_feedback_rows(engine, reviewer=args.reviewer)
    export = build_export(rows)
    write_json(export_path, export)
    print(f"Wrote feedback export: {export_path}")

    if args.final_output:
        final_output_path = Path(args.final_output).expanduser().resolve()
        combined_output_path = (
            Path(args.combined_output).expanduser().resolve()
            if args.combined_output
            else default_combined_output_path(final_output_path)
        )
        combined = combine_with_final_output(final_output_path, export, export_path)
        write_json(combined_output_path, combined)
        summary = combined.get("human_feedback_summary", {})
        print(f"Wrote enriched final output: {combined_output_path}")
        print(
            "Matched feedback queries: "
            f"{summary.get('matched_queries')} matched, "
            f"{summary.get('unmatched_queries')} unmatched"
        )


if __name__ == "__main__":
    main()
