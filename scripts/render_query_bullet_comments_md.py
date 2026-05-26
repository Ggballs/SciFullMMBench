from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_NAME = "query_bullet_original_comments.md"


def load_pipeline(path: Path) -> dict[str, Any]:
    if path.is_dir():
        candidates = [
            path / "final_pipeline_output.json",
            path / "ir_final.json",
            path / "pipeline_output.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        else:
            return load_stage_pipeline_dir(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "papers" not in data:
        if path.name == "03_queries.json":
            return load_stage_pipeline_dir(path.parent)
        raise ValueError(
            f"{path} does not look like a final pipeline output: missing 'papers'. "
            "Pass a directory containing 00_downloaded.json, 02_summarized.json, and 03_queries.json "
            "for stage-pipeline outputs."
        )
    return data


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_stage_pipeline_dir(path: Path) -> dict[str, Any]:
    downloaded_path = path / "00_downloaded.json"
    summarized_path = path / "02_summarized.json"
    queries_path = path / "03_queries.json"
    missing = [p.name for p in (downloaded_path, summarized_path, queries_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"No final pipeline JSON found in {path}, and stage files are incomplete. "
            f"Missing: {', '.join(missing)}."
        )

    downloaded = read_json(downloaded_path)
    summarized = read_json(summarized_path)
    queries = read_json(queries_path)

    openreview_by_id: dict[str, dict[str, Any]] = {}
    for record in downloaded.get("papers", []):
        paper_meta = record.get("paper", {})
        paper_id = paper_meta.get("id")
        if not paper_id:
            continue
        openreview_by_id[paper_id] = {
            "abstract": paper_meta.get("abstract"),
            "pdf_url": paper_meta.get("pdf_url"),
            "openreview_url": paper_meta.get("openreview_url"),
            "venue": paper_meta.get("venue"),
            "year": paper_meta.get("year"),
            "authors": paper_meta.get("authors"),
            "reviews": record.get("reviews", []),
            "rebuttals": record.get("rebuttals", []),
            "comments": record.get("comments", []),
            "decision": record.get("decision"),
        }

    summary_by_id: dict[str, dict[str, Any]] = {}
    for record in summarized.get("summaries", []):
        paper_id = record.get("paper_id")
        if paper_id:
            summary_by_id[paper_id] = record

    papers: list[dict[str, Any]] = []
    for record in queries.get("papers_queries", []):
        paper_id = record.get("paper_id")
        summary_record = summary_by_id.get(paper_id, {})
        papers.append(
            {
                "paper_id": paper_id,
                "paper_title": record.get("paper_title") or summary_record.get("paper_title"),
                "openreview": openreview_by_id.get(paper_id, {}),
                "summary_views": summary_record.get("summary_views")
                or summary_record.get("views")
                or [],
                "queries": record.get("queries_by_view", []),
            }
        )

    return {
        "artifact_type": "stage_pipeline_output",
        "generated_at": queries.get("generated_at"),
        "papers": papers,
    }


def md_escape_cell(text: Any) -> str:
    value = "" if text is None else str(text)
    return value.replace("|", "\\|").replace("\n", "<br>")


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def truncate(text: str, max_chars: int | None) -> str:
    if not max_chars or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def find_summary_view(paper: dict[str, Any], view_name: str | None) -> dict[str, Any] | None:
    for view in paper.get("summary_views", []):
        if view.get("view_name") == view_name:
            return view
    return None


def find_bullet(paper: dict[str, Any], view_name: str | None, bullet_index: Any) -> dict[str, Any] | None:
    view = find_summary_view(paper, view_name)
    if not view:
        return None
    for bullet in view.get("bullet_points", []):
        if bullet.get("index") == bullet_index:
            return bullet
    return None


SOURCE_REF_RE = re.compile(r"^(?P<section>[^/]+)/(?P<label>[^/]+)(?:/(?P<field>.+))?$")
NUMBER_RE = re.compile(r"(\d+)")


def get_number(label: str) -> int | None:
    match = NUMBER_RE.search(label)
    return int(match.group(1)) if match else None


def content_field(item: dict[str, Any], field: str | None) -> str:
    content = item.get("content", {})
    if not isinstance(content, dict):
        return ""
    if field and field in content:
        return clean_text(content.get(field))
    for fallback in ("comment", "review", "summary", "pros", "cons", "questions", "decision", "title"):
        if fallback in content and clean_text(content.get(fallback)):
            return clean_text(content.get(fallback))
    return clean_text(content)


def resolve_source_ref(paper: dict[str, Any], source_ref: str) -> tuple[str, str]:
    match = SOURCE_REF_RE.match(source_ref)
    if not match:
        return source_ref, ""

    section = match.group("section")
    label = match.group("label")
    field = match.group("field")
    openreview = paper.get("openreview", {})
    number = get_number(label)

    if section == "Reviews":
        for review in openreview.get("reviews", []):
            if review.get("number") == number:
                return source_ref, content_field(review, field)
    if section == "Comments":
        for comment in openreview.get("comments", []):
            if comment.get("number") == number:
                return source_ref, content_field(comment, field)
    if section == "Rebuttals":
        for rebuttal in openreview.get("rebuttals", []):
            if rebuttal.get("number") == number:
                return source_ref, content_field(rebuttal, field)
    if section == "Decision":
        decision = openreview.get("decision")
        if isinstance(decision, dict):
            return source_ref, content_field(decision, field)

    return source_ref, ""


def iter_queries(paper: dict[str, Any]) -> list[dict[str, Any]]:
    queries = paper.get("queries")
    if isinstance(queries, list):
        return queries

    # Fallback for earlier stage-3 outputs if someone passes a lightly transformed file.
    queries_by_view = paper.get("queries_by_view")
    if isinstance(queries_by_view, list):
        return queries_by_view
    return []


def render_report(data: dict[str, Any], max_comment_chars: int | None) -> str:
    lines: list[str] = [
        "# Query, Bullet Point, Original Comment View",
        "",
        f"Generated from artifact: `{data.get('artifact_type', 'unknown')}`",
        "",
    ]

    for paper in data.get("papers", []):
        paper_id = clean_text(paper.get("paper_id"))
        paper_title = clean_text(paper.get("paper_title"))
        title = paper_title or paper_id or "Untitled paper"
        lines.extend([f"## {title}", ""])
        if paper_id:
            lines.extend([f"`{paper_id}`", ""])

        for query in iter_queries(paper):
            query_text = clean_text(query.get("query_text"))
            source_view = query.get("source_view")
            bullet_index = query.get("related_bullet_indice")
            is_multimodal = query.get("is_multimodal")
            if is_multimodal is True:
                multimodal_label = "Yes"
            elif is_multimodal is False:
                multimodal_label = "No"
            else:
                multimodal_label = "Unknown"
            bullet = find_bullet(paper, source_view, bullet_index)

            lines.extend([f"### {query_text}", ""])
            lines.append(
                f"- **View:** `{source_view}`  \n"
                f"- **Based on multimodal info:** `{multimodal_label}`  \n"
                f"- **Related bullet index:** `{bullet_index}`"
            )

            if bullet:
                lines.extend(["", f"- **Bullet point:** {clean_text(bullet.get('text'))}", ""])
                if bullet.get("multimodal_ref"):
                    dependency = clean_text(bullet.get("multimodal_dependency") or "none")
                    dependency_rationale = clean_text(
                        bullet.get("multimodal_dependency_rationale")
                        or bullet.get("multimodal_rationale")
                        or ""
                    )
                    lines.append(f"- **Bullet multimodal dependency:** `{dependency}`")
                    if dependency_rationale:
                        lines.append(f"- **Bullet multimodal rationale:** {dependency_rationale}")
                    lines.append("")
                source_refs = bullet.get("source_refs", [])
            else:
                lines.extend(["", "- **Bullet point:** _Not found_", ""])
                source_refs = []

            if query.get("is_multimodal") and query.get("multimodal_rationale"):
                lines.extend(
                    [
                        f"- **Query multimodal rationale:** {clean_text(query.get('multimodal_rationale'))}",
                        "",
                    ]
                )

            if query.get("related_bullet_justification"):
                lines.extend(
                    [
                        f"- **Query-bullet justification:** {clean_text(query.get('related_bullet_justification'))}",
                        "",
                    ]
                )

            if source_refs:
                lines.extend(["| Source ref | Original comment |", "|---|---|"])
                for ref in source_refs:
                    label, comment = resolve_source_ref(paper, str(ref))
                    comment = truncate(comment, max_comment_chars)
                    lines.append(f"| {md_escape_cell(label)} | {md_escape_cell(comment)} |")
                lines.append("")
            else:
                lines.extend(["_No source refs found for this bullet._", ""])

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render a Markdown view linking each generated query to its related bullet "
            "and the original OpenReview comments used as evidence."
        )
    )
    parser.add_argument(
        "pipeline_path",
        type=Path,
        help="Path to final_pipeline_output.json, or a directory containing it.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=f"Markdown output path. Defaults to <pipeline_dir>/{DEFAULT_OUTPUT_NAME}.",
    )
    parser.add_argument(
        "--max-comment-chars",
        type=int,
        default=1800,
        help="Maximum characters to include per original comment. Use 0 for no truncation.",
    )
    args = parser.parse_args()

    data = load_pipeline(args.pipeline_path)
    source_path = args.pipeline_path
    output = args.output
    if output is None:
        output_dir = source_path if source_path.is_dir() else source_path.parent
        output = output_dir / DEFAULT_OUTPUT_NAME

    max_chars = None if args.max_comment_chars == 0 else args.max_comment_chars
    output.write_text(render_report(data, max_chars), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
