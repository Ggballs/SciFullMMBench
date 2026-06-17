from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_JSONL = (
    REPO_ROOT
    / "outputs/query_analysis/golden_view_classification_with_targets/golden_query_view_consensus.jsonl"
)
DEFAULT_ANNOTATIONS_CSV = (
    REPO_ROOT
    / "outputs/query_analysis/golden_view_classification_with_targets/final_human_consensus_annotations.csv"
)

VIEW_LABELS = ["motivation", "method", "experiment"]
PRIMARY_LABELS = ["motivation", "method", "experiment", "unclear"]
HUMAN_CONFIDENCE_LEVELS = ["High", "Medium", "Low"]
DECISIONS = ["accept", "fix", "unclear", "skip"]
MILESTONE_STEP = 50


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes", "y", "checked"}


def split_labels(value: Any) -> list[str]:
    if isinstance(value, list):
        labels = value
    else:
        labels = str(value or "").replace(",", "|").replace(";", "|").split("|")
    normalized: list[str] = []
    for label in labels:
        label = str(label).strip().lower()
        if label in VIEW_LABELS and label not in normalized:
            normalized.append(label)
    return normalized


def labels_text(labels: Any) -> str:
    parsed = split_labels(labels)
    return "|".join(parsed) if parsed else "unclear"


def clean_context_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def format_math_text(value: Any) -> str:
    text = str(value or "")
    parts: list[str] = []
    last = 0
    for match in re.finditer(r"\$(.+?)\$", text):
        parts.append(esc(text[last : match.start()]))
        parts.append(f"<span class='math-inline'>{esc(match.group(1))}</span>")
        last = match.end()
    parts.append(esc(text[last:]))
    return "".join(parts)


def format_query_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for marker in (
        "What I have:",
        "What I want to estimate:",
        "Edit 1:",
        "Edit 2:",
        "Interpretation 1:",
        "Interpretation 2:",
        "R code for interpretation 1:",
        "R code for interpretation 2:",
    ):
        text = text.replace(f" {marker}", f"\n\n{marker}")
    text = re.sub(r"\s+(library\([A-Za-z0-9_.]+\))", r"\n\n\1", text)
    text = re.sub(r"\s+(set\.seed\()", r"\n\1", text)
    text = re.sub(r"\s+(for \()", r"\n\1", text)
    text = re.sub(r"\s+(if \()", r"\n\1", text)
    text = re.sub(r"\s+(cat\()", r"\n\1", text)
    text = re.sub(r"\s+(>\s*cat\()", r"\n\1", text)
    sections = text.split("\n\n", 1)
    if len(sections) == 2:
        title_html = f"<div class='query-title'>{format_math_text(sections[0])}</div>"
        body_html = f"<div class='query-body'>{format_math_text(sections[1])}</div>"
        return title_html + body_html
    return f"<div class='query-body'>{format_math_text(text)}</div>"


def parse_target_papers(value: Any) -> list[dict[str, str]]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [{"title": value}]
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        value = [{"title": str(value)}]

    papers: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            paper = {
                "title": str(item.get("title") or item.get("paper_title") or item.get("name") or "").strip(),
                "url": str(item.get("url") or item.get("link") or "").strip(),
                "score": str(item.get("score") or "").strip(),
                "paper_urls": item.get("paper_urls") or item.get("paper_url") or "",
                "source_id": str(
                    item.get("source_id")
                    or item.get("corpusid")
                    or item.get("corpus_id")
                    or item.get("arxiv_id")
                    or item.get("paper_id")
                    or ""
                ).strip(),
                "abstract": str(item.get("abstract") or item.get("body") or "").strip(),
            }
        else:
            paper = {"title": str(item).strip(), "url": "", "source_id": "", "abstract": ""}
        if any(paper.values()):
            papers.append(paper)
    return papers


def target_papers_json(value: Any) -> str:
    return json.dumps(parse_target_papers(value), ensure_ascii=False)


def target_papers_text(value: Any) -> str:
    papers = parse_target_papers(value)
    if not papers:
        return "No reference context."
    lines: list[str] = []
    for index, paper in enumerate(papers, 1):
        lines.append(f"Reference item {index}:")
        if paper.get("title"):
            lines.append(f"  title: {paper['title']}")
        if paper.get("score"):
            lines.append(f"  score: {paper['score']}")
        if paper.get("paper_urls"):
            lines.append(f"  paper_urls: {paper['paper_urls']}")
        if paper.get("source_id"):
            lines.append(f"  source_id: {paper['source_id']}")
        if paper.get("url"):
            lines.append(f"  url: {paper['url']}")
        if paper.get("abstract"):
            lines.append(f"  context: {clean_context_text(paper['abstract'])}")
    return "\n".join(lines)


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id") or f"row_{index:04d}").strip()
            rows.append(
                {
                    **row,
                    "query_id": query_id,
                    "query_type": row.get("query_type", ""),
                    "source_file": row.get("source_file", ""),
                    "row_index": row.get("row_index", index),
                    "query": row.get("query", ""),
                    "target_papers": parse_target_papers(row.get("target_papers")),
                    "call1_labels": labels_text(row.get("call1_labels")),
                    "call1_primary_label": row.get("call1_primary_label") or "unclear",
                    "call1_ambiguous": parse_bool(row.get("call1_ambiguous")),
                    "call1_confidence": row.get("call1_confidence", ""),
                    "call1_rationale": row.get("call1_rationale", ""),
                    "call2_labels": labels_text(row.get("call2_labels")),
                    "call2_primary_label": row.get("call2_primary_label") or "unclear",
                    "call2_ambiguous": parse_bool(row.get("call2_ambiguous")),
                    "call2_confidence": row.get("call2_confidence", ""),
                    "call2_rationale": row.get("call2_rationale", ""),
                    "agreement_status": row.get("agreement_status", ""),
                    "bucket": row.get("bucket", ""),
                    "labels": labels_text(row.get("labels")),
                    "primary_label": row.get("primary_label") or "unclear",
                    "ambiguous": parse_bool(row.get("ambiguous")),
                    "confidence": row.get("confidence", ""),
                    "rationale": row.get("rationale", ""),
                }
            )
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_annotations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    annotations: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(path):
        query_id = str(row.get("query_id", "")).strip()
        if query_id:
            annotations[query_id] = row
    return annotations


ANNOTATION_FIELDNAMES = [
    "query_id",
    "query_type",
    "source_file",
    "row_index",
    "query",
    "call1_labels",
    "call1_primary_label",
    "call1_ambiguous",
    "call1_confidence",
    "call1_rationale",
    "call2_labels",
    "call2_primary_label",
    "call2_ambiguous",
    "call2_confidence",
    "call2_rationale",
    "agreement_status",
    "bucket",
    "consensus_labels",
    "consensus_primary_label",
    "consensus_ambiguous",
    "consensus_confidence",
    "consensus_rationale",
    "final_labels",
    "final_motivation",
    "final_method",
    "final_experiment",
    "final_primary_label",
    "final_ambiguous",
    "final_confidence",
    "final_decision",
    "final_notes",
    "annotator",
    "updated_at",
    "target_papers",
]


def atomic_write_annotations(path: Path, annotations: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    rows = sorted(annotations.values(), key=lambda row: str(row.get("query_id", "")))
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def annotation_for_row(row: dict[str, Any], annotations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    existing = annotations.get(row["query_id"])
    if existing:
        return existing

    labels = split_labels(row.get("labels"))
    bucket = str(row.get("bucket") or "")
    return {
        "final_labels": "|".join(labels),
        "final_motivation": "1" if "motivation" in labels else "",
        "final_method": "1" if "method" in labels else "",
        "final_experiment": "1" if "experiment" in labels else "",
        "final_primary_label": row.get("primary_label") or "unclear",
        "final_ambiguous": "1" if parse_bool(row.get("ambiguous")) else "",
        "final_confidence": "High" if bucket == "high_confidence" and not parse_bool(row.get("ambiguous")) else "Medium",
        "final_decision": "accept",
        "final_notes": "",
        "annotator": "",
    }


def query_panel(row: dict[str, Any]) -> str:
    meta = (
        f"{esc(row.get('query_id'))} · {esc(row.get('query_type'))} · "
        f"row {esc(row.get('row_index'))} · {esc(row.get('bucket'))} · "
        f"{esc(row.get('agreement_status'))}"
    )
    return (
        "<div class='query-card'>"
        f"<div class='query-meta'>{meta}</div>"
        f"<div class='query-text'>{format_query_text(row.get('query'))}</div>"
        f"<div class='source-path'>{esc(row.get('source_file'))}</div>"
        "</div>"
    )


def target_papers_panel(row: dict[str, Any]) -> str:
    papers = parse_target_papers(row.get("target_papers"))
    is_qa = str(row.get("query_type") or "").upper() == "QA"
    panel_title = "Reference Answers" if is_qa else "Target Papers"
    empty_text = "No reference answer context in this row." if is_qa else "No target paper context in this row."
    if not papers:
        return (
            "<div class='target-panel'>"
            f"<div class='panel-title'>{panel_title}</div>"
            f"<div class='empty-target'>{empty_text}</div>"
            "</div>"
        )

    paper_blocks = []
    for index, paper in enumerate(papers, 1):
        title = esc(paper.get("title"))
        source_id = esc(paper.get("source_id"))
        url = esc(paper.get("url"))
        score = esc(paper.get("score"))
        paper_urls = esc(paper.get("paper_urls"))
        abstract = esc(clean_context_text(paper.get("abstract")))
        url_html = f"<a href='{url}' target='_blank' rel='noreferrer'>{url}</a>" if url else ""
        source_html = f"<div class='target-paper-meta'>{source_id}</div>" if source_id else ""
        score_html = f"<div class='target-paper-meta'>score: {score}</div>" if score else ""
        paper_urls_html = (
            f"<div class='target-paper-citations'><b>paper_urls:</b> {paper_urls}</div>" if paper_urls else ""
        )
        link_html = f"<div class='target-paper-url'>{url_html}</div>" if url_html else ""
        abstract_html = f"<div class='target-paper-context'>{abstract}</div>" if abstract else ""
        item_title = f"Reference answer {index}" if is_qa else f"{index}. {title or 'Untitled target'}"
        if is_qa and title:
            item_title = f"{item_title}: {title}"
        paper_blocks.append(
            "<div class='target-paper'>"
            f"<div class='target-paper-title'>{item_title}</div>"
            f"{score_html}"
            f"{paper_urls_html}"
            f"{source_html}"
            f"{link_html}"
            f"{abstract_html}"
            "</div>"
        )
    return (
        "<div class='target-panel'>"
        f"<div class='panel-title'>{panel_title}</div>"
        + "".join(paper_blocks)
        + "</div>"
    )


def call_panel(title: str, row: dict[str, Any], prefix: str) -> str:
    return (
        "<div class='llm-panel'>"
        f"<div class='panel-title'>{esc(title)}</div>"
        f"<div><b>Labels:</b> {esc(row.get(f'{prefix}_labels'))}</div>"
        f"<div><b>Primary:</b> {esc(row.get(f'{prefix}_primary_label'))}</div>"
        f"<div><b>Ambiguous:</b> {esc(row.get(f'{prefix}_ambiguous'))}</div>"
        f"<div><b>Confidence:</b> {esc(row.get(f'{prefix}_confidence'))}</div>"
        f"<div><b>Rationale:</b> {esc(row.get(f'{prefix}_rationale'))}</div>"
        "</div>"
    )


def consensus_panel(row: dict[str, Any]) -> str:
    return (
        "<div class='consensus-panel'>"
        "<div class='panel-title'>Consensus</div>"
        f"<div><b>Labels:</b> {esc(row.get('labels'))}</div>"
        f"<div><b>Primary:</b> {esc(row.get('primary_label'))}</div>"
        f"<div><b>Ambiguous:</b> {esc(row.get('ambiguous'))}</div>"
        f"<div><b>Confidence:</b> {esc(row.get('confidence'))}</div>"
        f"<div><b>Rationale:</b> {esc(row.get('rationale'))}</div>"
        "</div>"
    )


def copy_text(row: dict[str, Any], annotation: dict[str, Any]) -> str:
    final_labels = annotation.get("final_labels") or "|".join(
        label for label in VIEW_LABELS if parse_bool(annotation.get(f"final_{label}"))
    )
    return "\n".join(
        [
            f"query_id: {row.get('query_id', '')}",
            f"query_type: {row.get('query_type', '')}",
            f"bucket: {row.get('bucket', '')}",
            f"agreement_status: {row.get('agreement_status', '')}",
            f"query: {row.get('query', '')}",
            "",
            "reference_context:" if str(row.get("query_type") or "").upper() == "QA" else "target_papers:",
            target_papers_text(row.get("target_papers")),
            "",
            f"call1_labels: {row.get('call1_labels', '')}",
            f"call1_primary_label: {row.get('call1_primary_label', '')}",
            f"call1_ambiguous: {row.get('call1_ambiguous', '')}",
            f"call1_confidence: {row.get('call1_confidence', '')}",
            f"call1_rationale: {row.get('call1_rationale', '')}",
            "",
            f"call2_labels: {row.get('call2_labels', '')}",
            f"call2_primary_label: {row.get('call2_primary_label', '')}",
            f"call2_ambiguous: {row.get('call2_ambiguous', '')}",
            f"call2_confidence: {row.get('call2_confidence', '')}",
            f"call2_rationale: {row.get('call2_rationale', '')}",
            "",
            f"consensus_labels: {row.get('labels', '')}",
            f"consensus_primary_label: {row.get('primary_label', '')}",
            f"consensus_ambiguous: {row.get('ambiguous', '')}",
            f"consensus_confidence: {row.get('confidence', '')}",
            f"consensus_rationale: {row.get('rationale', '')}",
            "",
            f"final_labels: {final_labels}",
            f"final_primary_label: {annotation.get('final_primary_label', '')}",
            f"final_ambiguous: {parse_bool(annotation.get('final_ambiguous'))}",
            f"final_confidence: {annotation.get('final_confidence', '')}",
            f"final_decision: {annotation.get('final_decision', '')}",
            f"final_notes: {annotation.get('final_notes', '')}",
        ]
    )


def progress_text(rows: list[dict[str, Any]], index: int, annotations: dict[str, dict[str, Any]]) -> str:
    total = len(rows)
    done = sum(1 for row in rows if row["query_id"] in annotations)
    if not total:
        return "No rows loaded."
    return f"Row {index + 1} of {total} · saved {done}/{total}"


def milestone_bonus(done: int, total: int) -> str:
    if done <= 0 or done % MILESTONE_STEP != 0:
        return ""
    remaining = max(0, total - done)
    if remaining:
        detail = f"{remaining} rows left."
    else:
        detail = "Dataset complete."
    return (
        "<div class='milestone-bonus'>"
        f"<b>Milestone bonus:</b> congratulations on completing {done} annotations. {detail}"
        "</div>"
    )


def render(
    rows: list[dict[str, Any]],
    index: int,
    annotations: dict[str, dict[str, Any]],
) -> tuple[Any, ...]:
    if not rows:
        return (
            "No rows loaded. Set the input JSONL path and click Load.",
            "",
            "",
            "",
            "",
            "",
            [],
            "unclear",
            False,
            "Medium",
            "skip",
            "",
            "",
            1,
            0,
        )

    index = max(0, min(int(index or 0), len(rows) - 1))
    row = rows[index]
    annotation = annotation_for_row(row, annotations or {})

    labels = split_labels(annotation.get("final_labels"))
    if not labels:
        labels = [label for label in VIEW_LABELS if parse_bool(annotation.get(f"final_{label}"))]

    primary = str(annotation.get("final_primary_label") or row.get("primary_label") or "unclear")
    if primary not in PRIMARY_LABELS:
        primary = "unclear"
    confidence = str(annotation.get("final_confidence") or "Medium")
    if confidence not in HUMAN_CONFIDENCE_LEVELS:
        confidence = "Medium"
    decision = str(annotation.get("final_decision") or "accept")
    if decision not in DECISIONS:
        decision = "fix"

    return (
        progress_text(rows, index, annotations or {}),
        query_panel(row),
        target_papers_panel(row),
        call_panel("LLM Call 1", row, "call1"),
        call_panel("LLM Call 2", row, "call2"),
        consensus_panel(row),
        labels,
        primary,
        parse_bool(annotation.get("final_ambiguous")),
        confidence,
        decision,
        str(annotation.get("final_notes") or ""),
        copy_text(row, annotation),
        index + 1,
        index,
    )


def load_dataset(input_path_text: str, annotations_path_text: str) -> tuple[Any, ...]:
    input_path = resolve_path(input_path_text)
    annotations_path = resolve_path(annotations_path_text)
    if not input_path.exists():
        message = f"Input JSONL not found: {input_path}"
        return ([], 0, {}, message, *render([], 0, {}))

    rows = read_jsonl_rows(input_path)
    annotations = read_annotations(annotations_path)
    rendered = render(rows, 0, annotations)
    return (rows, 0, annotations, f"Loaded {len(rows)} rows from {input_path}", *rendered)


def make_annotation_record(
    row: dict[str, Any],
    labels: list[str],
    primary_label: str,
    ambiguous: bool,
    final_confidence: str,
    decision: str,
    notes: str,
    annotator: str,
) -> dict[str, Any]:
    labels = [label for label in labels if label in VIEW_LABELS]
    if primary_label not in PRIMARY_LABELS:
        primary_label = labels[0] if labels else "unclear"
    if primary_label not in labels and primary_label != "unclear":
        labels = [primary_label] + labels
    if not labels and primary_label != "unclear":
        labels = [primary_label]
    final_confidence = final_confidence if final_confidence in HUMAN_CONFIDENCE_LEVELS else "Medium"
    decision = decision if decision in DECISIONS else "fix"
    return {
        "query_id": row["query_id"],
        "query_type": row.get("query_type", ""),
        "source_file": row.get("source_file", ""),
        "row_index": row.get("row_index", ""),
        "query": row.get("query", ""),
        "call1_labels": row.get("call1_labels", ""),
        "call1_primary_label": row.get("call1_primary_label", ""),
        "call1_ambiguous": row.get("call1_ambiguous", ""),
        "call1_confidence": row.get("call1_confidence", ""),
        "call1_rationale": row.get("call1_rationale", ""),
        "call2_labels": row.get("call2_labels", ""),
        "call2_primary_label": row.get("call2_primary_label", ""),
        "call2_ambiguous": row.get("call2_ambiguous", ""),
        "call2_confidence": row.get("call2_confidence", ""),
        "call2_rationale": row.get("call2_rationale", ""),
        "agreement_status": row.get("agreement_status", ""),
        "bucket": row.get("bucket", ""),
        "consensus_labels": row.get("labels", ""),
        "consensus_primary_label": row.get("primary_label", ""),
        "consensus_ambiguous": row.get("ambiguous", ""),
        "consensus_confidence": row.get("confidence", ""),
        "consensus_rationale": row.get("rationale", ""),
        "final_labels": "|".join(labels),
        "final_motivation": "1" if "motivation" in labels else "",
        "final_method": "1" if "method" in labels else "",
        "final_experiment": "1" if "experiment" in labels else "",
        "final_primary_label": primary_label,
        "final_ambiguous": "1" if ambiguous else "",
        "final_confidence": final_confidence,
        "final_decision": decision,
        "final_notes": str(notes or "").strip(),
        "annotator": str(annotator or "").strip(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "target_papers": target_papers_json(row.get("target_papers")),
    }


def save_current(
    rows: list[dict[str, Any]],
    index: int,
    annotations: dict[str, dict[str, Any]],
    annotations_path_text: str,
    labels: list[str],
    primary_label: str,
    ambiguous: bool,
    final_confidence: str,
    decision: str,
    notes: str,
    annotator: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    if not rows:
        return annotations or {}, "No rows loaded."
    index = max(0, min(int(index or 0), len(rows) - 1))
    row = rows[index]
    annotations = dict(annotations or {})
    previous_done = len(annotations)
    was_already_saved = row["query_id"] in annotations
    annotations[row["query_id"]] = make_annotation_record(
        row=row,
        labels=labels or [],
        primary_label=primary_label,
        ambiguous=ambiguous,
        final_confidence=final_confidence,
        decision=decision,
        notes=notes,
        annotator=annotator,
    )
    annotations_path = resolve_path(annotations_path_text)
    atomic_write_annotations(annotations_path, annotations)
    done = len(annotations)
    status = f"Saved {row['query_id']} to {annotations_path}"
    if not was_already_saved and done > previous_done:
        bonus = milestone_bonus(done, len(rows))
        if bonus:
            status = f"{status}\n\n{bonus}"
    return annotations, status


def save_and_move(
    rows: list[dict[str, Any]],
    index: int,
    annotations: dict[str, dict[str, Any]],
    annotations_path_text: str,
    labels: list[str],
    primary_label: str,
    ambiguous: bool,
    final_confidence: str,
    decision: str,
    notes: str,
    annotator: str,
    step: int,
) -> tuple[Any, ...]:
    annotations, status = save_current(
        rows,
        index,
        annotations,
        annotations_path_text,
        labels,
        primary_label,
        ambiguous,
        final_confidence,
        decision,
        notes,
        annotator,
    )
    next_index = max(0, min(int(index or 0) + step, max(0, len(rows) - 1)))
    rendered = render(rows, next_index, annotations)
    return (annotations, next_index, status, *rendered[:-1])


def move_without_saving(
    rows: list[dict[str, Any]],
    index: int,
    annotations: dict[str, dict[str, Any]],
    step: int,
) -> tuple[Any, ...]:
    next_index = max(0, min(int(index or 0) + step, max(0, len(rows) - 1)))
    rendered = render(rows, next_index, annotations or {})
    return (next_index, *rendered[:-1])


def jump_to(
    rows: list[dict[str, Any]],
    jump_row: int,
    annotations: dict[str, dict[str, Any]],
) -> tuple[Any, ...]:
    index = max(0, min(int(jump_row or 1) - 1, max(0, len(rows) - 1)))
    rendered = render(rows, index, annotations or {})
    return (index, *rendered[:-1])


def find_next_pending(
    rows: list[dict[str, Any]],
    index: int,
    annotations: dict[str, dict[str, Any]],
) -> tuple[Any, ...]:
    annotations = annotations or {}
    if not rows:
        return (0, *render(rows, 0, annotations))
    total = len(rows)
    start = (int(index or 0) + 1) % total
    for offset in range(total):
        candidate = (start + offset) % total
        if rows[candidate]["query_id"] not in annotations:
            rendered = render(rows, candidate, annotations)
            return (candidate, *rendered[:-1])
    rendered = render(rows, int(index or 0), annotations)
    return (int(index or 0), *rendered[:-1])


def export_jsonl(annotations: dict[str, dict[str, Any]], annotations_path_text: str) -> str:
    annotations_path = resolve_path(annotations_path_text)
    jsonl_path = annotations_path.with_suffix(".jsonl")
    rows = sorted((annotations or {}).values(), key=lambda row: str(row.get("query_id", "")))
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return f"Exported {len(rows)} annotations to {jsonl_path}"


CSS = """
.query-card {
  border: 1px solid #d8dee8;
  border-radius: 8px;
  padding: 14px 16px;
  background: #ffffff;
}
.query-meta {
  color: #475569;
  font-weight: 700;
  margin-bottom: 10px;
}
.query-text {
  white-space: pre-wrap;
  line-height: 1.55;
  font-size: 15px;
  max-height: 280px;
  overflow-y: auto;
}
.query-title {
  font-weight: 800;
  font-size: 16px;
  margin-bottom: 10px;
}
.query-body {
  white-space: pre-wrap;
}
.math-inline {
  display: inline-block;
  padding: 0 4px;
  border-radius: 4px;
  background: #eef6ff;
  color: #0f4c81;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.source-path {
  margin-top: 12px;
  color: #64748b;
  font-size: 12px;
  overflow-wrap: anywhere;
}
.llm-panel, .consensus-panel {
  border: 1px solid #d8dee8;
  border-radius: 8px;
  padding: 12px 14px;
  background: #f8fafc;
  line-height: 1.5;
  height: 100%;
}
.consensus-panel {
  background: #f7fbf6;
}
.target-panel {
  border: 1px solid #d8dee8;
  border-radius: 8px;
  padding: 12px 14px;
  background: #fffdf7;
  line-height: 1.45;
  max-height: 360px;
  overflow-y: auto;
}
.target-paper {
  border-top: 1px solid #e2e8f0;
  padding-top: 10px;
  margin-top: 10px;
}
.target-paper:first-of-type {
  border-top: 0;
  padding-top: 0;
  margin-top: 0;
}
.target-paper-title {
  font-weight: 700;
}
.target-paper-meta, .target-paper-url {
  color: #64748b;
  font-size: 12px;
  overflow-wrap: anywhere;
  margin-top: 4px;
}
.target-paper-citations {
  color: #b91c1c;
  font-weight: 800;
  font-size: 12px;
  overflow-wrap: anywhere;
  margin-top: 4px;
}
.target-paper-context {
  white-space: pre-wrap;
  margin-top: 8px;
  max-height: 180px;
  overflow-y: auto;
}
.empty-target {
  color: #64748b;
}
.milestone-bonus {
  margin-top: 8px;
  border: 1px solid #9ed3b1;
  border-radius: 8px;
  padding: 10px 12px;
  background: #f0fbf3;
  color: #14532d;
}
.panel-title {
  font-weight: 800;
  margin-bottom: 8px;
}
"""


def build_app(
    input_jsonl: Path = DEFAULT_INPUT_JSONL,
    annotations_csv: Path = DEFAULT_ANNOTATIONS_CSV,
) -> gr.Blocks:
    with gr.Blocks(title="Golden Query Consensus Review", css=CSS) as app:
        rows_state = gr.State([])
        index_state = gr.State(0)
        annotations_state = gr.State({})

        gr.Markdown("# Golden Query Consensus Review")
        gr.Markdown(
            "Review the two LLM calls and consensus label, then save the final human annotation. "
            "The copy box contains the full row context for quick notes or external records."
        )

        with gr.Row():
            input_path = gr.Textbox(label="Input consensus JSONL", value=str(input_jsonl), scale=3)
            annotations_path = gr.Textbox(label="Save final annotations CSV", value=str(annotations_csv), scale=3)
        with gr.Row():
            annotator = gr.Textbox(label="Annotator", placeholder="your name or initials", scale=2)
            load_button = gr.Button("Load", variant="primary", scale=1)
            export_button = gr.Button("Export JSONL", scale=1)

        status = gr.Markdown("")
        progress = gr.Markdown("No rows loaded.")
        query_html = gr.HTML("")
        target_papers_html = gr.HTML("")
        with gr.Row():
            call1_html = gr.HTML("")
            call2_html = gr.HTML("")
            consensus_html = gr.HTML("")

        with gr.Row():
            final_labels = gr.CheckboxGroup(
                choices=VIEW_LABELS,
                label="Final labels",
                info="Multi-label: select every view that applies.",
                scale=2,
            )
            with gr.Column(scale=1):
                final_primary = gr.Radio(choices=PRIMARY_LABELS, label="Final primary label", value="unclear")
                final_ambiguous = gr.Checkbox(label="Ambiguous")
                final_confidence = gr.Radio(choices=HUMAN_CONFIDENCE_LEVELS, label="Final confidence", value="Medium")
                final_decision = gr.Radio(choices=DECISIONS, label="Decision", value="accept")

        final_notes = gr.Textbox(label="Reviewer note", lines=3)

        with gr.Row():
            previous_button = gr.Button("Previous")
            save_button = gr.Button("Save")
            save_next_button = gr.Button("Save & Next", variant="primary")
            next_button = gr.Button("Next")
            pending_button = gr.Button("Next Pending")
        row_copy = gr.Textbox(
            label="Copy whole row content",
            lines=18,
            interactive=True,
            show_copy_button=True,
        )
        with gr.Row():
            jump_row = gr.Number(label="Jump to row", value=1, precision=0)
            jump_button = gr.Button("Go")

        render_outputs = [
            progress,
            query_html,
            target_papers_html,
            call1_html,
            call2_html,
            consensus_html,
            final_labels,
            final_primary,
            final_ambiguous,
            final_confidence,
            final_decision,
            final_notes,
            row_copy,
            jump_row,
            index_state,
        ]

        load_button.click(
            load_dataset,
            inputs=[input_path, annotations_path],
            outputs=[rows_state, index_state, annotations_state, status, *render_outputs],
        )
        save_button.click(
            save_current,
            inputs=[
                rows_state,
                index_state,
                annotations_state,
                annotations_path,
                final_labels,
                final_primary,
                final_ambiguous,
                final_confidence,
                final_decision,
                final_notes,
                annotator,
            ],
            outputs=[annotations_state, status],
        ).then(
            render,
            inputs=[rows_state, index_state, annotations_state],
            outputs=render_outputs,
        )
        save_next_button.click(
            save_and_move,
            inputs=[
                rows_state,
                index_state,
                annotations_state,
                annotations_path,
                final_labels,
                final_primary,
                final_ambiguous,
                final_confidence,
                final_decision,
                final_notes,
                annotator,
                gr.State(1),
            ],
            outputs=[annotations_state, index_state, status, *render_outputs[:-1]],
        )
        previous_button.click(
            move_without_saving,
            inputs=[rows_state, index_state, annotations_state, gr.State(-1)],
            outputs=[index_state, *render_outputs[:-1]],
        )
        next_button.click(
            move_without_saving,
            inputs=[rows_state, index_state, annotations_state, gr.State(1)],
            outputs=[index_state, *render_outputs[:-1]],
        )
        jump_button.click(
            jump_to,
            inputs=[rows_state, jump_row, annotations_state],
            outputs=[index_state, *render_outputs[:-1]],
        )
        pending_button.click(
            find_next_pending,
            inputs=[rows_state, index_state, annotations_state],
            outputs=[index_state, *render_outputs[:-1]],
        )
        export_button.click(
            export_jsonl,
            inputs=[annotations_state, annotations_path],
            outputs=[status],
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a Gradio UI for consensus golden-query review.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=7891, help="Server port.")
    parser.add_argument(
        "--input-jsonl",
        default=str(DEFAULT_INPUT_JSONL),
        help="Consensus JSONL to review.",
    )
    parser.add_argument(
        "--annotations-csv",
        default=str(DEFAULT_ANNOTATIONS_CSV),
        help="CSV path where final human annotations are saved.",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share URL.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    app = build_app(
        input_jsonl=resolve_path(args.input_jsonl),
        annotations_csv=resolve_path(args.annotations_csv),
    )
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
