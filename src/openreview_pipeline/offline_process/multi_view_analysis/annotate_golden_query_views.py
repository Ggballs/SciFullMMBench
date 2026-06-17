from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REVIEW_SAMPLE = REPO_ROOT / "outputs/query_analysis/golden_view_classification/golden_query_view_review_sample.csv"
DEFAULT_CLASSIFICATIONS = REPO_ROOT / "outputs/query_analysis/golden_view_classification/golden_query_view_classifications.csv"
DEFAULT_ANNOTATIONS = REPO_ROOT / "outputs/query_analysis/golden_view_classification/human_annotations.csv"
VIEW_LABELS = ["motivation", "method", "experiment"]
PRIMARY_LABELS = ["motivation", "method", "experiment", "unclear"]
HUMAN_CONFIDENCE_LEVELS = ["High", "Medium", "Low"]
DECISIONS = ["accept", "fix", "unclear", "skip"]
ANNOTATION_BLIND_MODE = False


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def split_labels(value: Any) -> list[str]:
    if isinstance(value, list):
        labels = value
    else:
        labels = str(value or "").replace(",", "|").replace(";", "|").split("|")
    normalized = []
    for label in labels:
        label = str(label).strip().lower()
        if label in VIEW_LABELS and label not in normalized:
            normalized.append(label)
    return normalized


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes", "y", "checked"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_rows(input_path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(input_path)
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        query_id = str(row.get("query_id") or f"row_{index:04d}").strip()
        llm_labels = row.get("llm_labels") or row.get("labels") or ""
        llm_primary = row.get("llm_primary_label") or row.get("primary_label") or ""
        llm_ambiguous = row.get("llm_ambiguous") or row.get("ambiguous") or ""
        llm_confidence = row.get("llm_confidence") or row.get("confidence") or ""
        llm_rationale = row.get("llm_rationale") or row.get("rationale") or ""
        normalized_rows.append(
            {
                **row,
                "query_id": query_id,
                "query_type": row.get("query_type", ""),
                "source_file": row.get("source_file", ""),
                "row_index": row.get("row_index", ""),
                "query": row.get("query", ""),
                "llm_labels": "|".join(split_labels(llm_labels)),
                "llm_primary_label": llm_primary if llm_primary in PRIMARY_LABELS else "unclear",
                "llm_ambiguous": parse_bool(llm_ambiguous),
                "llm_confidence": llm_confidence,
                "llm_rationale": llm_rationale,
            }
        )
    return normalized_rows


def read_annotations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    annotations: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(path):
        query_id = str(row.get("query_id", "")).strip()
        if query_id:
            annotations[query_id] = row
    return annotations


def atomic_write_annotations(path: Path, annotations: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query_type",
        "source_file",
        "row_index",
        "query",
        "llm_primary_label",
        "llm_labels",
        "llm_ambiguous",
        "llm_confidence",
        "llm_rationale",
        "human_labels",
        "human_motivation",
        "human_method",
        "human_experiment",
        "human_ambiguous",
        "human_primary_label",
        "human_confidence",
        "human_decision",
        "human_notes",
        "annotator",
        "updated_at",
    ]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    rows = sorted(annotations.values(), key=lambda row: str(row.get("query_id", "")))
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def annotation_for_row(row: dict[str, Any], annotations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    existing = annotations.get(row["query_id"])
    if existing:
        return existing
    if ANNOTATION_BLIND_MODE:
        return {
            "human_labels": "",
            "human_motivation": "",
            "human_method": "",
            "human_experiment": "",
            "human_ambiguous": "",
            "human_primary_label": "unclear",
            "human_confidence": "Medium",
            "human_decision": "fix",
            "human_notes": "",
            "annotator": "",
        }
    labels = split_labels(row.get("llm_labels"))
    return {
        "human_labels": "|".join(labels),
        "human_motivation": "1" if "motivation" in labels else "",
        "human_method": "1" if "method" in labels else "",
        "human_experiment": "1" if "experiment" in labels else "",
        "human_ambiguous": "1" if parse_bool(row.get("llm_ambiguous")) else "",
        "human_primary_label": row.get("llm_primary_label") or "unclear",
        "human_confidence": "High" if not parse_bool(row.get("llm_ambiguous")) else "Medium",
        "human_decision": "accept",
        "human_notes": "",
        "annotator": "",
    }


def llm_panel(row: dict[str, Any]) -> str:
    if ANNOTATION_BLIND_MODE:
        return ""
    labels = row.get("llm_labels") or "unclear"
    return (
        "<div class='llm-panel'>"
        f"<div><b>LLM labels:</b> {esc(labels)}</div>"
        f"<div><b>Primary:</b> {esc(row.get('llm_primary_label'))}</div>"
        f"<div><b>Ambiguous:</b> {esc(row.get('llm_ambiguous'))}</div>"
        f"<div><b>Confidence:</b> {esc(row.get('llm_confidence'))}</div>"
        f"<div><b>Rationale:</b> {esc(row.get('llm_rationale'))}</div>"
        "</div>"
    )


def query_panel(row: dict[str, Any]) -> str:
    meta = (
        f"{esc(row.get('query_id'))} · {esc(row.get('query_type'))} · "
        f"source row {esc(row.get('row_index'))}"
    )
    return (
        "<div class='query-card'>"
        f"<div class='query-meta'>{meta}</div>"
        f"<div class='query-text'>{esc(row.get('query'))}</div>"
        f"<div class='source-path'>{esc(row.get('source_file'))}</div>"
        "</div>"
    )


def progress_text(rows: list[dict[str, Any]], index: int, annotations: dict[str, dict[str, Any]]) -> str:
    total = len(rows)
    done = sum(1 for row in rows if row["query_id"] in annotations)
    if not total:
        return "No rows loaded."
    return f"Row {index + 1} of {total} · saved {done}/{total}"


def render(
    rows: list[dict[str, Any]],
    index: int,
    annotations: dict[str, dict[str, Any]],
) -> tuple[Any, ...]:
    if not rows:
        return (
            "No rows loaded. Set the input CSV path and click Load.",
            "",
            "",
            [],
            "unclear",
            False,
            "Medium",
            "skip",
            "",
            0,
            0,
        )

    index = max(0, min(int(index or 0), len(rows) - 1))
    row = rows[index]
    annotation = annotation_for_row(row, annotations)
    labels = split_labels(annotation.get("human_labels"))
    if not labels:
        labels = [
            label
            for label in VIEW_LABELS
            if parse_bool(annotation.get(f"human_{label}"))
        ]

    primary = str(annotation.get("human_primary_label") or row.get("llm_primary_label") or "unclear")
    if primary not in PRIMARY_LABELS:
        primary = "unclear"
    confidence = str(annotation.get("human_confidence") or "Medium")
    if confidence not in HUMAN_CONFIDENCE_LEVELS:
        confidence = "Medium"
    decision = str(annotation.get("human_decision") or "accept")
    if decision not in DECISIONS:
        decision = "fix"

    return (
        progress_text(rows, index, annotations),
        query_panel(row),
        llm_panel(row),
        labels,
        primary,
        parse_bool(annotation.get("human_ambiguous")),
        confidence,
        decision,
        str(annotation.get("human_notes") or ""),
        index + 1,
        index,
    )


def load_dataset(input_path_text: str, annotations_path_text: str) -> tuple[Any, ...]:
    input_path = resolve_path(input_path_text)
    annotations_path = resolve_path(annotations_path_text)
    if not input_path.exists():
        fallback = DEFAULT_CLASSIFICATIONS if DEFAULT_CLASSIFICATIONS.exists() else None
        if fallback is not None:
            input_path = fallback
        else:
            message = (
                f"Input CSV not found: {input_path}. Run classify_golden_query_views.py first, "
                "or point this app at an existing review/classification CSV."
            )
            return ([], 0, {}, message, "No rows loaded.", "", "", [], "unclear", False, "Medium", "skip", "", 0)

    rows = load_rows(input_path)
    annotations = read_annotations(annotations_path)
    rendered = render(rows, 0, annotations)
    return (rows, 0, annotations, f"Loaded {len(rows)} rows from {input_path}", *rendered[:-1])


def make_annotation_record(
    row: dict[str, Any],
    labels: list[str],
    primary_label: str,
    ambiguous: bool,
    human_confidence: str,
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
    human_confidence = human_confidence if human_confidence in HUMAN_CONFIDENCE_LEVELS else "Medium"
    decision = decision if decision in DECISIONS else "fix"
    return {
        "query_id": row["query_id"],
        "query_type": row.get("query_type", ""),
        "source_file": row.get("source_file", ""),
        "row_index": row.get("row_index", ""),
        "query": row.get("query", ""),
        "llm_primary_label": row.get("llm_primary_label", ""),
        "llm_labels": row.get("llm_labels", ""),
        "llm_ambiguous": row.get("llm_ambiguous", ""),
        "llm_confidence": row.get("llm_confidence", ""),
        "llm_rationale": row.get("llm_rationale", ""),
        "human_labels": "|".join(labels),
        "human_motivation": "1" if "motivation" in labels else "",
        "human_method": "1" if "method" in labels else "",
        "human_experiment": "1" if "experiment" in labels else "",
        "human_ambiguous": "1" if ambiguous else "",
        "human_primary_label": primary_label,
        "human_confidence": human_confidence,
        "human_decision": decision,
        "human_notes": str(notes or "").strip(),
        "annotator": str(annotator or "").strip(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_current(
    rows: list[dict[str, Any]],
    index: int,
    annotations: dict[str, dict[str, Any]],
    annotations_path_text: str,
    labels: list[str],
    primary_label: str,
    ambiguous: bool,
    human_confidence: str,
    decision: str,
    notes: str,
    annotator: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    if not rows:
        return annotations, "No rows loaded."
    index = max(0, min(int(index or 0), len(rows) - 1))
    row = rows[index]
    annotations = dict(annotations or {})
    annotations[row["query_id"]] = make_annotation_record(
        row=row,
        labels=labels or [],
        primary_label=primary_label,
        ambiguous=ambiguous,
        human_confidence=human_confidence,
        decision=decision,
        notes=notes,
        annotator=annotator,
    )
    annotations_path = resolve_path(annotations_path_text)
    atomic_write_annotations(annotations_path, annotations)
    return annotations, f"Saved {row['query_id']} to {annotations_path}"


def save_and_move(
    rows: list[dict[str, Any]],
    index: int,
    annotations: dict[str, dict[str, Any]],
    annotations_path_text: str,
    labels: list[str],
    primary_label: str,
    ambiguous: bool,
    human_confidence: str,
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
        human_confidence,
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
  border: 1px solid #e5e7eb;
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
  max-height: 360px;
  overflow-y: auto;
}
.source-path {
  margin-top: 12px;
  color: #64748b;
  font-size: 12px;
  overflow-wrap: anywhere;
}
.llm-panel {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 12px 14px;
  background: #f8fafc;
  line-height: 1.5;
}
"""


def build_app(
    input_csv: Path = DEFAULT_REVIEW_SAMPLE,
    annotations_csv: Path = DEFAULT_ANNOTATIONS,
    blind: bool = False,
) -> gr.Blocks:
    global ANNOTATION_BLIND_MODE
    ANNOTATION_BLIND_MODE = bool(blind)

    with gr.Blocks(title="Golden Query View Annotation", css=CSS) as app:
        rows_state = gr.State([])
        index_state = gr.State(0)
        annotations_state = gr.State({})

        gr.Markdown("# Golden Query View Annotation")
        if blind:
            gr.Markdown(
                "Blind-check query labels for `motivation`, `method`, and `experiment`. "
                "LLM labels are hidden and human fields start empty."
            )
        else:
            gr.Markdown(
                "Review LLM classifications for `motivation`, `method`, and `experiment`. "
                "The human fields are prefilled from the LLM prediction; change only what needs fixing."
            )

        with gr.Row():
            input_path = gr.Textbox(
                label="Input review/classification CSV",
                value=str(input_csv),
                scale=3,
            )
            annotations_path = gr.Textbox(
                label="Save annotations CSV",
                value=str(annotations_csv),
                scale=3,
            )
        with gr.Row():
            annotator = gr.Textbox(label="Annotator", placeholder="your name or initials", scale=2)
            load_button = gr.Button("Load", variant="primary", scale=1)
            export_button = gr.Button("Export JSONL", scale=1)

        status = gr.Markdown("")
        progress = gr.Markdown("No rows loaded.")
        query_html = gr.HTML("")
        llm_html = gr.HTML("")

        with gr.Row():
            human_labels = gr.CheckboxGroup(
                choices=VIEW_LABELS,
                label="Human labels",
                info="Multi-label: select every view that applies.",
                scale=2,
            )
            with gr.Column(scale=1):
                human_primary = gr.Radio(choices=PRIMARY_LABELS, label="Primary label", value="unclear")
                human_ambiguous = gr.Checkbox(label="Ambiguous")
                human_confidence = gr.Radio(
                    choices=HUMAN_CONFIDENCE_LEVELS,
                    label="Human confidence",
                    value="Medium",
                )
                human_decision = gr.Radio(choices=DECISIONS, label="Decision", value="accept")

        human_notes = gr.Textbox(label="Human notes", lines=3)

        with gr.Row():
            previous_button = gr.Button("Previous")
            save_button = gr.Button("Save")
            save_next_button = gr.Button("Save & Next", variant="primary")
            next_button = gr.Button("Next")
            pending_button = gr.Button("Next Pending")
        with gr.Row():
            jump_row = gr.Number(label="Jump to row", value=1, precision=0)
            jump_button = gr.Button("Go")

        render_outputs = [
            progress,
            query_html,
            llm_html,
            human_labels,
            human_primary,
            human_ambiguous,
            human_confidence,
            human_decision,
            human_notes,
            jump_row,
            index_state,
        ]

        load_button.click(
            load_dataset,
            inputs=[input_path, annotations_path],
            outputs=[rows_state, index_state, annotations_state, status, *render_outputs[:-1]],
        )
        save_button.click(
            save_current,
            inputs=[
                rows_state,
                index_state,
                annotations_state,
                annotations_path,
                human_labels,
                human_primary,
                human_ambiguous,
                human_confidence,
                human_decision,
                human_notes,
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
                human_labels,
                human_primary,
                human_ambiguous,
                human_confidence,
                human_decision,
                human_notes,
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
    parser = argparse.ArgumentParser(description="Launch a Gradio UI for golden-query view annotation.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=7862, help="Server port.")
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_REVIEW_SAMPLE),
        help="Initial review/classification CSV shown in the UI.",
    )
    parser.add_argument(
        "--annotations-csv",
        default=str(DEFAULT_ANNOTATIONS),
        help="CSV path where human annotations are saved.",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share URL.")
    parser.add_argument(
        "--blind",
        action="store_true",
        help="Hide LLM labels/rationales and start human labels empty.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    app = build_app(
        input_csv=resolve_path(args.input_csv),
        annotations_csv=resolve_path(args.annotations_csv),
        blind=args.blind,
    )
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
