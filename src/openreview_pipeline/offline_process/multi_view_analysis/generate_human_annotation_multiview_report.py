from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_IR_CSV = (
    REPO_ROOT
    / "outputs/query_analysis/golden_view_classification_with_targets/final_human_consensus_annotations.csv"
)
DEFAULT_QA_CSV = (
    REPO_ROOT
    / "outputs/query_analysis/golden_view_classification_with_targets_qa_filtered/final_human_consensus_annotations.csv"
)
DEFAULT_OUTPUT_MD = REPO_ROOT / "outputs/query_analysis/human_annotation_ir_qa_multiview_report.md"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "outputs/query_analysis/human_annotation_ir_qa_multiview_summary.json"

VIEW_LABELS = ("motivation", "method", "experiment")
PRIMARY_LABELS = VIEW_LABELS + ("unclear",)
USABLE_DECISIONS = {"accept", "fix"}


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_whitespace(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_label(value: Any) -> str:
    label = normalize_whitespace(value).lower()
    if label in {"experiments", "result", "results", "experiment/result"}:
        return "experiment"
    return label


def split_labels(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_labels = value
    else:
        text = str(value or "").strip()
        if not text:
            raw_labels = []
        else:
            try:
                parsed = json.loads(text)
                raw_labels = parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                raw_labels = re.split(r"[|,;]", text)

    normalized = {normalize_label(label) for label in raw_labels}
    return [label for label in VIEW_LABELS if label in normalized]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return normalize_whitespace(value).lower() in {"true", "1", "yes", "y", "checked"}


def labels_from_fields(row: dict[str, Any], prefix: str) -> list[str]:
    labels = split_labels(row.get(f"{prefix}_labels"))
    if labels:
        return labels

    field_labels = [
        label
        for label in VIEW_LABELS
        if parse_bool(row.get(f"{prefix}_{label}"))
    ]
    if field_labels:
        return field_labels

    primary = normalize_label(row.get(f"{prefix}_primary_label"))
    if primary in VIEW_LABELS:
        return [primary]
    return []


def label_set_text(labels: Iterable[str], *, sep: str = " + ") -> str:
    labels = [label for label in VIEW_LABELS if label in set(labels)]
    return sep.join(labels) if labels else "unlabeled"


def normalize_primary(value: Any) -> str:
    primary = normalize_label(value)
    return primary if primary in PRIMARY_LABELS else "unclear"


def parse_target_papers(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value or "").strip()
        if not text:
            return []
        try:
            raw_items = json.loads(text)
        except json.JSONDecodeError:
            raw_items = [{"title": text}]

    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        return []

    papers: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict):
            papers.append(item)
        elif item:
            papers.append({"title": str(item)})
    return papers


def source_stem(row: dict[str, Any]) -> str:
    source_file = normalize_whitespace(row.get("source_file"))
    if not source_file:
        return "unknown"
    stem = Path(source_file).stem
    return stem.replace("_with_targets", "")


def pct(part: int | float, whole: int | float) -> str:
    if not whole:
        return "0.0%"
    return f"{(part / whole) * 100:.1f}%"


def count_pct(part: int, whole: int) -> str:
    return f"{part} ({pct(part, whole)})"


def escape_cell(value: Any) -> str:
    text = normalize_whitespace(value)
    return text.replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(escape_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def trunc(value: Any, max_chars: int = 115) -> str:
    text = normalize_whitespace(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def load_records(path: Path, dataset_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in read_csv_rows(path):
        final_labels = labels_from_fields(row, "final")
        consensus_labels = labels_from_fields(row, "consensus")
        target_papers = parse_target_papers(row.get("target_papers"))
        final_primary = normalize_primary(row.get("final_primary_label"))
        consensus_primary = normalize_primary(row.get("consensus_primary_label"))
        record = {
            **row,
            "dataset": dataset_name,
            "source_csv": str(path),
            "query_type": normalize_whitespace(row.get("query_type")) or dataset_name,
            "source_stem": source_stem(row),
            "final_labels_list": final_labels,
            "final_label_set": label_set_text(final_labels),
            "consensus_labels_list": consensus_labels,
            "consensus_label_set": label_set_text(consensus_labels),
            "final_primary_norm": final_primary,
            "consensus_primary_norm": consensus_primary,
            "final_decision_norm": normalize_whitespace(row.get("final_decision")).lower(),
            "final_confidence_norm": normalize_whitespace(row.get("final_confidence")).title(),
            "final_ambiguous_bool": parse_bool(row.get("final_ambiguous")),
            "label_set_changed": tuple(final_labels) != tuple(consensus_labels),
            "primary_changed": final_primary != consensus_primary,
            "target_papers_list": target_papers,
            "target_paper_count": len(target_papers),
            "target_abstract_count": sum(
                1
                for paper in target_papers
                if normalize_whitespace(paper.get("abstract") or paper.get("body"))
            ),
        }
        records.append(record)
    return records


def subset(records: list[dict[str, Any]], query_type: str | None) -> list[dict[str, Any]]:
    if query_type is None:
        return records
    return [record for record in records if record["query_type"] == query_type]


def counter(records: list[dict[str, Any]], field: str) -> Counter[str]:
    return Counter(str(record.get(field) or "") for record in records)


def label_membership_counter(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for label in record["final_labels_list"]:
            counts[label] += 1
    return counts


def cardinality_counter(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        size = len(record["final_labels_list"])
        if size == 0:
            counts["unlabeled"] += 1
        elif size == 1:
            counts["single-label"] += 1
        else:
            counts["multi-label"] += 1
    return counts


def target_context_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [record["target_paper_count"] for record in records]
    abstracts = sum(record["target_abstract_count"] for record in records)
    total_targets = sum(counts)
    return {
        "total_target_papers": total_targets,
        "single_target_rows": sum(1 for count in counts if count == 1),
        "multi_target_rows": sum(1 for count in counts if count > 1),
        "mean": mean(counts) if counts else 0.0,
        "median": median(counts) if counts else 0.0,
        "p75": percentile(counts, 0.75),
        "p95": percentile(counts, 0.95),
        "max": max(counts) if counts else 0,
        "abstracts": abstracts,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    query_types = sorted(counter(records, "query_type"))
    by_type = {query_type: subset(records, query_type) for query_type in query_types}

    summary: dict[str, Any] = {
        "total_rows": len(records),
        "query_types": {query_type: len(rows) for query_type, rows in by_type.items()},
        "primary": dict(counter(records, "final_primary_norm")),
        "label_sets": dict(counter(records, "final_label_set")),
        "decisions": dict(counter(records, "final_decision_norm")),
        "confidence": dict(counter(records, "final_confidence_norm")),
        "ambiguous": sum(1 for record in records if record["final_ambiguous_bool"]),
        "label_set_changed": sum(1 for record in records if record["label_set_changed"]),
        "primary_changed": sum(1 for record in records if record["primary_changed"]),
        "target_context": target_context_summary(records),
        "by_query_type": {},
    }

    for query_type, rows in by_type.items():
        summary["by_query_type"][query_type] = {
            "rows": len(rows),
            "sources": dict(counter(rows, "source_stem")),
            "primary": dict(counter(rows, "final_primary_norm")),
            "label_sets": dict(counter(rows, "final_label_set")),
            "membership": dict(label_membership_counter(rows)),
            "cardinality": dict(cardinality_counter(rows)),
            "decisions": dict(counter(rows, "final_decision_norm")),
            "confidence": dict(counter(rows, "final_confidence_norm")),
            "ambiguous": sum(1 for record in rows if record["final_ambiguous_bool"]),
            "label_set_changed": sum(1 for record in rows if record["label_set_changed"]),
            "primary_changed": sum(1 for record in rows if record["primary_changed"]),
            "target_context": target_context_summary(rows),
        }
    return summary


def fixed_label_set_order(label_set_counts: Counter[str]) -> list[str]:
    preferred = [
        "motivation",
        "method",
        "experiment",
        "motivation + method",
        "motivation + experiment",
        "method + experiment",
        "motivation + method + experiment",
        "unlabeled",
    ]
    extras = sorted(set(label_set_counts) - set(preferred))
    return [item for item in preferred if item in label_set_counts] + extras


def build_report(
    records: list[dict[str, Any]],
    input_paths: list[Path],
    output_json: Path,
) -> str:
    summary = summarize_records(records)
    total = len(records)
    query_types = sorted(summary["query_types"])
    usable_rows = [
        record
        for record in records
        if record["final_decision_norm"] in USABLE_DECISIONS
        and record["final_primary_norm"] != "unclear"
    ]
    multi_label = sum(1 for record in records if len(record["final_labels_list"]) > 1)
    label_set_changed = summary["label_set_changed"]
    primary_changed = summary["primary_changed"]
    update_values = sorted(
        normalize_whitespace(record.get("updated_at"))
        for record in records
        if normalize_whitespace(record.get("updated_at"))
    )

    lines: list[str] = []
    lines.append("# Human Annotation Analysis for IR/QA Queries into Multi-View Aspects")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now().replace(microsecond=0).isoformat()}`")
    for path in input_paths:
        lines.append(f"- Source CSV: `{path}`")
    lines.append(f"- Machine-readable summary: `{output_json}`")
    if update_values:
        lines.append(f"- Annotation update window: `{update_values[0]}` to `{update_values[-1]}`")
    lines.append(f"- Total parsed annotation rows: {total}")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"- The current human-final annotation pool has **{total} rows**: "
        + ", ".join(
            f"**{summary['query_types'][query_type]} {query_type}**"
            for query_type in query_types
        )
        + "."
    )
    lines.append(
        "- The dominant final primary view is **method** "
        f"({count_pct(summary['primary'].get('method', 0), total)} overall), "
        "but QA is much more experiment-oriented than IR."
    )
    if "IR" in summary["by_query_type"] and "QA" in summary["by_query_type"]:
        ir_rows = summary["by_query_type"]["IR"]["rows"]
        qa_rows = summary["by_query_type"]["QA"]["rows"]
        lines.append(
            "- Final primary **experiment** rises from "
            f"{count_pct(summary['by_query_type']['IR']['primary'].get('experiment', 0), ir_rows)} in IR "
            f"to {count_pct(summary['by_query_type']['QA']['primary'].get('experiment', 0), qa_rows)} in QA; "
            "final primary **motivation** falls sharply in QA."
        )
    lines.append(
        f"- Multi-view labels are material: **{count_pct(multi_label, total)}** of rows carry more than one final aspect. "
        "This supports treating the task as multi-label when possible rather than collapsing immediately to one class."
    )
    lines.append(
        f"- Human review changed the LLM consensus label set for **{count_pct(label_set_changed, total)}** of rows "
        f"and changed the primary label for **{count_pct(primary_changed, total)}**."
    )
    lines.append(
        f"- **{count_pct(len(usable_rows), total)}** are directly usable for clean single-primary modeling after filtering to "
        "`accept`/`fix` and excluding `unclear` primaries."
    )
    lines.append("")

    lines.append("## Dataset Inventory")
    lines.append("")
    inventory_rows: list[list[Any]] = []
    for query_type in query_types:
        rows = subset(records, query_type)
        q_total = len(rows)
        card = cardinality_counter(rows)
        inventory_rows.append(
            [
                query_type,
                q_total,
                pct(q_total, total),
                count_pct(
                    sum(1 for row in rows if row["final_decision_norm"] in USABLE_DECISIONS),
                    q_total,
                ),
                count_pct(counter(rows, "final_confidence_norm").get("High", 0), q_total),
                count_pct(sum(1 for row in rows if row["final_ambiguous_bool"]), q_total),
                count_pct(card.get("multi-label", 0), q_total),
                count_pct(sum(1 for row in rows if row["label_set_changed"]), q_total),
                count_pct(sum(1 for row in rows if row["primary_changed"]), q_total),
            ]
        )
    lines.append(
        md_table(
            [
                "query type",
                "rows",
                "% all",
                "accept/fix",
                "high confidence",
                "ambiguous",
                "multi-label",
                "label-set changed",
                "primary changed",
            ],
            inventory_rows,
        )
    )
    lines.append("")

    source_rows: list[list[Any]] = []
    for query_type in query_types:
        rows = subset(records, query_type)
        sources = counter(rows, "source_stem")
        for source, count in sources.most_common():
            source_rows.append([query_type, source, count, pct(count, len(rows))])
    lines.append("### Source Breakdown")
    lines.append("")
    lines.append(md_table(["query type", "source", "rows", "% within type"], source_rows))
    lines.append("")

    lines.append("## Final Human Multi-View Labels")
    lines.append("")
    primary_rows: list[list[Any]] = []
    for label in PRIMARY_LABELS:
        row = [label, count_pct(counter(records, "final_primary_norm").get(label, 0), total)]
        for query_type in query_types:
            rows = subset(records, query_type)
            row.append(count_pct(counter(rows, "final_primary_norm").get(label, 0), len(rows)))
        primary_rows.append(row)
    lines.append("### Primary View Distribution")
    lines.append("")
    lines.append(md_table(["primary view", "overall"] + query_types, primary_rows))
    lines.append("")

    membership_rows: list[list[Any]] = []
    for label in VIEW_LABELS:
        row = [label, count_pct(label_membership_counter(records).get(label, 0), total)]
        for query_type in query_types:
            rows = subset(records, query_type)
            row.append(count_pct(label_membership_counter(rows).get(label, 0), len(rows)))
        membership_rows.append(row)
    lines.append("### Multi-Label Membership")
    lines.append("")
    lines.append(md_table(["view label", "overall"] + query_types, membership_rows))
    lines.append("")

    all_label_sets = fixed_label_set_order(counter(records, "final_label_set"))
    label_set_rows: list[list[Any]] = []
    for label_set in all_label_sets:
        row = [label_set, count_pct(counter(records, "final_label_set").get(label_set, 0), total)]
        for query_type in query_types:
            rows = subset(records, query_type)
            row.append(count_pct(counter(rows, "final_label_set").get(label_set, 0), len(rows)))
        label_set_rows.append(row)
    lines.append("### Final Label Sets")
    lines.append("")
    lines.append(md_table(["label set", "overall"] + query_types, label_set_rows))
    lines.append("")

    cardinality_rows: list[list[Any]] = []
    for label in ("single-label", "multi-label", "unlabeled"):
        row = [label, count_pct(cardinality_counter(records).get(label, 0), total)]
        for query_type in query_types:
            rows = subset(records, query_type)
            row.append(count_pct(cardinality_counter(rows).get(label, 0), len(rows)))
        cardinality_rows.append(row)
    lines.append("### Label Cardinality")
    lines.append("")
    lines.append(md_table(["cardinality", "overall"] + query_types, cardinality_rows))
    lines.append("")

    lines.append("## Human Verification Quality")
    lines.append("")
    decision_values = sorted(counter(records, "final_decision_norm"))
    decision_rows: list[list[Any]] = []
    for decision in decision_values:
        row = [decision or "blank", count_pct(counter(records, "final_decision_norm").get(decision, 0), total)]
        for query_type in query_types:
            rows = subset(records, query_type)
            row.append(count_pct(counter(rows, "final_decision_norm").get(decision, 0), len(rows)))
        decision_rows.append(row)
    lines.append("### Final Decisions")
    lines.append("")
    lines.append(md_table(["decision", "overall"] + query_types, decision_rows))
    lines.append("")

    confidence_rows: list[list[Any]] = []
    confidence_order = ["High", "Medium", "Low", ""]
    for confidence in confidence_order:
        if confidence not in counter(records, "final_confidence_norm"):
            continue
        row = [confidence or "blank", count_pct(counter(records, "final_confidence_norm").get(confidence, 0), total)]
        for query_type in query_types:
            rows = subset(records, query_type)
            row.append(count_pct(counter(rows, "final_confidence_norm").get(confidence, 0), len(rows)))
        confidence_rows.append(row)
    lines.append("### Final Confidence")
    lines.append("")
    lines.append(md_table(["confidence", "overall"] + query_types, confidence_rows))
    lines.append("")

    flagged_rows = [
        record
        for record in records
        if record["final_decision_norm"] not in USABLE_DECISIONS
        or record["final_primary_norm"] == "unclear"
        or not record["final_labels_list"]
    ]
    lines.append("### Rows Requiring Explicit Handling")
    lines.append("")
    if flagged_rows:
        lines.append(
            md_table(
                ["query_id", "type", "decision", "label set", "primary", "confidence", "query"],
                [
                    [
                        record.get("query_id"),
                        record["query_type"],
                        record["final_decision_norm"],
                        record["final_label_set"],
                        record["final_primary_norm"],
                        record["final_confidence_norm"],
                        trunc(record.get("query")),
                    ]
                    for record in flagged_rows
                ],
            )
        )
    else:
        lines.append("No rows require explicit handling under the current `accept`/`fix` plus non-unclear-primary filter.")
    lines.append("")

    lines.append("## Consensus-to-Human Movement")
    lines.append("")
    movement_rows: list[list[Any]] = [
        [
            "overall",
            total,
            count_pct(label_set_changed, total),
            count_pct(primary_changed, total),
        ]
    ]
    for query_type in query_types:
        rows = subset(records, query_type)
        movement_rows.append(
            [
                query_type,
                len(rows),
                count_pct(sum(1 for row in rows if row["label_set_changed"]), len(rows)),
                count_pct(sum(1 for row in rows if row["primary_changed"]), len(rows)),
            ]
        )
    lines.append(md_table(["subset", "rows", "label-set changed", "primary changed"], movement_rows))
    lines.append("")

    bucket_rows: list[list[Any]] = []
    for bucket, count in counter(records, "bucket").most_common():
        rows = [record for record in records if str(record.get("bucket") or "") == bucket]
        bucket_rows.append(
            [
                bucket or "blank",
                count,
                count_pct(sum(1 for row in rows if row["label_set_changed"]), len(rows)),
                count_pct(sum(1 for row in rows if row["primary_changed"]), len(rows)),
            ]
        )
    lines.append("### Movement by LLM Triage Bucket")
    lines.append("")
    lines.append(md_table(["bucket", "rows", "label-set changed", "primary changed"], bucket_rows))
    lines.append("")

    transition_rows: list[list[Any]] = []
    for consensus_primary in PRIMARY_LABELS:
        row = [consensus_primary]
        for final_primary in PRIMARY_LABELS:
            row.append(
                sum(
                    1
                    for record in records
                    if record["consensus_primary_norm"] == consensus_primary
                    and record["final_primary_norm"] == final_primary
                )
            )
        transition_rows.append(row)
    lines.append("### Primary Label Transition Matrix")
    lines.append("")
    lines.append(md_table(["consensus primary"] + [f"final {label}" for label in PRIMARY_LABELS], transition_rows))
    lines.append("")

    transition_counter: Counter[tuple[str, str]] = Counter()
    for record in records:
        if record["label_set_changed"]:
            transition_counter[(record["consensus_label_set"], record["final_label_set"])] += 1
    lines.append("### Largest Label-Set Transitions")
    lines.append("")
    lines.append(
        md_table(
            ["consensus label set", "final label set", "rows"],
            [
                [source, target, count]
                for (source, target), count in transition_counter.most_common(12)
            ],
        )
    )
    lines.append("")

    lines.append("## Target-Paper Context")
    lines.append("")
    target_rows: list[list[Any]] = []
    for label, rows in [("overall", records)] + [(query_type, subset(records, query_type)) for query_type in query_types]:
        context = target_context_summary(rows)
        row_count = len(rows)
        target_rows.append(
            [
                label,
                row_count,
                context["total_target_papers"],
                count_pct(context["single_target_rows"], row_count),
                count_pct(context["multi_target_rows"], row_count),
                f"{context['mean']:.2f}",
                f"{context['median']:.1f}",
                f"{context['p75']:.1f}",
                f"{context['p95']:.1f}",
                context["max"],
                context["abstracts"],
            ]
        )
    lines.append(
        md_table(
            [
                "subset",
                "rows",
                "target entries",
                "single-target rows",
                "multi-target rows",
                "mean",
                "median",
                "p75",
                "p95",
                "max",
                "abstracts populated",
            ],
            target_rows,
        )
    )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- **IR queries** remain method-heavy, but the human corrections show that many retrieval needs also encode "
        "the reason a work matters. The largest movement is from `method` consensus into `motivation + method`, "
        "which means a single-view classifier would hide a meaningful motivation component in many literature-search queries."
    )
    lines.append(
        "- **QA queries** are concentrated in method and experiment views. Their low motivation share suggests that answer-seeking "
        "questions usually ask for mechanisms, assumptions, comparisons, empirical behavior, or testing procedures, rather than only "
        "for research-gap framing."
    )
    lines.append(
        "- **Experiment/result** is the main IR-vs-QA separator: QA has a much higher experiment-primary rate and higher experiment "
        "membership, so downstream evaluation should stratify by query type if it compares IR and QA examples."
    )
    lines.append(
        "- **Human verification is consequential** even for high-confidence LLM buckets. The movement tables show that consensus labels "
        "are useful for pre-annotation, but final human labels should be treated as the ground truth for training and evaluation."
    )
    lines.append("")

    lines.append("## Recommended Use")
    lines.append("")
    lines.append("- Use `final_motivation`, `final_method`, and `final_experiment` as authoritative multi-label targets.")
    lines.append("- Use `final_primary_label` only when a single aspect is required, after filtering out `unclear` primaries.")
    lines.append("- For clean evaluation, filter to `final_decision in {accept, fix}`; keep `unclear` and `skip` only for uncertainty analysis.")
    lines.append("- Keep `final_ambiguous` as a separate uncertainty signal rather than folding it into the label classes.")
    lines.append("- If a downstream component expects `experiment/result`, map the final `experiment` label to that spelling at export time.")
    lines.append("- Preserve `query_type` in splits and reports because IR and QA have visibly different aspect distributions.")
    lines.append("")

    return "\n".join(lines)


def write_json_summary(path: Path, records: list[dict[str, Any]]) -> None:
    summary = summarize_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a human annotation analysis report for IR/QA multi-view query labels."
    )
    parser.add_argument("--ir-csv", type=Path, default=DEFAULT_IR_CSV)
    parser.add_argument("--qa-csv", type=Path, default=DEFAULT_QA_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ir_csv = resolve_path(args.ir_csv)
    qa_csv = resolve_path(args.qa_csv)
    output_md = resolve_path(args.output_md)
    output_json = resolve_path(args.output_json)

    records = load_records(ir_csv, "IR") + load_records(qa_csv, "QA")
    write_json_summary(output_json, records)
    report = build_report(records, [ir_csv, qa_csv], output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
