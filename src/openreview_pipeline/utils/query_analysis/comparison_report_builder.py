from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .analysis_single_builder import (
    SummaryMetrics,
    compute_distribution_based_human_closeness,
    extract_per_query_frame,
    extract_summary_metrics,
)


@dataclass
class DatasetArtifact:
    label: str
    path: Path
    data: Dict[str, Any]
    summary: SummaryMetrics
    per_query_frame: Optional[pd.DataFrame]
    queries: List[str]


def load_analysis_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _extract_example_queries(data: Dict[str, Any], limit: int = 5) -> List[str]:
    query_examples = data.get("query_examples", [])
    if isinstance(query_examples, list):
        cleaned = [str(item).strip() for item in query_examples if str(item).strip()]
        if cleaned:
            return cleaned[:limit]

    examples = []
    for item in data.get("representative_examples", []):
        if isinstance(item, dict):
            query = str(item.get("query", "")).strip()
            if query:
                examples.append(query)
    return examples[:limit]


def _try_extract_per_query_frame(data: Dict[str, Any]) -> Optional[pd.DataFrame]:
    try:
        return extract_per_query_frame(data)
    except Exception:
        return None


def load_dataset_artifact(label: str, path: Path) -> DatasetArtifact:
    data = load_analysis_json(path)
    return DatasetArtifact(
        label=label,
        path=path,
        data=data,
        summary=extract_summary_metrics(data),
        per_query_frame=_try_extract_per_query_frame(data),
        queries=_extract_example_queries(data),
    )


def merge_summary_metrics(label: str, artifacts: Iterable[DatasetArtifact]) -> DatasetArtifact:
    artifact_list = list(artifacts)
    total_queries = sum(item.summary.query_count for item in artifact_list)
    if total_queries <= 0:
        raise ValueError("Cannot merge summaries with zero total queries.")

    def weighted(getter: str) -> float:
        return sum(getattr(item.summary, getter) * item.summary.query_count for item in artifact_list) / total_queries

    summary = SummaryMetrics(
        query_count=total_queries,
        avg_chars=weighted("avg_chars"),
        avg_tokens=weighted("avg_tokens"),
        constraints_per_query=weighted("constraints_per_query"),
        specificity_score=weighted("specificity_score"),
        specificity_fit=weighted("specificity_fit"),
        lexical_score=weighted("lexical_score"),
        lexical_fit=weighted("lexical_fit"),
        unmatched_ratio=weighted("unmatched_ratio"),
    )

    merged_frame_parts = [item.per_query_frame for item in artifact_list if item.per_query_frame is not None]
    merged_frame = pd.concat(merged_frame_parts, ignore_index=True) if merged_frame_parts else None

    merged_queries: List[str] = []
    for item in artifact_list:
        merged_queries.extend(item.queries)

    synthetic_data = {
        "dataset": label,
        "metrics": {"combined": {}},
    }
    return DatasetArtifact(
        label=label,
        path=Path("<combined>"),
        data=synthetic_data,
        summary=summary,
        per_query_frame=merged_frame,
        queries=merged_queries[:5],
    )


def _format_float(value: Optional[float], digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    return f"{value:.{digits}f}"


def _format_percent(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    return f"{value * 100:.1f}%"


def _summary_table(artifacts: List[DatasetArtifact]) -> List[str]:
    lines = [
        "| Metric | " + " | ".join(item.label for item in artifacts) + " |",
        "| --- | " + " | ".join(["---:"] * len(artifacts)) + " |",
    ]
    rows = [
        ("Query count", lambda item: str(item.summary.query_count)),
        ("Avg chars", lambda item: _format_float(item.summary.avg_chars, 1)),
        ("Avg tokens", lambda item: _format_float(item.summary.avg_tokens, 1)),
        ("Constraints/query", lambda item: _format_float(item.summary.constraints_per_query)),
        ("Specificity", lambda item: _format_float(item.summary.specificity_score)),
        ("Specificity fit", lambda item: _format_float(item.summary.specificity_fit)),
        ("Naturalness", lambda item: _format_float(item.summary.lexical_score)),
        ("Naturalness fit", lambda item: _format_float(item.summary.lexical_fit)),
        ("Unmatched templates", lambda item: _format_percent(item.summary.unmatched_ratio)),
    ]
    for label, getter in rows:
        lines.append("| " + label + " | " + " | ".join(getter(item) for item in artifacts) + " |")
    return lines


def _hc_rows(human: DatasetArtifact, artifacts: List[DatasetArtifact]) -> List[str]:
    metrics = [
        "W_constraints",
        "HC_constraints",
        "W_specificity",
        "HC_specificity",
        "W_naturalness",
        "HC_naturalness",
        "HC_overall",
    ]
    computed: Dict[str, Dict[str, float]] = {}
    for item in artifacts:
        if item.label == human.label:
            continue
        if human.per_query_frame is None or item.per_query_frame is None:
            continue
        computed[item.label] = compute_distribution_based_human_closeness(
            human.per_query_frame,
            item.per_query_frame,
        )

    lines = [
        "| Metric | " + " | ".join(item.label for item in artifacts) + " |",
        "| --- | " + " | ".join(["---:"] * len(artifacts)) + " |",
    ]
    for metric in metrics:
        row = []
        for item in artifacts:
            if item.label == human.label:
                row.append("-")
                continue
            values = computed.get(item.label)
            row.append(_format_float(values.get(metric)) if values else "N/A")
        lines.append("| " + metric + " | " + " | ".join(row) + " |")
    return lines


def _examples_table(artifacts: List[DatasetArtifact], rows: int = 5) -> List[str]:
    lines = [
        "| Example | " + " | ".join(item.label for item in artifacts) + " |",
        "| --- | " + " | ".join(["---"] * len(artifacts)) + " |",
    ]
    for idx in range(rows):
        values = []
        for item in artifacts:
            values.append(item.queries[idx] if idx < len(item.queries) else "")
        escaped = [value.replace("\n", " ").replace("|", "\\|") for value in values]
        lines.append("| " + str(idx + 1) + " | " + " | ".join(escaped) + " |")
    return lines


def render_comparison_markdown(
    *,
    title: str,
    artifacts: List[DatasetArtifact],
    human_label: str,
) -> str:
    human_artifact = next(item for item in artifacts if item.label == human_label)
    lines = [
        f"# {title}",
        "",
        "## Inputs",
        "",
    ]
    for item in artifacts:
        lines.append(f"- {item.label}: `{item.path}`")

    lines.extend(["", "## Summary Metrics", ""])
    lines.extend(_summary_table(artifacts))

    lines.extend(["", "## Human-Closeness", ""])
    lines.append("- Uses the human reference distribution as the gold standard.")
    lines.append("- `HC_overall` is the mean of `HC_constraints`, `HC_specificity`, and `HC_naturalness`.")
    lines.extend([""])
    lines.extend(_hc_rows(human_artifact, artifacts))

    lines.extend(["", "## Query Examples", ""])
    lines.append("- Full query text is shown below.")
    lines.extend([""])
    lines.extend(_examples_table(artifacts))
    lines.append("")
    return "\n".join(lines)


def build_default_report_artifacts(new_analysis_path: Path, reference_root: Path) -> tuple[List[DatasetArtifact], List[DatasetArtifact]]:
    human = load_dataset_artifact("Human reference", reference_root / "03_litsearch_human" / "style_analysis.json")
    original = load_dataset_artifact("Original synthetic", reference_root / "01_original_synthetic" / "style_analysis.json")
    litsearch = load_dataset_artifact("LitSearch synthetic", reference_root / "05_litsearch_inline_200" / "style_analysis.json")
    pasa = load_dataset_artifact("PASA synthetic", reference_root / "04_pasa_autoscholar_100" / "style_analysis.json")
    new_synthetic = load_dataset_artifact("New synthetic", new_analysis_path)
    combined = merge_summary_metrics("Combined synthetic", [litsearch, pasa])
    report1 = [human, original, new_synthetic]
    report2 = [human, litsearch, pasa, combined, new_synthetic]
    return report1, report2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build canonical query-analysis comparison reports.")
    parser.add_argument("--new-analysis", required=True, help="Path to the new synthetic style_analysis.json.")
    parser.add_argument(
        "--reference-root",
        default="outputs/query_analysis_comparison",
        help="Directory containing canonical comparison reference analyses.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write markdown reports. Defaults to the new-analysis parent directory.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    new_analysis_path = Path(args.new_analysis).expanduser().resolve()
    reference_root = Path(args.reference_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else new_analysis_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report1_artifacts, report2_artifacts = build_default_report_artifacts(
        new_analysis_path=new_analysis_path,
        reference_root=reference_root,
    )

    report1 = render_comparison_markdown(
        title="Comparison 1: Human vs Original Synthetic vs New Synthetic",
        artifacts=report1_artifacts,
        human_label="Human reference",
    )
    report2 = render_comparison_markdown(
        title="Comparison 2: Human vs Prior Synthetic Baselines vs New Synthetic",
        artifacts=report2_artifacts,
        human_label="Human reference",
    )

    report1_path = output_dir / "comparison_1_human_original_new_synthetic.md"
    report2_path = output_dir / "comparison_2_human_prior_synthetic_new_synthetic.md"
    report1_path.write_text(report1, encoding="utf-8")
    report2_path.write_text(report2, encoding="utf-8")

    print(f"Saved report 1 to {report1_path}")
    print(f"Saved report 2 to {report2_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
