from __future__ import annotations

import argparse
import hmac
import html
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_FINAL_JSON = Path("outputs/test_single/final_pipeline_output.json")
DEFAULT_HUMAN_LITSEARCH_JSON = Path("outputs/query_analysis_comparison/03_litsearch_human/style_analysis.json")
DEFAULT_HUMAN_PASA_JSON = Path("outputs/query_analysis_comparison/02_pasa_realscholar/style_analysis.json")
DEFAULT_HUMAN_JUDGMENTS_JSON = Path("outputs/human_judgments.json")
MAX_CANDIDATE_JUDGE_ROWS = 8
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openreview_pipeline.app_logging import configure_project_logging  # noqa: E402
from openreview_pipeline.utils.db.human_feedback_mysql import (  # noqa: E402
    load_query_feedback,
    save_query_feedback,
)

configure_project_logging()

ADMIN_FEEDBACK_PASSWORD = "westlakenlp"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON must be an object: {path}")
    return data


def resolve_input_path(raw_path: Optional[str]) -> Optional[Path]:
    if not raw_path:
        return None
    candidate = Path(raw_path).expanduser()
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append((Path.cwd() / candidate).resolve())
        candidates.append((REPO_ROOT / candidate).resolve())
        candidates.append((Path(__file__).resolve().parent / candidate).resolve())
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else candidate.resolve()


def parse_feedback_users(raw_users: Optional[str]) -> Dict[str, str]:
    users = {}
    for item in str(raw_users or "").split(","):
        if ":" not in item:
            continue
        username, password = item.split(":", 1)
        username = username.strip()
        if username:
            users[username] = password
    return users


def authenticate_user(username: str, password: str) -> bool:
    username = str(username or "").strip()
    password = str(password or "")
    if not username:
        return False
    if hmac.compare_digest(password, ADMIN_FEEDBACK_PASSWORD):
        return True
    configured_users = parse_feedback_users(os.getenv("HUMAN_FEEDBACK_USERS"))
    expected_password = configured_users.get(username)
    return expected_password is not None and hmac.compare_digest(password, expected_password)


def request_username(request: Optional[gr.Request]) -> Optional[str]:
    if request is None:
        return None
    username = getattr(request, "username", None)
    username = str(username or "").strip()
    return username or None


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def fmt_score(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return "N/A"


def fmt_metric(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def fmt_table_pct(value: Any) -> str:
    return fmt_pct(value, 2)


def safe_get(data: Optional[Dict[str, Any]], *keys: str, default=None):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def badge(text: str, bg: str = "#eef2ff", fg: str = "#3730a3") -> str:
    return (
        "<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{bg};color:{fg};font-size:12px;font-weight:700;margin-right:6px;margin-bottom:6px'>"
        f"{esc(text)}</span>"
    )


def plain_meta(text: str) -> str:
    return f"<span style='display:inline-block;font-size:13px;font-weight:600;color:#374151;margin-right:10px;margin-bottom:6px'>{esc(text)}</span>"


def help_meta(label: str, value: Any, explanation: str) -> str:
    return (
        "<span style='display:inline-block;font-size:13px;font-weight:600;color:#374151;margin-right:10px;margin-bottom:6px'>"
        f"{esc(label)}: {esc(value)} "
        f"<span title=\"{esc(explanation)}\" style='cursor:help;text-decoration:underline dotted;color:#6b7280'>(?)</span>"
        "</span>"
    )


def click_help_details(label: str, value: Any, explanation: str) -> str:
    return (
        "<details style='display:inline-block;margin-right:10px;margin-bottom:6px;vertical-align:top'>"
        "<summary style='cursor:pointer;list-style:none;font-size:13px;font-weight:600;color:#374151'>"
        f"{esc(label)}: {esc(value)} <span style='text-decoration:underline dotted;color:#6b7280'>(?)</span>"
        "</summary>"
        f"<div style='margin-top:6px;padding:8px 10px;border:1px solid #e5e7eb;border-radius:10px;background:#fafafa;"
        f"font-size:12px;line-height:1.5;color:#4b5563;max-width:360px'>{esc(explanation)}</div>"
        "</details>"
    )


def icon_help_details(explanation: str) -> str:
    return (
        "<details style='display:inline-block;margin:0 10px 6px 2px;vertical-align:top'>"
        "<summary style='cursor:pointer;list-style:none;font-size:13px;font-weight:700;color:#6b7280;text-decoration:underline dotted'>"
        "(?)"
        "</summary>"
        f"<div style='margin-top:6px;padding:8px 10px;border:1px solid #e5e7eb;border-radius:10px;background:#fafafa;"
        f"font-size:12px;line-height:1.5;color:#4b5563;max-width:360px'>{esc(explanation)}</div>"
        "</details>"
    )


def decision_badge(decision: str) -> str:
    normalized = (decision or "").lower()
    if normalized == "keep":
        return badge(decision, "#dcfce7", "#166534")
    if normalized == "hard reject":
        return badge(decision, "#fee2e2", "#991b1b")
    return badge(decision or "N/A", "#e5e7eb", "#111827")


def score_badge(label: str, value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return badge(f"{label}: N/A", "#e5e7eb", "#111827")

    if numeric <= 2:
        bg, fg = "#dbeafe", "#1d4ed8"
    elif numeric == 3:
        bg, fg = "#dcfce7", "#166534"
    else:
        bg, fg = "#fef3c7", "#92400e"
    if float(numeric).is_integer():
        numeric_text = str(int(numeric))
    else:
        numeric_text = f"{numeric:.2f}"
    return badge(f"{label}: {numeric_text}", bg, fg)


def retrieval_badge(label: str, value: Any) -> str:
    text = str(value or "N/A")
    normalized = text.upper()
    if normalized in {"PASS", "LOW"}:
        return badge(f"{label}: {text}", "#dcfce7", "#166534")
    if normalized in {"FAIL", "HIGH"}:
        return badge(f"{label}: {text}", "#fee2e2", "#991b1b")
    return badge(f"{label}: {text}", "#e5e7eb", "#111827")


def build_paper_choices(final_data: Dict[str, Any]) -> List[Tuple[str, str]]:
    choices = []
    for idx, paper in enumerate(final_data.get("papers", []), start=1):
        query_count = len(paper.get("queries", []))
        label = f"[{idx}] {paper.get('paper_title', 'Untitled')} | {paper.get('paper_id', 'unknown')} | q={query_count}"
        choices.append((label, paper.get("paper_id", f"paper_{idx}")))
    return choices


def build_paper_stats_df(final_data: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for paper in final_data.get("papers", []):
        openreview = paper.get("openreview", {}) or {}
        queries = paper.get("queries", []) or []
        keep = sum(1 for query in queries if safe_get(query, "query_analysis", "decision") == "Keep")
        reject = sum(1 for query in queries if safe_get(query, "query_analysis", "decision") == "Hard Reject")
        rows.append(
            {
                "paper_id": paper.get("paper_id", ""),
                "paper_title": paper.get("paper_title", ""),
                "venue": openreview.get("venue", ""),
                "year": openreview.get("year", ""),
                "queries": len(queries),
                "keep": keep,
                "hard_reject": reject,
            }
        )
    return pd.DataFrame(rows)


def get_paper_by_id(final_data: Dict[str, Any], paper_id: str) -> Optional[Dict[str, Any]]:
    for paper in final_data.get("papers", []):
        if paper.get("paper_id") == paper_id:
            return paper
    return None


def _human_score_count_estimate(summary: Dict[str, Any], total: int) -> List[int]:
    mean = summary.get("mean")
    if mean is None or total <= 0:
        return [0, 0, 0, 0, 0]
    target = float(mean) * int(total)
    weights = [0, 0, 0, 0, 0]
    center = min(5, max(1, int(round(float(mean)))))
    weights[center - 1] = int(total)

    current = sum((idx + 1) * count for idx, count in enumerate(weights))
    diff = target - current
    step = 1 if diff > 0 else -1
    while abs(diff) >= 1 and sum(weights) > 0:
        moved = False
        indices = range(5) if step > 0 else range(4, -1, -1)
        for idx in indices:
            next_idx = idx + step
            if next_idx < 0 or next_idx > 4 or weights[idx] <= 0:
                continue
            weights[idx] -= 1
            weights[next_idx] += 1
            diff -= step
            moved = True
            if abs(diff) < 1:
                break
        if not moved:
            break
    return weights


def load_human_bundle(litsearch_path: Optional[Path], pasa_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    datasets = []
    for name, path in [("litsearch_human", litsearch_path), ("pasa_human", pasa_path)]:
        if path and path.exists():
            datasets.append({"name": name, "path": str(path), "data": load_json(path)})
    if not datasets:
        return None
    return {"datasets": datasets}


def _weighted_mean(values: List[Tuple[float, int]]) -> Optional[float]:
    total_weight = sum(weight for _, weight in values if weight > 0)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in values if weight > 0) / total_weight


def combine_human_style_summary(human_bundle: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not human_bundle:
        return {}

    def _dataset_total(data: Dict[str, Any]) -> int:
        return int(safe_get(data, "dataset_overview", "combined", "total_queries", default=0) or 0)

    def _qual(data: Dict[str, Any]) -> Dict[str, Any]:
        return safe_get(data, "metrics", "combined", "qualitative_metrics", default={}) or {}

    sources = human_bundle.get("datasets", [])
    total_queries = sum(_dataset_total(item["data"]) for item in sources)
    specificity_means = []
    naturalism_means = []
    specificity_fit_means = []
    naturalism_fit_means = []
    specificity_counts = [0, 0, 0, 0, 0]
    naturalism_counts = [0, 0, 0, 0, 0]
    word_counts: List[int] = []
    constraint_counts: List[int] = []
    unmatched_template_count = 0

    for item in sources:
        data = item["data"]
        total = _dataset_total(data)
        qual = _qual(data)
        templates = safe_get(data, "metrics", "combined", "question_templates", default={}) or {}
        spec = qual.get("specificity_calibration", {}) or {}
        nat = qual.get("lexical_naturalism", {}) or {}
        spec_fit = qual.get("specificity_calibration_fit", {}) or qual.get("specificity", {}) or {}
        nat_fit = qual.get("lexical_naturalism_fit", {}) or qual.get("naturalness", {}) or {}
        semantic_items = (
            safe_get(data, "metrics", "combined", "semantic_constraint_analysis", "per_query", default=[])
            or []
        )
        if spec.get("mean") is not None:
            specificity_means.append((float(spec["mean"]), total))
        if nat.get("mean") is not None:
            naturalism_means.append((float(nat["mean"]), total))
        if spec_fit.get("mean") is not None:
            specificity_fit_means.append((float(spec_fit["mean"]), total))
        if nat_fit.get("mean") is not None:
            naturalism_fit_means.append((float(nat_fit["mean"]), total))
        est_spec = _human_score_count_estimate(spec, total)
        est_nat = _human_score_count_estimate(nat, total)
        specificity_counts = [a + b for a, b in zip(specificity_counts, est_spec)]
        naturalism_counts = [a + b for a, b in zip(naturalism_counts, est_nat)]
        for row in semantic_items:
            query_text = str(row.get("query", "")).strip()
            if query_text:
                word_counts.append(len(query_text.split()))
            try:
                constraint_counts.append(int(row.get("semantic_constraint_count")))
            except Exception:
                pass
        try:
            unmatched_template_count += int(templates.get("unmatched_count", 0) or 0)
        except Exception:
            pass

    return {
        "total_queries": total_queries,
        "source_names": [item["name"] for item in sources],
        "source_paths": [item["path"] for item in sources],
        "loaded": bool(sources),
        "specificity_mean": _weighted_mean(specificity_means),
        "naturalism_mean": _weighted_mean(naturalism_means),
        "specificity_fit_mean": _weighted_mean(specificity_fit_means),
        "naturalism_fit_mean": _weighted_mean(naturalism_fit_means),
        "specificity_score_counts": specificity_counts,
        "naturalism_score_counts": naturalism_counts,
        "word_counts": word_counts,
        "word_count_mean": float(sum(word_counts) / len(word_counts)) if word_counts else None,
        "constraint_counts": constraint_counts,
        "constraint_count_mean": float(sum(constraint_counts) / len(constraint_counts)) if constraint_counts else None,
        "unmatched_template_count": unmatched_template_count,
        "unmatched_template_share": (
            float(unmatched_template_count / total_queries) if total_queries else None
        ),
    }


def extract_generated_style_scores(final_data: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for paper in final_data.get("papers", []):
        for query in paper.get("queries", []):
            llm = safe_get(query, "query_analysis", "style_evaluation", "llm_based", default={}) or {}
            rule_based = safe_get(query, "query_analysis", "style_evaluation", "rule_based", default={}) or {}
            rows.append(
                {
                    "paper_id": paper.get("paper_id", ""),
                    "paper_title": paper.get("paper_title", ""),
                    "query_text": query.get("query_text", ""),
                    "word_count": len(str(query.get("query_text", "")).split()),
                    "specificity": llm.get("specificity_calibration_score"),
                    "naturalism": llm.get("lexical_naturalism_score"),
                    "constraint_count": llm.get("semantic_constraint_count"),
                    "matched_template": rule_based.get("matched_template"),
                }
            )
    return pd.DataFrame(rows)


def compute_generated_style_summary(generated_df: pd.DataFrame) -> Dict[str, Any]:
    summary = {"total_queries": len(generated_df)}
    for metric in ["specificity", "naturalism"]:
        values = [int(v) for v in generated_df[metric].dropna().tolist() if str(v).isdigit()]
        counts = [values.count(score) for score in [1, 2, 3, 4, 5]]
        mean = float(sum((idx + 1) * count for idx, count in enumerate(counts)) / len(values)) if values else None
        fit_values = [max(0.0, 1.0 - abs(v - 3) / 2.0) for v in values]
        fit_mean = float(sum(fit_values) / len(fit_values)) if fit_values else None
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_fit_mean"] = fit_mean
        summary[f"{metric}_score_counts"] = counts
    word_counts = [int(v) for v in generated_df.get("word_count", pd.Series(dtype=int)).dropna().tolist()]
    constraint_counts = []
    for value in generated_df.get("constraint_count", pd.Series(dtype=int)).dropna().tolist():
        try:
            constraint_counts.append(int(value))
        except Exception:
            pass
    summary["word_counts"] = word_counts
    summary["word_count_mean"] = float(sum(word_counts) / len(word_counts)) if word_counts else None
    summary["constraint_counts"] = constraint_counts
    summary["constraint_count_mean"] = (
        float(sum(constraint_counts) / len(constraint_counts)) if constraint_counts else None
    )
    matched_values = generated_df.get("matched_template", pd.Series(dtype=bool)).dropna().tolist()
    unmatched_count = sum(1 for value in matched_values if str(value).lower() == "false")
    summary["unmatched_template_count"] = unmatched_count
    summary["unmatched_template_share"] = (
        float(unmatched_count / len(matched_values)) if matched_values else None
    )
    return summary


def build_overview_html(
    final_data: Dict[str, Any],
    human_summary: Dict[str, Any],
    generated_summary: Dict[str, Any],
) -> str:
    dataset = final_data.get("dataset_overview", {}) or {}
    query_analysis_summary = (
        final_data.get("query_analysis_summary")
        or final_data.get("stage5_summary")
        or {}
    )
    decision_counts = query_analysis_summary.get("decision_counts", {}) or {}
    if not decision_counts:
        decision_counts = {"Keep": 0, "Hard Reject": 0}
        for paper in final_data.get("papers", []):
            for query in paper.get("queries", []) or []:
                decision = safe_get(query, "query_analysis", "decision", default="")
                if decision in decision_counts:
                    decision_counts[decision] += 1

    cards = [
        ("Papers", fmt_int(len(final_data.get("papers", [])))),
        ("Queries", fmt_int(dataset.get("stage4_total_queries") or dataset.get("stage3_total_queries") or 0)),
        ("Keep Queries", fmt_int(decision_counts.get("Keep", 0))),
        ("Hard Reject Queries", fmt_int(decision_counts.get("Hard Reject", 0))),
        ("Hard Negatives", fmt_int(dataset.get("stage5_total_hard_negatives") or dataset.get("stage4_total_hard_negatives") or 0)),
    ]

    cards_html = "".join(
        f"""
        <div style="padding:14px;border:1px solid #e5e7eb;border-radius:14px;background:#fafafa">
            <div style="font-size:12px;color:#6b7280;margin-bottom:6px">{esc(label)}</div>
            <div style="font-size:20px;font-weight:800;word-break:break-word">{esc(value)}</div>
        </div>
        """
        for label, value in cards
    )

    human_status = (
        badge("Human reference loaded", "#dcfce7", "#166534")
        if human_summary.get("loaded")
        else badge("Human reference missing", "#fee2e2", "#991b1b")
    )

    return f"""
    <div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:14px">
            {cards_html}
        </div>
        <div style="border:1px solid #e5e7eb;border-radius:14px;padding:14px;background:white;margin-bottom:14px">
            <div style="font-size:16px;font-weight:800;margin-bottom:8px">Comparison Overview</div>
            <div style="margin-bottom:8px">{human_status}</div>
            <div style="margin-bottom:8px"><b>Human reference in code:</b> combined LitSearch human + PASA human; JSON files unchanged.</div>
            <div style="margin-bottom:8px"><b>Human query count:</b> {fmt_int(human_summary.get("total_queries", 0))}</div>
            <div style="margin-bottom:8px"><b>Decision Counts:</b> {esc(query_analysis_summary.get("decision_counts", {}))}</div>
            <div style="margin-bottom:8px"><b>Retrieval Summary:</b> {esc(query_analysis_summary.get("retrieval_summary", {}))}</div>
        </div>
    </div>
    """


def build_overview_footer_html(final_data: Dict[str, Any], human_summary: Dict[str, Any]) -> str:
    human_sources_html = "".join(
        f"<li><b>{esc(name)}</b>: <code>{esc(path)}</code></li>"
        for name, path in zip(human_summary.get("source_names", []), human_summary.get("source_paths", []))
    )
    path_items = "".join(
        f"<li><b>{esc(name)}</b>: <code>{esc(path)}</code></li>"
        for name, path in (final_data.get("paths", {}) or {}).items()
    )
    return f"""
    <div>
        <div style="border:1px solid #e5e7eb;border-radius:14px;padding:14px;background:white;margin-bottom:14px">
            <div style="font-size:16px;font-weight:800;margin-bottom:8px">Human Reference Sources</div>
            <ul style="margin:0 0 0 18px;line-height:1.6">{human_sources_html or '<li>No human style files loaded.</li>'}</ul>
        </div>
        <div style="border:1px solid #e5e7eb;border-radius:14px;padding:14px;background:white">
            <div style="font-size:16px;font-weight:800;margin-bottom:8px">Artifact Paths</div>
            <ul style="margin:0 0 0 18px;line-height:1.6">{path_items}</ul>
        </div>
    </div>
    """


def build_metric_summary_df(generated_summary: Dict[str, Any], human_summary: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "human_reference_combined",
                "queries": human_summary.get("total_queries"),
                "word_count_mean": fmt_metric(human_summary.get("word_count_mean")),
                "semantic_constraint_count_mean": fmt_metric(human_summary.get("constraint_count_mean")),
                "specificity_mean": fmt_metric(human_summary.get("specificity_mean")),
                "naturalism_mean": fmt_metric(human_summary.get("naturalism_mean")),
                "template_unmatched_percent": fmt_table_pct(human_summary.get("unmatched_template_share")),
            },
            {
                "dataset": "generated",
                "queries": generated_summary.get("total_queries"),
                "word_count_mean": fmt_metric(generated_summary.get("word_count_mean")),
                "semantic_constraint_count_mean": fmt_metric(generated_summary.get("constraint_count_mean")),
                "specificity_mean": fmt_metric(generated_summary.get("specificity_mean")),
                "naturalism_mean": fmt_metric(generated_summary.get("naturalism_mean")),
                "template_unmatched_percent": fmt_table_pct(generated_summary.get("unmatched_template_share")),
            },
        ]
    )


def plot_style_distribution(
    generated_summary: Dict[str, Any],
    human_summary: Dict[str, Any],
    *,
    metric: str,
    title: str,
) -> plt.Figure:
    score_range = [1, 2, 3, 4, 5]
    generated_counts = generated_summary.get(f"{metric}_score_counts", [0, 0, 0, 0, 0])
    human_counts = human_summary.get(f"{metric}_score_counts", [0, 0, 0, 0, 0])
    generated_total = sum(generated_counts)
    human_total = sum(human_counts)
    generated_pct = [(count / generated_total * 100.0) if generated_total else 0.0 for count in generated_counts]
    human_pct = [(count / human_total * 100.0) if human_total else 0.0 for count in human_counts]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    width = 0.36
    xs = list(range(len(score_range)))
    ax.bar([x - width / 2 for x in xs], human_pct, width=width, color="#059669", label="Human Reference")
    ax.bar([x + width / 2 for x in xs], generated_pct, width=width, color="#2563eb", label="Generated")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in score_range])
    ax.set_xlabel("Judge Score")
    ax.set_ylabel("Query Percentage (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_word_count_distribution(
    generated_summary: Dict[str, Any],
    human_summary: Dict[str, Any],
) -> plt.Figure:
    generated_values = [int(v) for v in generated_summary.get("word_counts", []) if int(v) >= 0]
    human_values = [int(v) for v in human_summary.get("word_counts", []) if int(v) >= 0]
    all_values = generated_values + human_values
    if not all_values:
        all_values = [0]

    max_word_count = max(all_values)
    bin_edges = list(range(0, max(40, max_word_count) + 5, 5))
    if bin_edges[-1] <= max_word_count:
        bin_edges.append(bin_edges[-1] + 5)
    labels = [f"{start + 1}-{end}" if start else f"0-{end}" for start, end in zip(bin_edges[:-1], bin_edges[1:])]

    def _pct(values: List[int]) -> List[float]:
        counts = [0 for _ in labels]
        for value in values:
            for idx, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                if start <= value <= end:
                    counts[idx] += 1
                    break
        total = sum(counts)
        return [(count / total * 100.0) if total else 0.0 for count in counts]

    human_pct = _pct(human_values)
    generated_pct = _pct(generated_values)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    width = 0.36
    xs = list(range(len(labels)))
    ax.bar([x - width / 2 for x in xs], human_pct, width=width, color="#059669", label="Human Reference")
    ax.bar([x + width / 2 for x in xs], generated_pct, width=width, color="#2563eb", label="Generated")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Query Percentage (%)")
    ax.set_title("Word Count: Human vs Generated")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig


def build_view_choices(paper: Dict[str, Any]) -> List[Tuple[str, str]]:
    views = [("all", "all")]
    seen = set()
    for query in paper.get("queries", []):
        view = query.get("source_view", "unknown")
        if view not in seen:
            seen.add(view)
            count = sum(1 for item in paper.get("queries", []) if item.get("source_view") == view)
            views.append((f"{view} ({count})", view))
    return views


def get_filtered_queries(paper: Dict[str, Any], source_view: str) -> List[Dict[str, Any]]:
    queries = paper.get("queries", []) or []
    if source_view == "all":
        return queries
    return [query for query in queries if query.get("source_view") == source_view]


def has_saved_feedback_payload(payload: Dict[str, Any]) -> bool:
    if not payload:
        return False
    query_relevance = payload.get("query_paper_relevance") or {}
    human_like = payload.get("human_like_search") or {}
    candidate_checks = payload.get("candidate_checks") or {}
    return bool(query_relevance or human_like or candidate_checks)


def has_saved_query_feedback(
    paper: Dict[str, Any],
    query: Dict[str, Any],
    reviewer_username: Optional[str],
) -> bool:
    reviewer_username = str(reviewer_username or "").strip()
    if not reviewer_username:
        return False
    context = {
        "query_key": build_query_key(paper, query),
        "paper_id": paper.get("paper_id", ""),
        "query_text": query.get("query_text", ""),
        "source_view": query.get("source_view", ""),
    }
    try:
        return has_saved_feedback_payload(load_query_feedback(context, reviewer_username))
    except Exception:
        return False


def build_query_label_map(
    paper: Dict[str, Any],
    source_view: str,
    reviewer_username: Optional[str] = None,
) -> List[Tuple[str, str]]:
    out = []
    for idx, query in enumerate(paper.get("queries", []), start=1):
        if source_view != "all" and query.get("source_view") != source_view:
            continue
        decision = safe_get(query, "query_analysis", "decision", default="N/A")
        decision_label = "𝐊𝐞𝐞𝐩" if str(decision).lower() == "keep" else str(decision)
        saved_label = " | [saved]" if has_saved_query_feedback(paper, query, reviewer_username) else ""
        label = f"Q{idx} | {query.get('source_view', 'N/A')} | {decision_label}"
        label = f"{label}{saved_label}"
        out.append((label, query.get("query_text", "")))
    return out


def get_query_by_text(paper: Dict[str, Any], query_text: str) -> Optional[Dict[str, Any]]:
    for query in paper.get("queries", []):
        if query.get("query_text") == query_text:
            return query
    return None


def build_paper_detail_html(paper: Dict[str, Any]) -> str:
    openreview = paper.get("openreview", {}) or {}
    filter_status = paper.get("filter_status", {}) or {}

    meta_badges = "".join(
        [
            badge(f"Venue: {openreview.get('venue', 'N/A')}"),
            badge(f"Year: {openreview.get('year', 'N/A')}"),
            badge(f"Reviews: {len(openreview.get('reviews', []) or [])}"),
            badge(f"Comments: {len(openreview.get('comments', []) or [])}"),
            badge(f"Queries: {len(paper.get('queries', []) or [])}", "#ecfeff", "#155e75"),
        ]
    )

    authors = ", ".join(openreview.get("authors", []) or [])
    return f"""
    <div>
        <div style="font-size:24px;font-weight:900;margin-bottom:8px">{esc(paper.get('paper_title', 'Untitled'))}</div>
        <div style="margin-bottom:8px">{meta_badges}</div>
        <div style="margin-bottom:8px"><b>Paper ID:</b> <code>{esc(paper.get('paper_id', ''))}</code></div>
        <div style="margin-bottom:8px"><b>Paper Dir:</b> <code>{esc(paper.get('paper_dir', 'N/A'))}</code></div>
        <div style="margin-bottom:8px"><b>OpenReview URL:</b> <a href="{esc(openreview.get('openreview_url', ''))}" target="_blank">{esc(openreview.get('openreview_url', ''))}</a></div>
        <div style="margin-bottom:8px"><b>PDF URL:</b> <a href="{esc(openreview.get('pdf_url', ''))}" target="_blank">{esc(openreview.get('pdf_url', ''))}</a></div>
        <div style="margin-bottom:8px"><b>Authors:</b> {esc(authors)}</div>
        <div style="margin-bottom:8px"><b>Filter Status:</b> {esc(filter_status)}</div>
        <div style="border:1px solid #e5e7eb;border-radius:14px;padding:14px;background:white;margin-bottom:14px">
            <div style="font-size:16px;font-weight:800;margin-bottom:8px">Abstract</div>
            <div style="white-space:pre-wrap;line-height:1.6">{esc(openreview.get('abstract', ''))}</div>
        </div>
    </div>
    """


def build_summary_views_html(paper: Dict[str, Any]) -> str:
    blocks = []
    for view in paper.get("summary_views", []):
        bullets = "".join(
            f"<li>{esc(item.get('text', ''))}</li>"
            for item in (view.get("bullet_points", []) or [])
        ) or "<li>No bullet points.</li>"
        blocks.append(
            f"""
            <div style="padding:12px;border:1px solid #e5e7eb;border-radius:12px;background:#fafafa;margin-bottom:10px">
                <div style="font-weight:800;margin-bottom:6px">{esc(view.get('view_name', 'View'))}</div>
                <div style="margin-bottom:6px">{esc(view.get('summary', ''))}</div>
                <ul style="margin:0 0 0 18px">{bullets}</ul>
            </div>
            """
        )
    return "".join(blocks) or "<div>No summary views found.</div>"


def build_forum_content_html(paper: Dict[str, Any]) -> str:
    openreview = paper.get("openreview", {}) or {}

    def _render_items(items: List[Dict[str, Any]], title: str) -> str:
        blocks = []
        for idx, item in enumerate(items or [], start=1):
            content = item.get("content", {}) if isinstance(item, dict) else {}
            blocks.append(
                f"""
                <div style="padding:12px;border:1px solid #e5e7eb;border-radius:12px;background:#fafafa;margin-bottom:10px">
                    <div style="font-weight:800;margin-bottom:6px">{esc(title)} {idx}</div>
                    <pre style="white-space:pre-wrap;margin:0;font-size:13px">{esc(json.dumps(content, ensure_ascii=False, indent=2))}</pre>
                </div>
                """
            )
        return "".join(blocks) or f"<div>No {esc(title.lower())}.</div>"

    decision = openreview.get("decision")
    decision_html = (
        f"<pre style='white-space:pre-wrap;margin:0;font-size:13px'>{esc(json.dumps(decision, ensure_ascii=False, indent=2))}</pre>"
        if decision
        else "<div>No decision.</div>"
    )

    return f"""
    <div style="border:1px solid #e5e7eb;border-radius:14px;padding:14px;background:white">
        <div style="font-size:16px;font-weight:800;margin-bottom:8px">Decision</div>
        {decision_html}
        <div style="font-size:16px;font-weight:800;margin:14px 0 8px 0">Reviews</div>
        {_render_items(openreview.get('reviews', []), 'Review')}
        <div style="font-size:16px;font-weight:800;margin:14px 0 8px 0">Comments</div>
        {_render_items(openreview.get('comments', []), 'Comment')}
        <div style="font-size:16px;font-weight:800;margin:14px 0 8px 0">Rebuttals</div>
        {_render_items(openreview.get('rebuttals', []), 'Rebuttal')}
    </div>
    """


def lookup_related_bullet(paper: Dict[str, Any], query: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    source_view = query.get("source_view")
    indice = query.get("related_bullet_indice")
    for view in paper.get("summary_views", []):
        if view.get("view_name") != source_view:
            continue
        for idx, bullet in enumerate(view.get("bullet_points", []), start=1):
            bullet_index = bullet.get("index", idx) if isinstance(bullet, dict) else idx
            if bullet_index == indice:
                source_refs = bullet.get("source_refs", []) if isinstance(bullet, dict) else []
                if not isinstance(source_refs, list):
                    source_refs = [source_refs]
                return (
                    view.get("view_name", ""),
                    bullet.get("text", "") if isinstance(bullet, dict) else str(bullet),
                    [str(ref).strip() for ref in source_refs if str(ref).strip()],
                )
    return str(source_view or "N/A"), "", []


def _unwrap_content_value(value: Any) -> str:
    if isinstance(value, dict) and "value" in value:
        value = value.get("value")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value if value is not None else "")


def _numbered_entry(items: List[Dict[str, Any]], entry_name: str) -> Optional[Dict[str, Any]]:
    try:
        number = int(str(entry_name).strip().split()[-1])
    except Exception:
        return None

    for item in items or []:
        try:
            if int(item.get("number", 0)) == number:
                return item
        except Exception:
            continue
    if 1 <= number <= len(items or []):
        return items[number - 1]
    return None


def lookup_source_ref_content(paper: Dict[str, Any], source_ref: str) -> str:
    parts = [part.strip() for part in str(source_ref).split("/") if part.strip()]
    if len(parts) < 3:
        return ""

    section, entry_name = parts[0], parts[1]
    field_name = "/".join(parts[2:])
    openreview = paper.get("openreview", {}) or {}

    section_items = {
        "Reviews": openreview.get("reviews", []) or [],
        "Comments": openreview.get("comments", []) or [],
        "Rebuttals": openreview.get("rebuttals", []) or [],
    }
    if section == "Decision":
        entry = openreview.get("decision") or {}
    else:
        entry = _numbered_entry(section_items.get(section, []), entry_name) or {}

    content = entry.get("content", {}) if isinstance(entry, dict) else {}
    if not isinstance(content, dict) or field_name not in content:
        return ""
    return _unwrap_content_value(content.get(field_name)).strip()


def build_source_refs_details(paper: Dict[str, Any], source_refs: List[str]) -> str:
    if not source_refs:
        return ""

    blocks = []
    for source_ref in source_refs:
        content = lookup_source_ref_content(paper, source_ref)
        blocks.append(
            f"""
            <div style="border-top:1px solid #e5e7eb;padding-top:10px;margin-top:10px">
                <div style="font-size:12px;font-weight:800;color:#374151;margin-bottom:6px">{esc(source_ref)}</div>
                <div style="white-space:pre-wrap;line-height:1.55;color:#111827">{esc(content or 'Source content not found.')}</div>
            </div>
            """
        )

    return f"""
    <details style="margin-top:10px">
        <summary style="cursor:pointer;display:inline-block;border:1px solid #d1d5db;border-radius:8px;padding:4px 9px;background:#f9fafb;font-size:12px;font-weight:800;color:#374151">
            Source evidence
        </summary>
        <div style="margin-top:8px">
            {''.join(blocks)}
        </div>
    </details>
    """


def score_similarity_to_human(score: Any, human_mean: Any) -> Optional[float]:
    try:
        score_f = float(score)
        mean_f = float(human_mean)
    except Exception:
        return None
    return max(0.0, 1.0 - abs(score_f - mean_f) / 2.0)


def stable_field_id(value: Any) -> str:
    text = str(value if value is not None else "")
    total = sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))
    return str(total % 1_000_000_007)


def load_human_judgments(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "judgments": {}}
    data = load_json(path)
    data.setdefault("schema_version", 1)
    data.setdefault("judgments", {})
    if not isinstance(data["judgments"], dict):
        data["judgments"] = {}
    return data


def save_human_judgments(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_query_key(paper: Dict[str, Any], query: Dict[str, Any]) -> str:
    paper_id = paper.get("paper_id", "unknown_paper")
    source_view = query.get("source_view", "unknown_view")
    return f"{paper_id}::{source_view}::{stable_field_id(query.get('query_text', ''))}"


def build_candidate_items(query: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not query:
        return []
    hard_neg = query.get("hard_negative_context", {}) or {}
    items = []
    for idx, item in enumerate(hard_neg.get("hard_negatives", []) or [], start=1):
        title = item.get("paper_title", "Untitled")
        items.append(
            {
                "key": f"hard_negative:{idx}",
                "label": f"Hard Negative {idx}: {title}",
                "expected_label": "Hard Negative",
                "group": "hard_negative",
                "group_title": "Hard Negatives",
                "group_start": idx == 1,
                "paper": item,
            }
        )
    for idx, item in enumerate(hard_neg.get("positives", []) or [], start=1):
        title = item.get("paper_title", "Untitled")
        items.append(
            {
                "key": f"positive:{idx}",
                "label": f"Positive {idx}: {title}",
                "expected_label": "Positive",
                "group": "positive",
                "group_title": "Positives",
                "group_start": idx == 1,
                "paper": item,
            }
        )
    return items


def build_candidate_choices(query: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    return [(item["label"], item["key"]) for item in build_candidate_items(query)]


def build_candidate_item_html(candidate: Dict[str, Any]) -> str:
    item = candidate.get("paper", {}) or {}
    reason = item.get("hard_negative_reason") or item.get("positive_reason") or ""
    group_description = ""
    if candidate.get("group") == "hard_negative" and candidate.get("group_start"):
        group_description = (
            "<div style='font-size:13px;line-height:1.45;color:#4b5563;margin:-2px 0 8px 0'>"
            "<b>Definition:</b> A hard negative is a paper close enough to look plausible for the query, but it misses a key "
            "requirement and should be a challenging near-miss negative."
            "</div>"
        )
    group_header = (
        f"<div style='font-size:16px;font-weight:900;margin:12px 0 8px 0'>{esc(candidate.get('group_title', 'Candidates'))}</div>"
        f"{group_description}"
        if candidate.get("group_start")
        else ""
    )
    return f"""
    {group_header}
    <div style="padding:12px;border:1px solid #e5e7eb;border-radius:12px;background:#fafafa;margin-bottom:10px">
        <div style="font-weight:800;margin-bottom:6px">{esc(candidate.get('label', 'Candidate'))}</div>
        <div style="margin-bottom:6px"><b>PDF URL:</b> <a href="{esc(item.get('pdf_url', ''))}" target="_blank">{esc(item.get('pdf_url', ''))}</a></div>
        <div style="margin-bottom:6px"><b>ArXiv:</b> {esc(item.get('arxiv_id', 'N/A'))}</div>
        <div style="margin-bottom:6px"><b>Rationale:</b> {esc(reason)}</div>
    </div>
    """


def build_query_context(paper: Optional[Dict[str, Any]], query: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not paper or not query:
        return {}
    candidate_items = build_candidate_items(query)
    candidate_choices = [(item["label"], item["key"]) for item in candidate_items]
    return {
        "query_key": build_query_key(paper, query),
        "paper_id": paper.get("paper_id", ""),
        "paper_title": paper.get("paper_title", ""),
        "query_text": query.get("query_text", ""),
        "source_view": query.get("source_view", ""),
        "related_bullet_indice": query.get("related_bullet_indice"),
        "candidate_items": [
            {
                "key": item["key"],
                "label": item["label"],
                "expected_label": item["expected_label"],
                "group": item["group"],
                "html": build_candidate_item_html(item),
            }
            for item in candidate_items
        ],
        "candidate_choices": candidate_choices,
        "selected_candidate_key": candidate_choices[0][1] if candidate_choices else None,
    }


def candidate_question_label(candidate: Dict[str, Any]) -> str:
    if candidate.get("group") == "hard_negative":
        return "Is this hard negative label correct?"
    if candidate.get("group") == "positive":
        return "Is this positive label correct?"
    return "Is this label correct?"


def candidate_wrong_label_choices(candidate: Dict[str, Any]) -> List[str]:
    if candidate.get("group") == "hard_negative":
        return [
            "Actually positive",
            "Actually irrelevant / unsupported",
            "Too easy / not a hard negative",
            "Only topically related",
            "Evidence unavailable",
            "Duplicate / same paper",
            "Other",
        ]
    if candidate.get("group") == "positive":
        return [
            "Actually hard negative",
            "Actually irrelevant / unsupported",
            "Only topically related",
            "Evidence unavailable",
            "Duplicate / same paper",
            "Other",
        ]
    return [
        "Wrong relevance class",
        "Only topically related",
        "Evidence unavailable",
        "Duplicate / same paper",
        "Other",
    ]


def build_candidate_review_header(context: Dict[str, Any]) -> str:
    query_text = context.get("query_text", "") if context else ""
    if not query_text:
        return ""
    words = len(str(query_text).split())
    chars = len(str(query_text))
    return f"""
    <div style="font-size:18px;font-weight:900;line-height:1.35;margin:10px 0 8px 0">
        {esc(query_text)}
    </div>
    <div style="font-size:12px;color:#6b7280;margin-bottom:8px">Query size: {fmt_int(words)} words / {fmt_int(chars)} chars</div>
    """


def human_judgment_updates(context: Dict[str, Any], reviewer_username: Optional[str] = None):
    item: Dict[str, Any] = {}
    if context and reviewer_username:
        item = load_query_feedback(context, reviewer_username)
    query_paper = item.get("query_paper_relevance", {}) or {}
    human_like = item.get("human_like_search", {}) or {}
    candidates = item.get("candidate_checks", {}) or {}
    candidate_items = context.get("candidate_items", []) if context else []
    real_researcher_search = human_like.get("real_researcher_search")
    show_non_human_type = real_researcher_search == "No"
    non_human_like_type = human_like.get("non_human_like_type", [])
    show_non_human_other = show_non_human_type and "Other" in (non_human_like_type or [])

    updates = [
        gr.update(value=query_paper.get("relevance")),
        gr.update(value=query_paper.get("notes", "")),
        gr.update(value=real_researcher_search),
        gr.update(value=non_human_like_type, visible=show_non_human_type),
        gr.update(value=human_like.get("non_human_like_other", ""), visible=show_non_human_other),
        gr.update(value=human_like.get("notes", "")),
    ]

    for idx in range(MAX_CANDIDATE_JUDGE_ROWS):
        if idx < len(candidate_items):
            candidate = candidate_items[idx]
            saved = candidates.get(candidate["key"], {}) or {}
            label_correct = saved.get("label_correct")
            show_wrong_type = label_correct == "No"
            updates.extend(
                [
                    gr.update(value=candidate["html"], visible=True),
                    gr.update(value=label_correct, visible=True, label=candidate_question_label(candidate)),
                    gr.update(
                        value=saved.get("wrong_label_type", []),
                        choices=candidate_wrong_label_choices(candidate),
                        visible=show_wrong_type,
                    ),
                    gr.update(value=saved.get("notes", ""), visible=True),
                ]
            )
        else:
            updates.extend(
                [
                    gr.update(value="", visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(value=[], visible=False),
                    gr.update(value="", visible=False),
                ]
            )
    return tuple(updates)


def mysql_status_message(reviewer_username: Optional[str], error: Optional[Exception] = None) -> str:
    if error is not None:
        return f"MySQL feedback load failed for `{esc(reviewer_username or 'unknown')}`: {esc(error)}"
    if reviewer_username:
        return f"Saving to MySQL as `{esc(reviewer_username)}`"
    return "Saving to MySQL"


def safe_human_judgment_updates(
    context: Dict[str, Any],
    reviewer_username: Optional[str],
) -> Tuple[Tuple[Any, ...], str]:
    try:
        return human_judgment_updates(context, reviewer_username), mysql_status_message(reviewer_username)
    except Exception as exc:
        return human_judgment_updates(context, None), mysql_status_message(reviewer_username, exc)


def build_human_feedback_payload(
    context: Dict[str, Any],
    query_relevance: Optional[str],
    query_notes: str,
    real_researcher_search: Optional[str],
    non_human_like_type: List[str],
    non_human_like_other: str,
    human_like_notes: str,
    *candidate_values: Any,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    if not context or not context.get("query_key"):
        return None, ["No query selected"]

    errors = []
    if not query_relevance:
        errors.append("query-paper relevance")
    if not real_researcher_search:
        errors.append("human-like search")
    if real_researcher_search == "No" and not non_human_like_type:
        errors.append("non human-like type")
    if real_researcher_search == "No" and "Other" in (non_human_like_type or []) and not str(non_human_like_other or "").strip():
        errors.append("non human-like other reason")

    candidate_items = context.get("candidate_items", [])[:MAX_CANDIDATE_JUDGE_ROWS]
    candidate_checks = {}
    for idx, candidate in enumerate(candidate_items):
        offset = idx * 3
        label_correct = candidate_values[offset] if offset < len(candidate_values) else None
        wrong_label_type = candidate_values[offset + 1] if offset + 1 < len(candidate_values) else []
        notes = candidate_values[offset + 2] if offset + 2 < len(candidate_values) else ""
        if not label_correct:
            errors.append(f"{candidate.get('label', f'candidate {idx + 1}')} label check")
        if label_correct == "No" and not wrong_label_type:
            errors.append(f"{candidate.get('label', f'candidate {idx + 1}')} wrong-label type")
        candidate_checks[candidate["key"]] = {
            "candidate_label": candidate.get("label", candidate["key"]),
            "expected_label": candidate.get("expected_label", ""),
            "label_correct": label_correct,
            "wrong_label_type": (wrong_label_type or []) if label_correct == "No" else [],
            "notes": notes or "",
        }

    if errors:
        return None, errors

    return (
        {
            "query_paper_relevance": {
                "relevance": query_relevance,
                "notes": query_notes or "",
            },
            "human_like_search": {
                "real_researcher_search": real_researcher_search,
                "non_human_like_type": (
                    (non_human_like_type or []) if real_researcher_search == "No" else []
                ),
                "non_human_like_other": (
                    str(non_human_like_other or "").strip()
                    if real_researcher_search == "No" and "Other" in (non_human_like_type or [])
                    else ""
                ),
                "notes": human_like_notes or "",
            },
            "candidate_checks": candidate_checks,
        },
        [],
    )


def save_current_human_judgment(
    reviewer_username: Optional[str],
    context: Dict[str, Any],
    query_relevance: Optional[str],
    query_notes: str,
    real_researcher_search: Optional[str],
    non_human_like_type: List[str],
    non_human_like_other: str,
    human_like_notes: str,
    *candidate_values: Any,
) -> str:
    reviewer_username = str(reviewer_username or "").strip()
    if not reviewer_username:
        return "Login required before saving feedback."

    feedback_payload, errors = build_human_feedback_payload(
        context,
        query_relevance,
        query_notes,
        real_researcher_search,
        non_human_like_type,
        non_human_like_other,
        human_like_notes,
        *candidate_values,
    )
    if errors:
        if errors == ["No query selected"]:
            return "No query selected; nothing saved."
        return "Missing required fields: " + "; ".join(errors)

    try:
        save_query_feedback(context, feedback_payload or {}, reviewer_username)
    except Exception as exc:
        return f"MySQL feedback save failed for `{reviewer_username}`: {exc}"
    return f"Saved human feedback to MySQL as {reviewer_username}"


def build_candidate_list_html(items: List[Dict[str, Any]], label: str, query: Optional[Dict[str, Any]] = None) -> str:
    if not items:
        return f"<div>No {esc(label.lower())}.</div>"
    blocks = []
    for idx, item in enumerate(items, start=1):
        reason = item.get("hard_negative_reason") or item.get("positive_reason") or ""
        blocks.append(
            f"""
            <div style="padding:12px;border:1px solid #e5e7eb;border-radius:12px;background:#fafafa;margin-bottom:10px">
                <div style="font-weight:800;margin-bottom:6px">{esc(label)} {idx}: {esc(item.get('paper_title', 'Untitled'))}</div>
                <div style="margin-bottom:6px"><b>PDF URL:</b> <a href="{esc(item.get('pdf_url', ''))}" target="_blank">{esc(item.get('pdf_url', ''))}</a></div>
                <div style="margin-bottom:6px"><b>ArXiv:</b> {esc(item.get('arxiv_id', 'N/A'))}</div>
                <div style="margin-bottom:6px"><b>Rationale:</b> {esc(reason)}</div>
            </div>
            """
        )
    return "".join(blocks)


def framed_group(title: str, content: str) -> str:
    return (
        "<div style='border:1px solid #e5e7eb;border-radius:14px;padding:14px;background:#ffffff;margin-bottom:14px'>"
        f"<div style='font-size:16px;font-weight:800;margin-bottom:10px'>{esc(title)}</div>"
        f"{content}"
        "</div>"
    )


def score_group_box(title: str, items_html: str) -> str:
    return (
        "<div style='border:1px solid #e5e7eb;border-radius:14px;padding:12px;background:#fafafa'>"
        f"<div style='font-size:13px;font-weight:800;color:#374151;margin-bottom:8px'>{esc(title)}</div>"
        f"<div>{items_html}</div>"
        "</div>"
    )


def _query_display_parts(query: Dict[str, Any], paper: Dict[str, Any]) -> Tuple[str, str, str, int, int]:
    query_text = str(query.get("query_text", ""))
    query_number = "?"
    for idx, paper_query in enumerate(paper.get("queries", []) or [], start=1):
        if paper_query.get("query_text") == query.get("query_text"):
            query_number = str(idx)
            break
    return query_text, query_number, f"Q{query_number}: {query_text}", len(query_text.split()), len(query_text)


def build_query_header_html(query: Dict[str, Any], paper: Dict[str, Any], human_summary: Dict[str, Any]) -> str:
    analysis = query.get("query_analysis", {}) or {}
    style = analysis.get("style_evaluation", {}) or {}
    llm = style.get("llm_based", {}) or {}
    retrieval = analysis.get("retrieval_evaluation", {}) or {}
    query_text, query_number, _, query_words, query_chars = _query_display_parts(query, paper)
    spec_score = llm.get("specificity_calibration_score")
    nat_score = llm.get("lexical_naturalism_score")
    score_header = (
        "<div style='display:grid;grid-template-columns:minmax(220px,0.8fr) minmax(0,2fr);gap:12px;margin-bottom:14px'>"
        + score_group_box(
            "Decision",
            "".join(
                [
                    decision_badge(analysis.get("decision", "N/A")),
                    plain_meta(f"View: {query.get('source_view', 'N/A')}"),
                    plain_meta(f"is_multimodal: {query.get('is_multimodal', False)}"),
                ]
            ),
        )
        + score_group_box(
            "Analysis",
            f"""
            <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px">
                <div style="border:1px solid #e5e7eb;border-radius:10px;padding:10px;background:white">
                    <div style="font-size:12px;font-weight:800;color:#4b5563;margin-bottom:6px">Style Analysis</div>
                    {score_badge("Specificity", spec_score)}
                    {score_badge("Naturalism", nat_score)}
                    {score_badge("Constraint Count", llm.get("semantic_constraint_count"))}
                </div>
                <div style="border:1px solid #e5e7eb;border-radius:10px;padding:10px;background:white">
                    <div style="font-size:12px;font-weight:800;color:#4b5563;margin-bottom:6px">Retrieval Effectiveness</div>
                    {retrieval_badge("Full-Paper Reliance", retrieval.get("full_paper_reliance"))}
                </div>
            </div>
            """,
        )
        + "</div>"
    )
    return f"""
    <div style="padding:14px;border:1px solid #e5e7eb;border-radius:14px;background:white;margin-bottom:14px">
        <div style="font-size:19px;font-weight:900;margin-bottom:4px">Q{esc(query_number)}: {esc(query_text)}</div>
        <div style="font-size:12px;color:#6b7280;margin-bottom:10px">Query size: {fmt_int(query_words)} words / {fmt_int(query_chars)} chars</div>
        {score_header}
    </div>
    """


def build_related_source_html(query: Dict[str, Any], paper: Dict[str, Any]) -> str:
    bullet_view, bullet_text, bullet_source_refs = lookup_related_bullet(paper, query)

    return framed_group(
        "Related Source",
        f"""
        <div style="margin-bottom:6px">{plain_meta(f'View: {bullet_view}')}{plain_meta(f'Bullet: {query.get("related_bullet_indice", "N/A")}')}</div>
        <div style="margin-bottom:6px;white-space:pre-wrap;line-height:1.6"><b>Summarized Bullet Point:</b> {esc(bullet_text or 'N/A')}</div>
        {build_source_refs_details(paper, bullet_source_refs)}
        <div style="white-space:pre-wrap;line-height:1.6"><b>Relevance Justification:</b> {esc(query.get('related_bullet_justification', ''))}</div>
        """,
    )


def build_query_analysis_html(query: Dict[str, Any], human_summary: Dict[str, Any]) -> str:
    analysis = query.get("query_analysis", {}) or {}
    style = analysis.get("style_evaluation", {}) or {}
    llm = style.get("llm_based", {}) or {}
    retrieval = analysis.get("retrieval_evaluation", {}) or {}
    spec_score = llm.get("specificity_calibration_score")
    nat_score = llm.get("lexical_naturalism_score")
    spec_human_mean = human_summary.get("specificity_mean")
    nat_human_mean = human_summary.get("naturalism_mean")
    spec_similarity = score_similarity_to_human(spec_score, spec_human_mean)
    nat_similarity = score_similarity_to_human(nat_score, nat_human_mean)
    similarity_help = "Computed as max(0, 1 - abs(query_score - human_mean) / 2). Higher means closer to the combined human-reference average."
    specificity_help = "Specificity scale: 1 to 5 means broad to specific; 3 means moderate."
    naturalism_help = "Naturalism scale: 1 to 5 means casual to formal or synthetic-like; 3 means moderate."

    return framed_group(
        "Analysis",
        f"""
        <div style="border:1px solid #eef2f7;border-radius:12px;padding:12px;background:#fafafa;margin-bottom:12px">
            <div style="font-size:14px;font-weight:800;margin-bottom:8px">Style Analysis</div>
            <div style="margin-bottom:6px">{score_badge('Specificity', spec_score)} {icon_help_details(specificity_help)} {plain_meta(f'Human Avg: {fmt_score(spec_human_mean, 2)}')} {click_help_details('Similarity', fmt_score(spec_similarity, 2), similarity_help)}</div>
            <div style="margin-bottom:10px;white-space:pre-wrap;line-height:1.6"><b>Rationale:</b> {esc(llm.get('specificity_calibration_rationale', ''))}</div>
            <div style="margin-bottom:6px">{score_badge('Naturalism', nat_score)} {icon_help_details(naturalism_help)} {plain_meta(f'Human Avg: {fmt_score(nat_human_mean, 2)}')} {click_help_details('Similarity', fmt_score(nat_similarity, 2), similarity_help)}</div>
            <div style="margin-bottom:10px;white-space:pre-wrap;line-height:1.6"><b>Rationale:</b> {esc(llm.get('lexical_naturalism_rationale', ''))}</div>
            <div style="margin-bottom:6px">{score_badge('Constraint Count', llm.get('semantic_constraint_count'))}</div>
            <div style="white-space:pre-wrap;line-height:1.6"><b>Rationale:</b> {esc(llm.get('semantic_constraint_rationale', ''))}</div>
        </div>
        <div style="border:1px solid #eef2f7;border-radius:12px;padding:12px;background:#fafafa">
            <div style="font-size:14px;font-weight:800;margin-bottom:8px">Retrieval Effectiveness</div>
            <div style="margin-bottom:6px">{retrieval_badge("Full-Paper Reliance", retrieval.get("full_paper_reliance"))}</div>
            <div style="white-space:pre-wrap;line-height:1.6"><b>Retrieval Reasoning:</b> {esc(retrieval.get('reasoning', ''))}</div>
        </div>
        """,
    )


def render_paper(
    final_data: Dict[str, Any],
    human_summary: Dict[str, Any],
    paper_id: str,
    source_view: Optional[str] = None,
    query_label: Optional[str] = None,
    reviewer_username: Optional[str] = None,
):
    paper = get_paper_by_id(final_data, paper_id)
    if paper is None:
        return (
            "<div>Paper not found.</div>",
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            "<div>No query selected.</div>",
            "<div></div>",
            "<div></div>",
            "<div></div>",
            "<div></div>",
        )

    view_choices = build_view_choices(paper)
    available_views = [value for _, value in view_choices]
    selected_view = source_view if source_view in available_views else "all"

    label_map = build_query_label_map(paper, selected_view, reviewer_username)
    if label_map:
        selected_label = query_label if query_label in {label for label, _ in label_map} else label_map[0][0]
        selected_query_text = dict(label_map).get(selected_label)
        selected_query = get_query_by_text(paper, selected_query_text)
    else:
        selected_label = None
        selected_query = None

    return (
        build_paper_detail_html(paper),
        gr.update(choices=view_choices, value=selected_view),
        gr.update(choices=[label for label, _ in label_map], value=selected_label),
        build_query_header_html(selected_query, paper, human_summary) if selected_query else "<div>No query selected.</div>",
        build_related_source_html(selected_query, paper) if selected_query else "<div></div>",
        build_query_analysis_html(selected_query, human_summary) if selected_query else "<div></div>",
        build_summary_views_html(paper),
        build_forum_content_html(paper),
    )


def get_selected_query_context(
    final_data: Dict[str, Any],
    paper_id: str,
    source_view: Optional[str] = None,
    query_label: Optional[str] = None,
    reviewer_username: Optional[str] = None,
) -> Dict[str, Any]:
    paper = get_paper_by_id(final_data, paper_id)
    if paper is None:
        return {}

    available_views = [value for _, value in build_view_choices(paper)]
    selected_view = source_view if source_view in available_views else "all"
    label_map = build_query_label_map(paper, selected_view, reviewer_username)
    if not label_map:
        return {}
    selected_label = query_label if query_label in {label for label, _ in label_map} else label_map[0][0]
    selected_query = get_query_by_text(paper, dict(label_map).get(selected_label, ""))
    return build_query_context(paper, selected_query)


def build_query_selector_update(
    final_data: Dict[str, Any],
    paper_id: str,
    source_view: Optional[str],
    query_label: Optional[str],
    reviewer_username: Optional[str],
    selected_query_text: Optional[str] = None,
):
    paper = get_paper_by_id(final_data, paper_id)
    if paper is None:
        return gr.update(choices=[], value=None)

    available_views = [value for _, value in build_view_choices(paper)]
    selected_view = source_view if source_view in available_views else "all"
    label_map = build_query_label_map(paper, selected_view, reviewer_username)
    if not label_map:
        return gr.update(choices=[], value=None)

    label_by_query_text = {query_text: label for label, query_text in label_map}
    if selected_query_text and selected_query_text in label_by_query_text:
        selected_label = label_by_query_text[selected_query_text]
    elif query_label in {label for label, _ in label_map}:
        selected_label = str(query_label)
    else:
        selected_label = label_map[0][0]
    return gr.update(choices=[label for label, _ in label_map], value=selected_label)


def launch_app(
    final_json_path: Path,
    human_litsearch_path: Optional[Path],
    human_pasa_path: Optional[Path],
    human_judgments_path: Path,
    port: int = 7860,
    share: bool = False,
) -> None:
    final_data = load_json(final_json_path)
    human_bundle = load_human_bundle(human_litsearch_path, human_pasa_path)
    human_summary = combine_human_style_summary(human_bundle)
    generated_df = extract_generated_style_scores(final_data)
    generated_summary = compute_generated_style_summary(generated_df)

    paper_choices = build_paper_choices(final_data)
    default_paper_id = paper_choices[0][1] if paper_choices else None

    with gr.Blocks(title="Final Pipeline Output Viewer") as demo:
        gr.Markdown("# Final Pipeline Output Viewer")
        gr.Markdown(f"Viewing `<code>{esc(final_json_path)}</code>`")

        with gr.Tab("Overview"):
            gr.HTML(build_overview_html(final_data, human_summary, generated_summary))
            gr.Dataframe(value=build_metric_summary_df(generated_summary, human_summary), interactive=False)
            with gr.Row():
                gr.Plot(
                    value=plot_style_distribution(
                        generated_summary,
                        human_summary,
                        metric="specificity",
                        title="Specificity Calibration: Human vs Generated",
                    )
                )
                gr.Plot(
                    value=plot_style_distribution(
                        generated_summary,
                        human_summary,
                        metric="naturalism",
                        title="Lexical Naturalism: Human vs Generated",
                    )
                )
                gr.Plot(
                    value=plot_word_count_distribution(generated_summary, human_summary)
                )
            gr.Dataframe(value=build_paper_stats_df(final_data), interactive=False, wrap=True)
            gr.HTML(build_overview_footer_html(final_data, human_summary))

        with gr.Tab("Paper Browser"):
            paper_selector = gr.Dropdown(choices=paper_choices, value=default_paper_id, label="Paper")
            if default_paper_id is not None:
                (
                    initial_html,
                    initial_view_update,
                    initial_query_update,
                    initial_query_header,
                    initial_related_source,
                    initial_analysis,
                    initial_summary_views,
                    initial_forum,
                ) = (
                    render_paper(final_data, human_summary, default_paper_id)
                )
                initial_context = get_selected_query_context(final_data, default_paper_id)
            else:
                initial_html = "<div>No papers available.</div>"
                initial_view_update = gr.update(choices=[], value=None)
                initial_query_update = gr.update(choices=[], value=None)
                initial_query_header = "<div>No query selected.</div>"
                initial_related_source = "<div></div>"
                initial_analysis = "<div></div>"
                initial_summary_views = "<div></div>"
                initial_forum = "<div></div>"
                initial_context = {}
            initial_judgment_updates = human_judgment_updates(initial_context)
            initial_candidate_review_header = build_candidate_review_header(initial_context)

            paper_html = gr.HTML(value=initial_html)
            with gr.Accordion("Summary Views", open=False):
                summary_views_html = gr.HTML(value=initial_summary_views)
            with gr.Accordion("Comments + Reviews + Rebuttals", open=False):
                forum_html = gr.HTML(value=initial_forum)

            with gr.Row():
                view_selector = gr.Dropdown(
                    choices=initial_view_update["choices"],
                    value=initial_view_update["value"],
                    label="View",
                    scale=1,
                )
                query_selector = gr.Radio(
                    choices=initial_query_update["choices"],
                    value=initial_query_update["value"],
                    label="Queries",
                    scale=3,
                )
            query_context_state = gr.State(value=initial_context)
            reviewer_username_state = gr.State(value="")
            query_header_html = gr.HTML(value=initial_query_header)
            with gr.Row():
                with gr.Column(scale=3):
                    related_source_html = gr.HTML(value=initial_related_source)
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("**Query-Paper Relevance**")
                        query_relevance = gr.Radio(
                            ["Yes", "No", "Unsure"],
                            value=initial_judgment_updates[0]["value"],
                            label="Does this paper/source support the query?",
                        )
                        query_notes = gr.Textbox(
                            value=initial_judgment_updates[1]["value"],
                            label="Reason / note (optional)",
                            lines=2,
                        )
            with gr.Row():
                with gr.Column(scale=3):
                    analysis_html = gr.HTML(value=initial_analysis)
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("**Human-Like Search**")
                        real_researcher_search = gr.Radio(
                            ["Yes", "No", "Unsure"],
                            value=initial_judgment_updates[2]["value"],
                            label="Would a real researcher search this?",
                        )
                        non_human_like_type = gr.CheckboxGroup(
                            ["Too broad", "Too specific", "Too casual", "Too formal / synthetic-like", "Other"],
                            value=initial_judgment_updates[3]["value"],
                            label="Non-human-like type",
                            visible=initial_judgment_updates[3].get("visible", False),
                        )
                        non_human_like_other = gr.Textbox(
                            value=initial_judgment_updates[4]["value"],
                            label="Other reason / note",
                            lines=1,
                            visible=initial_judgment_updates[4].get("visible", False),
                        )
                        human_like_notes = gr.Textbox(
                            value=initial_judgment_updates[5]["value"],
                            label="Reason / note (optional)",
                            lines=2,
                        )

            gr.Markdown("### Candidate Checks")
            candidate_review_header = gr.HTML(value=initial_candidate_review_header)
            candidate_components = []
            candidate_start = 6
            for idx in range(MAX_CANDIDATE_JUDGE_ROWS):
                offset = candidate_start + idx * 4
                with gr.Row():
                    with gr.Column(scale=3):
                        candidate_html = gr.HTML(
                            value=initial_judgment_updates[offset]["value"],
                            visible=initial_judgment_updates[offset].get("visible", False),
                        )
                    with gr.Column(scale=1):
                        candidate_label_correct = gr.Radio(
                            ["Yes", "No", "Unsure"],
                            value=initial_judgment_updates[offset + 1]["value"],
                            label=initial_judgment_updates[offset + 1].get("label", "Is this label correct?"),
                            visible=initial_judgment_updates[offset + 1].get("visible", False),
                        )
                        candidate_wrong_type = gr.CheckboxGroup(
                            initial_judgment_updates[offset + 2].get("choices", ["Wrong relevance class", "Other"]),
                            value=initial_judgment_updates[offset + 2]["value"],
                            label="Wrong-label type",
                            visible=initial_judgment_updates[offset + 2].get("visible", False),
                        )
                        candidate_notes = gr.Textbox(
                            value=initial_judgment_updates[offset + 3]["value"],
                            label="Reason / note (optional)",
                            lines=2,
                            visible=initial_judgment_updates[offset + 3].get("visible", False),
                        )
                candidate_components.append(
                    (candidate_html, candidate_label_correct, candidate_wrong_type, candidate_notes)
                )

            save_judgment_button = gr.Button("Save Human Judgment", variant="primary")
            judge_status = gr.Markdown("Saving to MySQL")

            def _render_all(
                paper_id: str,
                reviewer_username: str,
                source_view: Optional[str] = None,
                query_label: Optional[str] = None,
            ):
                rendered = render_paper(
                    final_data,
                    human_summary,
                    paper_id,
                    source_view=source_view,
                    query_label=query_label,
                    reviewer_username=reviewer_username,
                )
                view_update = rendered[1]
                query_update = rendered[2]
                context = get_selected_query_context(
                    final_data,
                    paper_id,
                    source_view=view_update["value"],
                    query_label=query_update["value"],
                    reviewer_username=reviewer_username,
                )
                feedback_updates, status = safe_human_judgment_updates(context, reviewer_username)
                return (
                    rendered
                    + (context, gr.update(value=build_candidate_review_header(context)))
                    + feedback_updates
                    + (gr.update(value=status),)
                )

            def _on_paper_change(paper_id: str, reviewer_username: str):
                return _render_all(paper_id, reviewer_username)

            def _on_view_change(paper_id: str, reviewer_username: str, source_view: str):
                return _render_all(paper_id, reviewer_username, source_view=source_view)

            def _on_query_change(
                paper_id: str,
                reviewer_username: str,
                source_view: str,
                query_label: str,
            ):
                return _render_all(
                    paper_id,
                    reviewer_username,
                    source_view=source_view,
                    query_label=query_label,
                )

            def _toggle_non_human_like(value: Optional[str]):
                return gr.update(visible=value == "No"), gr.update(visible=False)

            def _toggle_non_human_other(value: Optional[List[str]]):
                return gr.update(visible="Other" in (value or []))

            def _toggle_wrong_label_type(value: Optional[str]):
                return gr.update(visible=value == "No")

            def _save_judgment(
                reviewer_username: str,
                context: Dict[str, Any],
                paper_id: str,
                source_view: str,
                query_label: str,
                query_relevance_value: Optional[str],
                query_notes_value: str,
                real_researcher_search_value: Optional[str],
                non_human_like_type_value: List[str],
                non_human_like_other_value: str,
                human_like_notes_value: str,
                *candidate_values: Any,
            ):
                status = save_current_human_judgment(
                    reviewer_username,
                    context,
                    query_relevance_value,
                    query_notes_value,
                    real_researcher_search_value,
                    non_human_like_type_value,
                    non_human_like_other_value,
                    human_like_notes_value,
                    *candidate_values,
                )
                query_update = build_query_selector_update(
                    final_data,
                    paper_id or context.get("paper_id", ""),
                    source_view or context.get("source_view", "all"),
                    query_label,
                    reviewer_username,
                    selected_query_text=context.get("query_text"),
                )
                return status, query_update

            render_outputs = [
                paper_html,
                view_selector,
                query_selector,
                query_header_html,
                related_source_html,
                analysis_html,
                summary_views_html,
                forum_html,
                query_context_state,
                candidate_review_header,
                query_relevance,
                query_notes,
                real_researcher_search,
                non_human_like_type,
                non_human_like_other,
                human_like_notes,
            ]
            for components in candidate_components:
                render_outputs.extend(components)
            render_outputs.append(judge_status)

            feedback_outputs = [
                query_relevance,
                query_notes,
                real_researcher_search,
                non_human_like_type,
                non_human_like_other,
                human_like_notes,
            ]
            for components in candidate_components:
                feedback_outputs.extend(components)
            feedback_outputs.append(judge_status)

            def _on_load(
                context: Dict[str, Any],
                paper_id: str,
                source_view: str,
                query_label: str,
                request: gr.Request,
            ):
                reviewer_username = request_username(request)
                feedback_updates, status = safe_human_judgment_updates(context, reviewer_username)
                query_update = build_query_selector_update(
                    final_data,
                    paper_id or context.get("paper_id", ""),
                    source_view or context.get("source_view", "all"),
                    query_label,
                    reviewer_username,
                    selected_query_text=context.get("query_text"),
                )
                return (reviewer_username or "", query_update) + feedback_updates + (gr.update(value=status),)

            demo.load(
                _on_load,
                inputs=[query_context_state, paper_selector, view_selector, query_selector],
                outputs=[reviewer_username_state, query_selector] + feedback_outputs,
            )

            paper_selector.change(
                _on_paper_change,
                inputs=[paper_selector, reviewer_username_state],
                outputs=render_outputs,
            )
            view_selector.change(
                _on_view_change,
                inputs=[paper_selector, reviewer_username_state, view_selector],
                outputs=render_outputs,
            )
            query_selector.change(
                _on_query_change,
                inputs=[paper_selector, reviewer_username_state, view_selector, query_selector],
                outputs=render_outputs,
            )
            real_researcher_search.change(
                _toggle_non_human_like,
                inputs=real_researcher_search,
                outputs=[non_human_like_type, non_human_like_other],
            )
            non_human_like_type.change(
                _toggle_non_human_other,
                inputs=non_human_like_type,
                outputs=non_human_like_other,
            )
            for _, candidate_label_correct, candidate_wrong_type, _ in candidate_components:
                candidate_label_correct.change(
                    _toggle_wrong_label_type,
                    inputs=candidate_label_correct,
                    outputs=candidate_wrong_type,
                )
            save_judgment_button.click(
                _save_judgment,
                inputs=[
                    reviewer_username_state,
                    query_context_state,
                    paper_selector,
                    view_selector,
                    query_selector,
                    query_relevance,
                    query_notes,
                    real_researcher_search,
                    non_human_like_type,
                    non_human_like_other,
                    human_like_notes,
                ]
                + [
                    component
                    for _, candidate_label_correct, candidate_wrong_type, candidate_notes in candidate_components
                    for component in (candidate_label_correct, candidate_wrong_type, candidate_notes)
                ],
                outputs=[judge_status, query_selector],
            )

    demo.launch(
        auth=authenticate_user,
        share=share,
        server_name=os.getenv("GRADIO_SERVER_NAME"),
        server_port=port,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio viewer for final_pipeline_output.json")
    parser.add_argument(
        "--final-json",
        default=str(DEFAULT_FINAL_JSON),
        help="Path to final_pipeline_output.json",
    )
    parser.add_argument(
        "--human-litsearch-json",
        default=str(DEFAULT_HUMAN_LITSEARCH_JSON),
        help="Path to LitSearch human style_analysis.json",
    )
    parser.add_argument(
        "--human-pasa-json",
        default=str(DEFAULT_HUMAN_PASA_JSON),
        help="Path to PASA human style_analysis.json",
    )
    parser.add_argument(
        "--human-judgments-json",
        default=str(DEFAULT_HUMAN_JUDGMENTS_JSON),
        help="Path where interactive human judgments are saved",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio app on (default is 7860)"
    )
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    args = parser.parse_args()

    launch_app(
        final_json_path=resolve_input_path(args.final_json) or Path(args.final_json).resolve(),
        human_litsearch_path=resolve_input_path(args.human_litsearch_json),
        human_pasa_path=resolve_input_path(args.human_pasa_json),
        human_judgments_path=Path(args.human_judgments_json).expanduser(),
        port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
