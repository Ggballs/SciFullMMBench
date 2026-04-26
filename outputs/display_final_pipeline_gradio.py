from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_FINAL_JSON = Path("outputs/test_single/final_pipeline_output.json")
DEFAULT_HUMAN_LITSEARCH_JSON = Path("outputs/query_analysis_comparison/03_litsearch_human/style_analysis.json")
DEFAULT_HUMAN_PASA_JSON = Path("outputs/query_analysis_comparison/02_pasa_realscholar/style_analysis.json")
REPO_ROOT = Path(__file__).resolve().parents[1]


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

    for item in sources:
        data = item["data"]
        total = _dataset_total(data)
        qual = _qual(data)
        spec = qual.get("specificity_calibration", {}) or {}
        nat = qual.get("lexical_naturalism", {}) or {}
        spec_fit = qual.get("specificity_calibration_fit", {}) or qual.get("specificity", {}) or {}
        nat_fit = qual.get("lexical_naturalism_fit", {}) or qual.get("naturalness", {}) or {}
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
    }


def extract_generated_style_scores(final_data: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for paper in final_data.get("papers", []):
        for query in paper.get("queries", []):
            llm = safe_get(query, "query_analysis", "style_evaluation", "llm_based", default={}) or {}
            rows.append(
                {
                    "paper_id": paper.get("paper_id", ""),
                    "paper_title": paper.get("paper_title", ""),
                    "query_text": query.get("query_text", ""),
                    "specificity": llm.get("specificity_calibration_score"),
                    "naturalism": llm.get("lexical_naturalism_score"),
                    "constraint_count": llm.get("semantic_constraint_count"),
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
    return summary


def build_overview_html(final_data: Dict[str, Any], human_summary: Dict[str, Any]) -> str:
    dataset = final_data.get("dataset_overview", {}) or {}
    stage5 = final_data.get("stage5_summary", {}) or {}
    style_summary = stage5.get("style_summary", {}) or {}

    cards = [
        ("Papers", fmt_int(len(final_data.get("papers", [])))),
        ("Queries", fmt_int(dataset.get("stage5_total_queries") or dataset.get("stage3_total_queries") or 0)),
        ("Hard Negatives", fmt_int(dataset.get("stage4_total_hard_negatives", 0))),
        ("Generated Spec. Mean", fmt_score(style_summary.get("specificity_calibration_mean"))),
        ("Generated Nat. Mean", fmt_score(style_summary.get("lexical_naturalism_mean"))),
        ("Human Spec. Mean", fmt_score(human_summary.get("specificity_mean"))),
        ("Human Nat. Mean", fmt_score(human_summary.get("naturalism_mean"))),
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
            <div style="margin-bottom:8px"><b>Decision Counts:</b> {esc(stage5.get("decision_counts", {}))}</div>
            <div style="margin-bottom:8px"><b>Retrieval Summary:</b> {esc(stage5.get("retrieval_summary", {}))}</div>
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
                "specificity_mean": human_summary.get("specificity_mean"),
                "naturalism_mean": human_summary.get("naturalism_mean"),
            },
            {
                "dataset": "generated",
                "queries": generated_summary.get("total_queries"),
                "specificity_mean": generated_summary.get("specificity_mean"),
                "naturalism_mean": generated_summary.get("naturalism_mean"),
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


def build_query_label_map(paper: Dict[str, Any], source_view: str) -> List[Tuple[str, str]]:
    out = []
    for idx, query in enumerate(paper.get("queries", []), start=1):
        if source_view != "all" and query.get("source_view") != source_view:
            continue
        decision = safe_get(query, "query_analysis", "decision", default="N/A")
        label = f"Q{idx} | {query.get('source_view', 'N/A')} | {decision}"
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


def lookup_related_bullet(paper: Dict[str, Any], query: Dict[str, Any]) -> Tuple[str, str]:
    source_view = query.get("source_view")
    indice = query.get("related_bullet_indice")
    for view in paper.get("summary_views", []):
        if view.get("view_name") != source_view:
            continue
        for idx, bullet in enumerate(view.get("bullet_points", []), start=1):
            if idx == indice:
                return view.get("view_name", ""), bullet.get("text", "")
    return str(source_view or "N/A"), ""


def score_similarity_to_human(score: Any, human_mean: Any) -> Optional[float]:
    try:
        score_f = float(score)
        mean_f = float(human_mean)
    except Exception:
        return None
    return max(0.0, 1.0 - abs(score_f - mean_f) / 4.0)


def build_candidate_list_html(items: List[Dict[str, Any]], label: str) -> str:
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


def build_query_detail_html(query: Dict[str, Any], paper: Dict[str, Any], human_summary: Dict[str, Any]) -> str:
    analysis = query.get("query_analysis", {}) or {}
    style = analysis.get("style_evaluation", {}) or {}
    llm = style.get("llm_based", {}) or {}
    retrieval = analysis.get("retrieval_evaluation", {}) or {}
    hard_neg = query.get("hard_negative_context", {}) or {}
    bullet_view, bullet_text = lookup_related_bullet(paper, query)

    spec_score = llm.get("specificity_calibration_score")
    nat_score = llm.get("lexical_naturalism_score")
    spec_human_mean = human_summary.get("specificity_mean")
    nat_human_mean = human_summary.get("naturalism_mean")
    spec_similarity = score_similarity_to_human(spec_score, spec_human_mean)
    nat_similarity = score_similarity_to_human(nat_score, nat_human_mean)
    similarity_help = "Computed as max(0, 1 - abs(query_score - human_mean) / 4). Higher means closer to the combined human-reference average."

    score_header = (
        "<div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:14px'>"
        + score_group_box(
            "Decision",
            "".join([decision_badge(analysis.get("decision", "N/A")), plain_meta(f"View: {query.get('source_view', 'N/A')}")]),
        )
        + score_group_box(
            "Retrieval Effectiveness",
            "".join(
                [
                    retrieval_badge("Full-Paper Reliance", retrieval.get("full_paper_reliance")),
                    retrieval_badge("False-Negative Risk", retrieval.get("false_negative_risk")),
                ]
            ),
        )
        + score_group_box(
            "Style",
            "".join(
                [
                    score_badge("Specificity", spec_score),
                    score_badge("Naturalism", nat_score),
                    score_badge("Constraint Count", llm.get("semantic_constraint_count")),
                ]
            ),
        )
        + "</div>"
    )

    related_bullet_group = framed_group(
        "Related Bullet Point",
        f"""
        <div style="margin-bottom:6px">{plain_meta(f'View: {bullet_view}')}{plain_meta(f'Bullet: {query.get("related_bullet_indice", "N/A")}')}</div>
        <div style="margin-bottom:6px;white-space:pre-wrap;line-height:1.6"><b>Bullet Content:</b> {esc(bullet_text or 'N/A')}</div>
        <div style="white-space:pre-wrap;line-height:1.6"><b>Selection Justification:</b> {esc(query.get('related_bullet_justification', ''))}</div>
        """,
    )

    analysis_group = framed_group(
        "Analysis",
        f"""
        <div style="border:1px solid #eef2f7;border-radius:12px;padding:12px;background:#fafafa;margin-bottom:12px">
            <div style="font-size:14px;font-weight:800;margin-bottom:8px">Retrieval Effectiveness</div>
            <div style="margin-bottom:6px">{retrieval_badge("Full-Paper Reliance", retrieval.get("full_paper_reliance"))}{retrieval_badge("False-Negative Risk", retrieval.get("false_negative_risk"))}</div>
            <div style="white-space:pre-wrap;line-height:1.6"><b>Retrieval Reasoning:</b> {esc(retrieval.get('reasoning', ''))}</div>
        </div>
        <div style="border:1px solid #eef2f7;border-radius:12px;padding:12px;background:#fafafa">
            <div style="font-size:14px;font-weight:800;margin-bottom:8px">Style Analysis</div>
            <div style="margin-bottom:6px">{score_badge('Specificity', spec_score)} {plain_meta(f'Human Avg: {fmt_score(spec_human_mean, 2)}')} {click_help_details('Similarity', fmt_score(spec_similarity, 2), similarity_help)}</div>
            <div style="margin-bottom:10px;white-space:pre-wrap;line-height:1.6"><b>Rationale:</b> {esc(llm.get('specificity_calibration_rationale', ''))}</div>
            <div style="margin-bottom:6px">{score_badge('Naturalism', nat_score)} {plain_meta(f'Human Avg: {fmt_score(nat_human_mean, 2)}')} {click_help_details('Similarity', fmt_score(nat_similarity, 2), similarity_help)}</div>
            <div style="margin-bottom:10px;white-space:pre-wrap;line-height:1.6"><b>Rationale:</b> {esc(llm.get('lexical_naturalism_rationale', ''))}</div>
            <div style="margin-bottom:6px">{score_badge('Constraint Count', llm.get('semantic_constraint_count'))}</div>
            <div style="white-space:pre-wrap;line-height:1.6"><b>Rationale:</b> {esc(llm.get('semantic_constraint_rationale', ''))}</div>
        </div>
        """,
    )

    hard_negative_group = framed_group(
        "Hard Negative / Positive",
        f"""
        <div style="margin-bottom:8px"><b>Search Queries Used:</b> {esc(hard_neg.get('search_queries_used', []))}</div>
        <div style="margin-bottom:8px"><b>Retrieved Candidates:</b> {esc(hard_neg.get('retrieved_candidates', 0))}</div>
        <div style="font-size:15px;font-weight:800;margin:10px 0 6px 0">Hard Negatives</div>
        {build_candidate_list_html(hard_neg.get('hard_negatives', []) or [], 'Hard Negative')}
        <div style="font-size:15px;font-weight:800;margin:10px 0 6px 0">Positives</div>
        {build_candidate_list_html(hard_neg.get('positives', []) or [], 'Positive')}
        """,
    )

    return f"""
    <div style="padding:14px;border:1px solid #e5e7eb;border-radius:14px;background:white">
        <div style="font-size:18px;font-weight:900;margin-bottom:8px">{esc(query.get('query_text', ''))}</div>
        {score_header}
        {related_bullet_group}
        {analysis_group}
        {hard_negative_group}
    </div>
    """


def render_paper(
    final_data: Dict[str, Any],
    human_summary: Dict[str, Any],
    paper_id: str,
    source_view: Optional[str] = None,
    query_label: Optional[str] = None,
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
        )

    view_choices = build_view_choices(paper)
    available_views = [value for _, value in view_choices]
    selected_view = source_view if source_view in available_views else "all"

    label_map = build_query_label_map(paper, selected_view)
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
        build_query_detail_html(selected_query, paper, human_summary) if selected_query else "<div>No query selected.</div>",
        build_summary_views_html(paper),
        build_forum_content_html(paper),
    )


def launch_app(
    final_json_path: Path,
    human_litsearch_path: Optional[Path],
    human_pasa_path: Optional[Path],
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
            gr.HTML(build_overview_html(final_data, human_summary))
            gr.Dataframe(value=build_metric_summary_df(generated_summary, human_summary), interactive=False)
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
            gr.Dataframe(value=build_paper_stats_df(final_data), interactive=False, wrap=True)
            gr.HTML(build_overview_footer_html(final_data, human_summary))

        with gr.Tab("Paper Browser"):
            paper_selector = gr.Dropdown(choices=paper_choices, value=default_paper_id, label="Paper")
            if default_paper_id is not None:
                initial_html, initial_view_update, initial_query_update, initial_detail, initial_summary_views, initial_forum = (
                    render_paper(final_data, human_summary, default_paper_id)
                )
            else:
                initial_html = "<div>No papers available.</div>"
                initial_view_update = gr.update(choices=[], value=None)
                initial_query_update = gr.update(choices=[], value=None)
                initial_detail = "<div>No query selected.</div>"
                initial_summary_views = "<div></div>"
                initial_forum = "<div></div>"

            paper_html = gr.HTML(value=initial_html)
            with gr.Accordion("Summary Views", open=False):
                summary_views_html = gr.HTML(value=initial_summary_views)
            with gr.Accordion("Comments + Reviews + Rebuttals", open=False):
                forum_html = gr.HTML(value=initial_forum)

            with gr.Row():
                with gr.Column(scale=1):
                    view_selector = gr.Dropdown(
                        choices=initial_view_update["choices"],
                        value=initial_view_update["value"],
                        label="View",
                    )
                    query_selector = gr.Radio(
                        choices=initial_query_update["choices"],
                        value=initial_query_update["value"],
                        label="Queries",
                    )
                with gr.Column(scale=3):
                    query_detail_html = gr.HTML(value=initial_detail)

            def _on_paper_change(paper_id: str):
                return render_paper(final_data, human_summary, paper_id)

            def _on_view_change(paper_id: str, source_view: str):
                return render_paper(final_data, human_summary, paper_id, source_view=source_view)

            def _on_query_change(paper_id: str, source_view: str, query_label: str):
                return render_paper(final_data, human_summary, paper_id, source_view=source_view, query_label=query_label)

            paper_selector.change(
                _on_paper_change,
                inputs=paper_selector,
                outputs=[
                    paper_html,
                    view_selector,
                    query_selector,
                    query_detail_html,
                    summary_views_html,
                    forum_html,
                ],
            )
            view_selector.change(
                _on_view_change,
                inputs=[paper_selector, view_selector],
                outputs=[
                    paper_html,
                    view_selector,
                    query_selector,
                    query_detail_html,
                    summary_views_html,
                    forum_html,
                ],
            )
            query_selector.change(
                _on_query_change,
                inputs=[paper_selector, view_selector, query_selector],
                outputs=[
                    paper_html,
                    view_selector,
                    query_selector,
                    query_detail_html,
                    summary_views_html,
                    forum_html,
                ],
            )

    demo.launch(share=share)


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
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    args = parser.parse_args()

    launch_app(
        final_json_path=resolve_input_path(args.final_json) or Path(args.final_json).resolve(),
        human_litsearch_path=resolve_input_path(args.human_litsearch_json),
        human_pasa_path=resolve_input_path(args.human_pasa_json),
        share=args.share,
    )


if __name__ == "__main__":
    main()
