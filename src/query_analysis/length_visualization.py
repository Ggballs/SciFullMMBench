"""Helpers for visualizing query word-count distributions."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _word_counts(queries: List[str]) -> List[int]:
    return [len(str(query).split()) for query in queries]


def build_length_distribution_summary(
    query_sets: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Build per-dataset summary data for query word-count distributions."""
    populated_sets = {
        name: _word_counts(queries)
        for name, queries in query_sets.items()
        if queries
    }
    if not populated_sets:
        return {"datasets": {}}

    global_min = min(min(counts) for counts in populated_sets.values())
    global_max = max(max(counts) for counts in populated_sets.values())
    bin_edges = np.arange(global_min - 0.5, global_max + 1.5, 1.0)

    datasets = {}
    for name, counts in populated_sets.items():
        counts_array = np.array(counts)
        hist_counts, hist_edges = np.histogram(counts_array, bins=bin_edges)
        datasets[name] = {
            "total_queries": len(counts),
            "word_count_stats": {
                "mean": float(np.mean(counts_array)),
                "std": float(np.std(counts_array)),
                "min": int(np.min(counts_array)),
                "max": int(np.max(counts_array)),
                "median": float(np.median(counts_array)),
                "p25": float(np.percentile(counts_array, 25)),
                "p75": float(np.percentile(counts_array, 75)),
            },
            "histogram": {
                "bin_edges": [float(edge) for edge in hist_edges.tolist()],
                "counts": [int(value) for value in hist_counts.tolist()],
            },
        }

    return {"datasets": datasets}


def save_length_distribution_plot(
    query_sets: Dict[str, List[str]],
    output_path: Path,
) -> Dict[str, Any]:
    """Save histogram + boxplot visualization and return summary data."""
    summary = build_length_distribution_summary(query_sets)
    datasets = summary.get("datasets", {})
    if not datasets:
        raise ValueError("No query sets with content were provided for plotting.")

    try:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(Path(tempfile.gettempdir()) / "sci_full_mm_bench_matplotlib"),
        )
        import matplotlib

        matplotlib.use("Agg")
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for --plot_length_distribution. "
            "Install project dependencies to enable plotting."
        ) from exc

    colors = {
        "litsearch": "#2E6F95",
        "pasa": "#C8553D",
        "combined": "#3A7D44",
    }
    fallback_colors = ["#5B8E7D", "#8E6C8A", "#D9A441", "#607196"]

    plotted_names = list(datasets.keys())
    all_counts = [
        _word_counts(query_sets[name])
        for name in plotted_names
    ]

    global_edges = datasets[plotted_names[0]]["histogram"]["bin_edges"]
    bin_edges = np.array(global_edges)

    fig, (hist_ax, box_ax) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        gridspec_kw={"height_ratios": [3, 1.2]},
        constrained_layout=True,
    )

    for index, name in enumerate(plotted_names):
        counts = _word_counts(query_sets[name])
        color = colors.get(name, fallback_colors[index % len(fallback_colors)])
        hist_ax.hist(
            counts,
            bins=bin_edges,
            alpha=0.35,
            edgecolor=color,
            color=color,
            linewidth=1.2,
            label=f"{name} (n={len(counts)})",
        )

    hist_ax.set_title("Query Word Count Distribution")
    hist_ax.set_xlabel("Query word count")
    hist_ax.set_ylabel("Query count")
    hist_ax.legend(frameon=False)
    hist_ax.grid(axis="y", linestyle="--", alpha=0.25)

    boxplot = box_ax.boxplot(
        all_counts,
        vert=False,
        patch_artist=True,
        labels=plotted_names,
        widths=0.55,
    )
    for index, patch in enumerate(boxplot["boxes"]):
        name = plotted_names[index]
        patch.set_facecolor(colors.get(name, fallback_colors[index % len(fallback_colors)]))
        patch.set_alpha(0.45)
    for median in boxplot["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.4)

    box_ax.set_xlabel("Query word count")
    box_ax.set_ylabel("Dataset")
    box_ax.grid(axis="x", linestyle="--", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return summary
