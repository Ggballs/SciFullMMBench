from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from utils.llm import LLMBackend
from openreview_pipeline.schemas.schemas import DownloadedPapersDataset
from openreview_pipeline.schemas.schemas_queries import (
    GeneratedQueriesDataset,
    LLMStyleEvaluation,
    PaperQueryAnalysis,
    QueryAnalysisDataset,
    QueryAnalysisEntry,
    RetrievalEvaluation,
    RuleBasedStyleEvaluation,
    StyleEvaluation,
)
from openreview_pipeline.schemas.schemas_summarize import SummarizedPapersDataset
from utils import load_json, load_prompt_template, save_json
from utils.db.golden_query_embeddings import (
    GOLDEN_QUERY_EMBEDDINGS_TABLE,
    get_engine as get_golden_query_embedding_engine,
)
from evaluations.embedding.embeddings import build_text_embedder
from openreview_pipeline.query_analysis import llm_judge, rule_judge
from utils.project_paths import resolve_prompt_path

logger = logging.getLogger(__name__)

DEFAULT_REFERENCE_STYLE_PATHS = {
    "litsearch": Path("outputs/query_analysis_comparison/03_litsearch_human/style_analysis.json"),
    "pasa": Path("outputs/query_analysis_comparison/02_pasa_realscholar/style_analysis.json"),
}


def _safe_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_repo_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (_repo_root() / candidate).resolve()


def _load_stage4_embedding_config(config_path: Optional[Path | str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if config_path:
        resolved = Path(config_path).expanduser()
        if resolved.exists():
            with resolved.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
                if isinstance(loaded, dict):
                    config = loaded
    stages_config = config.get("stages", {}) if isinstance(config.get("stages"), dict) else {}
    generate_config = stages_config.get("generate_queries", {}) if isinstance(stages_config.get("generate_queries"), dict) else {}
    analysis_config = stages_config.get("query_analysis", {}) if isinstance(stages_config.get("query_analysis"), dict) else {}
    embedding_config = analysis_config.get("embedding_analysis", {}) if isinstance(analysis_config.get("embedding_analysis"), dict) else {}
    reference_paths = embedding_config.get("reference_style_paths", {}) if isinstance(embedding_config.get("reference_style_paths"), dict) else {}
    return {
        "enabled": bool(embedding_config.get("enabled", True)),
        "bge_model_path": str(
            embedding_config.get("bge_model_path")
            or generate_config.get("bge_model_path")
            or "/data3/yangyinghao/bge-m3"
        ),
        "bge_device": str(
            os.environ.get("SCIFULL_QUERY_ANALYSIS_EMBEDDING_DEVICE")
            or embedding_config.get("bge_device")
            or generate_config.get("bge_device")
            or "cuda:2"
        ),
        "embedding_service_url": os.environ.get(
            "SCIFULL_EMBEDDING_SERVICE_URL",
            str(
                embedding_config.get("embedding_service_url")
                or generate_config.get("embedding_service_url")
                or ""
            ),
        ).strip(),
        "embedding_service_timeout": float(
            embedding_config.get(
                "embedding_service_timeout",
                generate_config.get("embedding_service_timeout", 120.0),
            )
        ),
        "golden_embedding_db_url": os.environ.get(
            "SCIFULL_GOLDEN_EMBEDDING_DB_URL",
            str(
                embedding_config.get("golden_embedding_db_url")
                or generate_config.get("golden_embedding_db_url")
                or ""
            ),
        ).strip(),
        "reference_style_paths": {
            name: _resolve_repo_path(reference_paths.get(name) or default_path)
            for name, default_path in DEFAULT_REFERENCE_STYLE_PATHS.items()
        },
    }


def _extract_reference_queries(style_analysis_path: Path) -> list[str]:
    if not style_analysis_path.is_file():
        return []
    try:
        with style_analysis_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        logger.warning("Could not load reference style analysis %s: %s", style_analysis_path, exc)
        return []

    queries: list[str] = []
    semantic_items = (
        data.get("metrics", {})
        .get("combined", {})
        .get("semantic_constraint_analysis", {})
        .get("per_query", [])
    )
    if isinstance(semantic_items, list):
        for item in semantic_items:
            if isinstance(item, dict):
                text = str(item.get("query", "")).strip()
                if text:
                    queries.append(text)
    if not queries:
        for item in data.get("representative_examples", []) or []:
            if isinstance(item, dict):
                text = str(item.get("query", "")).strip()
                if text:
                    queries.append(text)
    return queries


def _parse_vector_text(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    text_value = str(value or "").strip()
    if text_value.startswith("[") and text_value.endswith("]"):
        text_value = text_value[1:-1]
    if not text_value:
        return []
    return [float(item) for item in text_value.split(",") if item.strip()]


def _golden_source_label(query_id: str, example_id: str) -> str:
    text_value = f"{query_id} {example_id}".lower()
    if "pasa" in text_value:
        return "pasa"
    if "litsearch" in text_value:
        return "litsearch"
    return "golden"


def _load_reference_records_from_golden_db(db_url: str) -> tuple[list[dict[str, Any]], list[list[float]]]:
    from sqlalchemy import text

    if not db_url:
        return [], []
    engine = get_golden_query_embedding_engine(db_url)
    stmt = text(
        f"""
        SELECT
          example_id,
          query_id,
          query_type,
          view_label,
          query_text,
          embedding::text AS embedding_text
        FROM {GOLDEN_QUERY_EMBEDDINGS_TABLE}
        WHERE query_type = 'IR'
          AND specific = 1
          AND (
            query_id ILIKE '%pasa%'
            OR example_id ILIKE '%pasa%'
            OR query_id ILIKE '%litsearch%'
            OR example_id ILIKE '%litsearch%'
          )
        ORDER BY query_id, view_label
        """
    )
    records: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []
    with engine.begin() as conn:
        rows = conn.execute(stmt).mappings()
        for row in rows:
            embedding = _parse_vector_text(row["embedding_text"])
            if not embedding:
                continue
            query_id = str(row["query_id"])
            example_id = str(row["example_id"])
            records.append(
                {
                    "dataset": _golden_source_label(query_id, example_id),
                    "paper_id": "",
                    "paper_title": "",
                    "source_view": str(row["view_label"]),
                    "query_type": str(row["query_type"]),
                    "query_id": query_id,
                    "example_id": example_id,
                    "query_text": str(row["query_text"]),
                    "reference_source": "golden_query_embeddings",
                }
            )
            embeddings.append(embedding)
    return records, embeddings


def _tsne_2d(vectors: list[list[float]]) -> list[list[float]]:
    import numpy as np
    from sklearn.manifold import TSNE

    if not vectors:
        return []
    matrix = np.array(vectors, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return []
    if matrix.shape[0] == 1:
        return [[0.0, 0.0]]
    if matrix.shape[0] == 2:
        return [[float(matrix[0, 0]), float(matrix[0, 1])] for _ in range(2)]
    n_components = 2
    perplexity = min(30, matrix.shape[0] - 1)
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(matrix)
    return [[float(x), float(y)] for x, y in coords[:, :2]]


def _centroid_distance_summary(points: list[dict[str, Any]], label_key: str) -> dict[str, Any]:
    import math
    from collections import defaultdict

    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for point in points:
        label = str(point.get(label_key, "") or "unknown")
        if not label or label == "unknown":
            continue
        groups[label].append((float(point.get("x", 0.0)), float(point.get("y", 0.0))))
    centroids = {
        label: (
            sum(x for x, _ in coords) / len(coords),
            sum(y for _, y in coords) / len(coords),
        )
        for label, coords in groups.items()
        if coords
    }
    within = {}
    for label, coords in groups.items():
        cx, cy = centroids[label]
        within[label] = (
            sum(math.dist((x, y), (cx, cy)) for x, y in coords) / len(coords)
            if coords
            else None
        )
    between = {}
    labels = sorted(centroids)
    for idx, left in enumerate(labels):
        for right in labels[idx + 1 :]:
            between[f"{left}__{right}"] = math.dist(centroids[left], centroids[right])
    within_values = [value for value in within.values() if value is not None]
    between_values = list(between.values())
    mean_within = sum(within_values) / len(within_values) if within_values else None
    mean_between = sum(between_values) / len(between_values) if between_values else None
    separation_ratio = (
        float(mean_between / mean_within)
        if mean_between is not None and mean_within not in (None, 0)
        else None
    )
    return {
        "group_counts": {label: len(coords) for label, coords in groups.items()},
        "centroids": {label: [float(x), float(y)] for label, (x, y) in centroids.items()},
        "within_cluster_mean_distance": within,
        "between_centroid_distance": between,
        "mean_within_cluster_distance": mean_within,
        "mean_between_centroid_distance": mean_between,
        "separation_ratio": separation_ratio,
    }


def _plot_embedding_projection(
    points: list[dict[str, Any]],
    *,
    color_key: str,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = [
        "#2563eb",
        "#dc2626",
        "#059669",
        "#7c3aed",
        "#d97706",
        "#0891b2",
        "#be123c",
        "#4b5563",
    ]
    labels = []
    for point in points:
        label = str(point.get(color_key, "") or "unknown")
        if label not in labels:
            labels.append(label)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for idx, label in enumerate(labels):
        subset = [point for point in points if str(point.get(color_key, "") or "unknown") == label]
        ax.scatter(
            [point["x"] for point in subset],
            [point["y"] for point in subset],
            s=46 if label == "generated" else 28,
            alpha=0.82 if label == "generated" else 0.55,
            color=colors[idx % len(colors)],
            label=f"{label} (n={len(subset)})",
            edgecolors="white",
            linewidths=0.4,
        )
    ax.axhline(0, color="#e5e7eb", linewidth=0.8)
    ax.axvline(0, color="#e5e7eb", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_query_embedding_analysis(
    *,
    artifact: QueryAnalysisDataset,
    output_dir: Path,
    config_path: Optional[Path | str],
) -> Optional[Path]:
    settings = _load_stage4_embedding_config(config_path)
    if not settings["enabled"]:
        return None

    generated_records = []
    for paper in artifact.papers:
        for query in paper.queries:
            generated_records.append(
                {
                    "dataset": "generated",
                    "paper_id": paper.paper_id,
                    "paper_title": paper.paper_title,
                    "source_view": query.source_view or "unknown",
                    "query_text": query.query_text,
                }
            )

    reference_records: list[dict[str, Any]] = []
    reference_embeddings: list[list[float]] = []
    try:
        reference_records, reference_embeddings = _load_reference_records_from_golden_db(
            str(settings["golden_embedding_db_url"] or "")
        )
    except Exception as exc:
        logger.warning(
            "Could not load golden reference embeddings from PostgreSQL; "
            "falling back to reference style JSON without view labels: %s",
            exc,
        )
    if not reference_records:
        for dataset_name, style_path in settings["reference_style_paths"].items():
            for query_text in _extract_reference_queries(Path(style_path)):
                reference_records.append(
                    {
                        "dataset": dataset_name,
                        "paper_id": "",
                        "paper_title": "",
                        "source_view": "",
                        "query_text": query_text,
                        "reference_source": "style_analysis_json",
                    }
                )

    records = generated_records + reference_records
    if len(records) < 2:
        logger.warning("Skipping query embedding analysis: only %s query records available.", len(records))
        return None

    texts = [record["query_text"] for record in generated_records]
    fallback_reference_texts = (
        [record["query_text"] for record in reference_records]
        if not reference_embeddings
        else []
    )
    batch_size = 32

    def _embed_with_device(device: str, input_texts: list[str]) -> list[list[float]]:
        embedder = build_text_embedder(
            model_path=str(settings["bge_model_path"]),
            device=device,
            service_url=str(settings["embedding_service_url"] or "") or None,
            timeout_seconds=float(settings["embedding_service_timeout"]),
        )
        out: list[list[float]] = []
        for start in range(0, len(input_texts), batch_size):
            out.extend(embedder.embed_texts(input_texts[start : start + batch_size]))
        return out

    embedding_device = str(settings["bge_device"])
    try:
        generated_embeddings = _embed_with_device(embedding_device, texts)
        fallback_reference_embeddings = (
            _embed_with_device(embedding_device, fallback_reference_texts)
            if fallback_reference_texts
            else []
        )
    except Exception as exc:
        if settings["embedding_service_url"] or embedding_device.lower() == "cpu":
            raise
        logger.warning(
            "Embedding on device %s failed (%s); retrying query embedding analysis on CPU.",
            embedding_device,
            exc,
        )
        embedding_device = "cpu"
        generated_embeddings = _embed_with_device(embedding_device, texts)
        fallback_reference_embeddings = (
            _embed_with_device(embedding_device, fallback_reference_texts)
            if fallback_reference_texts
            else []
        )

    embeddings = generated_embeddings + (reference_embeddings or fallback_reference_embeddings)

    if len(embeddings) != len(records):
        raise ValueError(
            f"Expected {len(records)} query embeddings, received {len(embeddings)}."
        )

    coords = _tsne_2d(embeddings)
    for record, coord in zip(records, coords):
        record["x"], record["y"] = coord

    dataset_plot = output_dir / "query_embedding_dataset_projection.png"
    view_plot = output_dir / "query_embedding_view_projection.png"
    _plot_embedding_projection(
        records,
        color_key="dataset",
        title="Query Embedding Space: Generated vs LitSearch vs PASA",
        output_path=dataset_plot,
    )
    _plot_embedding_projection(
        [record for record in records if record.get("source_view")],
        color_key="source_view",
        title="Query Embedding Space by View: Generated + LitSearch + PASA",
        output_path=view_plot,
    )

    dataset_summary = _centroid_distance_summary(records, "dataset")
    view_summary = _centroid_distance_summary(
        [record for record in records if record.get("source_view")],
        "source_view",
    )
    analysis = {
        "method": "BGE embeddings + t-SNE projection",
        "embedding_model": str(settings["bge_model_path"]),
        "embedding_device": embedding_device,
        "embedding_dimension": len(embeddings[0]) if embeddings else 0,
        "record_counts": dict(Counter(record["dataset"] for record in records)),
        "view_counts": dict(
            Counter(record["source_view"] for record in records if record.get("source_view"))
        ),
        "reference_source": (
            "golden_query_embeddings" if reference_embeddings else "style_analysis_json"
        ),
        "plots": {
            "dataset_projection": str(dataset_plot),
            "view_projection": str(view_plot),
        },
        "datasets": dataset_summary,
        "views": view_summary,
        "records": [
            {key: value for key, value in record.items() if key not in {"paper_title"}}
            for record in records
        ],
    }
    output_path = output_dir / "query_embedding_analysis.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, ensure_ascii=False)
    return output_path


class RetrievalEvaluator:
    def __init__(self, llm: LLMBackend, prompt_template: Optional[str] = None):
        self.llm = llm
        self._prompt_template = prompt_template

    def _get_prompt_template(self) -> str:
        if self._prompt_template:
            return self._prompt_template
        prompt_path = resolve_prompt_path("query_analysis", "retrieval_effectiveness.txt")
        return load_prompt_template(prompt_path)

    def _parse_llm_response(self, response: str) -> list[dict[str, Any]]:
        array_match = re.search(r"\[[\s\S]*\]", response)
        if array_match:
            try:
                payload = json.loads(array_match.group())
            except json.JSONDecodeError:
                payload = []
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]

        object_match = re.search(r"\{[\s\S]*\}", response)
        if object_match:
            try:
                payload = json.loads(object_match.group())
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                return [payload]

        logger.warning("Failed to parse retrieval evaluation response: %s", response[:300])
        return []

    def _to_retrieval_evaluation(self, parsed: dict[str, Any]) -> RetrievalEvaluation:
        dimensions = parsed.get("dimensions", {}) if isinstance(parsed.get("dimensions"), dict) else {}

        abstract_relevance = str(dimensions.get("abstract_relevance", "HIGH-LEXICAL-OVERLAP")).strip().upper()
        if abstract_relevance not in {"PASS", "FAIL", "HIGH-LEXICAL-OVERLAP", "LOW-LEXICAL-OVERLAP", "LOW-SEMANTIC-OVERLAP"}:
            abstract_relevance = "HIGH-LEXICAL-OVERLAP"

        return RetrievalEvaluation(
            abstract_relevance=abstract_relevance,
            false_negative_risk=None,
            reasoning=str(parsed.get("reasoning", "")).strip(),
        )

    def evaluate_queries(
        self,
        *,
        paper_title: str,
        abstract: str,
        query_texts: list[str],
    ) -> list[RetrievalEvaluation]:
        if not query_texts:
            return []

        prompt_template = self._get_prompt_template()
        numbered_queries = "\n".join(
            f"{index}. {query_text}"
            for index, query_text in enumerate(query_texts, start=1)
        )
        logger.info(
            "Retrieval evaluation scoring %s queries for %s in one paper-level request",
            len(query_texts),
            paper_title or "unknown paper",
        )
        prompt = prompt_template
        prompt = prompt.replace("{{paper_title}}", paper_title or "Unknown title")
        prompt = prompt.replace("{{abstract}}", abstract or "N/A")
        prompt = prompt.replace("{{queries}}", numbered_queries)

        raw = self.llm.generate(prompt)
        parsed_items = self._parse_llm_response(raw)
        return [
            self._to_retrieval_evaluation(parsed_items[offset] if offset < len(parsed_items) else {})
            for offset in range(len(query_texts))
        ]

    def evaluate_query(
        self,
        *,
        paper_title: str,
        abstract: str,
        query_text: str,
    ) -> RetrievalEvaluation:
        return self.evaluate_queries(
            paper_title=paper_title,
            abstract=abstract,
            query_texts=[query_text],
        )[0]


def _decision_for_retrieval(evaluation: RetrievalEvaluation) -> str:
    if evaluation.abstract_relevance == "HIGH-LEXICAL-OVERLAP":
        return "Hard Reject"
    return "Keep"


def _openreview_url_for_paper(paper_id: str) -> str:
    return f"https://openreview.net/forum?id={paper_id}"


def _query_key(paper_id: str, query: Any) -> tuple[str, str, str, str]:
    return (
        str(paper_id),
        str(getattr(query, "query_text", "")),
        str(getattr(query, "source_view", "")),
        str(getattr(query, "query_type", "IR")),
    )


def _analyzed_query_key(paper_id: str, query: QueryAnalysisEntry) -> tuple[str, str, str, str]:
    return (str(paper_id), str(query.query_text), str(query.source_view), str(query.query_type))


def _completed_analysis_paper_ids(
    existing: QueryAnalysisDataset,
    queries_dataset: GeneratedQueriesDataset,
) -> set[str]:
    expected_by_paper = {
        paper.paper_id: {
            _query_key(paper.paper_id, query)
            for query in paper.queries_by_view
        }
        for paper in queries_dataset.papers_queries
    }
    completed = set()
    for paper in existing.papers:
        expected = expected_by_paper.get(paper.paper_id)
        if not expected:
            continue
        actual = {_analyzed_query_key(paper.paper_id, query) for query in paper.queries}
        if expected and expected.issubset(actual):
            completed.add(paper.paper_id)
    return completed


def _paper_metadata_map(downloaded_dataset: Optional[DownloadedPapersDataset]) -> dict[str, dict[str, Any]]:
    if downloaded_dataset is None:
        return {}

    metadata = {}
    for item in downloaded_dataset.papers:
        paper = item.paper
        metadata[paper.id] = {
            "paper_title": paper.title,
            "abstract": paper.abstract,
            "pdf_url": paper.pdf_url,
            "openreview_url": _openreview_url_for_paper(paper.id),
            "venue": paper.venue,
            "year": paper.year,
            "authors": list(paper.authors),
        }
    return metadata


def _refresh_analysis_paper_metadata(
    paper: PaperQueryAnalysis,
    *,
    metadata: dict[str, Any],
    summary: Optional[Any],
) -> PaperQueryAnalysis:
    if metadata:
        paper.paper_title = metadata.get("paper_title") or paper.paper_title
        paper.abstract = metadata.get("abstract") or paper.abstract
        paper.pdf_url = metadata.get("pdf_url") or paper.pdf_url
        paper.openreview_url = metadata.get("openreview_url") or paper.openreview_url
        paper.venue = metadata.get("venue") or paper.venue
        paper.year = metadata.get("year") or paper.year
        paper.authors = list(metadata.get("authors", paper.authors))
    if summary is not None:
        paper.summary_views = list(summary.views)
    return paper


def _build_dataset_summary_from_papers(papers: list[PaperQueryAnalysis]) -> dict[str, Any]:
    decision_counts: Counter[str] = Counter()
    full_paper_counts: Counter[str] = Counter()
    token_lengths = []
    char_lengths = []
    templates: Counter[str] = Counter()
    specificity_scores = []
    lexical_scores = []
    semantic_counts = []

    for paper in papers:
        for query in paper.queries:
            decision_counts[query.decision] += 1
            full_paper_counts[query.retrieval_evaluation.abstract_relevance] += 1
            token_lengths.append(query.style_evaluation.rule_based.token_length)
            char_lengths.append(query.style_evaluation.rule_based.char_length)
            templates[query.style_evaluation.rule_based.question_template] += 1
            if query.style_evaluation.llm_based.specificity_calibration_score is not None:
                specificity_scores.append(float(query.style_evaluation.llm_based.specificity_calibration_score))
            if query.style_evaluation.llm_based.lexical_naturalism_score is not None:
                lexical_scores.append(float(query.style_evaluation.llm_based.lexical_naturalism_score))
            semantic_counts.append(float(query.style_evaluation.llm_based.semantic_constraint_count))

    return {
        "retrieval_summary": {
            "full_paper_reliance": dict(full_paper_counts),
        },
        "style_summary": {
            "char_length": {
                "mean": _safe_mean([float(value) for value in char_lengths]),
                "min": min(char_lengths) if char_lengths else None,
                "max": max(char_lengths) if char_lengths else None,
            },
            "token_length": {
                "mean": _safe_mean([float(value) for value in token_lengths]),
                "min": min(token_lengths) if token_lengths else None,
                "max": max(token_lengths) if token_lengths else None,
            },
            "question_templates": dict(templates),
            "specificity_calibration_mean": _safe_mean(specificity_scores),
            "lexical_naturalism_mean": _safe_mean(lexical_scores),
            "semantic_constraint_count_mean": _safe_mean(semantic_counts),
        },
        "decision_counts": dict(decision_counts),
    }


def _render_markdown(report: QueryAnalysisDataset) -> str:
    lines = [
        "# Query Analysis",
        "",
        f"- Total papers: {report.total_papers}",
        f"- Total queries: {report.total_queries}",
    ]

    decision_counts = report.dataset_summary.get("decision_counts", {})
    if decision_counts:
        lines.append(f"- Decisions: {decision_counts}")

    retrieval_summary = report.dataset_summary.get("retrieval_summary", {})
    if retrieval_summary:
        lines.extend(
            [
                "",
                "## Retrieval Summary",
                f"- Full-paper reliance: {retrieval_summary.get('full_paper_reliance', {})}",
            ]
        )

    style_summary = report.dataset_summary.get("style_summary", {})
    if style_summary:
        lines.extend(
            [
                "",
                "## Style Summary",
                f"- Token length stats: {style_summary.get('token_length', {})}",
                f"- Template stats: {style_summary.get('question_templates', {})}",
                f"- Specificity mean: {style_summary.get('specificity_calibration_mean')}",
                f"- Lexical naturalism mean: {style_summary.get('lexical_naturalism_mean')}",
                f"- Semantic constraint mean: {style_summary.get('semantic_constraint_count_mean')}",
            ]
        )

    for paper in report.papers:
        lines.extend(
            [
                "",
                f"## {paper.paper_title}",
                f"- Paper ID: {paper.paper_id}",
                f"- Venue: {paper.venue or 'N/A'} {paper.year or ''}".rstrip(),
                f"- OpenReview: {paper.openreview_url or 'N/A'}",
                f"- Queries: {len(paper.queries)}",
            ]
        )
        for query in paper.queries:
            lines.extend(
                [
                    "",
                    f"### {query.query_text}",
                    f"- Decision: {query.decision}",
                    f"- Source view: {query.source_view}",
                    f"- Full-paper reliance: {query.retrieval_evaluation.abstract_relevance}",
                    f"- Specificity calibration: {query.style_evaluation.llm_based.specificity_calibration_score}",
                    f"- Lexical naturalism: {query.style_evaluation.llm_based.lexical_naturalism_score}",
                    f"- Semantic constraint count: {query.style_evaluation.llm_based.semantic_constraint_count}",
                    f"- Template: {query.style_evaluation.rule_based.question_template}",
                ]
            )
    return "\n".join(lines).strip() + "\n"


def apply(
    *,
    llm: LLMBackend,
    summarized_dataset: SummarizedPapersDataset,
    queries_dataset: GeneratedQueriesDataset,
    downloaded_dataset: Optional[DownloadedPapersDataset] = None,
    config_path: Optional[Path | str] = None,
) -> QueryAnalysisDataset:
    paper_meta = _paper_metadata_map(downloaded_dataset)
    summary_by_id = {item.paper_id: item for item in summarized_dataset.summaries}

    all_queries = [
        query.query_text
        for paper in queries_dataset.papers_queries
        for query in paper.queries_by_view
    ]
    rule_report = rule_judge.analyze_queries(all_queries)

    llm_report = llm_judge.analyze_queries(all_queries, llm=llm)

    rule_per_query = {
        item["index"]: item
        for item in rule_report.get("per_query", [])
        if isinstance(item, dict)
    }
    llm_per_query = {
        item["index"]: item
        for item in llm_report.get("llm_judge", {}).get("per_query", [])
        if isinstance(item, dict)
    }
    semantic_per_query = {
        item["index"]: item
        for item in llm_report.get("semantic_constraint_analysis", {}).get("per_query", [])
        if isinstance(item, dict)
    }

    retrieval_evaluator = RetrievalEvaluator(llm=llm)
    papers: list[PaperQueryAnalysis] = []
    decision_counts: Counter[str] = Counter()
    full_paper_counts: Counter[str] = Counter()
    query_index = 1

    for paper_queries in queries_dataset.papers_queries:
        summary = summary_by_id.get(paper_queries.paper_id)
        meta = paper_meta.get(paper_queries.paper_id, {})
        paper_title = meta.get("paper_title") or paper_queries.paper_title
        abstract = meta.get("abstract")

        analyzed_queries: list[QueryAnalysisEntry] = []
        query_entries = list(paper_queries.queries_by_view)
        retrieval_evaluations = retrieval_evaluator.evaluate_queries(
            paper_title=paper_title,
            abstract=abstract or "",
            query_texts=[query.query_text for query in query_entries],
        )

        for query, retrieval_evaluation in zip(query_entries, retrieval_evaluations):
            decision = _decision_for_retrieval(retrieval_evaluation)

            rule_item = rule_per_query.get(query_index, {})
            llm_item = llm_per_query.get(query_index, {})
            semantic_item = semantic_per_query.get(query_index, {})

            style_evaluation = StyleEvaluation(
                rule_based=RuleBasedStyleEvaluation(
                    char_length=int(rule_item.get("char_length", len(query.query_text))),
                    token_length=int(rule_item.get("token_length", len(query.query_text.split()))),
                    question_template=str(rule_item.get("template", "other")),
                    matched_pattern=str(rule_item.get("matched_pattern", "other")),
                    matched_template=bool(rule_item.get("matched_template", False)),
                ),
                llm_based=LLMStyleEvaluation(
                    specificity_calibration_score=llm_item.get("specificity_calibration_score"),
                    specificity_calibration_rationale=str(
                        llm_item.get("specificity_calibration_rationale", "")
                    ).strip(),
                    lexical_naturalism_score=llm_item.get("lexical_naturalism_score"),
                    lexical_naturalism_rationale=str(
                        llm_item.get("lexical_naturalism_rationale", "")
                    ).strip(),
                    semantic_constraint_count=int(semantic_item.get("semantic_constraint_count", 0)),
                    semantic_constraint_rationale=str(
                        semantic_item.get("semantic_constraint_rationale", "")
                    ).strip(),
                ),
            )

            analyzed_queries.append(
                QueryAnalysisEntry(
                    query_text=query.query_text,
                    query_type=query.query_type,
                    source_view=query.source_view,
                    is_multimodal=query.is_multimodal,
                    related_bullet_indice=query.related_bullet_indice,
                    related_bullet_justification=query.related_bullet_justification,
                    multimodal_rationale=query.multimodal_rationale,
                    hard_negative_context=None,
                    retrieval_evaluation=retrieval_evaluation,
                    style_evaluation=style_evaluation,
                    decision=decision,
                )
            )

            decision_counts[decision] += 1
            full_paper_counts[retrieval_evaluation.abstract_relevance] += 1
            query_index += 1

        papers.append(
            PaperQueryAnalysis(
                paper_id=paper_queries.paper_id,
                paper_title=paper_title,
                abstract=abstract,
                pdf_url=meta.get("pdf_url"),
                openreview_url=meta.get("openreview_url") or _openreview_url_for_paper(paper_queries.paper_id),
                venue=meta.get("venue"),
                year=meta.get("year"),
                authors=list(meta.get("authors", [])),
                summary_views=list(summary.views) if summary else [],
                queries=analyzed_queries,
            )
        )

    specificity_scores = [
        query.style_evaluation.llm_based.specificity_calibration_score
        for paper in papers
        for query in paper.queries
        if query.style_evaluation.llm_based.specificity_calibration_score is not None
    ]
    lexical_scores = [
        query.style_evaluation.llm_based.lexical_naturalism_score
        for paper in papers
        for query in paper.queries
        if query.style_evaluation.llm_based.lexical_naturalism_score is not None
    ]
    semantic_counts = [
        query.style_evaluation.llm_based.semantic_constraint_count
        for paper in papers
        for query in paper.queries
    ]

    return QueryAnalysisDataset(
        papers=papers,
        total_papers=len(papers),
        total_queries=sum(len(paper.queries) for paper in papers),
        dataset_summary={
            "retrieval_summary": {
                "full_paper_reliance": dict(full_paper_counts),
            },
            "style_summary": {
                "char_length": rule_report.get("length_stats", {}).get("char_length", {}),
                "token_length": rule_report.get("length_stats", {}).get("token_length", {}),
                "question_templates": rule_report.get("question_templates", {}),
                "representative_examples": rule_report.get("representative_examples", []),
                "qualitative_metrics": llm_report.get("qualitative_metrics", {}),
                "semantic_constraint_analysis": llm_report.get("semantic_constraint_analysis", {}),
                "specificity_calibration_mean": _safe_mean(
                    [float(score) for score in specificity_scores if score is not None]
                ),
                "lexical_naturalism_mean": _safe_mean(
                    [float(score) for score in lexical_scores if score is not None]
                ),
                "semantic_constraint_count_mean": _safe_mean(
                    [float(score) for score in semantic_counts]
                ),
            },
            "decision_counts": dict(decision_counts),
        },
    )


def run(
    *,
    llm: LLMBackend,
    summarized_path: Path,
    queries_path: Path,
    output_dir: Path,
    config_path: Path | str,
    downloaded_path: Optional[Path] = None,
    max_concurrent_papers: int = 1,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summarized_dataset = load_json(summarized_path, SummarizedPapersDataset)
    queries_dataset = load_json(queries_path, GeneratedQueriesDataset)
    downloaded_dataset = (
        load_json(downloaded_path, DownloadedPapersDataset)
        if downloaded_path is not None and downloaded_path.is_file()
        else None
    )

    json_path = output_dir / "query_analysis.json"
    md_path = output_dir / "query_analysis.md"

    existing_papers: list[PaperQueryAnalysis] = []
    completed_paper_ids: set[str] = set()
    paper_meta = _paper_metadata_map(downloaded_dataset)
    summary_by_id = {summary.paper_id: summary for summary in summarized_dataset.summaries}
    if json_path.exists():
        try:
            existing_artifact = load_json(json_path, QueryAnalysisDataset)
            completed_paper_ids = _completed_analysis_paper_ids(existing_artifact, queries_dataset)
            existing_papers = [
                _refresh_analysis_paper_metadata(
                    paper,
                    metadata=paper_meta.get(paper.paper_id, {}),
                    summary=summary_by_id.get(paper.paper_id),
                )
                for paper in existing_artifact.papers
                if paper.paper_id in completed_paper_ids
            ]
            if completed_paper_ids:
                logger.info(
                    "Loaded %s existing query-analysis paper results from %s",
                    len(completed_paper_ids),
                    json_path,
                )
        except Exception as exc:
            logger.warning("Could not load query-analysis checkpoint %s: %s", json_path, exc)

    missing_query_papers = [
        paper for paper in queries_dataset.papers_queries if paper.paper_id not in completed_paper_ids
    ]
    new_papers_by_id = {}
    if missing_query_papers:
        existing_papers_by_id = {paper.paper_id: paper for paper in existing_papers}

        def _analyze_paper(missing_query_paper) -> list[PaperQueryAnalysis]:
            summary = summary_by_id.get(missing_query_paper.paper_id)
            missing_summaries = [summary] if summary else []
            paper_artifact = apply(
                llm=llm,
                summarized_dataset=SummarizedPapersDataset(
                    summaries=missing_summaries,
                    total_papers=len(missing_summaries),
                ),
                queries_dataset=GeneratedQueriesDataset(
                    papers_queries=[missing_query_paper],
                    total_papers=1,
                    total_queries=len(missing_query_paper.queries_by_view),
                ),
                downloaded_dataset=downloaded_dataset,
                config_path=config_path,
            )
            return paper_artifact.papers

        max_workers = min(max(1, int(max_concurrent_papers)), len(missing_query_papers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_analyze_paper, missing_query_paper): missing_query_paper
                for missing_query_paper in missing_query_papers
            }
            for future in as_completed(futures):
                for paper in future.result():
                    new_papers_by_id[paper.paper_id] = paper

                checkpoint_papers = []
                for paper_queries in queries_dataset.papers_queries:
                    paper = existing_papers_by_id.get(paper_queries.paper_id) or new_papers_by_id.get(paper_queries.paper_id)
                    if paper is not None:
                        checkpoint_papers.append(paper)
                checkpoint_artifact = QueryAnalysisDataset(
                    papers=checkpoint_papers,
                    total_papers=len(checkpoint_papers),
                    total_queries=sum(len(paper.queries) for paper in checkpoint_papers),
                    dataset_summary=_build_dataset_summary_from_papers(checkpoint_papers),
                )
                save_json(json_path, checkpoint_artifact)
                md_path.write_text(_render_markdown(checkpoint_artifact), encoding="utf-8")
    else:
        logger.info("Query-analysis stage found no missing papers to process.")

    existing_papers_by_id = {paper.paper_id: paper for paper in existing_papers}
    combined_papers = []
    for paper_queries in queries_dataset.papers_queries:
        paper = existing_papers_by_id.get(paper_queries.paper_id) or new_papers_by_id.get(paper_queries.paper_id)
        if paper is not None:
            combined_papers.append(paper)

    artifact = QueryAnalysisDataset(
        papers=combined_papers,
        total_papers=len(combined_papers),
        total_queries=sum(len(paper.queries) for paper in combined_papers),
        dataset_summary=_build_dataset_summary_from_papers(combined_papers),
    )
    save_json(json_path, artifact)
    md_path.write_text(_render_markdown(artifact), encoding="utf-8")
    logger.info(
        "Query-analysis stage success: %s/%s papers (%.1f%%), %s/%s queries (%.1f%%).",
        artifact.total_papers,
        queries_dataset.total_papers,
        (artifact.total_papers / queries_dataset.total_papers * 100) if queries_dataset.total_papers else 100.0,
        artifact.total_queries,
        queries_dataset.total_queries,
        (artifact.total_queries / queries_dataset.total_queries * 100) if queries_dataset.total_queries else 100.0,
    )
    paths: dict[str, Path] = {"json": json_path, "markdown": md_path}
    try:
        embedding_path = write_query_embedding_analysis(
            artifact=artifact,
            output_dir=output_dir,
            config_path=config_path,
        )
        if embedding_path is not None:
            paths["embedding_analysis"] = embedding_path
            paths["embedding_dataset_projection"] = output_dir / "query_embedding_dataset_projection.png"
            paths["embedding_view_projection"] = output_dir / "query_embedding_view_projection.png"
            logger.info("Query embedding analysis written to %s", embedding_path)
    except Exception as exc:
        logger.warning("Query embedding analysis failed; Stage 4 JSON/MD remain valid: %s", exc)
    return paths
