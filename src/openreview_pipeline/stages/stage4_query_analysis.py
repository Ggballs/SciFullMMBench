from __future__ import annotations

import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

from openreview_pipeline.llm import LLMBackend
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
from openreview_pipeline.utils import load_json, load_prompt_template, save_json
from openreview_pipeline.utils.query_analysis import llm_judge, rule_judge

logger = logging.getLogger(__name__)


def _safe_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


class RetrievalEvaluator:
    def __init__(self, llm: LLMBackend, prompt_template: Optional[str] = None):
        self.llm = llm
        self._prompt_template = prompt_template

    def _get_prompt_template(self) -> str:
        if self._prompt_template:
            return self._prompt_template
        prompt_path = (
            Path(__file__).resolve().parents[3]
            / "prompts"
            / "query_analysis"
            / "retrieval_effectiveness.txt"
        )
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

        full_paper_reliance = str(dimensions.get("full_paper_reliance", "FAIL")).strip().upper()
        if full_paper_reliance not in {"PASS", "FAIL"}:
            full_paper_reliance = "FAIL"

        return RetrievalEvaluation(
            full_paper_reliance=full_paper_reliance,
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
    if evaluation.full_paper_reliance == "FAIL":
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
            full_paper_counts[query.retrieval_evaluation.full_paper_reliance] += 1
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
                    f"- Full-paper reliance: {query.retrieval_evaluation.full_paper_reliance}",
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
                    hard_negative_context=None,
                    retrieval_evaluation=retrieval_evaluation,
                    style_evaluation=style_evaluation,
                    decision=decision,
                )
            )

            decision_counts[decision] += 1
            full_paper_counts[retrieval_evaluation.full_paper_reliance] += 1
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
    return {"json": json_path, "markdown": md_path}
