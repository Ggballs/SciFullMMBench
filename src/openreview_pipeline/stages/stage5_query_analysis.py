from __future__ import annotations

import json
import logging
import re
from collections import Counter
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
    QueryHardNegativeContext,
    RetrievalEvaluation,
    RuleBasedStyleEvaluation,
    StyleEvaluation,
)
from openreview_pipeline.schemas.schemas_summarize import SummarizedPapersDataset
from openreview_pipeline.stages.stage4_hard_negative_mining import HardNegativeMiningDataset
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

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        array_match = re.search(r"\[[\s\S]*\]", response)
        if array_match:
            try:
                payload = json.loads(array_match.group())
            except json.JSONDecodeError:
                payload = []
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                return payload[0]

        object_match = re.search(r"\{[\s\S]*\}", response)
        if object_match:
            try:
                payload = json.loads(object_match.group())
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                return payload

        logger.warning("Failed to parse retrieval evaluation response: %s", response[:300])
        return {}

    def evaluate_query(
        self,
        *,
        paper_title: str,
        abstract: str,
        hard_negative_context: Optional[QueryHardNegativeContext],
        query_text: str,
    ) -> RetrievalEvaluation:
        hard_negative_lines = []
        if hard_negative_context:
            for idx, item in enumerate(hard_negative_context.hard_negatives, start=1):
                title = str(item.get("paper_title", "")).strip() or "Untitled"
                abstract_text = str(item.get("abstract", "")).strip() or "No abstract available."
                hard_negative_lines.append(f"{idx}. {title}: {abstract_text}")

        prompt = self._get_prompt_template()
        prompt = prompt.replace("{{paper_title}}", paper_title or "Unknown title")
        prompt = prompt.replace("{{abstract}}", abstract or "N/A")
        prompt = prompt.replace(
            "{{hard_negatives}}",
            "\n".join(hard_negative_lines) if hard_negative_lines else "None provided",
        )
        prompt = prompt.replace("{{queries}}", f"- {query_text}")

        raw = self.llm.generate(prompt)
        parsed = self._parse_llm_response(raw)
        dimensions = parsed.get("dimensions", {}) if isinstance(parsed.get("dimensions"), dict) else {}

        full_paper_reliance = str(dimensions.get("full_paper_reliance", "FAIL")).strip().upper()
        if full_paper_reliance not in {"PASS", "FAIL"}:
            full_paper_reliance = "FAIL"

        false_negative_risk = str(dimensions.get("false_negative_risk", "HIGH")).strip().upper()
        if false_negative_risk not in {"LOW", "HIGH"}:
            false_negative_risk = "HIGH"

        return RetrievalEvaluation(
            full_paper_reliance=full_paper_reliance,
            false_negative_risk=false_negative_risk,
            reasoning=str(parsed.get("reasoning", "")).strip(),
        )


def _decision_for_retrieval(evaluation: RetrievalEvaluation) -> str:
    if evaluation.full_paper_reliance == "FAIL":
        return "Hard Reject"
    if evaluation.false_negative_risk == "HIGH":
        return "Hard Reject"
    return "Keep"


def _openreview_url_for_paper(paper_id: str) -> str:
    return f"https://openreview.net/forum?id={paper_id}"


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


def _hard_negative_map(dataset: Optional[HardNegativeMiningDataset]) -> dict[tuple[str, str, str], QueryHardNegativeContext]:
    if dataset is None:
        return {}

    result = {}
    for item in dataset.results:
        key = (item.paper_id, item.query, item.source_view)
        result[key] = QueryHardNegativeContext(
            hard_negatives=[paper.model_dump(mode="json") for paper in item.hard_negatives],
            positives=[paper.model_dump(mode="json") for paper in item.positives],
            keywords_extracted=list(item.keywords_extracted),
            search_queries_used=list(item.search_queries_used),
            retrieved_candidates=int(item.retrieved_candidates),
            mining_method=item.mining_method,
        )
    return result


def _legacy_hard_negative_match(
    dataset: Optional[HardNegativeMiningDataset],
    query_text: str,
    source_view: str,
) -> Optional[QueryHardNegativeContext]:
    if dataset is None:
        return None

    for item in dataset.results:
        if item.query == query_text and item.source_view == source_view:
            return QueryHardNegativeContext(
                hard_negatives=[paper.model_dump(mode="json") for paper in item.hard_negatives],
                positives=[paper.model_dump(mode="json") for paper in item.positives],
                keywords_extracted=list(item.keywords_extracted),
                search_queries_used=list(item.search_queries_used),
                retrieved_candidates=int(item.retrieved_candidates),
                mining_method=item.mining_method,
            )
    return None


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
                f"- False-negative risk: {retrieval_summary.get('false_negative_risk', {})}",
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
                    f"- False-negative risk: {query.retrieval_evaluation.false_negative_risk}",
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
    hard_negative_dataset: Optional[HardNegativeMiningDataset] = None,
    config_path: Optional[Path | str] = None,
    llm_batch_size: int = 25,
    llm_judge_mode: str = "batch",
    llm_max_concurrency: int = 1,
) -> QueryAnalysisDataset:
    paper_meta = _paper_metadata_map(downloaded_dataset)
    summary_by_id = {item.paper_id: item for item in summarized_dataset.summaries}
    hard_negative_by_query = _hard_negative_map(hard_negative_dataset)

    all_queries = [
        query.query_text
        for paper in queries_dataset.papers_queries
        for query in paper.queries_by_view
    ]
    rule_report = rule_judge.analyze_queries(all_queries)

    if config_path is None:
        raise ValueError("config_path is required for stage-5 LLM-based analysis")
    llm_config = llm_judge.load_llm_config(Path(config_path))
    llm_report = llm_judge.analyze_queries(
        all_queries,
        base_url=llm_config["base_url"],
        api_token=llm_config["api_token"],
        model=llm_config["model"],
        batch_size=llm_batch_size,
        judge_mode=llm_judge_mode,
        max_concurrency=llm_max_concurrency,
    )

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
    false_negative_counts: Counter[str] = Counter()
    query_index = 1

    for paper_queries in queries_dataset.papers_queries:
        summary = summary_by_id.get(paper_queries.paper_id)
        meta = paper_meta.get(paper_queries.paper_id, {})
        paper_title = meta.get("paper_title") or paper_queries.paper_title
        abstract = meta.get("abstract")

        analyzed_queries: list[QueryAnalysisEntry] = []
        for query in paper_queries.queries_by_view:
            hard_negative_context = hard_negative_by_query.get(
                (paper_queries.paper_id, query.query_text, query.source_view)
            ) or _legacy_hard_negative_match(hard_negative_dataset, query.query_text, query.source_view)

            retrieval_evaluation = retrieval_evaluator.evaluate_query(
                paper_title=paper_title,
                abstract=abstract or "",
                hard_negative_context=hard_negative_context,
                query_text=query.query_text,
            )
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
                    source_view=query.source_view,
                    is_multimodal=query.is_multimodal,
                    related_bullet_indice=query.related_bullet_indice,
                    related_bullet_justification=query.related_bullet_justification,
                    hard_negative_context=hard_negative_context,
                    retrieval_evaluation=retrieval_evaluation,
                    style_evaluation=style_evaluation,
                    decision=decision,
                )
            )

            decision_counts[decision] += 1
            full_paper_counts[retrieval_evaluation.full_paper_reliance] += 1
            false_negative_counts[retrieval_evaluation.false_negative_risk] += 1
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
                "false_negative_risk": dict(false_negative_counts),
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
    hard_negatives_path: Optional[Path] = None,
    llm_batch_size: int = 25,
    llm_judge_mode: str = "batch",
    llm_max_concurrency: int = 1,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summarized_dataset = load_json(summarized_path, SummarizedPapersDataset)
    queries_dataset = load_json(queries_path, GeneratedQueriesDataset)
    downloaded_dataset = (
        load_json(downloaded_path, DownloadedPapersDataset)
        if downloaded_path is not None and downloaded_path.is_file()
        else None
    )
    hard_negative_dataset = (
        load_json(hard_negatives_path, HardNegativeMiningDataset)
        if hard_negatives_path is not None and hard_negatives_path.is_file()
        else None
    )

    artifact = apply(
        llm=llm,
        summarized_dataset=summarized_dataset,
        queries_dataset=queries_dataset,
        downloaded_dataset=downloaded_dataset,
        hard_negative_dataset=hard_negative_dataset,
        config_path=config_path,
        llm_batch_size=llm_batch_size,
        llm_judge_mode=llm_judge_mode,
        llm_max_concurrency=llm_max_concurrency,
    )

    json_path = output_dir / "query_analysis.json"
    md_path = output_dir / "query_analysis.md"
    save_json(json_path, artifact)
    md_path.write_text(_render_markdown(artifact), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
