from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from openreview_pipeline.schemas.schemas_queries import (
    GeneratedQueriesDataset,
    GeneratedQueriesForPaper,
    RetrievalQuery,
)
from openreview_pipeline.schemas.schemas_summarize import SummarizedPapersDataset
from utils import load_json, load_prompt_template, save_json
from utils.llm import LLMBackend
from utils.project_paths import resolve_prompt_path

logger = logging.getLogger(__name__)

PROMPT_PATH = resolve_prompt_path("decontextualize_query.txt")
MAX_ABSTRACT_CHARS = 3500


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split()).strip()


class QueryDecontextualizer:
    def __init__(
        self,
        llm: LLMBackend,
        *,
        prompt_template: Optional[str] = None,
        max_concurrent_papers: int = 1,
    ):
        self.llm = llm
        self.prompt_template = prompt_template or load_prompt_template(PROMPT_PATH)
        self.max_concurrent_papers = max(1, int(max_concurrent_papers))

    def _build_prompt(
        self,
        *,
        paper_title: str,
        paper_abstract: str,
        source_view: str,
        original_query: str,
    ) -> str:
        abstract_text = _normalize_whitespace(paper_abstract)
        if len(abstract_text) > MAX_ABSTRACT_CHARS:
            abstract_text = abstract_text[: MAX_ABSTRACT_CHARS - 3].rstrip() + "..."
        return self.prompt_template.format(
            paper_title=_normalize_whitespace(paper_title) or "(missing title)",
            paper_abstract=abstract_text or "(missing abstract)",
            source_view=_normalize_whitespace(source_view) or "unknown",
            original_query=_normalize_whitespace(original_query),
        )

    def _rewrite_query(
        self,
        *,
        paper_title: str,
        paper_abstract: str,
        query: RetrievalQuery,
    ) -> RetrievalQuery:
        original_query = query.original_query_text or query.query_text
        prompt = self._build_prompt(
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            source_view=query.source_view,
            original_query=original_query,
        )
        raw = self.llm.generate_json(prompt)
        rewritten_query = _normalize_whitespace(str(raw.get("decontextualized_query") or ""))
        if not rewritten_query:
            raise ValueError(f"Missing decontextualized_query for source query: {original_query}")
        rationale = _normalize_whitespace(str(raw.get("rewrite_rationale") or ""))
        return query.model_copy(
            update={
                "query_text": rewritten_query,
                "original_query_text": original_query,
                "decontextualization_note": rationale or None,
            }
        )

    def _rewrite_paper(
        self,
        paper: GeneratedQueriesForPaper,
        summary_by_paper_id: dict[str, object],
    ) -> GeneratedQueriesForPaper:
        summary = summary_by_paper_id.get(paper.paper_id)
        if summary is None:
            raise KeyError(f"Missing summarized paper context for paper_id={paper.paper_id}")
        rewritten_queries = [
            self._rewrite_query(
                paper_title=str(getattr(summary, "paper_title", "") or paper.paper_title),
                paper_abstract=str(getattr(summary, "abstract", "") or ""),
                query=query,
            )
            for query in paper.queries_by_view
        ]
        return GeneratedQueriesForPaper(
            paper_id=paper.paper_id,
            paper_title=paper.paper_title,
            queries_by_view=rewritten_queries,
            generated_at=paper.generated_at,
        )

    def apply(
        self,
        queries_dataset: GeneratedQueriesDataset,
        summarized_dataset: SummarizedPapersDataset,
        checkpoint_path: Optional[Path] = None,
    ) -> GeneratedQueriesDataset:
        logger.info("Decontextualizing queries for %s papers", queries_dataset.total_papers)
        summary_by_paper_id = {summary.paper_id: summary for summary in summarized_dataset.summaries}

        papers_queries = []
        completed_ids = set()
        target_ids = {paper.paper_id for paper in queries_dataset.papers_queries}
        if checkpoint_path and checkpoint_path.exists():
            try:
                checkpoint = load_json(checkpoint_path, GeneratedQueriesDataset)
                papers_queries = [
                    paper for paper in checkpoint.papers_queries if paper.paper_id in target_ids
                ]
                completed_ids = {paper.paper_id for paper in papers_queries}
                if completed_ids:
                    logger.info(
                        "Loaded %s existing decontextualized query results from %s",
                        len(completed_ids),
                        checkpoint_path,
                    )
            except Exception as exc:
                logger.warning(
                    "Could not load decontextualization checkpoint %s: %s",
                    checkpoint_path,
                    exc,
                )

        work_items = [paper for paper in queries_dataset.papers_queries if paper.paper_id not in completed_ids]
        max_workers = min(self.max_concurrent_papers, len(work_items)) if work_items else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._rewrite_paper, paper, summary_by_paper_id): paper
                for paper in work_items
            }
            for future in as_completed(futures):
                rewritten_paper = future.result()
                papers_queries.append(rewritten_paper)
                completed_ids.add(rewritten_paper.paper_id)
                if checkpoint_path:
                    save_json(
                        checkpoint_path,
                        GeneratedQueriesDataset(
                            papers_queries=papers_queries,
                            total_papers=len(papers_queries),
                            total_queries=sum(len(item.queries_by_view) for item in papers_queries),
                        ),
                    )

        papers_by_id = {paper.paper_id: paper for paper in papers_queries}
        ordered_papers = [
            papers_by_id[paper.paper_id]
            for paper in queries_dataset.papers_queries
            if paper.paper_id in papers_by_id
        ]
        result = GeneratedQueriesDataset(
            papers_queries=ordered_papers,
            total_papers=len(ordered_papers),
            total_queries=sum(len(paper.queries_by_view) for paper in ordered_papers),
        )
        logger.info(
            "Decontextualize-queries stage success: %s/%s papers, %s queries.",
            len(ordered_papers),
            queries_dataset.total_papers,
            result.total_queries,
        )
        return result

    def run(self, summarized_path: Path, queries_path: Path, output_path: Path) -> None:
        logger.info(
            "Running decontextualize-queries stage: %s + %s -> %s",
            summarized_path,
            queries_path,
            output_path,
        )
        summarized_dataset = load_json(summarized_path, SummarizedPapersDataset)
        queries_dataset = load_json(queries_path, GeneratedQueriesDataset)
        result = self.apply(queries_dataset, summarized_dataset, checkpoint_path=output_path)
        save_json(output_path, result)
