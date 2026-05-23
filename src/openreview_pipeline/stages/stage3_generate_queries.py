from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from tqdm.auto import tqdm

from openreview_pipeline.llm import LLMBackend
from openreview_pipeline.schemas.schemas_summarize import SummarizedPapersDataset
from openreview_pipeline.schemas.schemas_queries import (
    GeneratedQueriesDataset,
    GeneratedQueriesForPaper,
    RetrievalQuery,
)
from openreview_pipeline.utils import load_json, load_prompt_template, save_json
from openreview_pipeline.utils.db.golden_query_embeddings import (
    GoldenQueryExample,
    get_engine,
    retrieve_golden_query_examples,
)
from openreview_pipeline.utils.embeddings import TextEmbedder, build_text_embedder
from openreview_pipeline.utils.golden_retrieval_icl import normalize_view_label

logger = logging.getLogger(__name__)

QUERY_TYPES = ("IR", "QA")
VIEW_DEFINITIONS = {
    "motivation": "research problem, need, gap, goal, hypothesis, or why the work matters",
    "method": "proposed approach, model, algorithm, system, dataset construction process, or implementation design",
    "experiment/result": "evaluation setup, benchmark, test data, metric, baseline, ablation, empirical finding, comparison, result, or observed limitation",
}


class GoldenExampleRetriever(Protocol):
    def retrieve(
        self,
        *,
        query_type: str,
        view_label: str,
        embedding: list[float],
        limit: int,
        exclude_litsearch: bool = True,
    ) -> list[GoldenQueryExample]:
        ...


@dataclass(frozen=True)
class BulletContext:
    index: int
    text: str
    multimodal_ref: list[str]


@dataclass(frozen=True)
class RetrievedPromptContext:
    bullets: list[BulletContext]
    examples: list[GoldenQueryExample]


class PostgresGoldenExampleRetriever:
    def __init__(self, db_url: str):
        self.engine = get_engine(db_url)

    def retrieve(
        self,
        *,
        query_type: str,
        view_label: str,
        embedding: list[float],
        limit: int,
        exclude_litsearch: bool = True,
    ) -> list[GoldenQueryExample]:
        return retrieve_golden_query_examples(
            self.engine,
            query_type=query_type,
            view_label=view_label,
            embedding=embedding,
            limit=limit,
            exclude_litsearch=exclude_litsearch,
        )


class QueryGenerator:
    def __init__(
        self,
        llm: LLMBackend,
        prompt_template: Optional[str] = None,
        max_concurrent_papers: int = 1,
        *,
        example_retriever: Optional[GoldenExampleRetriever] = None,
        embedder: Optional[TextEmbedder] = None,
        golden_embedding_db_url: Optional[str] = None,
        golden_examples_k: int = 5,
        queries_per_type_view: int | dict[str, dict[str, int]] = 3,
        bge_model_path: str = "/data3/yangyinghao/bge-m3",
        bge_device: str = "cuda:2",
        embedding_service_url: Optional[str] = None,
        embedding_service_timeout: float = 120.0,
    ):
        self.llm = llm
        self._prompt_template = prompt_template
        self.max_concurrent_papers = max(1, int(max_concurrent_papers))
        self.golden_examples_k = max(1, int(golden_examples_k))
        self._raw_queries_per_type_view = queries_per_type_view
        if isinstance(queries_per_type_view, dict):
            self._queries_per_type_view_map = queries_per_type_view
        else:
            self._queries_per_type_view_map = {}
            self._default_queries_per_type_view = max(1, int(queries_per_type_view))
        self.embedder = embedder or build_text_embedder(
            model_path=bge_model_path,
            device=bge_device,
            service_url=embedding_service_url,
            timeout_seconds=embedding_service_timeout,
        )
        if example_retriever is not None:
            self.example_retriever = example_retriever
        elif golden_embedding_db_url:
            self.example_retriever = PostgresGoldenExampleRetriever(golden_embedding_db_url)
        else:
            raise ValueError("QueryGenerator requires golden_embedding_db_url or example_retriever.")

    def _get_queries_count(self, query_type: str, view_name: str) -> int:
        if self._queries_per_type_view_map:
            return self._queries_per_type_view_map.get(query_type, {}).get(view_name, 3)
        return self._default_queries_per_type_view

    def _extract_related_bullet_indice(self, query_data: dict) -> Optional[int]:
        candidates = [
            query_data.get("related_bullet_indice"),
            query_data.get("related_bullet_indices"),
            query_data.get("related_bulletpoint_indices"),
            query_data.get("related_bullet_point_indices"),
            query_data.get("bullet_indice"),
            query_data.get("bullet_indices"),
            query_data.get("bullet_point_indices"),
        ]

        for value in candidates:
            if value is None:
                continue
            if not isinstance(value, list):
                value = [value]
            for item in value:
                try:
                    index = int(item)
                except (TypeError, ValueError):
                    continue
                if index > 0:
                    return index
        return None

    def _extract_related_bullet_justification(self, query_data: dict) -> Optional[str]:
        return (
            query_data.get("related_bullet_justification")
            or query_data.get("related_bulletpoint_justification")
            or query_data.get("related_bullet_point_justification")
            or query_data.get("justification")
            or query_data.get("relation_justification")
        )

    def _view_bullets(self, view) -> list[tuple[int, str, list[str]]]:
        bullets = []
        for idx, bullet in enumerate(getattr(view, "bullet_points", []) or [], 1):
            bullet_index = int(getattr(bullet, "index", idx) or idx)
            bullet_text = str(getattr(bullet, "text", bullet)).strip()
            if not bullet_text:
                continue
            multimodal_ref = getattr(bullet, "multimodal_ref", []) or []
            bullets.append((bullet_index, bullet_text, [str(ref) for ref in multimodal_ref]))
        return bullets

    def _retrieve_prompt_context(
        self,
        *,
        query_type: str,
        paper_title: str,
        view,
    ) -> RetrievedPromptContext:
        view_label = normalize_view_label(str(view.view_name))
        raw_bullets = self._view_bullets(view)
        bullets = [
            BulletContext(index=index, text=text, multimodal_ref=multimodal_ref)
            for index, text, multimodal_ref in raw_bullets
        ]
        if not raw_bullets:
            return RetrievedPromptContext(bullets=[], examples=[])

        view_summary = str(getattr(view, "summary", "") or "").strip()
        retrieval_text = " ".join(
            part
            for part in [
                str(paper_title or "").strip(),
                view_summary,
                " ".join(text for _, text, _ in raw_bullets),
            ]
            if part
        )
        embedding = self.embedder.embed_texts([retrieval_text])[0]
        examples = self.example_retriever.retrieve(
            query_type=query_type,
            view_label=view_label,
            embedding=embedding,
            limit=self.golden_examples_k,
        )
        if not examples:
            raise ValueError(
                "No golden query examples found for "
                f"query_type={query_type!r}, view_label={view_label!r}. "
                "Run scripts/import_golden_query_embeddings.py or inspect the golden classifications."
            )
        if len(examples) < self.golden_examples_k:
            logger.warning(
                "Only %s golden examples found for query_type=%s view=%s.",
                len(examples),
                query_type,
                view_label,
            )
        return RetrievedPromptContext(bullets=bullets, examples=examples)

    def _format_bullets(self, bullets: list[BulletContext]) -> str:
        parts = []
        for bullet in bullets:
            refs = ", ".join(bullet.multimodal_ref) if bullet.multimodal_ref else "none"
            parts.append(f"Bullet {bullet.index}: {bullet.text}")
            parts.append(f"multimodal_ref: {refs}")
            parts.append("")
        return "\n".join(parts).strip()

    def _format_retrieved_examples(self, examples: list[GoldenQueryExample]) -> str:
        return "\n\n".join(
            f"Example {idx}:\n{example.retrieval_content or example.query_text}"
            for idx, example in enumerate(examples, 1)
        ).strip()

    def _prompt_template_for_query_type(self, query_type: str) -> str:
        if self._prompt_template:
            return self._prompt_template
        prompt_name = "generate_queries_ir.txt" if query_type == "IR" else "generate_queries_qa.txt"
        prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / prompt_name
        return load_prompt_template(prompt_path)

    def _build_prompt(
        self,
        *,
        query_type: str,
        paper_title: str,
        view,
        prompt_context: RetrievedPromptContext,
    ) -> str:
        view_name = normalize_view_label(str(view.view_name))
        view_summary = str(getattr(view, "summary", "") or "N/A").strip()
        template = self._prompt_template_for_query_type(query_type)
        return template.format(
            num_queries=self._get_queries_count(query_type, view_name),
            retrieved_examples=self._format_retrieved_examples(prompt_context.examples),
            paper_title=paper_title,
            view_label=view_name,
            view_summary=view_summary,
            bullets=self._format_bullets(prompt_context.bullets),
        )

    def _parse_response(self, response: str, *, query_type: str, view_name: str) -> list[RetrievalQuery]:
        queries: list[RetrievalQuery] = []
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return queries

        data = json.loads(json_match.group())
        raw_queries = data.get("queries", [])
        if not isinstance(raw_queries, list):
            return queries

        for item in raw_queries[: self._get_queries_count(query_type, view_name)]:
            if isinstance(item, dict):
                query_text = str(item.get("query_text", item.get("Q", ""))).strip()
                is_multimodal = bool(item.get("is_multimodal", False))
                related_bullet_indice = self._extract_related_bullet_indice(item)
                related_bullet_justification = self._extract_related_bullet_justification(item)
            else:
                query_text = str(item).strip()
                is_multimodal = False
                related_bullet_indice = None
                related_bullet_justification = None

            query_text = query_text.replace("(multimodal)", "").strip()
            if query_text:
                queries.append(
                    RetrievalQuery(
                        query_text=query_text,
                        query_type=query_type,
                        is_multimodal=is_multimodal,
                        source_view=view_name,
                        related_bullet_indice=related_bullet_indice,
                        related_bullet_justification=related_bullet_justification,
                    )
                )
        return queries

    def generate_queries_for_paper(self, paper_id: str, paper_title: str, summary) -> GeneratedQueriesForPaper:
        logger.debug("Generating queries for paper: %s", paper_id)

        queries: list[RetrievalQuery] = []
        expected_query_count = 0
        for view in summary.views:
            view_name = normalize_view_label(str(view.view_name))
            if view_name not in VIEW_DEFINITIONS:
                continue
            for query_type in QUERY_TYPES:
                prompt_context = self._retrieve_prompt_context(
                    query_type=query_type,
                    paper_title=paper_title,
                    view=view,
                )
                if not prompt_context.bullets:
                    continue
                expected_query_count += self._get_queries_count(query_type, view_name)
                prompt = self._build_prompt(
                    query_type=query_type,
                    paper_title=paper_title,
                    view=view,
                    prompt_context=prompt_context,
                )
                response = self.llm.generate(prompt)
                try:
                    parsed_queries = self._parse_response(
                        response,
                        query_type=query_type,
                        view_name=view_name,
                    )
                    expected = self._get_queries_count(query_type, view_name)
                    if len(parsed_queries) != expected:
                        raise ValueError(
                            f"Expected {expected} queries, got {len(parsed_queries)}"
                        )
                    queries.extend(parsed_queries)
                except Exception as exc:
                    logger.warning("Failed to parse %s/%s query response: %s", query_type, view_name, exc)
                    logger.debug("Raw response: %s", response[:500])

        if len(queries) != expected_query_count:
            raise ValueError(
                f"Generated {len(queries)} queries for paper {paper_id}, "
                f"expected {expected_query_count}."
            )
        if not queries:
            raise ValueError(f"No queries generated for paper {paper_id}.")

        return GeneratedQueriesForPaper(
            paper_id=paper_id,
            paper_title=paper_title,
            queries_by_view=queries,
        )

    def apply(
        self,
        dataset: SummarizedPapersDataset,
        checkpoint_path: Optional[Path] = None,
    ) -> GeneratedQueriesDataset:
        logger.info("Generating queries for %s papers", dataset.total_papers)

        papers_queries = []
        completed_ids = set()
        target_ids = {summary.paper_id for summary in dataset.summaries}
        if checkpoint_path and checkpoint_path.exists():
            try:
                checkpoint = load_json(checkpoint_path, GeneratedQueriesDataset)
                papers_queries = [
                    paper for paper in checkpoint.papers_queries if paper.paper_id in target_ids
                ]
                completed_ids = {paper.paper_id for paper in papers_queries}
                if completed_ids:
                    logger.info(
                        "Loaded %s existing query-generation results from %s",
                        len(completed_ids),
                        checkpoint_path,
                    )
            except Exception as exc:
                logger.warning("Could not load query-generation checkpoint %s: %s", checkpoint_path, exc)

        work_items = [summary for summary in dataset.summaries if summary.paper_id not in completed_ids]
        progress = tqdm(total=len(work_items), desc="Generating queries", unit="paper", dynamic_ncols=True)

        def _generate_for_summary(summary) -> GeneratedQueriesForPaper:
            return self.generate_queries_for_paper(
                paper_id=summary.paper_id,
                paper_title=summary.paper_title,
                summary=summary,
            )

        max_workers = min(self.max_concurrent_papers, len(work_items)) if work_items else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_generate_for_summary, summary): summary for summary in work_items}
            for future in as_completed(futures):
                paper_queries = future.result()
                papers_queries.append(paper_queries)
                completed_ids.add(paper_queries.paper_id)
                if checkpoint_path:
                    save_json(
                        checkpoint_path,
                        GeneratedQueriesDataset(
                            papers_queries=papers_queries,
                            total_papers=len(papers_queries),
                            total_queries=sum(len(paper.queries_by_view) for paper in papers_queries),
                        ),
                    )
                progress.update(1)
                progress.set_postfix_str(
                    f"queries={sum(len(paper.queries_by_view) for paper in papers_queries)}"
                )
        progress.close()

        papers_by_id = {paper.paper_id: paper for paper in papers_queries}
        papers_queries = [
            papers_by_id[summary.paper_id]
            for summary in dataset.summaries
            if summary.paper_id in papers_by_id
        ]
        result = GeneratedQueriesDataset(
            papers_queries=papers_queries,
            total_papers=len(papers_queries),
            total_queries=sum(len(paper.queries_by_view) for paper in papers_queries),
        )
        logger.info(
            "Generate-queries stage success: %s/%s papers (%.1f%%), %s queries.",
            len(papers_queries),
            dataset.total_papers,
            (len(papers_queries) / dataset.total_papers * 100) if dataset.total_papers else 100.0,
            result.total_queries,
        )
        return result

    def run(self, input_path: Path, output_path: Path) -> None:
        logger.info("Running generate-queries stage: %s -> %s", input_path, output_path)
        dataset = load_json(input_path, SummarizedPapersDataset)
        result = self.apply(dataset, checkpoint_path=output_path)
        save_json(output_path, result)
