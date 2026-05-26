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

QUERY_TYPES = ("IR",)
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
        exclude_litsearch: bool = False,
    ) -> list[GoldenQueryExample]:
        ...


@dataclass(frozen=True)
class BulletContext:
    index: int
    text: str
    multimodal_ref: list[str]
    multimodal_dependency: str = "none"
    multimodal_dependency_rationale: str = ""


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
        exclude_litsearch: bool = False,
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
            return max(0, int(self._queries_per_type_view_map.get(query_type, {}).get(view_name, 0)))
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

    def _extract_multimodal_rationale(self, query_data: dict) -> Optional[str]:
        return (
            query_data.get("multimodal_rationale")
            or query_data.get("is_multimodal_rationale")
            or query_data.get("multimodal_label_rationale")
            or query_data.get("multimodal_justification")
        )

    def _view_bullets(self, view) -> list[tuple[int, str, list[str], str, str]]:
        bullets = []
        for idx, bullet in enumerate(getattr(view, "bullet_points", []) or [], 1):
            bullet_index = int(getattr(bullet, "index", idx) or idx)
            bullet_text = str(getattr(bullet, "text", bullet)).strip()
            if not bullet_text:
                continue
            multimodal_ref = getattr(bullet, "multimodal_ref", []) or []
            multimodal_dependency = str(
                getattr(bullet, "multimodal_dependency", "none") or "none"
            ).strip().lower()
            if multimodal_dependency not in {"none", "incidental", "supportive", "necessary"}:
                multimodal_dependency = "none"
            multimodal_dependency_rationale = str(
                getattr(bullet, "multimodal_dependency_rationale", "") or ""
            ).strip()
            bullets.append(
                (
                    bullet_index,
                    bullet_text,
                    [str(ref) for ref in multimodal_ref],
                    multimodal_dependency,
                    multimodal_dependency_rationale,
                )
            )
        return bullets

    def _bullet_contexts_from_raw(
        self,
        raw_bullets: list[tuple[int, str, list[str], str, str]],
    ) -> list[BulletContext]:
        return [
            BulletContext(
                index=index,
                text=text,
                multimodal_ref=multimodal_ref,
                multimodal_dependency=multimodal_dependency,
                multimodal_dependency_rationale=multimodal_dependency_rationale,
            )
            for index, text, multimodal_ref, multimodal_dependency, multimodal_dependency_rationale in raw_bullets
        ]

    def _retrieve_prompt_context(
        self,
        *,
        query_type: str,
        paper_title: str,
        view,
        raw_bullets: Optional[list[tuple[int, str, list[str], str, str]]] = None,
    ) -> RetrievedPromptContext:
        view_label = normalize_view_label(str(view.view_name))
        raw_bullets = raw_bullets if raw_bullets is not None else self._view_bullets(view)
        bullets = self._bullet_contexts_from_raw(raw_bullets)
        if not raw_bullets:
            return RetrievedPromptContext(bullets=[], examples=[])

        view_summary = str(getattr(view, "summary", "") or "").strip()
        retrieval_text = " ".join(
            part
            for part in [
                str(paper_title or "").strip(),
                view_summary,
                " ".join(text for _, text, _, _, _ in raw_bullets),
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
            parts.append(f"multimodal_dependency: {bullet.multimodal_dependency}")
            if bullet.multimodal_ref:
                parts.append(
                    "Bullet multimodal rationale: "
                    f"{bullet.multimodal_dependency_rationale or 'not provided'}"
                )
            parts.append("")
        return "\n".join(parts).strip()

    def _format_retrieved_examples(self, examples: list[GoldenQueryExample]) -> str:
        return "\n\n".join(
            f"Example {idx}:\n{example.retrieval_content or example.query_text}"
            for idx, example in enumerate(examples, 1)
        ).strip()

    def _serialize_retrieved_examples(self, examples: list[GoldenQueryExample]) -> list[dict[str, Any]]:
        return [
            {
                "rank": idx,
                "example_id": example.example_id,
                "query_id": example.query_id,
                "query_type": example.query_type,
                "view_label": example.view_label,
                "query_text": example.query_text,
                "retrieval_content": example.retrieval_content,
                "distance": example.distance,
            }
            for idx, example in enumerate(examples, 1)
        ]

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
        num_queries: int,
        query_mode_instructions: str,
    ) -> str:
        view_name = normalize_view_label(str(view.view_name))
        view_summary = str(getattr(view, "summary", "") or "N/A").strip()
        template = self._prompt_template_for_query_type(query_type)
        return template.format(
            num_queries=num_queries,
            query_mode_instructions=query_mode_instructions,
            retrieved_examples=self._format_retrieved_examples(prompt_context.examples),
            paper_title=paper_title,
            view_label=view_name,
            view_summary=view_summary,
            bullets=self._format_bullets(prompt_context.bullets),
        )

    def _parse_response(
        self,
        response: str,
        *,
        query_type: str,
        view_name: str,
        limit: int,
    ) -> list[RetrievalQuery]:
        queries: list[RetrievalQuery] = []

        cleaned = str(response or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        data: Any = None
        for candidate in [
            cleaned,
            (re.search(r"\{[\s\S]*\}", cleaned).group() if re.search(r"\{[\s\S]*\}", cleaned) else ""),
            (re.search(r"\[[\s\S]*\]", cleaned).group() if re.search(r"\[[\s\S]*\]", cleaned) else ""),
        ]:
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if data is None:
            return queries

        raw_queries = data if isinstance(data, list) else data.get("queries", [])
        if not isinstance(raw_queries, list):
            return queries

        for item in raw_queries[:limit]:
            if isinstance(item, dict):
                query_text = str(item.get("query_text", item.get("Q", ""))).strip()
                is_multimodal = bool(item.get("is_multimodal", False))
                related_bullet_indice = self._extract_related_bullet_indice(item)
                related_bullet_justification = self._extract_related_bullet_justification(item)
                multimodal_rationale = (
                    self._extract_multimodal_rationale(item) if is_multimodal else None
                )
            else:
                query_text = str(item).strip()
                is_multimodal = False
                related_bullet_indice = None
                related_bullet_justification = None
                multimodal_rationale = None

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
                        multimodal_rationale=multimodal_rationale,
                    )
                )
        return queries

    def _clean_query_text(self, text: str) -> str:
        text = text.replace("(multimodal)", "").strip()
        text = re.sub(
            r"\b(?:fig(?:ure)?|table|eq(?:uation)?|alg(?:orithm)?)\.?\s*\(?\s*[A-Za-z]?\d+(?:\.\d+)?\s*\)?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\bappendix\b\s*[A-Za-z0-9. -]*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([?.!,])", r"\1", text)
        return text.strip()

    def _force_plain_queries(
        self,
        queries: list[RetrievalQuery],
        *,
        multimodal: bool,
        bullet: Optional[BulletContext] = None,
    ) -> list[RetrievalQuery]:
        fixed = []
        for query in queries:
            query.query_text = self._clean_query_text(query.query_text)
            if multimodal:
                query.is_multimodal = True
                if bullet is not None:
                    query.related_bullet_indice = bullet.index
                    if not query.multimodal_rationale:
                        rationale = bullet.multimodal_dependency_rationale or (
                            "The selected bullet depends on concrete multimodal evidence."
                        )
                        query.multimodal_rationale = (
                            f"The selected bullet is multimodal-relevant: {rationale}"
                        )
            else:
                query.is_multimodal = False
                query.multimodal_rationale = None
            if query.query_text:
                fixed.append(query)
        return fixed

    def _generate_with_context(
        self,
        *,
        paper_id: str,
        paper_title: str,
        view,
        query_type: str,
        prompt_context: RetrievedPromptContext,
        expected: int,
        query_mode_instructions: str,
        all_bullets: Optional[list[BulletContext]] = None,
    ) -> list[RetrievalQuery]:
        if expected <= 0:
            return []
        prompt = self._build_prompt(
            query_type=query_type,
            paper_title=paper_title,
            view=view,
            prompt_context=prompt_context,
            num_queries=expected,
            query_mode_instructions=query_mode_instructions,
        )
        view_name = normalize_view_label(str(view.view_name))
        parsed_queries: list[RetrievalQuery] = []
        best_queries: list[RetrievalQuery] = []
        response = ""
        for attempt in range(3):
            response = self.llm.generate(prompt)
            try:
                parsed_queries = self._parse_response(
                    response,
                    query_type=query_type,
                    view_name=view_name,
                    limit=expected,
                )
                if all_bullets is not None:
                    parsed_queries = self._apply_multimodal_flags(parsed_queries, all_bullets)
                if len(parsed_queries) > len(best_queries):
                    best_queries = parsed_queries
                if len(parsed_queries) < expected:
                    raise ValueError(f"Expected {expected} queries, got {len(parsed_queries)}")
                break
            except Exception as exc:
                logger.warning(
                    "Failed to parse %s/%s query response on attempt %s: %s",
                    query_type,
                    view_name,
                    attempt + 1,
                    exc,
                )
                logger.debug("Raw response: %s", response[:500])
                parsed_queries = []
        if not parsed_queries and best_queries:
            logger.warning(
                "Using %s/%s parsed %s/%s queries for paper %s after retries.",
                len(best_queries),
                expected,
                query_type,
                view_name,
                paper_id,
            )
            parsed_queries = best_queries
        retrieved_examples = self._serialize_retrieved_examples(prompt_context.examples)
        for query in parsed_queries:
            query.retrieved_golden_queries = retrieved_examples
        return parsed_queries

    def _apply_multimodal_flags(
        self,
        queries: list[RetrievalQuery],
        all_bullets: list[BulletContext],
    ) -> list[RetrievalQuery]:
        """Assign is_multimodal per query based on related_bullet_indice matching a multimodal bullet."""
        bullets_by_index: dict[int, BulletContext] = {b.index: b for b in all_bullets}
        fixed: list[RetrievalQuery] = []
        for query in queries:
            query.query_text = self._clean_query_text(query.query_text)
            bullet_idx = query.related_bullet_indice
            bullet = bullets_by_index.get(bullet_idx) if bullet_idx else None
            if bullet is not None and bullet.multimodal_ref:
                query.is_multimodal = True
                query.related_bullet_indice = bullet.index
                if not query.multimodal_rationale:
                    query.multimodal_rationale = (
                        bullet.multimodal_dependency_rationale
                        or f"The selected bullet relates to {', '.join(bullet.multimodal_ref)}."
                    )
            else:
                query.is_multimodal = False
                query.multimodal_rationale = None
            if query.query_text:
                fixed.append(query)
        return fixed

    def generate_queries_for_paper(self, paper_id: str, paper_title: str, summary) -> GeneratedQueriesForPaper:
        logger.debug("Generating queries for paper: %s", paper_id)

        queries: list[RetrievalQuery] = []
        expected_query_count = 0
        for view in summary.views:
            view_name = normalize_view_label(str(view.view_name))
            if view_name not in VIEW_DEFINITIONS:
                continue
            for query_type in QUERY_TYPES:
                raw_bullets = self._view_bullets(view)
                if not raw_bullets:
                    continue

                text_bullets = [b for b in raw_bullets if not b[2]]
                multimodal_bullets = [b for b in raw_bullets if b[2]]

                text_expected = min(self._get_queries_count(query_type, view_name), len(text_bullets))
                multimodal_expected = len(multimodal_bullets)
                total_expected = text_expected + multimodal_expected
                if total_expected == 0:
                    continue

                # Single prompt context with ALL bullets — one LLM call per view.
                prompt_context = self._retrieve_prompt_context(
                    query_type=query_type,
                    paper_title=paper_title,
                    view=view,
                    raw_bullets=raw_bullets,
                )

                instruction_parts = []
                if text_expected:
                    instruction_parts.append(
                        f"Generate {text_expected} plain IR queries from the text-only bullets "
                        "(multimodal_ref=none). Set is_multimodal=false for these."
                    )
                if multimodal_expected:
                    instruction_parts.append(
                        f"Generate {multimodal_expected} plain IR queries, one per multimodal bullet. "
                        "Do NOT mention exact figure/table/equation/algorithm labels in query_text. "
                        "Set is_multimodal=true and use the bullet's multimodal rationale."
                    )

                expected_query_count += total_expected
                all_bullets = self._bullet_contexts_from_raw(raw_bullets)
                queries.extend(
                    self._generate_with_context(
                        paper_id=paper_id,
                        paper_title=paper_title,
                        view=view,
                        query_type=query_type,
                        prompt_context=prompt_context,
                        expected=total_expected,
                        query_mode_instructions=" ".join(instruction_parts),
                        all_bullets=all_bullets,
                    )
                )

        if len(queries) != expected_query_count:
            logger.warning(
                "Generated %s queries for paper %s, expected %s; continuing with parsed queries.",
                len(queries),
                paper_id,
                expected_query_count,
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
