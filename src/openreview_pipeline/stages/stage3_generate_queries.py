import logging
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

from openreview_pipeline.llm import LLMBackend
from openreview_pipeline.schemas.schemas_summarize import SummarizedPapersDataset
from openreview_pipeline.schemas.schemas_queries import GeneratedQueriesDataset, GeneratedQueriesForPaper, RetrievalQuery
from openreview_pipeline.utils import load_json, save_json, load_prompt_template

logger = logging.getLogger(__name__)


class QueryGenerator:
    def __init__(self, llm: LLMBackend, prompt_template: Optional[str] = None):
        self.llm = llm
        self._prompt_template = prompt_template

    def _get_prompt_template(self) -> str:
        if self._prompt_template:
            return self._prompt_template
        prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / "generate_queries.txt"
        return load_prompt_template(prompt_path)

    def _build_summary_text(self, summary) -> str:
        parts = []
        for view in summary.views:
            parts.append(f"### {view.view_name}")
            if getattr(view, "summary", None):
                parts.append(f"Summary: {view.summary}")
            for i, bp in enumerate(view.bullet_points, 1):
                bullet_index = getattr(bp, "index", i)
                bullet_text = getattr(bp, "text", str(bp))
                parts.append(f"{bullet_index}. {bullet_text}")
            parts.append("")
        return "\n".join(parts)

    def _extract_related_bullet_indices(self, query_data: dict) -> list[int]:
        return (
            query_data.get("related_bullet_indices")
            or query_data.get("related_bulletpoint_indices")
            or query_data.get("related_bullet_point_indices")
            or query_data.get("bullet_indices")
            or query_data.get("bullet_point_indices")
            or []
        )

    def _extract_related_bullet_justification(self, query_data: dict) -> Optional[str]:
        return (
            query_data.get("related_bullet_justification")
            or query_data.get("related_bulletpoint_justification")
            or query_data.get("related_bullet_point_justification")
            or query_data.get("justification")
            or query_data.get("relation_justification")
        )

    def generate_queries_for_paper(self, paper_id: str, paper_title: str, summary) -> GeneratedQueriesForPaper:
        logger.debug(f"Generating queries for paper: {paper_id}")

        summary_text = self._build_summary_text(summary)

        prompt = self._get_prompt_template().format(
            paper_title=paper_title,
            summary_by_views=summary_text,
        )

        response = self.llm.generate(prompt)

        queries = []
        try:
            import json
            import re

            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                for view_name, view_queries in data.items():
                    if isinstance(view_queries, list):
                        for q in view_queries:
                            if isinstance(q, dict):
                                query_text = q.get("query_text", q.get("Q", ""))
                                is_multimodal = q.get("is_multimodal", "(multimodal)" in query_text)
                                related_bullet_indices = self._extract_related_bullet_indices(q)
                                related_bullet_justification = self._extract_related_bullet_justification(q)
                            else:
                                query_text = str(q)
                                is_multimodal = "(multimodal)" in query_text
                                related_bullet_indices = []
                                related_bullet_justification = None

                            query_text = query_text.replace("(multimodal)", "").strip()
                            if query_text:
                                queries.append(RetrievalQuery(
                                    query_text=query_text,
                                    is_multimodal=is_multimodal,
                                    source_view=view_name,
                                    related_bullet_indices=related_bullet_indices,
                                    related_bullet_justification=related_bullet_justification,
                                ))
        except Exception as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response[:500]}")

        if not queries:
            queries.append(RetrievalQuery(
                query_text=response[:200],
                is_multimodal=False,
                source_view="general",
                related_bullet_indices=[],
                related_bullet_justification=None,
            ))

        return GeneratedQueriesForPaper(
            paper_id=paper_id,
            paper_title=paper_title,
            queries_by_view=queries,
        )

    def apply(self, dataset: SummarizedPapersDataset) -> GeneratedQueriesDataset:
        logger.info(f"Generating queries for {dataset.total_papers} papers")

        papers_queries = []
        total_queries = 0

        progress = tqdm(
            dataset.summaries,
            total=dataset.total_papers,
            desc="Generating queries",
            unit="paper",
            dynamic_ncols=True,
        )
        for summary in progress:
            paper_queries = self.generate_queries_for_paper(
                paper_id=summary.paper_id,
                paper_title=summary.paper_title,
                summary=summary,
            )
            papers_queries.append(paper_queries)
            total_queries += len(paper_queries.queries_by_view)
            progress.set_postfix_str(f"queries={total_queries}")
        progress.close()

        return GeneratedQueriesDataset(
            papers_queries=papers_queries,
            total_papers=len(papers_queries),
            total_queries=total_queries,
        )

    def run(self, input_path: Path, output_path: Path) -> None:
        logger.info(f"Running generate-queries stage: {input_path} -> {output_path}")
        dataset = load_json(input_path, SummarizedPapersDataset)
        result = self.apply(dataset)
        save_json(output_path, result)
