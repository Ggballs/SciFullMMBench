import logging
from pathlib import Path
from typing import Optional, Tuple

from tqdm.auto import tqdm

from openreview_pipeline.llm import LLMBackend
from openreview_pipeline.schemas.schemas_queries import GeneratedQueriesDataset, FilteredQueriesDataset, FilteredQuery, QueryDimensions
from openreview_pipeline.utils import load_json, save_json, load_prompt_template

logger = logging.getLogger(__name__)


class QueryFilter:
    def __init__(
        self,
        llm: LLMBackend,
        prompt_template: Optional[str] = None,
        threshold: Optional[float] = None,
    ):
        self.llm = llm
        self._prompt_template = prompt_template
        self.threshold = threshold

    def _get_prompt_template(self) -> str:
        if self._prompt_template:
            return self._prompt_template
        prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / "filter_queries.txt"
        return load_prompt_template(prompt_path)

    def _parse_llm_response(self, response: str) -> list:
        import json
        import re

        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from response: {response[:200]}")
                return []
        return []

    def score_queries_for_paper(self, paper_title: str, abstract: str, hard_negatives: list, queries: list) -> list:
        logger.debug(f"Scoring {len(queries)} queries for paper: {paper_title[:50]}...")

        queries_text = "\n".join([f"- {q}" for q in queries])

        hard_negatives_text = ""
        if hard_negatives:
            for i, neg in enumerate(hard_negatives, 1):
                hard_negatives_text += f"{i}. {neg}\n"

        prompt = self._get_prompt_template().replace("{{abstract}}", abstract or "N/A")
        prompt = prompt.replace("{{hard_negatives}}", hard_negatives_text or "None provided")
        prompt = prompt.replace("{{queries}}", queries_text)

        response = self.llm.generate(prompt)

        results = self._parse_llm_response(response)

        scored_queries = []
        for i, q in enumerate(queries):
            if i < len(results) and isinstance(results[i], dict):
                dims = results[i].get("dimensions", {})
                scored_queries.append(FilteredQuery(
                    original_query=q.get("query_text", q if isinstance(q, str) else ""),
                    is_multimodal=q.get("is_multimodal", False) if isinstance(q, dict) else "(multimodal)" in str(q),
                    source_view=q.get("source_view", "unknown") if isinstance(q, dict) else "unknown",
                    dimensions=QueryDimensions(
                        full_paper_reliance=dims.get("full_paper_reliance", "FAIL"),
                        authenticity=dims.get("authenticity", "FAIL"),
                        relevance=dims.get("relevance", "FAIL"),
                        difficulty=dims.get("difficulty", "TOO_HARD"),
                        false_negative_risk=dims.get("false_negative_risk", "HIGH"),
                    ),
                    reasoning=results[i].get("reasoning", ""),
                    verdict=results[i].get("verdict", "Hard Reject"),
                    revised_query=results[i].get("revised_query"),
                ))
            else:
                scored_queries.append(FilteredQuery(
                    original_query=q.get("query_text", q if isinstance(q, str) else ""),
                    is_multimodal=q.get("is_multimodal", False) if isinstance(q, dict) else "(multimodal)" in str(q),
                    source_view=q.get("source_view", "unknown") if isinstance(q, dict) else "unknown",
                    dimensions=QueryDimensions(
                        full_paper_reliance="FAIL",
                        authenticity="FAIL",
                        relevance="FAIL",
                        difficulty="TOO_HARD",
                        false_negative_risk="HIGH",
                    ),
                    reasoning="Failed to parse LLM response",
                    verdict="Hard Reject",
                    revised_query=None,
                ))

        return scored_queries

    def apply(self, dataset: GeneratedQueriesDataset) -> FilteredQueriesDataset:
        logger.info(f"Filtering queries for {dataset.total_papers} papers")

        all_results = []
        passed_count = 0
        total_queries = sum(len(paper_queries.queries_by_view) for paper_queries in dataset.papers_queries)

        progress = tqdm(
            total=total_queries,
            desc="Filtering queries",
            unit="query",
            dynamic_ncols=True,
        )
        for paper_queries in dataset.papers_queries:
            queries_to_score = []
            for q in paper_queries.queries_by_view:
                queries_to_score.append({
                    "query_text": q.query_text,
                    "is_multimodal": q.is_multimodal,
                    "source_view": q.source_view,
                })

            if queries_to_score:
                scored = self.score_queries_for_paper(
                    paper_title=paper_queries.paper_title,
                    abstract="",
                    hard_negatives=[],
                    queries=queries_to_score,
                )
                for sq in scored:
                    if sq.verdict == "Keep":
                        passed_count += 1
                    all_results.append(sq)
                progress.update(len(queries_to_score))
                progress.set_postfix_str(f"kept={passed_count}")
        progress.close()

        logger.info(f"Query filtering complete: {passed_count}/{len(all_results)} passed (Keep verdict)")

        return FilteredQueriesDataset(
            results=all_results,
            total_input=len(all_results),
            total_passed=passed_count,
            total_filtered=len(all_results) - passed_count,
        )

    def run(self, input_path: Path, output_path: Path) -> None:
        logger.info(f"Running filter-queries stage: {input_path} -> {output_path}")
        dataset = load_json(input_path, GeneratedQueriesDataset)
        result = self.apply(dataset)
        save_json(output_path, result)
