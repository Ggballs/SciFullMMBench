import logging
from pathlib import Path
from typing import Optional, List

from openreview_pipeline.llm import LLMBackend
from openreview_pipeline.schemas.schemas_filter import FilteredPapersDataset
from openreview_pipeline.schemas.schemas_summarize import SummarizedPapersDataset, PaperSummary, ViewBulletPoints
from openreview_pipeline.utils import load_json, save_json, load_prompt_template

logger = logging.getLogger(__name__)

DEFAULT_VIEWS = ["contribution", "method/dataset", "experiment/result", "limitation/scope"]


class Summarizer:
    def __init__(self, llm: LLMBackend, views: Optional[List[str]] = None, prompt_template: Optional[str] = None, llm_limit: Optional[int] = None):
        self.llm = llm
        self.views = views or DEFAULT_VIEWS
        self._prompt_template = prompt_template
        self.llm_limit = llm_limit

    def _get_prompt_template(self) -> str:
        if self._prompt_template:
            return self._prompt_template
        prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / "summarize_by_view.txt"
        return load_prompt_template(prompt_path)

    def _build_openreview_content(self, paper_meta) -> str:
        parts = []

        parts.append("=" * 50)
        parts.append("REVIEWS")
        parts.append("=" * 50)

        for review in paper_meta.reviews:
            content = review.content
            parts.append(f"\n--- Review {review.number} ---")

            for key, value in content.items():
                if isinstance(value, dict):
                    val = value.get("value", "")
                else:
                    val = str(value) if value else ""
                if val:
                    parts.append(f"{key}: {val}")

        if paper_meta.rebuttals:
            parts.append("\n" + "=" * 50)
            parts.append("REBUTTALS")
            parts.append("=" * 50)
            for rebuttal in paper_meta.rebuttals:
                content = rebuttal.content
                parts.append(f"\n--- Rebuttal {rebuttal.number} ---")
                for key, value in content.items():
                    if isinstance(value, dict):
                        val = value.get("value", "")
                    else:
                        val = str(value) if value else ""
                    if val:
                        parts.append(f"{key}: {val}")

        if paper_meta.comments:
            parts.append("\n" + "=" * 50)
            parts.append("COMMENTS")
            parts.append("=" * 50)
            for comment in paper_meta.comments:
                content = comment.content
                parts.append(f"\n--- Comment {comment.number} ---")
                for key, value in content.items():
                    if isinstance(value, dict):
                        val = value.get("value", "")
                    else:
                        val = str(value) if value else ""
                    if val:
                        parts.append(f"{key}: {val}")

        if paper_meta.decision:
            parts.append("\n" + "=" * 50)
            parts.append("AREA CHAIR META REVIEW / DECISION")
            parts.append("=" * 50)
            content = paper_meta.decision.content
            for key, value in content.items():
                if isinstance(value, dict):
                    val = value.get("value", "")
                else:
                    val = str(value) if value else ""
                if val:
                    parts.append(f"{key}: {val}")

        return "\n".join(parts)

    def summarize_paper(self, paper_id: str, paper_title: str, paper_abstract: str, paper_meta) -> PaperSummary:
        logger.debug(f"Summarizing paper: {paper_id}")

        openreview_content = self._build_openreview_content(paper_meta)

        prompt = self._get_prompt_template().format(
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            full_openreview_content=openreview_content,
        )

        response = self.llm.generate(prompt)

        views = []
        raw_summary = ""

        try:
            import json
            import re

            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                for view_name in self.views:
                    bullet_points = data.get(view_name.replace("/", "_").replace(" ", "_"), [])
                    if isinstance(bullet_points, list):
                        views.append(ViewBulletPoints(view_name=view_name, bullet_points=bullet_points))
                    else:
                        bps = data.get(view_name, [])
                        if isinstance(bps, list):
                            views.append(ViewBulletPoints(view_name=view_name, bullet_points=bps))

                raw_summary = data.get("summary", "")
        except Exception as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response[:500]}")

        if not views:
            views.append(ViewBulletPoints(view_name="general", bullet_points=[response[:500]]))

        return PaperSummary(
            paper_id=paper_id,
            paper_title=paper_title,
            views=views,
            raw_summary=raw_summary,
        )

    def apply(self, dataset: FilteredPapersDataset) -> SummarizedPapersDataset:
        passed_papers = [r for r in dataset.results if r.passed]
        total = len(passed_papers)
        limit = self.llm_limit if self.llm_limit else total
        logger.info(f"Summarizing {min(limit, total)} of {total} passed papers (llm_limit={self.llm_limit})")

        summaries = []
        for i, result in enumerate(passed_papers[:limit]):

            paper = result.paper.paper
            summary = self.summarize_paper(
                paper_id=paper.id,
                paper_title=paper.title,
                paper_abstract=paper.abstract,
                paper_meta=result.paper,
            )
            summaries.append(summary)

        return SummarizedPapersDataset(summaries=summaries, total_papers=len(summaries))

    def run(self, input_path: Path, output_path: Path) -> None:
        logger.info(f"Running summarize stage: {input_path} -> {output_path}")
        dataset = load_json(input_path, FilteredPapersDataset)
        result = self.apply(dataset)
        save_json(output_path, result)
