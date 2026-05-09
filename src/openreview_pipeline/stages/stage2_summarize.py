import logging
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List

from tqdm.auto import tqdm

from openreview_pipeline.llm import LLMBackend
from openreview_pipeline.schemas.schemas_filter import FilteredPapersDataset
from openreview_pipeline.schemas.schemas_summarize import SummarizedPapersDataset, PaperSummary, ViewBulletPoints
from openreview_pipeline.utils import load_json, save_json, load_prompt_template

logger = logging.getLogger(__name__)

DEFAULT_VIEWS = ["motivation", "method", "experiment"]


class Summarizer:
    def __init__(
        self,
        llm: LLMBackend,
        views: Optional[List[str]] = None,
        prompt_template: Optional[str] = None,
        llm_limit: Optional[int] = None,
        max_concurrent_papers: int = 1,
    ):
        self.llm = llm
        self.views = views or DEFAULT_VIEWS
        self._prompt_template = prompt_template
        self.llm_limit = llm_limit
        self.max_concurrent_papers = max(1, int(max_concurrent_papers))

    def _get_prompt_template(self) -> str:
        if self._prompt_template:
            return self._prompt_template
        prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / "summarize_by_view.txt"
        return load_prompt_template(prompt_path)

    def _build_openreview_content(self, paper_meta) -> str:
        parts = []

        def append_content_entries(section: str, entry_name: str, content: dict) -> None:
            for key, value in content.items():
                if isinstance(value, dict):
                    val = value.get("value", "")
                else:
                    val = str(value) if value else ""
                if val:
                    source_ref = f"{section}/{entry_name}/{key}"
                    parts.append(f"[{source_ref}] {key}: {val}")

        parts.append("=" * 50)
        parts.append("REVIEWS")
        parts.append("=" * 50)

        for review in paper_meta.reviews:
            content = review.content
            parts.append(f"\n--- Review {review.number} ---")
            append_content_entries("Reviews", f"Review {review.number}", content)

        if paper_meta.rebuttals:
            parts.append("\n" + "=" * 50)
            parts.append("REBUTTALS")
            parts.append("=" * 50)
            for rebuttal in paper_meta.rebuttals:
                content = rebuttal.content
                parts.append(f"\n--- Rebuttal {rebuttal.number} ---")
                append_content_entries("Rebuttals", f"Rebuttal {rebuttal.number}", content)

        if paper_meta.comments:
            parts.append("\n" + "=" * 50)
            parts.append("COMMENTS")
            parts.append("=" * 50)
            for comment in paper_meta.comments:
                content = comment.content
                parts.append(f"\n--- Comment {comment.number} ---")
                append_content_entries("Comments", f"Comment {comment.number}", content)

        if paper_meta.decision:
            parts.append("\n" + "=" * 50)
            parts.append("AREA CHAIR META REVIEW / DECISION")
            parts.append("=" * 50)
            content = paper_meta.decision.content
            append_content_entries("Decision", "Decision", content)

        return "\n".join(parts)

    def _view_key(self, view_name: str) -> str:
        return view_name.replace("/", "_").replace(" ", "_")

    def _parse_view_output(self, view_name: str, value) -> ViewBulletPoints:
        summary_text = None
        bullet_points = []

        if isinstance(value, dict):
            summary_text = value.get("summary")
            bullet_points = value.get("bullet_points", [])
        elif isinstance(value, list):
            bullet_points = value

        return ViewBulletPoints(
            view_name=view_name,
            summary=summary_text,
            bullet_points=bullet_points,
        )

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

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                for view_name in self.views:
                    key = self._view_key(view_name)
                    views.append(
                        self._parse_view_output(
                            view_name=view_name,
                            value=data.get(key, data.get(view_name)),
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response[:500]}")

        if not views:
            views.append(
                ViewBulletPoints(
                    view_name="general",
                    summary=None,
                    bullet_points=[{"text": response[:500], "source_refs": []}],
                )
            )

        return PaperSummary(
            paper_id=paper_id,
            paper_title=paper_title,
            abstract=paper_abstract,
            views=views,
        )

    def apply(
        self,
        dataset: FilteredPapersDataset,
        checkpoint_path: Optional[Path] = None,
    ) -> SummarizedPapersDataset:
        passed_papers = [r for r in dataset.results if r.passed]
        total = len(passed_papers)
        limit = self.llm_limit if self.llm_limit else total
        logger.info(f"Summarizing {min(limit, total)} of {total} passed papers (llm_limit={self.llm_limit})")

        summaries = []
        completed_ids = set()
        target_ids = {result.paper.paper.id for result in passed_papers[:limit]}
        if checkpoint_path and checkpoint_path.exists():
            try:
                checkpoint = load_json(checkpoint_path, SummarizedPapersDataset)
                summaries = [
                    summary for summary in checkpoint.summaries if summary.paper_id in target_ids
                ]
                completed_ids = {summary.paper_id for summary in summaries}
                if completed_ids:
                    logger.info("Loaded %s existing summaries from %s", len(completed_ids), checkpoint_path)
            except Exception as exc:
                logger.warning("Could not load summarize checkpoint %s: %s", checkpoint_path, exc)

        work_items = [result for result in passed_papers[:limit] if result.paper.paper.id not in completed_ids]
        progress = tqdm(total=len(work_items), desc="Summarizing papers", unit="paper", dynamic_ncols=True)

        def _summarize_result(result) -> PaperSummary:
            paper = result.paper.paper
            return self.summarize_paper(
                paper_id=paper.id,
                paper_title=paper.title,
                paper_abstract=paper.abstract,
                paper_meta=result.paper,
            )

        max_workers = min(self.max_concurrent_papers, len(work_items)) if work_items else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_summarize_result, result): result for result in work_items}
            for future in as_completed(futures):
                summary = future.result()
                summaries.append(summary)
                completed_ids.add(summary.paper_id)
                if checkpoint_path:
                    save_json(
                        checkpoint_path,
                        SummarizedPapersDataset(summaries=summaries, total_papers=len(summaries)),
                    )
                progress.update(1)
                progress.set_postfix_str(f"done={len(summaries)}")
        progress.close()

        summaries_by_id = {summary.paper_id: summary for summary in summaries}
        summaries = [
            summaries_by_id[result.paper.paper.id]
            for result in passed_papers[:limit]
            if result.paper.paper.id in summaries_by_id
        ]
        logger.info(
            "Summarize stage success: %s/%s papers (%.1f%%).",
            len(summaries),
            min(limit, total),
            (len(summaries) / min(limit, total) * 100) if min(limit, total) else 100.0,
        )
        return SummarizedPapersDataset(summaries=summaries, total_papers=len(summaries))

    def run(self, input_path: Path, output_path: Path) -> None:
        logger.info(f"Running summarize stage: {input_path} -> {output_path}")
        dataset = load_json(input_path, FilteredPapersDataset)
        result = self.apply(dataset, checkpoint_path=output_path)
        save_json(output_path, result)
