import logging
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Any

from tqdm.auto import tqdm

from openreview_pipeline.llm import LLMBackend
from openreview_pipeline.schemas.schemas_filter import FilteredPapersDataset
from openreview_pipeline.schemas.schemas_summarize import SummarizedPapersDataset, PaperSummary, ViewBulletPoints
from openreview_pipeline.utils import load_json, save_json, load_prompt_template
from openreview_pipeline.utils.multimodal_evidence import group_multimodal_evidence, extract_multimodal_evidence_snippets

logger = logging.getLogger(__name__)

DEFAULT_VIEWS = ["motivation", "method", "experiment/result"]


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
        self._multimodal_prompt_template: Optional[str] = None

    def _get_prompt_template(self) -> str:
        if self._prompt_template:
            return self._prompt_template
        prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / "summarize_by_view.txt"
        return load_prompt_template(prompt_path)

    def _get_multimodal_prompt_template(self) -> str:
        if self._multimodal_prompt_template:
            return self._multimodal_prompt_template
        prompt_path = (
            Path(__file__).parent.parent.parent.parent
            / "prompts"
            / "summarize_multimodal_by_evidence.txt"
        )
        self._multimodal_prompt_template = load_prompt_template(prompt_path)
        return self._multimodal_prompt_template

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

    def _parse_response_views(self, response: str) -> list[ViewBulletPoints]:
        cleaned = str(response or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        data = None
        for candidate in [
            cleaned,
            (re.search(r"\{[\s\S]*\}", cleaned).group() if re.search(r"\{[\s\S]*\}", cleaned) else ""),
        ]:
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if not isinstance(data, dict):
            return []

        views_payload = data.get("views")
        if isinstance(views_payload, dict):
            data = views_payload
        elif isinstance(views_payload, list):
            by_name = {}
            for item in views_payload:
                if isinstance(item, dict):
                    name = item.get("view_name") or item.get("name") or item.get("view")
                    if name:
                        by_name[str(name)] = item
            data = by_name

        views = []
        for view_name in self.views:
            key = self._view_key(view_name)
            views.append(
                self._parse_view_output(
                    view_name=view_name,
                    value=data.get(key, data.get(view_name)),
                )
            )
        return views

    def _parse_multimodal_bullet_response(self, response: str) -> list[dict[str, Any]]:
        cleaned = str(response or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        data = None
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
        if isinstance(data, dict):
            bullets = data.get("bullet_points", [])
        elif isinstance(data, list):
            bullets = data
        else:
            return []
        return [item for item in bullets if isinstance(item, dict)]

    def _format_evidence_snippets(self, group) -> str:
        lines = []
        for idx, snippet in enumerate(group.meaningful_snippets, 1):
            text = re.sub(r"\s+", " ", snippet.text).strip()
            if len(text) > 1500:
                text = text[:1500].rstrip() + "..."
            lines.append(f"{idx}. [{snippet.source_ref}] {text}")
        return "\n".join(lines)

    def _generate_multimodal_bullets(
        self,
        *,
        paper_title: str,
        paper_abstract: str,
        paper_meta,
    ) -> list[dict[str, Any]]:
        snippets = extract_multimodal_evidence_snippets(paper_meta)
        groups = group_multimodal_evidence(snippets, meaningful_only=True)
        bullets: list[dict[str, Any]] = []
        for group in groups:
            if not group.meaningful_snippets:
                continue
            prompt = self._get_multimodal_prompt_template().format(
                paper_title=paper_title,
                paper_abstract=paper_abstract,
                multimodal_ref=group.multimodal_ref,
                evidence_snippets=self._format_evidence_snippets(group),
            )
            try:
                response = self.llm.generate(prompt)
            except Exception as exc:
                logger.warning(
                    "Failed to generate multimodal bullets for %s: %s",
                    group.multimodal_ref,
                    exc,
                )
                continue
            for item in self._parse_multimodal_bullet_response(response):
                item = dict(item)
                target_view = str(item.pop("target_view", "") or "").strip()
                if target_view == "experiment":
                    target_view = "experiment/result"
                if target_view not in self.views:
                    target_view = "experiment/result"
                item["target_view"] = target_view
                item["multimodal_ref"] = [group.multimodal_ref]
                item.setdefault(
                    "source_refs",
                    [snippet.source_ref for snippet in group.meaningful_snippets],
                )
                item.setdefault("multimodal_dependency", "supportive")
                item.setdefault(
                    "multimodal_dependency_rationale",
                    f"{group.multimodal_ref} is explicitly discussed in OpenReview evidence snippets.",
                )
                if str(item.get("text", "")).strip():
                    bullets.append(item)
        return bullets

    def _bullet_to_dict(self, bullet) -> dict[str, Any]:
        if isinstance(bullet, dict):
            return dict(bullet)
        return {
            "text": getattr(bullet, "text", ""),
            "source_refs": list(getattr(bullet, "source_refs", []) or []),
            "multimodal_ref": list(getattr(bullet, "multimodal_ref", []) or []),
            "multimodal_dependency": getattr(bullet, "multimodal_dependency", "none") or "none",
            "multimodal_dependency_rationale": getattr(bullet, "multimodal_dependency_rationale", None),
        }

    def _token_jaccard(self, left: str, right: str) -> float:
        left_tokens = {token for token in re.findall(r"[a-z0-9]+", left.lower()) if len(token) > 2}
        right_tokens = {token for token in re.findall(r"[a-z0-9]+", right.lower()) if len(token) > 2}
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _is_duplicate_bullet(self, candidate: dict[str, Any], existing: list[dict[str, Any]]) -> bool:
        candidate_refs = set(candidate.get("source_refs") or [])
        candidate_mm_refs = set(candidate.get("multimodal_ref") or [])
        candidate_text = str(candidate.get("text", ""))
        for bullet in existing:
            source_refs = set(bullet.get("source_refs") or [])
            mm_refs = set(bullet.get("multimodal_ref") or [])
            if candidate_mm_refs and candidate_mm_refs == mm_refs and candidate_refs & source_refs:
                return True
            if self._token_jaccard(candidate_text, str(bullet.get("text", ""))) >= 0.82:
                return True
        return False

    def _merge_multimodal_bullets(
        self,
        views: list[ViewBulletPoints],
        multimodal_bullets: list[dict[str, Any]],
    ) -> list[ViewBulletPoints]:
        view_payloads: dict[str, dict[str, Any]] = {
            view.view_name: {
                "summary": view.summary,
                "bullet_points": [self._bullet_to_dict(bullet) for bullet in view.bullet_points],
            }
            for view in views
        }
        for view_name in self.views:
            view_payloads.setdefault(view_name, {"summary": None, "bullet_points": []})

        for bullet in multimodal_bullets:
            view_name = str(bullet.pop("target_view", "") or "experiment/result")
            if view_name not in view_payloads:
                view_name = "experiment/result"
            existing = view_payloads[view_name]["bullet_points"]
            if not self._is_duplicate_bullet(bullet, existing):
                existing.append(bullet)

        return [
            ViewBulletPoints(
                view_name=view_name,
                summary=view_payloads[view_name]["summary"],
                bullet_points=view_payloads[view_name]["bullet_points"],
            )
            for view_name in self.views
        ]

    def summarize_paper(self, paper_id: str, paper_title: str, paper_abstract: str, paper_meta) -> PaperSummary:
        logger.debug(f"Summarizing paper: {paper_id}")

        openreview_content = self._build_openreview_content(paper_meta)

        prompt = self._get_prompt_template().format(
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            full_openreview_content=openreview_content,
        )

        response = ""
        views = []
        for attempt in range(2):
            response = self.llm.generate(prompt)
            try:
                views = self._parse_response_views(response)
            except Exception as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"Raw response: {response[:500]}")
                views = []
            if any(view.bullet_points for view in views):
                break
            logger.warning(
                "Empty parsed summary for paper %s on attempt %s; retrying once.",
                paper_id,
                attempt + 1,
            )

        if not views:
            views.append(
                ViewBulletPoints(
                    view_name="general",
                    summary=None,
                    bullet_points=[{"text": response[:500], "source_refs": []}],
                )
            )
        multimodal_bullets = self._generate_multimodal_bullets(
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            paper_meta=paper_meta,
        )
        if multimodal_bullets:
            views = self._merge_multimodal_bullets(views, multimodal_bullets)

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
