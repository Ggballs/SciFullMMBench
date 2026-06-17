from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from openreview_pipeline.schemas import PipelineOutput, PipelinePaper, PipelineQuery
from openreview_pipeline.schemas.schemas_summarize import ViewBulletPoints, dump_view_bullet_points_compact
from utils import save_json


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def load_optional_json_file(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    return load_json_file(path)


def _extract_stage0_paper_id(item: dict[str, Any]) -> Optional[str]:
    paper_meta = item.get("paper", {}) if isinstance(item, dict) else {}
    paper_id = paper_meta.get("id")
    return str(paper_id) if paper_id else None


def _extract_stage1_paper_id(result: dict[str, Any]) -> Optional[str]:
    paper_bundle = result.get("paper", {}) if isinstance(result, dict) else {}
    paper_meta = paper_bundle.get("paper", {}) if isinstance(paper_bundle, dict) else {}
    paper_id = paper_meta.get("id")
    return str(paper_id) if paper_id else None


def _extract_paper_dir(
    query_analysis_dir: Optional[Path],
    hard_negatives_path: Optional[Path],
    paper_id: str,
) -> Optional[str]:
    candidates = []
    if query_analysis_dir is not None:
        candidates.append(query_analysis_dir / paper_id)
    if hard_negatives_path is not None:
        candidates.append(hard_negatives_path.parent / f"{hard_negatives_path.stem}_pdfs" / paper_id)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _build_openreview_payload(stage0_item: Optional[dict[str, Any]], paper_id: str, query_analysis_item: Optional[dict[str, Any]]) -> dict[str, Any]:
    if stage0_item and isinstance(stage0_item, dict):
        paper_meta = stage0_item.get("paper", {}) if isinstance(stage0_item.get("paper"), dict) else {}
        return {
            "abstract": paper_meta.get("abstract"),
            "pdf_url": paper_meta.get("pdf_url"),
            "openreview_url": f"https://openreview.net/forum?id={paper_id}",
            "venue": paper_meta.get("venue"),
            "year": paper_meta.get("year"),
            "authors": paper_meta.get("authors", []),
            "reviews": stage0_item.get("reviews", []),
            "rebuttals": stage0_item.get("rebuttals", []),
            "comments": stage0_item.get("comments", []),
            "decision": stage0_item.get("decision"),
        }
    if query_analysis_item and isinstance(query_analysis_item, dict):
        return {
            "abstract": query_analysis_item.get("abstract"),
            "pdf_url": query_analysis_item.get("pdf_url"),
            "openreview_url": query_analysis_item.get("openreview_url") or f"https://openreview.net/forum?id={paper_id}",
            "venue": query_analysis_item.get("venue"),
            "year": query_analysis_item.get("year"),
            "authors": query_analysis_item.get("authors", []),
            "reviews": [],
            "rebuttals": [],
            "comments": [],
            "decision": None,
        }
    return {
        "abstract": None,
        "pdf_url": None,
        "openreview_url": f"https://openreview.net/forum?id={paper_id}",
        "venue": None,
        "year": None,
        "authors": [],
        "reviews": [],
        "rebuttals": [],
        "comments": [],
        "decision": None,
    }


def _build_filter_status(stage1_item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not stage1_item or not isinstance(stage1_item, dict):
        return None
    reasons = stage1_item.get("reason") or stage1_item.get("reasons")
    return {
        "passed": bool(stage1_item.get("passed", True)),
        "reason": reasons,
        "score": stage1_item.get("score"),
    }


def _build_summary_views(stage2_item: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(stage2_item, dict):
        return []

    views = stage2_item.get("views", [])
    if not isinstance(views, list):
        return []

    normalized_views: list[dict[str, Any]] = []
    for view in views:
        if not isinstance(view, dict):
            continue
        normalized_views.append(dump_view_bullet_points_compact(ViewBulletPoints.model_validate(view)))
    return normalized_views


def build_pipeline_output(
    *,
    downloaded_path: Optional[Path] = None,
    filtered_path: Optional[Path] = None,
    summarized_path: Optional[Path] = None,
    queries_path: Optional[Path] = None,
    query_analysis_output_dir: Optional[Path] = None,
    hard_negatives_path: Optional[Path] = None,
) -> PipelineOutput:
    stage0 = load_optional_json_file(downloaded_path)
    stage1 = load_optional_json_file(filtered_path)
    stage2 = load_optional_json_file(summarized_path)
    stage3 = load_optional_json_file(queries_path)
    query_analysis = load_optional_json_file(
        (query_analysis_output_dir / "query_analysis.json") if query_analysis_output_dir else None
    )
    hard_negatives = load_optional_json_file(hard_negatives_path)

    stage0_by_id = {
        paper_id: item
        for item in (stage0.get("papers", []) if stage0 else [])
        if isinstance(item, dict) and (paper_id := _extract_stage0_paper_id(item))
    }
    stage1_by_id = {
        paper_id: item
        for item in (stage1.get("results", []) if stage1 else [])
        if isinstance(item, dict) and (paper_id := _extract_stage1_paper_id(item))
    }
    stage2_by_id = {
        str(item["paper_id"]): item
        for item in (stage2.get("summaries", []) if stage2 else [])
        if isinstance(item, dict) and item.get("paper_id")
    }
    stage3_by_id = {
        str(item["paper_id"]): item
        for item in (stage3.get("papers_queries", []) if stage3 else [])
        if isinstance(item, dict) and item.get("paper_id")
    }
    hard_negatives_by_query = {
        (
            str(item.get("paper_id", "")),
            str(item.get("query", "")),
            str(item.get("source_view", "")),
            str(item.get("query_type", "IR")),
        ): item
        for item in (hard_negatives.get("results", []) if hard_negatives else [])
        if isinstance(item, dict)
    }
    query_analysis_by_id = {
        str(item["paper_id"]): item
        for item in (query_analysis.get("papers", []) if query_analysis else [])
        if isinstance(item, dict) and item.get("paper_id")
    }

    ordered_paper_ids: list[str] = []
    seen_ids: set[str] = set()
    for collection in [
        list(stage0_by_id.keys()),
        list(stage1_by_id.keys()),
        list(stage2_by_id.keys()),
        list(stage3_by_id.keys()),
        list(query_analysis_by_id.keys()),
    ]:
        for paper_id in collection:
            if paper_id not in seen_ids:
                seen_ids.add(paper_id)
                ordered_paper_ids.append(paper_id)

    papers: list[PipelinePaper] = []
    for paper_id in ordered_paper_ids:
        stage0_item = stage0_by_id.get(paper_id)
        stage1_item = stage1_by_id.get(paper_id)
        stage2_item = stage2_by_id.get(paper_id)
        stage3_item = stage3_by_id.get(paper_id)
        query_analysis_item = query_analysis_by_id.get(paper_id)

        paper_title = ""
        if stage0_item and isinstance(stage0_item.get("paper"), dict):
            paper_title = str(stage0_item["paper"].get("title", ""))
        elif stage3_item:
            paper_title = str(stage3_item.get("paper_title", ""))
        elif query_analysis_item:
            paper_title = str(query_analysis_item.get("paper_title", ""))

        query_analysis_queries = {
            (
                str(item.get("query_text", "")),
                str(item.get("source_view", "")),
                str(item.get("query_type", "IR")),
            ): item
            for item in (query_analysis_item.get("queries", []) if isinstance(query_analysis_item, dict) else [])
            if isinstance(item, dict)
        }

        queries: list[PipelineQuery] = []
        stage3_queries = stage3_item.get("queries_by_view", []) if isinstance(stage3_item, dict) else []
        for query in stage3_queries if isinstance(stage3_queries, list) else []:
            if not isinstance(query, dict):
                continue
            key = (
                str(query.get("query_text", "")),
                str(query.get("source_view", "")),
                str(query.get("query_type", "IR")),
            )
            queries.append(
                PipelineQuery(
                    query_text=key[0],
                    query_type=key[2],
                    is_multimodal=bool(query.get("is_multimodal", False)),
                    source_view=key[1],
                    original_query_text=query.get("original_query_text"),
                    decontextualization_note=query.get("decontextualization_note"),
                    related_bullet_indice=query.get("related_bullet_indice"),
                    related_bullet_justification=query.get("related_bullet_justification"),
                    multimodal_rationale=query.get("multimodal_rationale"),
                    retrieved_golden_queries=query.get("retrieved_golden_queries") or [],
                    hard_negative_context=hard_negatives_by_query.get((paper_id, key[0], key[1], key[2])),
                    query_analysis=query_analysis_queries.get(key),
                )
            )

        papers.append(
            PipelinePaper(
                paper_id=paper_id,
                paper_title=paper_title,
                paper_dir=_extract_paper_dir(query_analysis_output_dir, hard_negatives_path, paper_id),
                openreview=_build_openreview_payload(stage0_item, paper_id, query_analysis_item),
                filter_status=_build_filter_status(stage1_item),
                summary_views=_build_summary_views(stage2_item),
                queries=queries,
            )
        )

    return PipelineOutput(
        paths={
            "downloaded_dataset": str(downloaded_path.resolve()) if downloaded_path and downloaded_path.is_file() else None,
            "filtered_dataset": str(filtered_path.resolve()) if filtered_path and filtered_path.is_file() else None,
            "summarized_dataset": str(summarized_path.resolve()) if summarized_path and summarized_path.is_file() else None,
            "queries_dataset": str(queries_path.resolve()) if queries_path and queries_path.is_file() else None,
            "hard_negatives_dataset": str(hard_negatives_path.resolve()) if hard_negatives_path and hard_negatives_path.is_file() else None,
            "query_analysis_dataset": (
                str((query_analysis_output_dir / "query_analysis.json").resolve())
                if query_analysis_output_dir and (query_analysis_output_dir / "query_analysis.json").is_file()
                else None
            ),
        },
        dataset_overview={
            "stage0_total_papers": stage0.get("total_count") if stage0 else None,
            "stage1_total_input": stage1.get("total_input") if stage1 else None,
            "stage1_total_passed": stage1.get("total_passed") if stage1 else None,
            "stage2_total_papers": stage2.get("total_papers") if stage2 else None,
            "stage3_total_papers": stage3.get("total_papers") if stage3 else None,
            "stage3_total_queries": stage3.get("total_queries") if stage3 else None,
            "stage4_total_papers": query_analysis.get("total_papers") if query_analysis else None,
            "stage4_total_queries": query_analysis.get("total_queries") if query_analysis else None,
            "stage5_total_queries": hard_negatives.get("total_queries") if hard_negatives else None,
            "stage5_total_hard_negatives": hard_negatives.get("total_hard_negatives") if hard_negatives else None,
            "stage5_total_positives": hard_negatives.get("total_positives") if hard_negatives else None,
            # Legacy keys kept for existing viewers that still read the old stage labels.
            "stage4_total_hard_negatives": hard_negatives.get("total_hard_negatives") if hard_negatives else None,
            "stage4_total_positives": hard_negatives.get("total_positives") if hard_negatives else None,
            "stage5_total_papers": query_analysis.get("total_papers") if query_analysis else None,
        },
        query_analysis_summary=query_analysis.get("dataset_summary", {}) if query_analysis else {},
        stage5_summary=query_analysis.get("dataset_summary", {}) if query_analysis else {},
        papers=papers,
    )


def write_pipeline_output(path: Path, output: PipelineOutput) -> None:
    save_json(path, output)
