from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import yaml
from utils.project_paths import DEFAULT_CONFIG_PATH, REPO_ROOT

from utils.llm import OpenAICompatibleBackend
from openreview_pipeline.schemas import DownloadedPapersDataset
from openreview_pipeline.schemas.schemas_filter import FilteredPapersDataset, FilterResult, FilterRuleResult
from openreview_pipeline.schemas.schemas_queries import GeneratedQueriesDataset, GeneratedQueriesForPaper
from utils import load_json, save_json
from openreview_pipeline.stage0_download import DatasetDownloader, parse_forum_ids
from openreview_pipeline.stage1_filter import RuleBasedFilter
from openreview_pipeline.stage2_summarize import Summarizer
from openreview_pipeline.stage3_generate_queries import QueryGenerator
from openreview_pipeline.stage4_query_analysis import run as run_stage4_query_analysis
from openreview_pipeline.stage5_hard_negative_mining import (
    HardNegativeMiner,
    build_google_scholar_client,
    resolve_hard_negative_llm_settings,
)
from openreview_pipeline.stage5_worker.queue import (
    DEFAULT_BATCH_SIZE as STAGE5_QUEUE_DEFAULT_BATCH_SIZE,
    DEFAULT_DOCLING_POOL_SIZE as STAGE5_QUEUE_DEFAULT_DOCLING_POOL_SIZE,
    DEFAULT_HTTP_PROXY as STAGE5_QUEUE_DEFAULT_HTTP_PROXY,
    DEFAULT_HTTPS_PROXY as STAGE5_QUEUE_DEFAULT_HTTPS_PROXY,
    DEFAULT_MONITOR_INTERVAL as STAGE5_QUEUE_DEFAULT_MONITOR_INTERVAL,
    DEFAULT_QUEUE_TABLE_NAME as STAGE5_QUEUE_DEFAULT_TABLE_NAME,
    DEFAULT_SCHEDULER_MODE as STAGE5_QUEUE_DEFAULT_SCHEDULER_MODE,
    filter_queries_by_paper_ids,
    run_stage5_queue_mode,
)

logger = logging.getLogger(__name__)

LOGICAL_STAGE_ORDER = [
    "download",
    "filter",
    "summarize",
    "generate_queries",
    "query_analysis",
    "hard_negative_mining",
]

STAGE_ALIASES = {
    "0": "download",
    "download": "download",
    "1": "filter",
    "filter": "filter",
    "2": "summarize",
    "summarize": "summarize",
    "3": "generate_queries",
    "generate_queries": "generate_queries",
    "generate-queries": "generate_queries",
    "4": "query_analysis",
    "query_analysis": "query_analysis",
    "query-analysis": "query_analysis",
    "5": "hard_negative_mining",
    "hard_negative_mining": "hard_negative_mining",
    "hard-negative-mining": "hard_negative_mining",
}

DEFAULT_STAGE_FILENAMES = {
    "download": "00_downloaded.json",
    "filter": "01_filtered.json",
    "summarize": "02_summarized.json",
    "generate_queries": "03_queries.json",
    "hard_negative_mining": "05_hard_negatives.json",
}


@dataclass(frozen=True)
class PipelinePaths:
    output_dir: Path
    downloaded_path: Path
    filtered_path: Path
    summarized_path: Path
    queries_path: Path
    query_analysis_output_dir: Path
    hard_negatives_path: Path


def _normalize_path(path: Optional[Path | str]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def load_config(config_path: Optional[Path | str] = None) -> dict:
    resolved_config_path = _normalize_path(config_path) or DEFAULT_CONFIG_PATH
    if resolved_config_path.exists():
        with open(resolved_config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def resolve_llm_settings(
    config_path: Optional[Path | str] = None,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    stage_name: Optional[str] = None,
) -> dict[str, object]:
    config = load_config(config_path)
    llm_config = config.get("llm", {})
    if not isinstance(llm_config, dict):
        llm_config = {}
    stage_llm_config = {}
    if stage_name:
        stages_config = config.get("stages", {})
        if isinstance(stages_config, dict):
            stage_config = stages_config.get(stage_name, {})
            if isinstance(stage_config, dict) and isinstance(stage_config.get("llm"), dict):
                stage_llm_config = stage_config["llm"]

    settings = {
        "base_url": llm_config.get("base_url", ""),
        "model": llm_config.get("model", "gpt-4o-mini"),
        "api_tokens": llm_config.get("api_tokens", []),
        "per_key_request_interval_seconds": llm_config.get("per_key_request_interval_seconds", 0.0),
        "per_key_max_concurrent_requests": llm_config.get("per_key_max_concurrent_requests", 1),
        "max_retries": llm_config.get("max_retries", 3),
        "retry_backoff_seconds": llm_config.get("retry_backoff_seconds", 8.0),
        "max_tokens": llm_config.get("max_tokens", 4096),
        "temperature": llm_config.get("temperature", 0.0),
        "seed": llm_config.get("seed"),
    }
    for key, value in stage_llm_config.items():
        if value is not None:
            settings[key] = value

    if base_url:
        settings["base_url"] = base_url
    if model:
        settings["model"] = model

    if not settings["base_url"]:
        raise ValueError("Missing llm.base_url in config.yaml")
    api_tokens = settings.get("api_tokens")
    if not isinstance(api_tokens, list) or not [token for token in api_tokens if str(token).strip()]:
        raise ValueError("Missing llm.api_tokens in config.yaml; expected a non-empty list")

    return settings


def resolve_stage_settings(
    config_path: Optional[Path | str] = None,
) -> dict[str, int]:
    config = load_config(config_path)
    stages_config = config.get("stages", {})
    if not isinstance(stages_config, dict):
        stages_config = {}
    hard_negative_config = stages_config.get("hard_negative_mining", {})
    if not isinstance(hard_negative_config, dict):
        hard_negative_config = {}
    return {
        "max_concurrent_papers": max(1, int(stages_config.get("max_concurrent_papers", 1))),
        "hard_negative_review_max_workers": max(1, int(hard_negative_config.get("review_max_workers", 1))),
    }


def resolve_generate_query_settings(
    config_path: Optional[Path | str] = None,
) -> dict[str, object]:
    config = load_config(config_path)
    stages_config = config.get("stages", {})
    if not isinstance(stages_config, dict):
        stages_config = {}
    generate_config = stages_config.get("generate_queries", {})
    if not isinstance(generate_config, dict):
        generate_config = {}

    return {
        "golden_embedding_db_url": os.environ.get(
            "SCIFULL_GOLDEN_EMBEDDING_DB_URL",
            str(
                generate_config.get(
                    "golden_embedding_db_url",
                    "postgresql+psycopg://scifull:westlakenlp@127.0.0.1:5432/scifullmmbench",
                )
            ),
        ),
        "golden_examples_k": max(1, int(generate_config.get("golden_examples_k", 5))),
        "queries_per_type_view": generate_config.get("queries_per_type_view", 3),
        "bge_model_path": generate_config.get("bge_model_path", "/data3/yangyinghao/bge-m3"),
        "bge_device": generate_config.get("bge_device", "cuda:2"),
        "embedding_service_url": os.environ.get(
            "SCIFULL_EMBEDDING_SERVICE_URL",
            str(generate_config.get("embedding_service_url", "") or ""),
        ),
        "embedding_service_timeout": float(generate_config.get("embedding_service_timeout", 120.0)),
        "golden_classifications_path": generate_config.get(
            "golden_classifications_path",
            "outputs/query_analysis/golden_retrieval_icl_examples.json",
        ),
        "include_multimodal_queries": bool(generate_config.get("include_multimodal_queries", True)),
    }


def resolve_search_settings(
    config_path: Optional[Path | str] = None,
    *,
    provider: Optional[str] = None,
    serpapi_api_key: Optional[str] = None,
    max_results: Optional[int] = None,
    language: Optional[str] = None,
) -> dict[str, object]:
    config = load_config(config_path)
    search_config = config.get("search", {})
    if not isinstance(search_config, dict):
        search_config = {}

    resolved_provider = provider or search_config.get("provider")
    if not resolved_provider:
        resolved_provider = "serpapi" if (serpapi_api_key or search_config.get("serpapi_api_key")) else "scholarly"

    resolved_max_results = max_results or search_config.get("max_results", 10)
    resolved_language = language or search_config.get("language", "en")
    cache_dir = os.environ.get("SCIFULL_SEARCH_CACHE_DIR")
    if not cache_dir:
        configured_cache_dir = search_config.get("cache_dir")
        cache_dir = str(configured_cache_dir).strip() if configured_cache_dir else ""
    semantic_scholar_api_keys = os.environ.get("SCIFULL_SEMANTIC_SCHOLAR_API_KEYS")
    if semantic_scholar_api_keys:
        resolved_semantic_scholar_api_keys = [
            token.strip()
            for token in semantic_scholar_api_keys.replace("\n", ",").split(",")
            if token.strip()
        ]
    else:
        configured_semantic_keys = search_config.get("semantic_scholar_api_keys", [])
        if isinstance(configured_semantic_keys, str):
            resolved_semantic_scholar_api_keys = [
                token.strip()
                for token in configured_semantic_keys.replace("\n", ",").split(",")
                if token.strip()
            ]
        elif isinstance(configured_semantic_keys, list):
            resolved_semantic_scholar_api_keys = [
                str(token).strip() for token in configured_semantic_keys if str(token).strip()
            ]
        else:
            resolved_semantic_scholar_api_keys = []

    return {
        "provider": resolved_provider,
        "serpapi_api_key": serpapi_api_key or search_config.get("serpapi_api_key", ""),
        "semantic_scholar_api_keys": resolved_semantic_scholar_api_keys,
        "max_results": int(resolved_max_results),
        "language": str(resolved_language),
        "timeout_seconds": float(
            os.environ.get(
                "SCIFULL_SEARCH_TIMEOUT_SECONDS",
                search_config.get("timeout_seconds", 30.0),
            )
        ),
        "min_interval_seconds": float(
            os.environ.get(
                "SCIFULL_SEARCH_MIN_INTERVAL_SECONDS",
                search_config.get("min_interval_seconds", 0.0),
            )
        ),
        "max_retries": int(
            os.environ.get(
                "SCIFULL_SEARCH_MAX_RETRIES",
                search_config.get("max_retries", 3),
            )
        ),
        "retry_backoff_seconds": float(
            os.environ.get(
                "SCIFULL_SEARCH_RETRY_BACKOFF_SECONDS",
                search_config.get("retry_backoff_seconds", 3.0),
            )
        ),
        "retry_backoff_multiplier": float(
            os.environ.get(
                "SCIFULL_SEARCH_RETRY_BACKOFF_MULTIPLIER",
                search_config.get("retry_backoff_multiplier", 2.0),
            )
        ),
        "cache_dir": cache_dir,
    }


def resolve_openreview_credentials(
    config_path: Optional[Path | str] = None,
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
    token: Optional[str] = None,
) -> dict[str, str]:
    config = load_config(config_path)
    openreview_config = config.get("openreview", {})
    return {
        "username": username if username is not None else openreview_config.get("username", ""),
        "password": password if password is not None else openreview_config.get("password", ""),
        "token": token if token is not None else openreview_config.get("token", ""),
    }


def build_llm_backend(
    config_path: Optional[Path | str] = None,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    stage_name: Optional[str] = None,
):
    settings = resolve_llm_settings(
        config_path,
        base_url=base_url,
        model=model,
        stage_name=stage_name,
    )
    seed = settings.get("seed")
    return OpenAICompatibleBackend(
        base_url=str(settings["base_url"]),
        api_tokens=[str(token) for token in settings["api_tokens"]],
        model=str(settings["model"]),
        max_tokens=int(settings.get("max_tokens", 4096)),
        temperature=float(settings.get("temperature", 0.0)),
        seed=int(seed) if seed is not None else None,
        reasoning_effort=(
            str(settings.get("reasoning_effort")).strip()
            if settings.get("reasoning_effort") is not None
            else None
        ),
        per_key_request_interval_seconds=float(settings.get("per_key_request_interval_seconds", 0.0)),
        per_key_max_concurrent_requests=int(settings.get("per_key_max_concurrent_requests", 1)),
        max_retries=int(settings.get("max_retries", 3)),
        retry_backoff_seconds=float(settings.get("retry_backoff_seconds", 8.0)),
    )


def build_hard_negative_llm_backend(
    config_path: Optional[Path | str] = None,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    per_key_request_interval_seconds_override: Optional[float] = None,
    per_key_max_concurrent_requests_override: Optional[int] = None,
):
    settings = resolve_hard_negative_llm_settings(
        load_config(config_path),
        base_url=base_url,
        model=model,
    )
    if per_key_request_interval_seconds_override is not None:
        settings["per_key_request_interval_seconds"] = float(per_key_request_interval_seconds_override)
    if per_key_max_concurrent_requests_override is not None:
        settings["per_key_max_concurrent_requests"] = max(
            1,
            int(per_key_max_concurrent_requests_override),
        )
    seed = settings.get("seed")
    return OpenAICompatibleBackend(
        base_url=str(settings["base_url"]),
        api_tokens=[str(token) for token in settings["api_tokens"]],
        model=str(settings["model"]),
        max_tokens=int(settings.get("max_tokens", 4096)),
        temperature=float(settings.get("temperature", 0.0)),
        seed=int(seed) if seed is not None else None,
        reasoning_effort=(
            str(settings.get("reasoning_effort")).strip()
            if settings.get("reasoning_effort") is not None
            else None
        ),
        per_key_request_interval_seconds=float(settings.get("per_key_request_interval_seconds", 0.0)),
        per_key_max_concurrent_requests=int(settings.get("per_key_max_concurrent_requests", 1)),
        max_retries=int(settings.get("max_retries", 3)),
        retry_backoff_seconds=float(settings.get("retry_backoff_seconds", 8.0)),
    )


def normalize_stage_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized in STAGE_ALIASES:
        return STAGE_ALIASES[normalized]
    raise ValueError(f"Unknown stage: {name}")


def parse_stage_spec(stage_spec: str | Sequence[str]) -> list[str]:
    if isinstance(stage_spec, str):
        raw_tokens = [token.strip() for token in stage_spec.split(",") if token.strip()]
    else:
        raw_tokens = [str(token).strip() for token in stage_spec if str(token).strip()]

    selected: list[str] = []
    for token in raw_tokens:
        lowered = token.lower()
        if lowered in STAGE_ALIASES:
            selected.append(STAGE_ALIASES[lowered])
            continue
        if "-" in token:
            start_token, end_token = token.split("-", 1)
            start_stage = normalize_stage_name(start_token)
            end_stage = normalize_stage_name(end_token)
            start_index = LOGICAL_STAGE_ORDER.index(start_stage)
            end_index = LOGICAL_STAGE_ORDER.index(end_stage)
            if start_index > end_index:
                raise ValueError(f"Invalid descending stage range: {token}")
            selected.extend(LOGICAL_STAGE_ORDER[start_index : end_index + 1])
            continue
        raise ValueError(f"Unknown stage token: {token}")

    ordered = [stage for stage in LOGICAL_STAGE_ORDER if stage in set(selected)]
    if not ordered:
        raise ValueError("No stages selected.")

    indices = [LOGICAL_STAGE_ORDER.index(stage) for stage in ordered]
    expected = list(range(indices[0], indices[-1] + 1))
    if indices != expected:
        raise ValueError(
            "Selected stages must form a contiguous slice of the pipeline "
            f"(got: {', '.join(ordered)})."
        )
    return ordered


def resolve_pipeline_paths(
    *,
    output_dir: Optional[Path | str] = None,
    downloaded_path: Optional[Path | str] = None,
    filtered_path: Optional[Path | str] = None,
    summarized_path: Optional[Path | str] = None,
    queries_path: Optional[Path | str] = None,
    query_analysis_output_dir: Optional[Path | str] = None,
    hard_negatives_path: Optional[Path | str] = None,
) -> PipelinePaths:
    normalized_overrides = [
        _normalize_path(downloaded_path),
        _normalize_path(filtered_path),
        _normalize_path(summarized_path),
        _normalize_path(queries_path),
        _normalize_path(query_analysis_output_dir),
        _normalize_path(hard_negatives_path),
    ]
    base_dir = _normalize_path(output_dir)
    if base_dir is None:
        first_override = next((path for path in normalized_overrides if path is not None), None)
        base_dir = first_override.parent if first_override else (REPO_ROOT / "data").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    def stage_path(override: Optional[Path | str], stage_name: str) -> Path:
        normalized = _normalize_path(override)
        return normalized or (base_dir / DEFAULT_STAGE_FILENAMES[stage_name])

    return PipelinePaths(
        output_dir=base_dir,
        downloaded_path=stage_path(downloaded_path, "download"),
        filtered_path=stage_path(filtered_path, "filter"),
        summarized_path=stage_path(summarized_path, "summarize"),
        queries_path=stage_path(queries_path, "generate_queries"),
        query_analysis_output_dir=_normalize_path(query_analysis_output_dir) or (base_dir / "04_query_analysis"),
        hard_negatives_path=stage_path(hard_negatives_path, "hard_negative_mining"),
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _require_input(stage_name: str, input_path: Optional[Path]) -> Path:
    if input_path is None:
        raise ValueError(f"Stage '{stage_name}' requires an input path.")
    resolved = input_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Input file not found for stage '{stage_name}': {resolved}")
    return resolved


def _filter_queries_for_hard_negative_mining(
    *,
    query_dataset: GeneratedQueriesDataset,
    query_analysis_output_dir: Optional[Path],
) -> GeneratedQueriesDataset:
    if query_analysis_output_dir is None:
        return query_dataset

    analysis_path = query_analysis_output_dir / "query_analysis.json"
    if not analysis_path.is_file():
        return query_dataset

    with analysis_path.open("r", encoding="utf-8") as handle:
        raw_analysis = json.load(handle)

    required_low_query_views = {"motivation", "method", "experiment/result"}
    low_overlap_labels = {"LOW-LEXICAL-OVERLAP", "LOW-SEMANTIC-OVERLAP"}
    eligible_paper_ids = set()
    for paper in raw_analysis.get("papers", []):
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("paper_id", "")).strip()
        if not paper_id:
            continue
        low_query_views = {
            str(query.get("source_view", "")).strip()
            for query in paper.get("queries", [])
            if isinstance(query, dict)
            and str(((query.get("retrieval_evaluation") or {}).get("abstract_relevance", ""))).strip()
            in low_overlap_labels
        }
        if required_low_query_views.issubset(low_query_views):
            eligible_paper_ids.add(paper_id)

    keep_keys = {
        (
            str(paper.get("paper_id", "")),
            str(query.get("query_text", "")),
            str(query.get("source_view", "")),
            str(query.get("query_type", "IR")),
        )
        for paper in raw_analysis.get("papers", [])
        if isinstance(paper, dict) and str(paper.get("paper_id", "")).strip() in eligible_paper_ids
        for query in paper.get("queries", [])
        if isinstance(query, dict) and str(query.get("decision", "")).strip() == "Keep"
    }
    if not keep_keys:
        logger.info(
            "Stage-4 analysis produced no surviving queries after requiring low-query coverage in all 3 views; "
            "hard-negative mining will be skipped."
        )
        return GeneratedQueriesDataset(papers_queries=[], total_papers=0, total_queries=0)

    filtered_papers = []
    for paper in query_dataset.papers_queries:
        kept_queries = [
            query
            for query in paper.queries_by_view
            if (paper.paper_id, query.query_text, query.source_view, query.query_type) in keep_keys
        ]
        if not kept_queries:
            continue
        filtered_papers.append(
            GeneratedQueriesForPaper(
                paper_id=paper.paper_id,
                paper_title=paper.paper_title,
                queries_by_view=kept_queries,
                generated_at=paper.generated_at,
            )
        )

    logger.info(
        "Filtered hard-negative mining input from %s papers / %s queries to %s eligible papers / %s surviving queries "
        "using stage-4 paper-level low-query coverage and query-level Keep decisions.",
        query_dataset.total_papers,
        query_dataset.total_queries,
        len(filtered_papers),
        sum(len(paper.queries_by_view) for paper in filtered_papers),
    )
    return GeneratedQueriesDataset(
        papers_queries=filtered_papers,
        total_papers=len(filtered_papers),
        total_queries=sum(len(paper.queries_by_view) for paper in filtered_papers),
        generated_at=query_dataset.generated_at,
    )


def _resolve_stage5_queue_paths(
    output_path: Path,
    *,
    pdf_output_dir: Optional[Path | str] = None,
    monitor_file: Optional[Path | str] = None,
    retrieve_done_flag: Optional[Path | str] = None,
) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    queue_work_dir = output_path.parent / f"{output_path.stem}_queue"
    resolved_pdf_output_dir = (
        Path(pdf_output_dir).expanduser().resolve()
        if pdf_output_dir is not None
        else (queue_work_dir / "hard_negative_pdfs")
    )
    resolved_monitor_file = (
        Path(monitor_file).expanduser().resolve()
        if monitor_file is not None
        else (queue_work_dir / "stage5_queue_monitor.json")
    )
    resolved_retrieve_done_flag = (
        Path(retrieve_done_flag).expanduser().resolve()
        if retrieve_done_flag is not None
        else (queue_work_dir / "retrieve_done.flag")
    )
    return resolved_pdf_output_dir, resolved_monitor_file, resolved_retrieve_done_flag


def run_download_stage(
    *,
    output_path: Path | str,
    venue: str = "ICLR",
    year: Optional[int] = None,
    config_path: Optional[Path | str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    token: Optional[str] = None,
    limit: Optional[int] = None,
    forum_id: Optional[str] = None,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    _ensure_parent(output_path)

    target_year = year or datetime.now().year
    credentials = resolve_openreview_credentials(
        config_path,
        username=username,
        password=password,
        token=token,
    )

    downloader = DatasetDownloader(
        venue=venue,
        year_threshold=target_year,
        output_dir=str(output_path.parent),
    )
    downloader.set_openreview_credentials(
        username=credentials["username"],
        password=credentials["password"],
        token=credentials["token"],
    )
    if not any(credentials.values()):
        logger.info("No OpenReview credentials configured. Using public OpenReview API client.")

    forum_ids = parse_forum_ids(forum_id)
    downloader.run(output_path, limit=limit, forum_ids=forum_ids)
    return output_path


def run_filter_stage(
    *,
    input_path: Path | str,
    output_path: Path | str,
    rules_config_path: Optional[Path | str] = None,
    limit: Optional[int] = None,
) -> Path:
    input_path = _require_input("filter", _normalize_path(input_path))
    output_path = Path(output_path).expanduser().resolve()
    _ensure_parent(output_path)

    filter_stage = RuleBasedFilter(
        config_path=_normalize_path(rules_config_path),
        limit=limit,
    )
    filter_stage.run(input_path, output_path)
    return output_path


def run_skip_filter_stage(
    *,
    input_path: Path | str,
    output_path: Path | str,
) -> Path:
    input_path = _require_input("skip_filter", _normalize_path(input_path))
    output_path = Path(output_path).expanduser().resolve()
    _ensure_parent(output_path)

    dataset = load_json(input_path, DownloadedPapersDataset)
    target_ids = {paper.paper.id for paper in dataset.papers}
    existing_results = []
    completed_ids = set()
    if output_path.exists():
        try:
            checkpoint = load_json(output_path, FilteredPapersDataset)
            existing_results = [
                result for result in checkpoint.results if result.paper.paper.id in target_ids
            ]
            completed_ids = {result.paper.paper.id for result in existing_results}
            if completed_ids:
                logger.info(
                    "Loaded %s existing skip-filter results from %s",
                    len(completed_ids),
                    output_path,
                )
        except Exception as exc:
            logger.warning("Could not load skip-filter checkpoint %s: %s", output_path, exc)
    results = [
        FilterResult(
            paper=paper,
            passed=True,
            details=FilterRuleResult(
                accepted=True,
                similar_paper=False,
                multimodal_info=True,
            ),
        )
        for paper in dataset.papers
        if paper.paper.id not in completed_ids
    ]
    results_by_id = {result.paper.paper.id: result for result in existing_results + results}
    results = [
        results_by_id[paper.paper.id]
        for paper in dataset.papers
        if paper.paper.id in results_by_id
    ]
    result = FilteredPapersDataset(
        results=results,
        total_input=len(results),
        total_passed=len(results),
        total_filtered=0,
    )
    logger.info(
        "Skip-filter stage success: %s/%s papers passed (%.1f%%).",
        len(results),
        len(results),
        100.0,
    )
    save_json(output_path, result)
    return output_path


def run_summarize_stage(
    *,
    input_path: Path | str,
    output_path: Path | str,
    config_path: Optional[Path | str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    llm_limit: Optional[int] = None,
    llm_backend=None,
) -> Path:
    input_path = _require_input("summarize", _normalize_path(input_path))
    output_path = Path(output_path).expanduser().resolve()
    _ensure_parent(output_path)

    llm_backend = llm_backend or build_llm_backend(
        config_path,
        base_url=base_url,
        model=model,
    )
    config = load_config(config_path)
    stages_config = config.get("stages", {}) if isinstance(config.get("stages"), dict) else {}
    summarize_config = stages_config.get("summarize", {}) if isinstance(stages_config.get("summarize"), dict) else {}
    stage_settings = resolve_stage_settings(config_path)
    summarizer = Summarizer(
        llm=llm_backend,
        llm_limit=llm_limit,
        max_concurrent_papers=int(stage_settings["max_concurrent_papers"]),
        include_multimodal_evidence=bool(summarize_config.get("include_multimodal_evidence", True)),
    )
    summarizer.run(input_path, output_path)
    return output_path


def run_generate_queries_stage(
    *,
    input_path: Path | str,
    output_path: Path | str,
    config_path: Optional[Path | str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    llm_backend=None,
) -> Path:
    input_path = _require_input("generate_queries", _normalize_path(input_path))
    output_path = Path(output_path).expanduser().resolve()
    _ensure_parent(output_path)

    llm_backend = llm_backend or build_llm_backend(
        config_path,
        base_url=base_url,
        model=model,
    )
    stage_settings = resolve_stage_settings(config_path)
    generate_settings = resolve_generate_query_settings(config_path)
    generator = QueryGenerator(
        llm=llm_backend,
        max_concurrent_papers=int(stage_settings["max_concurrent_papers"]),
        golden_embedding_db_url=str(generate_settings["golden_embedding_db_url"]),
        golden_classifications_path=str(generate_settings["golden_classifications_path"]),
        golden_examples_k=int(generate_settings["golden_examples_k"]),
        queries_per_type_view=generate_settings["queries_per_type_view"],
        bge_model_path=str(generate_settings["bge_model_path"]),
        bge_device=str(generate_settings["bge_device"]),
        embedding_service_url=str(generate_settings.get("embedding_service_url") or "").strip()
        or None,
        embedding_service_timeout=float(generate_settings["embedding_service_timeout"]),
        include_multimodal_queries=bool(generate_settings.get("include_multimodal_queries", True)),
    )
    generator.run(input_path, output_path)
    return output_path


def run_hard_negative_mining_stage(
    *,
    input_path: Path | str,
    output_path: Path | str,
    query_analysis_output_dir: Optional[Path | str] = None,
    config_path: Optional[Path | str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    scholar_provider: Optional[str] = None,
    serpapi_api_key: Optional[str] = None,
    scholar_max_results: Optional[int] = None,
    scholar_language: Optional[str] = None,
    mode: str = "direct",
    queue_table_name: str = STAGE5_QUEUE_DEFAULT_TABLE_NAME,
    scheduler_mode: str = STAGE5_QUEUE_DEFAULT_SCHEDULER_MODE,
    batch_size: int = STAGE5_QUEUE_DEFAULT_BATCH_SIZE,
    review_max_workers: Optional[int] = None,
    docling_pool_size: int = STAGE5_QUEUE_DEFAULT_DOCLING_POOL_SIZE,
    http_proxy: str = STAGE5_QUEUE_DEFAULT_HTTP_PROXY,
    https_proxy: str = STAGE5_QUEUE_DEFAULT_HTTPS_PROXY,
    pdf_output_dir: Optional[Path | str] = None,
    monitor_file: Optional[Path | str] = None,
    monitor_interval: int = STAGE5_QUEUE_DEFAULT_MONITOR_INTERVAL,
    retrieve_done_flag: Optional[Path | str] = None,
    query_subset_paper_ids: Optional[Sequence[str]] = None,
    apply_query_analysis_filter: bool = True,
    retrieval_only: bool = False,
    parse_only: bool = False,
    review_only: bool = False,
    download_only: bool = False,
    parse_min_query_candidates: int = 25,
    retrieval_search_max_workers: int = 1,
    retrieval_rerank_max_workers: int = 1,
    task_filter: Optional[str] = None,
    download_workers: int = 1,
    parse_workers: int = 8,
    review_workers: int = 200,
    llm_backend=None,
) -> Path:
    input_path = _require_input("hard_negative_mining", _normalize_path(input_path))
    output_path = Path(output_path).expanduser().resolve()
    _ensure_parent(output_path)
    normalized_mode = str(mode).strip().lower() or "direct"
    if normalized_mode not in {"direct", "queue"}:
        raise ValueError(f"Unsupported hard-negative mining mode: {mode!r}")

    llm_backend_overrides: dict[str, object] = {}
    if normalized_mode == "queue" and retrieval_only:
        hard_negative_settings = resolve_hard_negative_llm_settings(
            load_config(config_path),
            base_url=base_url,
            model=model,
        )
        key_count = max(1, len([str(token) for token in hard_negative_settings.get("api_tokens", [])]))
        rerank_workers = max(1, int(retrieval_rerank_max_workers))
        per_key_max_concurrent_requests = math.ceil(rerank_workers / key_count) + 1
        llm_backend_overrides = {
            "per_key_request_interval_seconds_override": 1.0,
            "per_key_max_concurrent_requests_override": per_key_max_concurrent_requests,
        }
        logger.info(
            "Tuning hard-negative retrieval LLM concurrency: rerank_workers=%s key_count=%s "
            "per_key_max_concurrent_requests=%s per_key_request_interval_seconds=1.0",
            rerank_workers,
            key_count,
            per_key_max_concurrent_requests,
        )

    llm_backend = llm_backend or build_hard_negative_llm_backend(
        config_path,
        base_url=base_url,
        model=model,
        **llm_backend_overrides,
    )
    stage_settings = resolve_stage_settings(config_path)
    search_settings = resolve_search_settings(
        config_path,
        provider=scholar_provider,
        serpapi_api_key=serpapi_api_key,
        max_results=scholar_max_results,
        language=scholar_language,
    )
    scholar_client = build_google_scholar_client(
        str(search_settings["provider"]),
        serpapi_api_key=str(search_settings["serpapi_api_key"]),
        semantic_scholar_api_keys=[
            str(token)
            for token in search_settings.get("semantic_scholar_api_keys", [])
            if str(token).strip()
        ],
        language=str(search_settings["language"]),
        timeout_seconds=float(search_settings["timeout_seconds"]),
        min_interval_seconds=float(search_settings["min_interval_seconds"]),
        max_retries=int(search_settings["max_retries"]),
        retry_backoff_seconds=float(search_settings["retry_backoff_seconds"]),
        retry_backoff_multiplier=float(search_settings["retry_backoff_multiplier"]),
        cache_dir=Path(str(search_settings["cache_dir"])).expanduser().resolve()
        if str(search_settings["cache_dir"]).strip()
        else None,
    )
    query_dataset = load_json(input_path, GeneratedQueriesDataset)
    filtered_dataset = filter_queries_by_paper_ids(query_dataset, query_subset_paper_ids)
    if apply_query_analysis_filter:
        filtered_dataset = _filter_queries_for_hard_negative_mining(
            query_dataset=filtered_dataset,
            query_analysis_output_dir=_normalize_path(query_analysis_output_dir),
        )

    if normalized_mode == "queue":
        resolved_review_max_workers = int(review_max_workers or stage_settings["hard_negative_review_max_workers"])
        resolved_pdf_output_dir, resolved_monitor_file, resolved_retrieve_done_flag = _resolve_stage5_queue_paths(
            output_path,
            pdf_output_dir=pdf_output_dir,
            monitor_file=monitor_file,
            retrieve_done_flag=retrieve_done_flag,
        )
        return run_stage5_queue_mode(
            query_dataset=filtered_dataset,
            llm_backend=llm_backend,
            search_settings=search_settings,
            output_path=output_path,
            queue_table_name=queue_table_name,
            scheduler_mode=scheduler_mode,
            batch_size=int(batch_size),
            review_max_workers=resolved_review_max_workers,
            docling_pool_size=int(docling_pool_size),
            http_proxy=str(http_proxy),
            https_proxy=str(https_proxy),
            pdf_output_dir=resolved_pdf_output_dir,
            monitor_file=resolved_monitor_file,
            monitor_interval=int(monitor_interval),
            retrieve_done_flag=resolved_retrieve_done_flag,
            skip_existing_queries=bool(retrieval_only),
            retrieval_only=bool(retrieval_only),
            parse_only=bool(parse_only),
            review_only=bool(review_only),
            download_only=bool(download_only),
            parse_min_query_candidates=int(parse_min_query_candidates),
            retrieval_search_max_workers=int(retrieval_search_max_workers),
            retrieval_rerank_max_workers=int(retrieval_rerank_max_workers),
            download_workers=int(download_workers),
            parse_workers=int(parse_workers),
            review_workers=int(review_workers),
            task_filter=task_filter,
        )

    miner = HardNegativeMiner(
        llm=llm_backend,
        scholar_client=scholar_client,
        scholar_max_results=int(search_settings["max_results"]),
        review_max_workers=int(review_max_workers or stage_settings["hard_negative_review_max_workers"]),
    )
    save_json(output_path, miner.apply(filtered_dataset, checkpoint_path=output_path))
    return output_path


def run_query_analysis_stage(
    *,
    summarized_path: Path | str,
    queries_path: Path | str,
    output_dir: Path | str,
    config_path: Optional[Path | str] = None,
    downloaded_path: Optional[Path | str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    analysis_modes: Optional[Sequence[str]] = None,
    llm_backend=None,
) -> Path:
    summarized_path = _require_input("query_analysis", _normalize_path(summarized_path))
    queries_path = _require_input("query_analysis", _normalize_path(queries_path))
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_backend = llm_backend or build_llm_backend(
        config_path,
        base_url=base_url,
        model=model,
        stage_name="query_analysis",
    )
    stage_settings = resolve_stage_settings(config_path)
    run_stage4_query_analysis(
        llm=llm_backend,
        summarized_path=summarized_path,
        queries_path=queries_path,
        output_dir=output_dir,
        config_path=_normalize_path(config_path) or DEFAULT_CONFIG_PATH,
        downloaded_path=_normalize_path(downloaded_path),
        max_concurrent_papers=int(stage_settings["max_concurrent_papers"]),
        analysis_modes=list(analysis_modes) if analysis_modes else None,
    )
    return output_dir


def _generate_simplified_queries(queries_path: Path) -> None:
    try:
        from scripts.simplify_queries import simplify
        output = queries_path.parent / "03_queries_simplified.json"
        if queries_path.exists():
            simplify(str(queries_path), str(output))
    except Exception:
        pass


def run_selected_stages(
    stage_spec: str | Sequence[str],
    *,
    input_path: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    downloaded_path: Optional[Path | str] = None,
    filtered_path: Optional[Path | str] = None,
    summarized_path: Optional[Path | str] = None,
    queries_path: Optional[Path | str] = None,
    query_analysis_output_dir: Optional[Path | str] = None,
    hard_negatives_path: Optional[Path | str] = None,
    venue: str = "ICLR",
    year: Optional[int] = None,
    forum_id: Optional[str] = None,
    download_limit: Optional[int] = None,
    filter_limit: Optional[int] = None,
    llm_limit: Optional[int] = None,
    rules_config_path: Optional[Path | str] = None,
    config_path: Optional[Path | str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    query_analysis_modes: Optional[Sequence[str]] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    token: Optional[str] = None,
    scholar_provider: Optional[str] = None,
    serpapi_api_key: Optional[str] = None,
    scholar_max_results: Optional[int] = None,
    scholar_language: Optional[str] = None,
    skip_filter: bool = False,
) -> PipelinePaths:
    stages = parse_stage_spec(stage_spec)
    current_input = _normalize_path(input_path)
    if stages[0] != "download" and current_input is None:
        raise ValueError(
            f"An input path is required when starting from stage '{stages[0]}'."
        )

    paths = resolve_pipeline_paths(
        output_dir=output_dir,
        downloaded_path=downloaded_path,
        filtered_path=filtered_path,
        summarized_path=summarized_path,
        queries_path=queries_path,
        query_analysis_output_dir=query_analysis_output_dir,
        hard_negatives_path=hard_negatives_path,
    )

    downloaded_source = paths.downloaded_path if paths.downloaded_path.is_file() else None
    filtered_source = paths.filtered_path if paths.filtered_path.is_file() else None
    summarized_source = paths.summarized_path if paths.summarized_path.is_file() else None
    queries_source = paths.queries_path if paths.queries_path.is_file() else None

    for stage in stages:
        if stage == "download":
            current_input = run_download_stage(
                output_path=paths.downloaded_path,
                venue=venue,
                year=year,
                config_path=config_path,
                username=username,
                password=password,
                token=token,
                limit=download_limit,
                forum_id=forum_id,
            )
            downloaded_source = current_input
        elif stage == "filter":
            if skip_filter:
                current_input = run_skip_filter_stage(
                    input_path=_require_input(stage, current_input),
                    output_path=paths.filtered_path,
                )
            else:
                current_input = run_filter_stage(
                    input_path=_require_input(stage, current_input),
                    output_path=paths.filtered_path,
                    rules_config_path=rules_config_path,
                    limit=filter_limit,
                )
            filtered_source = current_input
        elif stage == "summarize":
            current_input = run_summarize_stage(
                input_path=_require_input(stage, current_input),
                output_path=paths.summarized_path,
                config_path=config_path,
                base_url=base_url,
                model=model,
                llm_limit=llm_limit,
            )
            summarized_source = current_input
        elif stage == "generate_queries":
            current_input = run_generate_queries_stage(
                input_path=_require_input(stage, current_input),
                output_path=paths.queries_path,
                config_path=config_path,
                base_url=base_url,
                model=model,
            )
            queries_source = current_input
            _generate_simplified_queries(paths.queries_path)
        elif stage == "query_analysis":
            summary_input = summarized_source or paths.summarized_path
            query_input = queries_source or paths.queries_path
            run_query_analysis_stage(
                summarized_path=_require_input(stage, summary_input),
                queries_path=_require_input(stage, query_input),
                output_dir=paths.query_analysis_output_dir,
                downloaded_path=downloaded_source,
                config_path=config_path,
                base_url=base_url,
                model=model,
                analysis_modes=query_analysis_modes,
            )
            current_input = query_input
        elif stage == "hard_negative_mining":
            query_input = _require_input(
                stage,
                queries_source or current_input or paths.queries_path,
            )
            run_hard_negative_mining_stage(
                input_path=query_input,
                output_path=paths.hard_negatives_path,
                query_analysis_output_dir=(
                    paths.query_analysis_output_dir
                    if paths.query_analysis_output_dir.is_dir()
                    else None
                ),
                config_path=config_path,
                base_url=base_url,
                model=model,
                scholar_provider=scholar_provider,
                serpapi_api_key=serpapi_api_key,
                scholar_max_results=scholar_max_results,
                scholar_language=scholar_language,
            )
            current_input = query_input

    return paths
