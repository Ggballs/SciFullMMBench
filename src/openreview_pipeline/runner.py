from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import yaml

from openreview_pipeline.llm import OpenAICompatibleBackend
from openreview_pipeline.schemas import DownloadedPapersDataset
from openreview_pipeline.schemas.schemas_filter import FilteredPapersDataset, FilterResult, FilterRuleResult
from openreview_pipeline.schemas.schemas_queries import GeneratedQueriesDataset, GeneratedQueriesForPaper
from openreview_pipeline.utils import load_json, save_json
from openreview_pipeline.stages import (
    DatasetDownloader,
    HardNegativeMiner,
    QueryGenerator,
    RuleBasedFilter,
    Summarizer,
    build_google_scholar_client,
    resolve_hard_negative_llm_settings,
    run as run_stage4_query_analysis,
)
from openreview_pipeline.stages.stage0_download import parse_forum_ids

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

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
) -> dict[str, object]:
    config = load_config(config_path)
    llm_config = config.get("llm", {})
    if not isinstance(llm_config, dict):
        llm_config = {}

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

    return {
        "provider": resolved_provider,
        "serpapi_api_key": serpapi_api_key or search_config.get("serpapi_api_key", ""),
        "max_results": int(resolved_max_results),
        "language": str(resolved_language),
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
):
    settings = resolve_llm_settings(
        config_path,
        base_url=base_url,
        model=model,
    )
    seed = settings.get("seed")
    return OpenAICompatibleBackend(
        base_url=str(settings["base_url"]),
        api_tokens=[str(token) for token in settings["api_tokens"]],
        model=str(settings["model"]),
        max_tokens=int(settings.get("max_tokens", 4096)),
        temperature=float(settings.get("temperature", 0.0)),
        seed=int(seed) if seed is not None else None,
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
):
    settings = resolve_hard_negative_llm_settings(
        load_config(config_path),
        base_url=base_url,
        model=model,
    )
    seed = settings.get("seed")
    return OpenAICompatibleBackend(
        base_url=str(settings["base_url"]),
        api_tokens=[str(token) for token in settings["api_tokens"]],
        model=str(settings["model"]),
        max_tokens=int(settings.get("max_tokens", 4096)),
        temperature=float(settings.get("temperature", 0.0)),
        seed=int(seed) if seed is not None else None,
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
    keep_keys = {
        (
            str(paper.get("paper_id", "")),
            str(query.get("query_text", "")),
            str(query.get("source_view", "")),
            str(query.get("query_type", "IR")),
        )
        for paper in raw_analysis.get("papers", [])
        if isinstance(paper, dict)
        for query in paper.get("queries", [])
        if isinstance(query, dict) and str(query.get("decision", "")).strip() == "Keep"
    }
    if not keep_keys:
        logger.info("Stage-4 analysis produced no surviving queries; hard-negative mining will be skipped.")
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
        "Filtered hard-negative mining input from %s to %s queries using stage-4 Keep decisions.",
        query_dataset.total_queries,
        sum(len(paper.queries_by_view) for paper in filtered_papers),
    )
    return GeneratedQueriesDataset(
        papers_queries=filtered_papers,
        total_papers=len(filtered_papers),
        total_queries=sum(len(paper.queries_by_view) for paper in filtered_papers),
        generated_at=query_dataset.generated_at,
    )


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
    stage_settings = resolve_stage_settings(config_path)
    summarizer = Summarizer(
        llm=llm_backend,
        llm_limit=llm_limit,
        max_concurrent_papers=int(stage_settings["max_concurrent_papers"]),
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
        golden_examples_k=int(generate_settings["golden_examples_k"]),
        queries_per_type_view=generate_settings["queries_per_type_view"],
        bge_model_path=str(generate_settings["bge_model_path"]),
        bge_device=str(generate_settings["bge_device"]),
        embedding_service_url=str(generate_settings.get("embedding_service_url") or "").strip()
        or None,
        embedding_service_timeout=float(generate_settings["embedding_service_timeout"]),
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
    download_selected_pdfs: bool = False,
    llm_backend=None,
) -> Path:
    input_path = _require_input("hard_negative_mining", _normalize_path(input_path))
    output_path = Path(output_path).expanduser().resolve()
    _ensure_parent(output_path)

    llm_backend = llm_backend or build_hard_negative_llm_backend(
        config_path,
        base_url=base_url,
        model=model,
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
        language=str(search_settings["language"]),
    )
    miner = HardNegativeMiner(
        llm=llm_backend,
        scholar_client=scholar_client,
        scholar_max_results=int(search_settings["max_results"]),
        download_selected_pdfs=download_selected_pdfs,
        review_max_workers=int(stage_settings["hard_negative_review_max_workers"]),
    )

    query_dataset = load_json(input_path, GeneratedQueriesDataset)
    filtered_dataset = _filter_queries_for_hard_negative_mining(
        query_dataset=query_dataset,
        query_analysis_output_dir=_normalize_path(query_analysis_output_dir),
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
    )
    return output_dir


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
    username: Optional[str] = None,
    password: Optional[str] = None,
    token: Optional[str] = None,
    scholar_provider: Optional[str] = None,
    serpapi_api_key: Optional[str] = None,
    scholar_max_results: Optional[int] = None,
    scholar_language: Optional[str] = None,
    download_selected_pdfs: bool = False,
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
                download_selected_pdfs=download_selected_pdfs,
            )
            current_input = query_input

    return paths
