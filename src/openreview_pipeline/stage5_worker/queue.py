from __future__ import annotations

import json
import logging
import os
import resource
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from openreview_pipeline.schemas.schemas_queries import GeneratedQueriesDataset, GeneratedQueriesForPaper
from openreview_pipeline.stage5_worker.core import HardNegativeMiner, ScholarCandidatePaper, build_google_scholar_client
from openreview_pipeline.stage5_worker.runtime import (
    configure_stage5_runtime,
    get_docling_runtime_snapshot,
    get_pdf_download_runtime_snapshot,
)
from utils.db.stage5_candidate_queue_postgres import (
    batch_update_download_results,
    build_candidate_key,
    build_query_key,
    claim_download_candidates,
    claim_parse_candidates,
    claim_pending_candidates,
    close_satisfied_query_pending_candidates,
    ensure_schema,
    get_engine,
    load_label_summary,
    load_queue_snapshot,
    load_status_summary,
    reset_stale_processing_candidates,
    resolve_queue_storage_names,
    update_candidate_result,
    update_download_result,
    upsert_candidates,
)

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_TABLE_NAME = ""
DEFAULT_SCHEDULER_MODE = "independent_worker"
DEFAULT_HTTP_PROXY = "http://127.0.0.1:7890"
DEFAULT_HTTPS_PROXY = "http://127.0.0.1:7890"
DEFAULT_DOCLING_POOL_SIZE = 4
DEFAULT_MONITOR_INTERVAL = 5
DEFAULT_STALE_PROCESSING_SECONDS = 600
DEFAULT_BATCH_SIZE = 20


@dataclass(frozen=True)
class QueueRunPaths:
    work_dir: Path
    effective_queries_path: Path
    monitor_file: Path
    retrieve_done_flag: Path
    pdf_output_dir: Path


def build_queue_run_paths(
    *,
    output_path: Path,
    pdf_output_dir: Optional[Path | str] = None,
    monitor_file: Optional[Path | str] = None,
    retrieve_done_flag: Optional[Path | str] = None,
) -> QueueRunPaths:
    work_dir = output_path.parent / f"{output_path.stem}_queue"
    work_dir.mkdir(parents=True, exist_ok=True)
    return QueueRunPaths(
        work_dir=work_dir,
        effective_queries_path=work_dir / "03_queries_effective.json",
        monitor_file=Path(monitor_file).expanduser().resolve() if monitor_file else (work_dir / "stage5_queue_monitor.json"),
        retrieve_done_flag=Path(retrieve_done_flag).expanduser().resolve() if retrieve_done_flag else (work_dir / "retrieve_done.flag"),
        pdf_output_dir=Path(pdf_output_dir).expanduser().resolve() if pdf_output_dir else (work_dir / "hard_negative_pdfs"),
    )


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_generated_queries_dataset(path: Path, dataset: GeneratedQueriesDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")


def filter_queries_by_paper_ids(
    query_dataset: GeneratedQueriesDataset,
    paper_ids: Optional[Sequence[str]],
) -> GeneratedQueriesDataset:
    normalized = {str(paper_id).strip() for paper_id in (paper_ids or []) if str(paper_id).strip()}
    if not normalized:
        return query_dataset

    filtered_papers = [paper for paper in query_dataset.papers_queries if paper.paper_id in normalized]
    return GeneratedQueriesDataset(
        papers_queries=filtered_papers,
        total_papers=len(filtered_papers),
        total_queries=sum(len(paper.queries_by_view) for paper in filtered_papers),
        generated_at=query_dataset.generated_at,
    )


def _row_to_candidate(row: dict[str, Any]) -> ScholarCandidatePaper:
    review_payload = row.get("review_payload") or {}
    if not isinstance(review_payload, dict):
        review_payload = {}
    return ScholarCandidatePaper(
        paper_title=str(row.get("candidate_title") or ""),
        arxiv_id=str(row.get("candidate_arxiv_id")).strip() if row.get("candidate_arxiv_id") else None,
        abstract=str(row.get("candidate_abstract")).strip() if row.get("candidate_abstract") else None,
        venue=str(row.get("candidate_venue")).strip() if row.get("candidate_venue") else None,
        year=int(row["candidate_year"]) if row.get("candidate_year") is not None else None,
        authors=[str(author) for author in (row.get("candidate_authors") or [])],
        url=str(row.get("candidate_url")).strip() if row.get("candidate_url") else None,
        pdf_url=str(row.get("candidate_pdf_url")).strip() if row.get("candidate_pdf_url") else None,
        pdf_path=str(review_payload.get("candidate_pdf_path")).strip() if review_payload.get("candidate_pdf_path") else None,
        full_text_path=(
            str(review_payload.get("candidate_full_text_path")).strip()
            if review_payload.get("candidate_full_text_path")
            else None
        ),
        citations=int(row["candidate_citations"]) if row.get("candidate_citations") is not None else None,
        source="stage5_queue",
    )


@dataclass
class WorkerSummary:
    worker_id: int
    processed_rows: int = 0


@dataclass
class WorkerRuntime:
    worker_id: int
    status: str = "starting"
    stage: str = "starting"
    stage_started_at: float = 0.0
    row_started_at: float = 0.0
    last_update_at: float = 0.0
    current_row_id: Optional[int] = None
    current_query_key: Optional[str] = None
    current_rank: Optional[int] = None
    current_title: Optional[str] = None
    processed_rows: int = 0
    last_result: Optional[str] = None
    last_parse_status: Optional[str] = None
    last_error: Optional[str] = None
    docling_initialized: bool = False
    docling_parse_count: int = 0
    last_docling_parse_seconds: float = 0.0
    last_docling_error: Optional[str] = None
    waiting_for_docling: bool = False
    holding_docling_slot: bool = False
    docling_slot_id: Optional[int] = None

    def as_snapshot(self, now_monotonic: float, stale_after_seconds: float, stuck_after_seconds: float) -> dict[str, Any]:
        last_update_age = max(0.0, now_monotonic - self.last_update_at) if self.last_update_at else 0.0
        stage_elapsed = max(0.0, now_monotonic - self.stage_started_at) if self.stage_started_at else 0.0
        if self.status == "exited":
            health = "exited"
        elif last_update_age >= stale_after_seconds:
            health = "silent"
        elif stage_elapsed >= stuck_after_seconds and self.stage not in {"idle", "claimed"}:
            health = "long_running"
        else:
            health = "healthy"
        return {
            "worker_id": self.worker_id,
            "health": health,
            "status": self.status,
            "stage": self.stage,
            "stage_elapsed_seconds": stage_elapsed,
            "row_elapsed_seconds": max(0.0, now_monotonic - self.row_started_at) if self.row_started_at else 0.0,
            "last_update_age_seconds": last_update_age,
            "current_row_id": self.current_row_id,
            "current_query_key": self.current_query_key,
            "current_rank": self.current_rank,
            "current_title": self.current_title,
            "processed_rows": self.processed_rows,
            "last_result": self.last_result,
            "last_parse_status": self.last_parse_status,
            "last_error": self.last_error,
            "docling_initialized": self.docling_initialized,
            "docling_parse_count": self.docling_parse_count,
            "last_docling_parse_seconds": self.last_docling_parse_seconds,
            "last_docling_error": self.last_docling_error,
            "waiting_for_docling": self.waiting_for_docling,
            "holding_docling_slot": self.holding_docling_slot,
            "docling_slot_id": self.docling_slot_id,
        }


class WorkerMonitor:
    def __init__(self, output_path: Path) -> None:
        self._lock = threading.Lock()
        self._output_path = output_path
        self._workers: dict[int, WorkerRuntime] = {}
        self._queue_counts: dict[str, int] = {}
        self._latest_summary: dict[str, float] = {}
        self._docling_runtime: dict[str, Any] = {}
        self._pdf_runtime: dict[str, Any] = {}
        self._process_runtime: dict[str, Any] = {}
        self._parse_runtime: dict[str, Any] = {}
        self._phase: str = "review"
        self._rss_history_mb: list[float] = []
        self._updated_at_iso: Optional[str] = None

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase
            self._updated_at_iso = datetime.now(timezone.utc).isoformat()

    def worker_started(self, worker_id: int, row: Optional[dict[str, Any]] = None) -> None:
        with self._lock:
            runtime = self._workers.get(worker_id) or WorkerRuntime(worker_id=worker_id)
            self._workers[worker_id] = runtime
            now = time.monotonic()
            runtime.status = "running"
            runtime.stage = "claimed"
            runtime.stage_started_at = now
            runtime.row_started_at = now
            runtime.last_update_at = now
            if row is not None:
                runtime.current_row_id = int(row["id"])
                runtime.current_query_key = str(row.get("query_key") or "")
                runtime.current_rank = int(row["retrieval_rank"]) if row.get("retrieval_rank") is not None else None
                runtime.current_title = str(row.get("candidate_title") or "")

    def worker_stage(self, worker_id: int, stage: str, *, row: Optional[dict[str, Any]] = None, error: Optional[str] = None) -> None:
        with self._lock:
            runtime = self._workers.setdefault(worker_id, WorkerRuntime(worker_id=worker_id))
            now = time.monotonic()
            runtime.status = "running"
            runtime.stage = stage
            runtime.stage_started_at = now
            runtime.last_update_at = now
            if row is not None:
                runtime.current_row_id = int(row["id"])
                runtime.current_query_key = str(row.get("query_key") or "")
                runtime.current_rank = int(row["retrieval_rank"]) if row.get("retrieval_rank") is not None else None
                runtime.current_title = str(row.get("candidate_title") or "")
                if stage == "claimed":
                    runtime.row_started_at = now
            if error:
                runtime.last_error = error

    def worker_completed_row(self, worker_id: int, payload: dict[str, Any]) -> None:
        with self._lock:
            runtime = self._workers.setdefault(worker_id, WorkerRuntime(worker_id=worker_id))
            now = time.monotonic()
            runtime.processed_rows += 1
            runtime.last_result = str(payload.get("review_status") or "")
            runtime.last_parse_status = str(payload.get("parse_status") or "")
            runtime.last_error = payload.get("error_message")
            runtime.stage = "idle"
            runtime.stage_started_at = now
            runtime.last_update_at = now
            runtime.waiting_for_docling = False
            runtime.holding_docling_slot = False
            runtime.docling_slot_id = None
            runtime.current_row_id = None
            runtime.current_query_key = None
            runtime.current_rank = None
            runtime.current_title = None
            runtime.row_started_at = 0.0

    def worker_exited(self, worker_id: int, processed_rows: int) -> None:
        with self._lock:
            runtime = self._workers.setdefault(worker_id, WorkerRuntime(worker_id=worker_id))
            now = time.monotonic()
            runtime.status = "exited"
            runtime.stage = "exited"
            runtime.stage_started_at = now
            runtime.last_update_at = now
            runtime.processed_rows = processed_rows
            runtime.waiting_for_docling = False
            runtime.holding_docling_slot = False
            runtime.docling_slot_id = None
            runtime.current_row_id = None
            runtime.current_query_key = None
            runtime.current_rank = None
            runtime.current_title = None
            runtime.row_started_at = 0.0

    def update_summary(
        self,
        *,
        queue_counts: dict[str, int],
        latest_summary: dict[str, float],
        docling_runtime: Optional[dict[str, Any]] = None,
        pdf_runtime: Optional[dict[str, Any]] = None,
        process_runtime: Optional[dict[str, Any]] = None,
        parse_runtime: Optional[dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._queue_counts = dict(queue_counts)
            self._latest_summary = dict(latest_summary)
            if docling_runtime is not None:
                self._docling_runtime = dict(docling_runtime)
                worker_states = docling_runtime.get("workers") or {}
                for worker_id_raw, state in worker_states.items():
                    worker_id = int(worker_id_raw)
                    runtime = self._workers.setdefault(worker_id, WorkerRuntime(worker_id=worker_id))
                    runtime.docling_initialized = bool(state.get("converter_initialized", False))
                    runtime.docling_parse_count = int(state.get("parse_count", 0) or 0)
                    runtime.last_docling_parse_seconds = float(state.get("last_parse_seconds", 0.0) or 0.0)
                    runtime.last_docling_error = state.get("last_error")
                    runtime.waiting_for_docling = bool(state.get("waiting_for_docling", False))
                    runtime.holding_docling_slot = bool(state.get("holding_docling_slot", False))
                    runtime.docling_slot_id = int(state["docling_slot_id"]) if state.get("docling_slot_id") is not None else None
            if pdf_runtime is not None:
                self._pdf_runtime = dict(pdf_runtime)
            if process_runtime is not None:
                self._process_runtime = dict(process_runtime)
                current_rss_mb = float(process_runtime.get("rss_mb", 0.0) or 0.0)
                if current_rss_mb > 0:
                    self._rss_history_mb.append(current_rss_mb)
                    self._rss_history_mb = self._rss_history_mb[-12:]
                    baseline = min(self._rss_history_mb)
                    rising_mb = current_rss_mb - baseline
                    self._process_runtime["rss_window_baseline_mb"] = round(baseline, 2)
                    self._process_runtime["rss_window_growth_mb"] = round(rising_mb, 2)
                    self._process_runtime["rss_growth_alert"] = rising_mb >= 4096.0
            if parse_runtime is not None:
                self._parse_runtime = dict(parse_runtime)
            self._updated_at_iso = datetime.now(timezone.utc).isoformat()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            workers = [
                runtime.as_snapshot(now, stale_after_seconds=300.0, stuck_after_seconds=1800.0)
                for runtime in sorted(self._workers.values(), key=lambda item: item.worker_id)
            ]
            return {
                "phase": self._phase,
                "updated_at": self._updated_at_iso,
                "queue_counts": dict(self._queue_counts),
                "summary": dict(self._latest_summary),
                "docling": dict(self._docling_runtime),
                "pdf_gate": dict(self._pdf_runtime),
                "process": dict(self._process_runtime),
                "parse_status": dict(self._parse_runtime),
                "workers": workers,
            }

    def write_snapshot(self) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.snapshot(), ensure_ascii=True))
            handle.write("\n")


class ProgressStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active_workers = 0
        self.completed_rows = 0
        self.failed_rows = 0
        self.rows_in_interval = 0
        self.download_seconds = 0.0
        self.docling_wait_seconds = 0.0
        self.docling_parse_seconds = 0.0
        self.deepseek_wait_seconds = 0.0
        self.deepseek_request_seconds = 0.0
        self.total_row_seconds = 0.0

    def worker_started(self) -> None:
        with self._lock:
            self.active_workers += 1

    def worker_exited(self) -> None:
        with self._lock:
            self.active_workers = max(0, self.active_workers - 1)

    def record_row(self, payload: dict[str, Any]) -> None:
        timings = payload.get("timings") or {}
        with self._lock:
            self.rows_in_interval += 1
            if payload.get("review_status") == "completed":
                self.completed_rows += 1
            else:
                self.failed_rows += 1
            self.download_seconds += float(timings.get("download_seconds", 0.0) or 0.0)
            self.docling_wait_seconds += float(timings.get("docling_wait_seconds", 0.0) or 0.0)
            self.docling_parse_seconds += float(timings.get("docling_parse_seconds", 0.0) or 0.0)
            self.deepseek_wait_seconds += float(timings.get("deepseek_wait_seconds", 0.0) or 0.0)
            self.deepseek_request_seconds += float(timings.get("deepseek_request_seconds", 0.0) or 0.0)
            self.total_row_seconds += float(timings.get("total_row_seconds", 0.0) or 0.0)

    def snapshot_interval(self) -> dict[str, float]:
        with self._lock:
            rows = self.rows_in_interval
            snapshot = {
                "active_workers": float(self.active_workers),
                "rows_in_interval": float(rows),
                "avg_download_seconds": self.download_seconds / rows if rows else 0.0,
                "avg_docling_wait_seconds": self.docling_wait_seconds / rows if rows else 0.0,
                "avg_docling_parse_seconds": self.docling_parse_seconds / rows if rows else 0.0,
                "avg_deepseek_wait_seconds": self.deepseek_wait_seconds / rows if rows else 0.0,
                "avg_deepseek_request_seconds": self.deepseek_request_seconds / rows if rows else 0.0,
                "avg_total_row_seconds": self.total_row_seconds / rows if rows else 0.0,
            }
            self.rows_in_interval = 0
            self.download_seconds = 0.0
            self.docling_wait_seconds = 0.0
            self.docling_parse_seconds = 0.0
            self.deepseek_wait_seconds = 0.0
            self.deepseek_request_seconds = 0.0
            self.total_row_seconds = 0.0
        return snapshot


def _read_process_runtime() -> dict[str, Any]:
    rss_mb = 0.0
    hwm_mb = 0.0
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        try:
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss_mb = int(line.split()[1]) / 1024.0
                elif line.startswith("VmHWM:"):
                    hwm_mb = int(line.split()[1]) / 1024.0
        except Exception:
            pass
    if rss_mb <= 0.0:
        try:
            maxrss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if maxrss_kb > 0:
                hwm_mb = max(hwm_mb, maxrss_kb / 1024.0)
                rss_mb = max(rss_mb, hwm_mb)
        except Exception:
            pass
    return {
        "pid": os.getpid(),
        "rss_mb": round(rss_mb, 2),
        "peak_rss_mb": round(max(hwm_mb, rss_mb), 2),
    }


def _build_parse_runtime(status_summary: dict[str, Any]) -> dict[str, Any]:
    completed_parse_counts = dict(status_summary.get("completed_parse_counts") or {})
    latest_completed_parse_counts = dict(status_summary.get("latest_completed_parse_counts") or {})
    canonical_label_counts = dict(status_summary.get("canonical_labels") or {})
    latest_completed_label_counts: dict[str, int] = {}
    for row in status_summary.get("latest_completed") or []:
        review_label = row.get("review_label")
        if review_label is None:
            continue
        key = str(review_label)
        latest_completed_label_counts[key] = latest_completed_label_counts.get(key, 0) + 1
    completed_total = int(sum(int(value) for value in completed_parse_counts.values()))
    latest_total = int(sum(int(value) for value in latest_completed_parse_counts.values()))
    metadata_only_total = int(completed_parse_counts.get("metadata_only", 0) or 0)
    latest_metadata_only = int(latest_completed_parse_counts.get("metadata_only", 0) or 0)
    return {
        "canonical_label_counts": canonical_label_counts,
        "latest_completed_label_counts": latest_completed_label_counts,
        "completed_parse_counts": completed_parse_counts,
        "latest_completed_parse_counts": latest_completed_parse_counts,
        "completed_parsed_ratio": round((int(completed_parse_counts.get("parsed", 0) or 0) / completed_total), 4) if completed_total else 0.0,
        "latest_parsed_ratio": round((int(latest_completed_parse_counts.get("parsed", 0) or 0) / latest_total), 4) if latest_total else 0.0,
        "metadata_only_alert": metadata_only_total > 0,
        "latest_metadata_only_alert": latest_metadata_only > 0,
    }


def _load_existing_query_keys(db) -> set[str]:
    queue_table_name, _ = resolve_queue_storage_names()
    from sqlalchemy import text

    with db.begin() as conn:
        rows = conn.execute(text(f"select distinct query_key from {queue_table_name}")).fetchall()
    return {str(row[0]) for row in rows if row and row[0]}


def _write_retrieval_monitor(
    *,
    monitor_file: Path,
    queue_table_name: str,
    canonical_view_name: str,
    processed_queries: int,
    total_queries: int,
    upserted_rows: int,
    skipped_queries: int,
    latest_paper_id: Optional[str] = None,
    latest_query_text: Optional[str] = None,
    note: Optional[str] = None,
) -> None:
    queue_counts = load_status_summary(canonical=True)
    payload = {
        "phase": "retrieval",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "queue_table": queue_table_name,
        "canonical_view": canonical_view_name,
        "retrieval": {
            "processed_queries": processed_queries,
            "total_queries": total_queries,
            "upserted_rows": upserted_rows,
            "skipped_queries": skipped_queries,
            "latest_paper_id": latest_paper_id,
            "latest_query_text": latest_query_text,
            "note": note,
        },
        "queue_counts": queue_counts,
        "docling": get_docling_runtime_snapshot(),
        "pdf_gate": get_pdf_download_runtime_snapshot(),
    }
    write_json_file(monitor_file, payload)


def enqueue_stage5_queue_candidates(
    *,
    query_dataset: GeneratedQueriesDataset,
    llm_backend,
    search_settings: dict[str, Any],
    monitor_file: Path,
    skip_existing_queries: bool = False,
    rerank_max_workers: int = 1,
) -> dict[str, Any]:
    db = ensure_schema()
    queue_table_name, canonical_view_name = resolve_queue_storage_names()
    scholar_client = build_google_scholar_client(
        provider=str(search_settings["provider"]),
        serpapi_api_key=str(search_settings.get("serpapi_api_key") or ""),
        semantic_scholar_api_keys=[
            str(token).strip()
            for token in search_settings.get("semantic_scholar_api_keys", [])
            if str(token).strip()
        ],
        language=str(search_settings["language"]),
        timeout_seconds=float(search_settings["timeout_seconds"]),
        min_interval_seconds=float(search_settings["min_interval_seconds"]),
        max_retries=int(search_settings["max_retries"]),
        retry_backoff_seconds=float(search_settings["retry_backoff_seconds"]),
        retry_backoff_multiplier=float(search_settings["retry_backoff_multiplier"]),
        cache_dir=Path(str(search_settings["cache_dir"])).expanduser().resolve() if str(search_settings["cache_dir"]).strip() else None,
    )
    miner = HardNegativeMiner(
        llm=llm_backend,
        scholar_client=scholar_client,
        scholar_max_results=int(search_settings["max_results"]),
        review_max_workers=1,
    )

    total_queries = sum(len(paper.queries_by_view) for paper in query_dataset.papers_queries)
    existing_query_keys = _load_existing_query_keys(db) if skip_existing_queries else set()
    processed_queries = 0
    upserted_rows = 0
    skipped_queries = 0
    rerank_max_workers = max(1, int(rerank_max_workers))

    _write_retrieval_monitor(
        monitor_file=monitor_file,
        queue_table_name=queue_table_name,
        canonical_view_name=canonical_view_name,
        processed_queries=processed_queries,
        total_queries=total_queries,
        upserted_rows=upserted_rows,
        skipped_queries=skipped_queries,
        note="retrieval_started",
    )

    def _rerank_and_build_rows(payload: dict[str, Any]) -> dict[str, Any]:
        review_candidates = miner._rank_candidates_by_relevance(
            payload["query_text"],
            payload["candidates"],
            top_n=miner.DEFAULT_RERANK_TOP_N,
        )
        rows = []
        for rank, candidate in enumerate(review_candidates, start=1):
            candidate_key = build_candidate_key(
                query_key=payload["query_key"],
                candidate_title=candidate.paper_title,
                candidate_arxiv_id=candidate.arxiv_id,
                candidate_url=candidate.url,
            )
            rows.append(
                {
                    "candidate_key": candidate_key,
                    "query_key": payload["query_key"],
                    "paper_id": payload["paper_id"],
                    "paper_title": payload["paper_title"],
                    "query_text": payload["query_text"],
                    "query_type": payload["query_type"],
                    "source_view": payload["source_view"],
                    "is_multimodal": payload["is_multimodal"],
                    "related_bullet_indice": payload["related_bullet_indice"],
                    "related_bullet_justification": payload["related_bullet_justification"],
                    "multimodal_rationale": payload["multimodal_rationale"],
                    "keywords_extracted": payload["keywords"],
                    "search_query": " || ".join(payload["search_queries"]),
                    "retrieval_rank": rank,
                    "candidate_title": candidate.paper_title,
                    "candidate_arxiv_id": candidate.arxiv_id,
                    "candidate_url": candidate.url,
                    "candidate_pdf_url": candidate.pdf_url,
                    "candidate_venue": candidate.venue,
                    "candidate_year": candidate.year,
                    "candidate_authors": candidate.authors,
                    "candidate_abstract": candidate.abstract,
                    "candidate_citations": candidate.citations,
                    "parse_status": "pending",
                    "review_status": "pending",
                    "review_label": None,
                    "review_reason": None,
                    "error_message": None,
                    "review_payload": None,
                }
            )
        return {
            "query_key": payload["query_key"],
            "paper_id": payload["paper_id"],
            "query_text": payload["query_text"],
            "retrieved_count": len(payload["candidates"]),
            "reranked_count": len(review_candidates),
            "rows": rows,
        }

    pending_futures: dict[Future, dict[str, Any]] = {}

    def _flush_completed(*, wait_for_one: bool = False) -> None:
        nonlocal upserted_rows
        if not pending_futures:
            return
        completed = [future for future in pending_futures if future.done()]
        if wait_for_one and not completed:
            completed = [next(as_completed(list(pending_futures.keys())))]
        for future in completed:
            meta = pending_futures.pop(future)
            try:
                result = future.result()
            except Exception as exc:
                logger.exception(
                    "stage5_queue_rerank_failed query=%s/%s paper_id=%s query_key=%s error=%s",
                    meta["processed_queries"],
                    total_queries,
                    meta["paper_id"],
                    meta["query_key"],
                    exc,
                )
                _write_retrieval_monitor(
                    monitor_file=monitor_file,
                    queue_table_name=queue_table_name,
                    canonical_view_name=canonical_view_name,
                    processed_queries=processed_queries,
                    total_queries=total_queries,
                    upserted_rows=upserted_rows,
                    skipped_queries=skipped_queries,
                    latest_paper_id=meta["paper_id"],
                    latest_query_text=meta["query_text"],
                    note="rerank_failed",
                )
                continue

            current_upserted = upsert_candidates(result["rows"], engine=db)
            upserted_rows += current_upserted
            existing_query_keys.add(result["query_key"])
            logger.info(
                "stage5_queue_retrieval query=%s/%s paper_id=%s retrieved=%s reranked_pending=%s upserted=%s",
                meta["processed_queries"],
                total_queries,
                result["paper_id"],
                result["retrieved_count"],
                result["reranked_count"],
                current_upserted,
            )
            _write_retrieval_monitor(
                monitor_file=monitor_file,
                queue_table_name=queue_table_name,
                canonical_view_name=canonical_view_name,
                processed_queries=processed_queries,
                total_queries=total_queries,
                upserted_rows=upserted_rows,
                skipped_queries=skipped_queries,
                latest_paper_id=result["paper_id"],
                latest_query_text=result["query_text"],
                note="retrieval_progress",
            )

    logger.info(
        "stage5_queue_retrieval_start total_queries=%s semantic_search_workers=1 rerank_workers=%s",
        total_queries,
        rerank_max_workers,
    )
    with ThreadPoolExecutor(max_workers=rerank_max_workers) as executor:
        for paper in query_dataset.papers_queries:
            for query in paper.queries_by_view:
                processed_queries += 1
                query_key = build_query_key(paper.paper_id, query.query_text, query.source_view, query.query_type)
                if query_key in existing_query_keys:
                    skipped_queries += 1
                    _write_retrieval_monitor(
                        monitor_file=monitor_file,
                        queue_table_name=queue_table_name,
                        canonical_view_name=canonical_view_name,
                        processed_queries=processed_queries,
                        total_queries=total_queries,
                        upserted_rows=upserted_rows,
                        skipped_queries=skipped_queries,
                        latest_paper_id=paper.paper_id,
                        latest_query_text=query.query_text,
                        note="skipped_existing_query",
                    )
                    continue

                keywords = miner.extract_keywords(query.query_text)
                search_queries, candidates = miner.retrieve_candidates_for_query(query.query_text, keywords)
                future = executor.submit(
                    _rerank_and_build_rows,
                    {
                        "query_key": query_key,
                        "paper_id": paper.paper_id,
                        "paper_title": paper.paper_title,
                        "query_text": query.query_text,
                        "query_type": query.query_type,
                        "source_view": query.source_view,
                        "is_multimodal": "true" if query.is_multimodal else "false",
                        "related_bullet_indice": query.related_bullet_indice,
                        "related_bullet_justification": query.related_bullet_justification,
                        "multimodal_rationale": query.multimodal_rationale,
                        "keywords": keywords,
                        "search_queries": search_queries,
                        "candidates": candidates,
                    },
                )
                pending_futures[future] = {
                    "processed_queries": processed_queries,
                    "paper_id": paper.paper_id,
                    "query_key": query_key,
                    "query_text": query.query_text,
                }
                _flush_completed(wait_for_one=False)
                if len(pending_futures) >= max(1, rerank_max_workers * 2):
                    _flush_completed(wait_for_one=True)

        while pending_futures:
            _flush_completed(wait_for_one=True)

    _write_retrieval_monitor(
        monitor_file=monitor_file,
        queue_table_name=queue_table_name,
        canonical_view_name=canonical_view_name,
        processed_queries=processed_queries,
        total_queries=total_queries,
        upserted_rows=upserted_rows,
        skipped_queries=skipped_queries,
        note="retrieval_completed",
    )
    return {
        "processed_queries": processed_queries,
        "total_queries": total_queries,
        "upserted_rows": upserted_rows,
        "skipped_queries": skipped_queries,
        "queue_table_name": queue_table_name,
        "canonical_view_name": canonical_view_name,
    }


def enqueue_stage5_queue_candidates_parallel_search(
    *,
    query_dataset: GeneratedQueriesDataset,
    llm_backend,
    search_settings: dict[str, Any],
    monitor_file: Path,
    skip_existing_queries: bool = False,
    search_max_workers: int = 2,
    rerank_max_workers: int = 1,
) -> dict[str, Any]:
    db = ensure_schema()
    queue_table_name, canonical_view_name = resolve_queue_storage_names()
    provider_name = str(search_settings["provider"]).strip().lower()
    semantic_scholar_api_keys = [
        str(token).strip()
        for token in search_settings.get("semantic_scholar_api_keys", [])
        if str(token).strip()
    ]
    cache_dir = (
        Path(str(search_settings["cache_dir"])).expanduser().resolve()
        if str(search_settings["cache_dir"]).strip()
        else None
    )

    def _build_search_miner(*, api_keys: Optional[list[str]] = None) -> HardNegativeMiner:
        scholar_client = build_google_scholar_client(
            provider=str(search_settings["provider"]),
            serpapi_api_key=str(search_settings.get("serpapi_api_key") or ""),
            semantic_scholar_api_keys=api_keys,
            language=str(search_settings["language"]),
            timeout_seconds=float(search_settings["timeout_seconds"]),
            min_interval_seconds=float(search_settings["min_interval_seconds"]),
            max_retries=int(search_settings["max_retries"]),
            retry_backoff_seconds=float(search_settings["retry_backoff_seconds"]),
            retry_backoff_multiplier=float(search_settings["retry_backoff_multiplier"]),
            cache_dir=cache_dir,
        )
        return HardNegativeMiner(
            llm=llm_backend,
            scholar_client=scholar_client,
            scholar_max_results=int(search_settings["max_results"]),
            review_max_workers=1,
        )

    rerank_miner = _build_search_miner(api_keys=semantic_scholar_api_keys or None)
    total_queries = sum(len(paper.queries_by_view) for paper in query_dataset.papers_queries)
    existing_query_keys = _load_existing_query_keys(db) if skip_existing_queries else set()
    processed_queries = 0
    upserted_rows = 0
    skipped_queries = 0
    requested_search_workers = max(1, int(search_max_workers))
    rerank_max_workers = max(1, int(rerank_max_workers))

    if provider_name in {"semantic_scholar", "semanticscholar", "s2"} and semantic_scholar_api_keys:
        actual_search_workers = min(requested_search_workers, len(semantic_scholar_api_keys))
    else:
        actual_search_workers = 1

    search_miners = (
        [_build_search_miner(api_keys=[semantic_scholar_api_keys[i]]) for i in range(actual_search_workers)]
        if actual_search_workers > 1
        else [rerank_miner]
    )

    _write_retrieval_monitor(
        monitor_file=monitor_file,
        queue_table_name=queue_table_name,
        canonical_view_name=canonical_view_name,
        processed_queries=processed_queries,
        total_queries=total_queries,
        upserted_rows=upserted_rows,
        skipped_queries=skipped_queries,
        note="retrieval_started",
    )

    def _search_query(payload: dict[str, Any], search_miner: HardNegativeMiner) -> dict[str, Any]:
        keywords = search_miner.extract_keywords(payload["query_text"])
        search_queries, candidates = search_miner.retrieve_candidates_for_query(payload["query_text"], keywords)
        merged = dict(payload)
        merged["keywords"] = keywords
        merged["search_queries"] = search_queries
        merged["candidates"] = candidates
        return merged

    def _rerank_and_build_rows(payload: dict[str, Any]) -> dict[str, Any]:
        review_candidates = rerank_miner._rank_candidates_by_relevance(
            payload["query_text"],
            payload["candidates"],
            top_n=rerank_miner.DEFAULT_RERANK_TOP_N,
        )
        rows = []
        for rank, candidate in enumerate(review_candidates, start=1):
            candidate_key = build_candidate_key(
                query_key=payload["query_key"],
                candidate_title=candidate.paper_title,
                candidate_arxiv_id=candidate.arxiv_id,
                candidate_url=candidate.url,
            )
            rows.append(
                {
                    "candidate_key": candidate_key,
                    "query_key": payload["query_key"],
                    "paper_id": payload["paper_id"],
                    "paper_title": payload["paper_title"],
                    "query_text": payload["query_text"],
                    "query_type": payload["query_type"],
                    "source_view": payload["source_view"],
                    "is_multimodal": payload["is_multimodal"],
                    "related_bullet_indice": payload["related_bullet_indice"],
                    "related_bullet_justification": payload["related_bullet_justification"],
                    "multimodal_rationale": payload["multimodal_rationale"],
                    "keywords_extracted": payload["keywords"],
                    "search_query": " || ".join(payload["search_queries"]),
                    "retrieval_rank": rank,
                    "candidate_title": candidate.paper_title,
                    "candidate_arxiv_id": candidate.arxiv_id,
                    "candidate_url": candidate.url,
                    "candidate_pdf_url": candidate.pdf_url,
                    "candidate_venue": candidate.venue,
                    "candidate_year": candidate.year,
                    "candidate_authors": candidate.authors,
                    "candidate_abstract": candidate.abstract,
                    "candidate_citations": candidate.citations,
                    "parse_status": "pending",
                    "review_status": "pending",
                    "review_label": None,
                    "review_reason": None,
                    "error_message": None,
                    "review_payload": None,
                }
            )
        return {
            "query_key": payload["query_key"],
            "paper_id": payload["paper_id"],
            "query_text": payload["query_text"],
            "retrieved_count": len(payload["candidates"]),
            "reranked_count": len(review_candidates),
            "rows": rows,
        }

    pending_search_futures: dict[Future, dict[str, Any]] = {}
    pending_rerank_futures: dict[Future, dict[str, Any]] = {}

    def _flush_rerank_completed(*, wait_for_one: bool = False) -> None:
        nonlocal upserted_rows
        if not pending_rerank_futures:
            return
        completed = [future for future in pending_rerank_futures if future.done()]
        if wait_for_one and not completed:
            completed = [next(as_completed(list(pending_rerank_futures.keys())))]
        for future in completed:
            meta = pending_rerank_futures.pop(future)
            try:
                result = future.result()
            except Exception as exc:
                logger.exception(
                    "stage5_queue_rerank_failed query=%s/%s paper_id=%s query_key=%s error=%s",
                    meta["processed_queries"],
                    total_queries,
                    meta["paper_id"],
                    meta["query_key"],
                    exc,
                )
                _write_retrieval_monitor(
                    monitor_file=monitor_file,
                    queue_table_name=queue_table_name,
                    canonical_view_name=canonical_view_name,
                    processed_queries=processed_queries,
                    total_queries=total_queries,
                    upserted_rows=upserted_rows,
                    skipped_queries=skipped_queries,
                    latest_paper_id=meta["paper_id"],
                    latest_query_text=meta["query_text"],
                    note="rerank_failed",
                )
                continue

            current_upserted = upsert_candidates(result["rows"], engine=db)
            upserted_rows += current_upserted
            existing_query_keys.add(result["query_key"])
            logger.info(
                "stage5_queue_retrieval query=%s/%s paper_id=%s retrieved=%s reranked_pending=%s upserted=%s",
                meta["processed_queries"],
                total_queries,
                result["paper_id"],
                result["retrieved_count"],
                result["reranked_count"],
                current_upserted,
            )
            _write_retrieval_monitor(
                monitor_file=monitor_file,
                queue_table_name=queue_table_name,
                canonical_view_name=canonical_view_name,
                processed_queries=processed_queries,
                total_queries=total_queries,
                upserted_rows=upserted_rows,
                skipped_queries=skipped_queries,
                latest_paper_id=result["paper_id"],
                latest_query_text=result["query_text"],
                note="retrieval_progress",
            )

    def _flush_search_completed(*, rerank_executor: ThreadPoolExecutor, wait_for_one: bool = False) -> None:
        if not pending_search_futures:
            return
        completed = [future for future in pending_search_futures if future.done()]
        if wait_for_one and not completed:
            completed = [next(as_completed(list(pending_search_futures.keys())))]
        for future in completed:
            meta = pending_search_futures.pop(future)
            try:
                payload = future.result()
            except Exception as exc:
                logger.exception(
                    "stage5_queue_search_failed query=%s/%s paper_id=%s query_key=%s error=%s",
                    meta["processed_queries"],
                    total_queries,
                    meta["paper_id"],
                    meta["query_key"],
                    exc,
                )
                _write_retrieval_monitor(
                    monitor_file=monitor_file,
                    queue_table_name=queue_table_name,
                    canonical_view_name=canonical_view_name,
                    processed_queries=processed_queries,
                    total_queries=total_queries,
                    upserted_rows=upserted_rows,
                    skipped_queries=skipped_queries,
                    latest_paper_id=meta["paper_id"],
                    latest_query_text=meta["query_text"],
                    note="search_failed",
                )
                continue

            rerank_future = rerank_executor.submit(_rerank_and_build_rows, payload)
            pending_rerank_futures[rerank_future] = dict(meta)
            _flush_rerank_completed(wait_for_one=False)

    logger.info(
        "stage5_queue_retrieval_start total_queries=%s semantic_search_workers=%s rerank_workers=%s",
        total_queries,
        actual_search_workers,
        rerank_max_workers,
    )
    with ThreadPoolExecutor(max_workers=actual_search_workers) as search_executor, ThreadPoolExecutor(max_workers=rerank_max_workers) as rerank_executor:
        for paper in query_dataset.papers_queries:
            for query in paper.queries_by_view:
                processed_queries += 1
                query_key = build_query_key(paper.paper_id, query.query_text, query.source_view, query.query_type)
                if query_key in existing_query_keys:
                    skipped_queries += 1
                    _write_retrieval_monitor(
                        monitor_file=monitor_file,
                        queue_table_name=queue_table_name,
                        canonical_view_name=canonical_view_name,
                        processed_queries=processed_queries,
                        total_queries=total_queries,
                        upserted_rows=upserted_rows,
                        skipped_queries=skipped_queries,
                        latest_paper_id=paper.paper_id,
                        latest_query_text=query.query_text,
                        note="skipped_existing_query",
                    )
                    continue

                search_future = search_executor.submit(
                    _search_query,
                    {
                        "query_key": query_key,
                        "paper_id": paper.paper_id,
                        "paper_title": paper.paper_title,
                        "query_text": query.query_text,
                        "query_type": query.query_type,
                        "source_view": query.source_view,
                        "is_multimodal": "true" if query.is_multimodal else "false",
                        "related_bullet_indice": query.related_bullet_indice,
                        "related_bullet_justification": query.related_bullet_justification,
                        "multimodal_rationale": query.multimodal_rationale,
                    },
                    search_miners[(processed_queries - 1) % len(search_miners)],
                )
                pending_search_futures[search_future] = {
                    "processed_queries": processed_queries,
                    "paper_id": paper.paper_id,
                    "query_key": query_key,
                    "query_text": query.query_text,
                }
                _flush_search_completed(rerank_executor=rerank_executor, wait_for_one=False)
                _flush_rerank_completed(wait_for_one=False)
                if len(pending_search_futures) >= max(1, actual_search_workers * 2):
                    _flush_search_completed(rerank_executor=rerank_executor, wait_for_one=True)
                if len(pending_rerank_futures) >= max(1, rerank_max_workers * 2):
                    _flush_rerank_completed(wait_for_one=True)

        while pending_search_futures:
            _flush_search_completed(rerank_executor=rerank_executor, wait_for_one=True)
        while pending_rerank_futures:
            _flush_rerank_completed(wait_for_one=True)

    _write_retrieval_monitor(
        monitor_file=monitor_file,
        queue_table_name=queue_table_name,
        canonical_view_name=canonical_view_name,
        processed_queries=processed_queries,
        total_queries=total_queries,
        upserted_rows=upserted_rows,
        skipped_queries=skipped_queries,
        note="retrieval_completed",
    )
    return {
        "processed_queries": processed_queries,
        "total_queries": total_queries,
        "upserted_rows": upserted_rows,
        "skipped_queries": skipped_queries,
        "queue_table_name": queue_table_name,
        "canonical_view_name": canonical_view_name,
    }


def run_stage5_review_pass(
    *,
    llm_backend,
    review_max_workers: int,
    scheduler_mode: str,
    batch_size: int,
    pdf_output_dir: Path,
    monitor_file: Path,
    monitor_interval: int,
    stale_processing_seconds: int,
    task_filter: Optional[str] = None,
    db: Any = None,
) -> dict[str, Any]:
    """Per-paper review pass: claims single parsed rows, reviews with shared miner.
    Only processes rows where parse_status='parsed' — no download or parsing needed."""
    if db is None:
        db = get_engine()
    queue_table_name, canonical_view_name = resolve_queue_storage_names()
    max_workers = max(1, int(review_max_workers))

    reset_count = reset_stale_processing_candidates(
        stale_after_seconds=max(1, int(stale_processing_seconds)),
        engine=db,
    )
    logger.info(
        "stage5_queue_review_start per_paper max_workers=%s queue_table=%s stale_reset=%s task=%s",
        max_workers, queue_table_name, reset_count, task_filter or "",
    )

    _counters = {"hn": 0, "pos": 0, "ignored": 0}
    counter_lock = threading.Lock()

    shared_miner = HardNegativeMiner(
        llm=llm_backend,
        scholar_client=None,  # type: ignore[arg-type]
        scholar_max_results=10,
        review_max_workers=1,
        pdf_output_dir=pdf_output_dir,
    )

    def _process_row(worker_id: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        started_at = time.monotonic()
        candidate = _row_to_candidate(row)
        try:
            shared_miner.bind_worker_context(worker_id)
            review = shared_miner._review_single_candidate(
                str(row["query_text"]), candidate, on_stage_update=None,
            )
        except Exception as exc:
            logger.exception("stage5_queue_row_failed id=%s error=%s", row.get("id"), exc)
            return int(row["id"]), {
                "parse_status": "parsed", "review_status": "failed",
                "review_label": "", "review_reason": "", "error_message": f"review_exception: {exc}",
                "review_payload": None, "timings": {"total_row_seconds": time.monotonic() - started_at},
            }
        if not review:
            return int(row["id"]), {
                "parse_status": "parsed", "review_status": "failed",
                "review_label": "", "review_reason": "", "error_message": "response_parse_failed",
                "review_payload": None, "timings": {"total_row_seconds": time.monotonic() - started_at},
            }
        timings = dict(review.get("timings") or {})
        timings.setdefault("total_row_seconds", time.monotonic() - started_at)
        return int(row["id"]), {
            "parse_status": "parsed",
            "review_status": str(review.get("review_status") or "completed"),
            "review_label": str(review.get("label") or ""),
            "review_reason": str(review.get("reason") or ""),
            "error_message": str(review.get("error_message")) if review.get("error_message") else None,
            "review_payload": {
                "candidate_title": candidate.paper_title,
                "candidate_arxiv_id": candidate.arxiv_id,
                "candidate_pdf_url": candidate.pdf_url,
                "candidate_pdf_path": candidate.pdf_path,
                "candidate_full_text_path": candidate.full_text_path,
                "label": review.get("label"),
                "reason": review.get("reason"),
            },
            "timings": timings,
        }

    # Batch fetcher + in-memory queue (same pattern as parse)
    import queue as queue_module
    review_queue: queue_module.Queue = queue_module.Queue(maxsize=1000)
    fetched_count = 0

    def review_batch_fetcher():
        nonlocal fetched_count
        while not stop_event.is_set():
            try:
                free = review_queue.maxsize - review_queue.qsize()
                claim_n = min(500, max(10, free))
                rows = claim_pending_candidates(claim_n, engine=db, task_filter=task_filter or None, parse_status_filter="parsed")
                if rows:
                    for row in rows:
                        review_queue.put(row, timeout=60)
                    fetched_count += len(rows)
                else:
                    time.sleep(5)
            except Exception as exc:
                logger.warning("review_batch_fetcher_error: %s", exc)
                time.sleep(5)

    def worker_loop(worker_id: int) -> WorkerSummary:
        summary = WorkerSummary(worker_id=worker_id)
        try:
            while not stop_event.is_set():
                try:
                    row = review_queue.get(timeout=30)
                except queue_module.Empty:
                    continue
                candidate_id, payload = _process_row(worker_id, row)
                update_candidate_result(candidate_id, engine=db, **{
                    k: v for k, v in payload.items() if k not in {"timings", "_worker_id"}
                })
                close_satisfied_query_pending_candidates(str(row.get("query_key") or ""), engine=db)
                review_queue.task_done()
                summary.processed_rows += 1
                label = payload.get("review_label", "")
                with counter_lock:
                    if label == "hard_negative": _counters["hn"] += 1
                    elif label == "positive": _counters["pos"] += 1
                    elif label == "ignored": _counters["ignored"] += 1
        finally:
            shared_miner.unbind_worker_context()
        return summary

    stop_event = threading.Event()
    fetcher_thread = threading.Thread(target=review_batch_fetcher, daemon=True)
    fetcher_thread.start()

    worker_count = max_workers
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(worker_loop, wid) for wid in range(1, worker_count + 1)]
        for future in as_completed(futures):
            future.result()

    stop_event.set()
    fetcher_thread.join(timeout=1.0)

    final_snapshot = load_queue_snapshot(engine=db, stale_processing_seconds=max(1, int(stale_processing_seconds)))
    final_counts = dict(final_snapshot.get("canonical_counts") or {})
    return {
        "processed_rows": _counters["hn"] + _counters["pos"] + _counters["ignored"],
        "total_hard_negatives": _counters["hn"],
        "total_positives": _counters["pos"],
        "total_ignored": _counters["ignored"],
        "queue_drained": int(final_counts.get("pending", 0)) == 0 and int(final_counts.get("processing", 0)) == 0,
    }


def _run_download_pass(
    *,
    pdf_output_dir: Path,
    task_filter: str,
    http_proxy: str = "",
    https_proxy: str = "",
    db: Any = None,
) -> dict[str, Any]:
    """Batch download loop for tagged rows. Claims batches, checks disk, downloads if needed."""
    if db is None:
        db = get_engine()
    queue_table_name, _ = resolve_queue_storage_names()
    miner = HardNegativeMiner(
        llm=None,  # type: ignore[arg-type]
        scholar_client=None,
        scholar_max_results=10,
        review_max_workers=1,
        pdf_output_dir=pdf_output_dir,
    )

    logger.info(
        "stage5_download_pass_start task=%s queue_table=%s pdf_output_dir=%s http_proxy=%r https_proxy=%r",
        task_filter,
        queue_table_name,
        str(pdf_output_dir),
        http_proxy,
        https_proxy,
    )

    downloaded = 0
    skipped = 0
    failed = 0
    processed = 0
    pdf_dir = Path(miner.pdf_output_dir) if miner.pdf_output_dir else (Path("outputs") / "hard_negative_pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 200
    while True:
        rows = claim_download_candidates(BATCH_SIZE, task_filter=task_filter, engine=db)
        if not rows:
            break
        batch_updates: list[dict[str, Any]] = []
        for row in rows:
            row_id = int(row["id"])
            candidate_title = str(row.get("candidate_title") or "")
            pdf_url = str(row.get("candidate_pdf_url") or "").strip()

            slug = miner._pdf_cache_slug(candidate_title)
            pdf_hash = miner._pdf_url_hash(pdf_url)
            pdf_path = pdf_dir / f"{slug}-{pdf_hash}.pdf"

            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                batch_updates.append({"candidate_id": row_id, "pdf_path": str(pdf_path), "parse_status": "downloaded"})
                skipped += 1
            else:
                try:
                    from openreview_pipeline.stage5_worker.runtime import _download_pdf_to_path

                    _download_pdf_to_path(pdf_url, pdf_path, timeout_seconds=45)
                    batch_updates.append({"candidate_id": row_id, "pdf_path": str(pdf_path), "parse_status": "downloaded"})
                    downloaded += 1
                except Exception as exc:
                    logger.warning("stage5_download_failed id=%s title=%r pdf_url=%s error=%s", row_id, candidate_title, pdf_url, exc)
                    batch_updates.append({"candidate_id": row_id, "pdf_path": "", "parse_status": "failed"})
                    failed += 1
            processed += 1

        # Batch write all results at once
        batch_update_download_results(batch_updates, engine=db)
        logger.info(
            "stage5_download_pass_progress task=%s processed=%s downloaded=%s skipped=%s failed=%s",
            task_filter, processed, downloaded, skipped, failed,
        )

    summary = {
        "task": task_filter,
        "processed": processed,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }
    logger.info("stage5_download_pass_done task=%s summary=%s", task_filter, summary)
    return summary


def run_stage5_parse_pass(
    *,
    llm_backend,
    parse_max_workers: int,
    pdf_output_dir: Path,
    monitor_file: Path,
    monitor_interval: int,
    min_query_candidates: int = 25,
    task_filter: Optional[str] = None,
    db: Any = None,
) -> dict[str, Any]:
    if db is None:
        db = get_engine()
    queue_table_name, canonical_view_name = resolve_queue_storage_names()
    max_workers = max(1, int(parse_max_workers))
    miner = HardNegativeMiner(
        llm=llm_backend,
        scholar_client=None,  # type: ignore[arg-type]
        scholar_max_results=10,
        review_max_workers=max_workers,
        pdf_output_dir=pdf_output_dir,
    )

    logger.info(
        "stage5_queue_parse_start max_workers=%s min_query_candidates=%s queue_table=%s canonical_view=%s",
        max_workers,
        min_query_candidates,
        queue_table_name,
        canonical_view_name,
    )

    monitor = WorkerMonitor(monitor_file)
    monitor.set_phase("parse")

    # Reset ALL processing rows for THIS parser at startup — any leftover
    # rows belong to a dead previous run (we are the only live process).
    if task_filter:
        import os as _os
        _parser_id = _os.environ.get("SCIFULL_PARSER_ID", "A")
        _processing_status = f"processing_parser_{_parser_id}"
        from sqlalchemy import text as sa_text
        with db.begin() as conn:
            result = conn.execute(
                sa_text(
                    f"UPDATE {queue_table_name} SET parse_status = 'downloaded', updated_at = CURRENT_TIMESTAMP "
                    f"WHERE task = :task AND parse_status = :pstatus"
                ),
                {"task": task_filter, "pstatus": _processing_status},
            )
        if result.rowcount:
            logger.info("stage5_parse_reset_%s count=%s", _processing_status, result.rowcount)

    stop_event = threading.Event()
    processed_counter = 0
    processed_lock = threading.Lock()

    def claim_next_row() -> Optional[dict[str, Any]]:
        nonlocal processed_counter
        rows = claim_parse_candidates(
            1,
            engine=db,
            min_query_candidates=max(1, int(min_query_candidates)),
            task_filter=task_filter,
        )
        if not rows:
            return None
        with processed_lock:
            processed_counter += 1
        row = rows[0]
        logger.info(
            "stage5_queue_parse_row_claimed id=%s query_key=%s retrieval_rank=%s title=%r processed_counter=%s",
            row.get("id"),
            row.get("query_key"),
            row.get("retrieval_rank"),
            row.get("candidate_title"),
            processed_counter,
        )
        return row

    def process_row(worker_id: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        started_at = time.monotonic()
        candidate = _row_to_candidate(row)
        monitor.worker_stage(worker_id, "parsing", row=row)

        def on_stage_update(stage: str, metadata: dict[str, Any]) -> None:
            if stage == "download_wait":
                monitor.worker_stage(worker_id, "download_wait", row=row)
            elif stage == "download":
                monitor.worker_stage(worker_id, "download", row=row)
            elif stage == "download_cooldown":
                monitor.worker_stage(worker_id, "download_cooldown", row=row)
            elif stage in {"docling", "docling_fallback"}:
                monitor.worker_stage(worker_id, "docling", row=row)

        try:
            miner.bind_worker_context(worker_id)
            if candidate.full_text_path and Path(candidate.full_text_path).exists():
                return int(row["id"]), {
                    "parse_status": "parsed",
                    "review_status": "pending",
                    "error_message": None,
                    "review_payload": {
                        "candidate_title": candidate.paper_title,
                        "candidate_arxiv_id": candidate.arxiv_id,
                        "candidate_pdf_url": candidate.pdf_url,
                        "candidate_pdf_path": candidate.pdf_path,
                        "candidate_full_text_path": candidate.full_text_path,
                        "parse_status": "parsed",
                        "parse_only": True,
                    },
                    "timings": {"total_row_seconds": time.monotonic() - started_at},
                }

            pdf_url = candidate.pdf_url
            if not pdf_url:
                return int(row["id"]), {
                    "parse_status": "no_pdf",
                    "review_status": "pending",
                    "error_message": None,
                    "review_payload": {
                        "candidate_title": candidate.paper_title,
                        "candidate_arxiv_id": candidate.arxiv_id,
                        "candidate_pdf_url": None,
                        "candidate_pdf_path": None,
                        "candidate_full_text_path": None,
                        "parse_status": "no_pdf",
                        "parse_only": True,
                    },
                    "timings": {"total_row_seconds": time.monotonic() - started_at},
                }

            pdf_result = miner._download_and_parse_pdf(
                pdf_url,
                candidate.paper_title,
                worker_id=worker_id,
                on_stage_update=on_stage_update,
            )
            candidate.pdf_path = str(pdf_result.get("pdf_path") or "") or candidate.pdf_path
            candidate.full_text_path = str(pdf_result.get("full_text_path") or "") or candidate.full_text_path
            timings = dict(pdf_result.get("metrics") or {})
            failure_stage = str(pdf_result.get("failure_stage") or "")
            parse_status = "parsed" if candidate.full_text_path else ("failed" if failure_stage else "metadata_only")
            error_message = str(timings.get("docling_error") or failure_stage) if failure_stage else None
            return int(row["id"]), {
                "parse_status": parse_status,
                "review_status": "pending",
                "error_message": error_message,
                "review_payload": {
                    "candidate_title": candidate.paper_title,
                    "candidate_arxiv_id": candidate.arxiv_id,
                    "candidate_pdf_url": candidate.pdf_url,
                    "candidate_pdf_path": candidate.pdf_path,
                    "candidate_full_text_path": candidate.full_text_path,
                    "parse_status": parse_status,
                    "parse_only": True,
                },
                "timings": {
                    **timings,
                    "total_row_seconds": time.monotonic() - started_at,
                },
            }
        except Exception as exc:
            logger.exception("stage5_queue_parse_row_failed id=%s error=%s", row.get("id"), exc)
            return int(row["id"]), {
                "parse_status": "failed",
                "review_status": "pending",
                "error_message": f"parse_exception: {exc}",
                "review_payload": None,
                "timings": {"total_row_seconds": time.monotonic() - started_at},
            }

    def persist_row_result(worker_id: int, candidate_id: int, payload: dict[str, Any]) -> None:
        update_candidate_result(
            candidate_id,
            engine=db,
            parse_status=str(payload.get("parse_status") or "pending"),
            review_status=str(payload.get("review_status") or "pending"),
            error_message=str(payload.get("error_message")) if payload.get("error_message") else None,
            review_payload=payload.get("review_payload"),
        )
        monitor.worker_completed_row(worker_id, payload)

    import queue as queue_module
    work_queue: queue_module.Queue = queue_module.Queue(maxsize=2000)
    fetched_count = 0
    parsed_count = 0
    counter_lock = threading.Lock()
    stop_event = threading.Event()

    # --- DEBUG: signal handler to dump queue state ---
    def _debug_queue_state(_signum, _frame):
        db_count = -1
        try:
            from sqlalchemy import text as _sat
            import os as _os2
            _pid = _os2.environ.get("SCIFULL_PARSER_ID", "A")
            with db.begin() as _conn:
                db_count = _conn.execute(
                    _sat(
                        f"SELECT COUNT(*) FROM {queue_table_name} "
                        f"WHERE task = :t AND parse_status = :ps"
                    ),
                    {"t": task_filter, "ps": f"processing_parser_{_pid}"},
                ).scalar()
        except Exception as _e:
            db_count = f"err:{type(_e).__name__}:{_e}"
        logger.warning(
            "DEBUG_QUEUE qsize=%d qmax=%d db_processing=%s parsed=%d fetched=%d",
            work_queue.qsize(), work_queue.maxsize, str(db_count), parsed_count, fetched_count,
        )
    import signal as _signal
    _signal.signal(_signal.SIGUSR1, _debug_queue_state)
    # --- END DEBUG ---

    def batch_fetcher_loop():
        nonlocal fetched_count
        while not stop_event.is_set():
            try:
                free = work_queue.maxsize - work_queue.qsize()
                claim_n = min(500, max(10, free))
                rows = claim_parse_candidates(claim_n, engine=db, task_filter=task_filter, min_query_candidates=1)
                if rows:
                    for row in rows:
                        work_queue.put(row, timeout=60)
                    with counter_lock:
                        fetched_count += len(rows)
                    logger.info("fetcher: claimed=%d queue=%d", len(rows), work_queue.qsize())
                else:
                    time.sleep(10)
            except Exception as exc:
                logger.warning("fetcher_error: %s", exc)
                time.sleep(5)

    def worker_loop(worker_id: int) -> WorkerSummary:
        nonlocal parsed_count
        summary = WorkerSummary(worker_id=worker_id)
        monitor.worker_started(worker_id)
        try:
            while not stop_event.is_set():
                try:
                    row = work_queue.get(timeout=30)
                except queue_module.Empty:
                    continue
                candidate_id, payload = process_row(worker_id, row)
                persist_row_result(worker_id, candidate_id, payload)
                work_queue.task_done()
                summary.processed_rows += 1
                with counter_lock:
                    parsed_count += 1
        finally:
            miner.unbind_worker_context()
            monitor.worker_exited(worker_id, summary.processed_rows)
        return summary

    def summary_loop() -> None:
        interval = max(1, int(monitor_interval))
        def emit_summary() -> None:
            qs = load_queue_snapshot(engine=db)
            counts = dict(qs.get("canonical_counts") or {})
            counts["stale_processing"] = int(qs.get("stale_processing_count", 0) or 0)
            monitor.update_summary(
                queue_counts=counts,
                latest_summary={"fetched": fetched_count, "parsed": parsed_count, "queue": work_queue.qsize()},
                docling_runtime=get_docling_runtime_snapshot(),
                pdf_runtime=get_pdf_download_runtime_snapshot(),
                process_runtime=_read_process_runtime(),
                parse_runtime=_build_parse_runtime(qs),
            )
            monitor.write_snapshot()
        emit_summary()
        while not stop_event.wait(interval):
            emit_summary()

    summary_thread = threading.Thread(target=summary_loop, daemon=True)
    summary_thread.start()
    fetcher_thread = threading.Thread(target=batch_fetcher_loop, daemon=True)
    fetcher_thread.start()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_loop, wid) for wid in range(1, max_workers + 1)]
        for future in as_completed(futures):
            future.result()

    stop_event.set()
    summary_thread.join(timeout=1.0)
    fetcher_thread.join(timeout=1.0)
    final_snapshot = load_queue_snapshot(engine=db)
    final_counts = dict(final_snapshot.get("canonical_counts") or {})
    final_counts["stale_processing"] = int(final_snapshot.get("stale_processing_count", 0) or 0)
    monitor.update_summary(
        queue_counts=final_counts,
        latest_summary={"rows_processed": processed_counter},
        docling_runtime=get_docling_runtime_snapshot(),
        pdf_runtime=get_pdf_download_runtime_snapshot(),
        process_runtime=_read_process_runtime(),
        parse_runtime=_build_parse_runtime(final_snapshot),
    )
    monitor.write_snapshot()
    return {
        "processed_rows": int(sum(summary.processed_rows for summary in (future.result() for future in futures))),
        "queue_drained": False,
    }


def run_stage5_queue_mode(
    *,
    query_dataset: GeneratedQueriesDataset,
    llm_backend,
    search_settings: dict[str, Any],
    output_path: Path,
    queue_table_name: str = DEFAULT_QUEUE_TABLE_NAME,
    scheduler_mode: str = DEFAULT_SCHEDULER_MODE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    review_max_workers: int = 1,
    docling_pool_size: int = DEFAULT_DOCLING_POOL_SIZE,
    http_proxy: str = DEFAULT_HTTP_PROXY,
    https_proxy: str = DEFAULT_HTTPS_PROXY,
    pdf_output_dir: Optional[Path | str] = None,
    monitor_file: Optional[Path | str] = None,
    monitor_interval: int = DEFAULT_MONITOR_INTERVAL,
    retrieve_done_flag: Optional[Path | str] = None,
    skip_existing_queries: bool = False,
    retrieval_only: bool = False,
    parse_only: bool = False,
    review_only: bool = False,
    download_only: bool = False,
    parse_min_query_candidates: int = 25,
    retrieval_search_max_workers: int = 1,
    retrieval_rerank_max_workers: int = 1,
    download_workers: int = 1,
    parse_workers: int = 8,
    review_workers: int = 200,
    task_filter: Optional[str] = None,
) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paths = build_queue_run_paths(
        output_path=output_path,
        pdf_output_dir=pdf_output_dir,
        monitor_file=monitor_file,
        retrieve_done_flag=retrieve_done_flag,
    )

    if queue_table_name:
        os.environ["SCIFULL_STAGE5_QUEUE_TABLE"] = str(queue_table_name).strip()
    configure_stage5_runtime(
        docling_pool_size=docling_pool_size,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
    )

    save_generated_queries_dataset(paths.effective_queries_path, query_dataset)
    db = ensure_schema()
    queue_table_name_resolved, canonical_view_name = resolve_queue_storage_names()

    # ── Download-Only (standalone) ──
    if download_only:
        download_summary = _run_download_pass(
            pdf_output_dir=paths.pdf_output_dir,
            task_filter=task_filter or "",
            http_proxy=http_proxy,
            https_proxy=https_proxy,
            db=db,
        )
        output_path.write_text(json.dumps({"mode": "queue", "download_only": True, "summary": download_summary}, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    # ── Three-Phase Pipeline (when --task is set, no other mode flags) ──
    if task_filter and not (review_only or parse_only or retrieval_only):
        from concurrent.futures import ThreadPoolExecutor

        summaries: dict[str, Any] = {}

        def _run_download():
            summaries["download"] = _run_download_pass(
                pdf_output_dir=paths.pdf_output_dir,
                task_filter=task_filter,
                http_proxy=http_proxy,
                https_proxy=https_proxy,
                db=db,
            )

        def _run_parse():
            summaries["parse"] = run_stage5_parse_pass(
                llm_backend=llm_backend,
                parse_max_workers=max(1, int(parse_workers)),
                pdf_output_dir=paths.pdf_output_dir,
                monitor_file=paths.monitor_file,
                monitor_interval=monitor_interval,
                min_query_candidates=1,
                task_filter=task_filter,
                db=db,
            )

        def _run_review():
            summaries["review"] = run_stage5_review_pass(
                llm_backend=llm_backend,
                review_max_workers=max(1, int(review_workers)),
                scheduler_mode=scheduler_mode,
                db=db,
                batch_size=batch_size,
                pdf_output_dir=paths.pdf_output_dir,
                monitor_file=paths.monitor_file,
                monitor_interval=monitor_interval,
                stale_processing_seconds=DEFAULT_STALE_PROCESSING_SECONDS,
                task_filter=task_filter,
            )

        # Pipeline-level monitor: writes 3-phase progress every monitor_interval
        _monitor_stop = threading.Event()
        _monitor_prev = {"downloaded": 0, "parsed": 0, "completed": 0}
        _monitor_last_at = time.monotonic()

        def _pipeline_monitor():
            nonlocal _monitor_prev, _monitor_last_at
            interval = max(1, int(monitor_interval))
            while not _monitor_stop.wait(interval):
                now = time.monotonic()
                elapsed = now - _monitor_last_at
                _monitor_last_at = now
                snapshot = load_queue_snapshot(engine=db, stale_processing_seconds=DEFAULT_STALE_PROCESSING_SECONDS)
                raw = dict(snapshot.get("raw_counts") or {})
                labels = dict(snapshot.get("canonical_labels") or {})
                # Count by parse_status from the DB directly
                parse_counts = db.execute(
                    text(
                        f"SELECT parse_status, COUNT(*) FROM {queue_table_name_resolved} "
                        f"WHERE task = :task GROUP BY parse_status"
                    ),
                    {"task": task_filter},
                ).mappings().all()
                parse_map = {str(r["parse_status"]): int(r["count"]) for r in parse_counts}
                review_pending = int(raw.get("pending", 0))
                downloaded = parse_map.get("downloaded", 0)
                parsed = parse_map.get("parsed", 0)
                completed = int(raw.get("completed", 0))

                def _rate(current, prev):
                    return round((current - prev) / elapsed * 60.0, 1) if elapsed > 0 else 0.0
                def _eta(current, total, rpm):
                    remain = total - current
                    return round(remain / rpm, 1) if rpm > 0 else None

                dl_rpm = _rate(downloaded, _monitor_prev["downloaded"])
                pa_rpm = _rate(parsed, _monitor_prev["parsed"])
                rv_rpm = _rate(completed, _monitor_prev["completed"])
                total_rows = downloaded + parsed + review_pending + completed + parse_map.get("failed", 0) + parse_map.get("pending", 0) + parse_map.get("downloading", 0) + parse_map.get("processing", 0) + raw.get("processing", 0) + raw.get("failed", 0)
                if total_rows == 0:
                    total_rows = sum(parse_map.values()) + sum(raw.values())

                mon = {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "download": {
                        "pending": parse_map.get("pending", 0), "downloaded": downloaded,
                        "failed": parse_map.get("failed", 0),
                        "speed_rpm": dl_rpm, "eta_min": _eta(downloaded, total_rows, dl_rpm),
                    },
                    "parse": {
                        "pending": downloaded - parsed if downloaded > parsed else 0,
                        "parsed": parsed, "failed": 0,
                        "speed_rpm": pa_rpm, "eta_min": _eta(parsed, total_rows, pa_rpm),
                    },
                    "review": {
                        "pending": review_pending, "completed": completed,
                        "failed": int(raw.get("failed", 0)),
                        "speed_rpm": rv_rpm, "eta_min": _eta(completed, total_rows, rv_rpm),
                        "hard_negative": int(labels.get("hard_negative", 0)),
                        "positive": int(labels.get("positive", 0)),
                        "ignored": int(labels.get("ignored", 0)),
                    },
                    "process": _read_process_runtime(),
                }
                _monitor_prev = {"downloaded": downloaded, "parsed": parsed, "completed": completed}
                mon_path = Path(paths.monitor_file)
                mon_path.parent.mkdir(parents=True, exist_ok=True)
                with open(mon_path, "a") as fh:
                    fh.write(json.dumps(mon, ensure_ascii=True) + "\n")

        monitor_thread = threading.Thread(target=_pipeline_monitor, daemon=True)
        monitor_thread.start()

        with ThreadPoolExecutor(max_workers=3) as pipeline_executor:
            futures = [
                pipeline_executor.submit(_run_download),
                pipeline_executor.submit(_run_parse),
                pipeline_executor.submit(_run_review),
            ]
            for f in as_completed(futures):
                f.result()

        _monitor_stop.set()
        monitor_thread.join(timeout=1.0)

        final_snapshot = load_queue_snapshot(engine=db, stale_processing_seconds=DEFAULT_STALE_PROCESSING_SECONDS)
        output_payload = {
            "mode": "queue",
            "pipeline": "download_parse_review_async",
            "task": task_filter,
            "queue_table_name": queue_table_name_resolved,
            "canonical_view_name": canonical_view_name,
            "effective_queries_path": str(paths.effective_queries_path),
            "pdf_output_dir": str(paths.pdf_output_dir),
            "download_summary": summaries.get("download"),
            "parse_summary": summaries.get("parse"),
            "review_summary": summaries.get("review"),
            "final_counts": dict(final_snapshot.get("canonical_counts") or {}),
            "final_label_counts": load_label_summary(engine=db, canonical=True),
            "query_count": query_dataset.total_queries,
            "paper_count": query_dataset.total_papers,
            "parameters": {
                "download_workers": download_workers,
                "parse_workers": parse_workers,
                "review_workers": review_workers,
                "docling_pool_size": docling_pool_size,
            },
        }
        output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    if review_only:
        while True:
            counts = load_status_summary(engine=db, canonical=True)
            pending = int(counts.get("pending", 0) or 0)
            processing = int(counts.get("processing", 0) or 0)
            if pending == 0 and processing == 0:
                break
            run_stage5_review_pass(
                llm_backend=llm_backend,
                review_max_workers=review_max_workers,
                scheduler_mode=scheduler_mode,
                batch_size=batch_size,
                pdf_output_dir=paths.pdf_output_dir,
                monitor_file=paths.monitor_file,
                monitor_interval=monitor_interval,
                stale_processing_seconds=DEFAULT_STALE_PROCESSING_SECONDS,
                task_filter=task_filter,
                db=db,
            )
            counts = load_status_summary(engine=db, canonical=True)
            if int(counts.get("pending", 0) or 0) == 0 and int(counts.get("processing", 0) or 0) == 0:
                break

        final_snapshot = load_queue_snapshot(engine=db, stale_processing_seconds=DEFAULT_STALE_PROCESSING_SECONDS)
        output_payload = {
            "mode": "queue",
            "review_only": True,
            "queue_table_name": queue_table_name_resolved,
            "canonical_view_name": canonical_view_name,
            "effective_queries_path": str(paths.effective_queries_path),
            "monitor_file": str(paths.monitor_file),
            "retrieve_done_flag": str(paths.retrieve_done_flag),
            "pdf_output_dir": str(paths.pdf_output_dir),
            "final_counts": dict(final_snapshot.get("canonical_counts") or {}),
            "final_label_counts": load_label_summary(engine=db, canonical=True),
            "latest_completed": list(final_snapshot.get("latest_completed") or []),
            "completed_parse_counts": dict(final_snapshot.get("completed_parse_counts") or {}),
            "latest_completed_parse_counts": dict(final_snapshot.get("latest_completed_parse_counts") or {}),
            "query_count": query_dataset.total_queries,
            "paper_count": query_dataset.total_papers,
            "review_parameters": {
                "scheduler_mode": scheduler_mode,
                "batch_size": batch_size,
                "review_max_workers": review_max_workers,
                "docling_pool_size": docling_pool_size,
                "monitor_interval": monitor_interval,
                "review_only": True,
            },
        }
        output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    if parse_only:
        parse_summaries: list[dict[str, Any]] = []
        while True:
            parse_summary = run_stage5_parse_pass(
                llm_backend=llm_backend,
                parse_max_workers=max(1, int(parse_workers)),
                pdf_output_dir=paths.pdf_output_dir,
                monitor_file=paths.monitor_file,
                monitor_interval=monitor_interval,
                min_query_candidates=int(parse_min_query_candidates),
                task_filter=task_filter,
                db=db,
            )
            parse_summaries.append(parse_summary)
            retrieval_done = paths.retrieve_done_flag.exists()
            if int(parse_summary.get("processed_rows", 0) or 0) > 0:
                continue
            if retrieval_done:
                break
            time.sleep(max(5, int(monitor_interval)))

        final_snapshot = load_queue_snapshot(engine=db, stale_processing_seconds=DEFAULT_STALE_PROCESSING_SECONDS)
        output_payload = {
            "mode": "queue",
            "parse_only": True,
            "queue_table_name": queue_table_name_resolved,
            "canonical_view_name": canonical_view_name,
            "effective_queries_path": str(paths.effective_queries_path),
            "monitor_file": str(paths.monitor_file),
            "retrieve_done_flag": str(paths.retrieve_done_flag),
            "pdf_output_dir": str(paths.pdf_output_dir),
            "parse_summaries": parse_summaries,
            "final_counts": dict(final_snapshot.get("canonical_counts") or {}),
            "completed_parse_counts": dict(final_snapshot.get("completed_parse_counts") or {}),
            "latest_completed_parse_counts": dict(final_snapshot.get("latest_completed_parse_counts") or {}),
            "query_count": query_dataset.total_queries,
            "paper_count": query_dataset.total_papers,
            "parse_parameters": {
                "parse_max_workers": review_max_workers,
                "docling_pool_size": docling_pool_size,
                "monitor_interval": monitor_interval,
                "parse_min_query_candidates": parse_min_query_candidates,
            },
        }
        output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    if int(retrieval_search_max_workers) > 1:
        retrieval_summary = enqueue_stage5_queue_candidates_parallel_search(
            query_dataset=query_dataset,
            llm_backend=llm_backend,
            search_settings=search_settings,
            monitor_file=paths.monitor_file,
            skip_existing_queries=skip_existing_queries,
            search_max_workers=int(retrieval_search_max_workers),
            rerank_max_workers=int(retrieval_rerank_max_workers),
        )
    else:
        retrieval_summary = enqueue_stage5_queue_candidates(
            query_dataset=query_dataset,
            llm_backend=llm_backend,
            search_settings=search_settings,
            monitor_file=paths.monitor_file,
            skip_existing_queries=skip_existing_queries,
            rerank_max_workers=int(retrieval_rerank_max_workers),
        )
    paths.retrieve_done_flag.parent.mkdir(parents=True, exist_ok=True)
    paths.retrieve_done_flag.write_text("done\n", encoding="utf-8")

    if retrieval_only:
        final_snapshot = load_queue_snapshot(engine=db, stale_processing_seconds=DEFAULT_STALE_PROCESSING_SECONDS)
        output_payload = {
            "mode": "queue",
            "retrieval_only": True,
            "queue_table_name": queue_table_name_resolved,
            "canonical_view_name": canonical_view_name,
            "effective_queries_path": str(paths.effective_queries_path),
            "monitor_file": str(paths.monitor_file),
            "retrieve_done_flag": str(paths.retrieve_done_flag),
            "pdf_output_dir": str(paths.pdf_output_dir),
            "retrieval_summary": retrieval_summary,
            "final_counts": dict(final_snapshot.get("canonical_counts") or {}),
            "final_label_counts": load_label_summary(engine=db, canonical=True),
            "latest_completed": list(final_snapshot.get("latest_completed") or []),
            "completed_parse_counts": dict(final_snapshot.get("completed_parse_counts") or {}),
            "latest_completed_parse_counts": dict(final_snapshot.get("latest_completed_parse_counts") or {}),
            "query_count": query_dataset.total_queries,
            "paper_count": query_dataset.total_papers,
            "review_parameters": {
                "scheduler_mode": scheduler_mode,
                "batch_size": batch_size,
                "review_max_workers": review_max_workers,
                "docling_pool_size": docling_pool_size,
                "monitor_interval": monitor_interval,
                "retrieval_only": True,
                "retrieval_search_max_workers": retrieval_search_max_workers,
                "retrieval_rerank_max_workers": retrieval_rerank_max_workers,
            },
        }
        output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    while True:
        counts = load_status_summary(engine=db, canonical=True)
        pending = int(counts.get("pending", 0) or 0)
        processing = int(counts.get("processing", 0) or 0)
        if pending == 0 and processing == 0:
            break
        run_stage5_review_pass(
            llm_backend=llm_backend,
            review_max_workers=review_max_workers,
            scheduler_mode=scheduler_mode,
            batch_size=batch_size,
            pdf_output_dir=paths.pdf_output_dir,
            monitor_file=paths.monitor_file,
            monitor_interval=monitor_interval,
            stale_processing_seconds=DEFAULT_STALE_PROCESSING_SECONDS,
            task_filter=task_filter,
            db=db,
        )
        counts = load_status_summary(engine=db, canonical=True)
        if int(counts.get("pending", 0) or 0) == 0 and int(counts.get("processing", 0) or 0) == 0:
            break

    final_snapshot = load_queue_snapshot(engine=db, stale_processing_seconds=DEFAULT_STALE_PROCESSING_SECONDS)
    output_payload = {
        "mode": "queue",
        "queue_table_name": queue_table_name_resolved,
        "canonical_view_name": canonical_view_name,
        "effective_queries_path": str(paths.effective_queries_path),
        "monitor_file": str(paths.monitor_file),
        "retrieve_done_flag": str(paths.retrieve_done_flag),
        "pdf_output_dir": str(paths.pdf_output_dir),
        "retrieval_summary": retrieval_summary,
        "final_counts": dict(final_snapshot.get("canonical_counts") or {}),
        "final_label_counts": load_label_summary(engine=db, canonical=True),
        "latest_completed": list(final_snapshot.get("latest_completed") or []),
        "completed_parse_counts": dict(final_snapshot.get("completed_parse_counts") or {}),
        "latest_completed_parse_counts": dict(final_snapshot.get("latest_completed_parse_counts") or {}),
        "query_count": query_dataset.total_queries,
        "paper_count": query_dataset.total_papers,
        "review_parameters": {
            "scheduler_mode": scheduler_mode,
            "batch_size": batch_size,
            "review_max_workers": review_max_workers,
            "docling_pool_size": docling_pool_size,
            "http_proxy": http_proxy,
            "https_proxy": https_proxy,
            "retrieval_search_max_workers": retrieval_search_max_workers,
            "retrieval_rerank_max_workers": retrieval_rerank_max_workers,
            "scholar_max_results": int(search_settings["max_results"]),
            "rerank_top_n": HardNegativeMiner.DEFAULT_RERANK_TOP_N,
            "max_hard_negatives": HardNegativeMiner.DEFAULT_MAX_HARD_NEGATIVES,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_file(output_path, output_payload)
    return output_path
