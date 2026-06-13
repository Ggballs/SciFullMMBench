#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from openreview_pipeline.app_logging import configure_project_logging
from openreview_pipeline.runner import build_hard_negative_llm_backend, resolve_stage_settings
from openreview_pipeline.stages.stage5_hard_negative_mining import HardNegativeMiner, ScholarCandidatePaper
from openreview_pipeline.utils.db.stage5_candidate_queue_postgres import (
    claim_pending_candidates,
    ensure_schema,
    load_status_summary,
    update_candidate_result,
)

configure_project_logging()
logger = logging.getLogger(__name__)


def _row_to_candidate(row: dict) -> ScholarCandidatePaper:
    return ScholarCandidatePaper(
        paper_title=str(row.get("candidate_title") or ""),
        arxiv_id=str(row.get("candidate_arxiv_id")).strip() if row.get("candidate_arxiv_id") else None,
        abstract=str(row.get("candidate_abstract")).strip() if row.get("candidate_abstract") else None,
        venue=str(row.get("candidate_venue")).strip() if row.get("candidate_venue") else None,
        year=int(row["candidate_year"]) if row.get("candidate_year") is not None else None,
        authors=[str(author) for author in (row.get("candidate_authors") or [])],
        url=str(row.get("candidate_url")).strip() if row.get("candidate_url") else None,
        pdf_url=str(row.get("candidate_pdf_url")).strip() if row.get("candidate_pdf_url") else None,
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
    last_error: Optional[str] = None

    def as_snapshot(self, now_monotonic: float) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "status": self.status,
            "stage": self.stage,
            "stage_elapsed_seconds": max(0.0, now_monotonic - self.stage_started_at) if self.stage_started_at else 0.0,
            "row_elapsed_seconds": max(0.0, now_monotonic - self.row_started_at) if self.row_started_at else 0.0,
            "last_update_age_seconds": max(0.0, now_monotonic - self.last_update_at) if self.last_update_at else 0.0,
            "current_row_id": self.current_row_id,
            "current_query_key": self.current_query_key,
            "current_rank": self.current_rank,
            "current_title": self.current_title,
            "processed_rows": self.processed_rows,
            "last_result": self.last_result,
            "last_error": self.last_error,
        }


class WorkerMonitor:
    def __init__(self, output_path: Optional[Path]) -> None:
        self._lock = threading.Lock()
        self._output_path = output_path
        self._workers: dict[int, WorkerRuntime] = {}
        self._queue_counts: dict[str, int] = {}
        self._latest_summary: dict[str, float] = {}
        self._updated_at_iso: Optional[str] = None

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
            runtime.last_error = payload.get("error_message")
            runtime.stage = "idle"
            runtime.stage_started_at = now
            runtime.last_update_at = now
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
            runtime.current_row_id = None
            runtime.current_query_key = None
            runtime.current_rank = None
            runtime.current_title = None
            runtime.row_started_at = 0.0

    def update_summary(self, *, queue_counts: dict[str, int], latest_summary: dict[str, float]) -> None:
        with self._lock:
            self._queue_counts = dict(queue_counts)
            self._latest_summary = dict(latest_summary)
            self._updated_at_iso = datetime.now(timezone.utc).isoformat()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            workers = [runtime.as_snapshot(now) for runtime in sorted(self._workers.values(), key=lambda item: item.worker_id)]
            return {
                "updated_at": self._updated_at_iso,
                "queue_counts": dict(self._queue_counts),
                "summary": dict(self._latest_summary),
                "workers": workers,
            }

    def write_snapshot(self) -> None:
        if self._output_path is None:
            return
        snapshot = self.snapshot()
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(json.dumps(snapshot, ensure_ascii=True, indent=2), encoding="utf-8")


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Review queued Stage5 candidates from PostgreSQL.")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--db-url", default="", help="PostgreSQL URL override")
    parser.add_argument(
        "--scheduler-mode",
        choices=["independent_worker", "batch_size"],
        default="independent_worker",
        help="How workers claim queue rows",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Rows to process when scheduler mode is batch_size")
    parser.add_argument("--max-workers", type=int, default=0, help="Override review worker count")
    parser.add_argument("--pdf-output-dir", default="", help="Optional writable PDF cache directory")
    parser.add_argument("--summary-interval", type=int, default=30, help="Periodic summary interval in seconds")
    parser.add_argument(
        "--monitor-file",
        default="outputs/stage5_review_monitor.json",
        help="Optional JSON snapshot path for worker health monitoring",
    )
    parser.add_argument("--monitor-interval", type=int, default=5, help="Worker monitor snapshot interval in seconds")
    args = parser.parse_args()

    config_path = Path(args.config)
    db = ensure_schema(db_url=args.db_url or None)
    stage_settings = resolve_stage_settings(config_path)
    max_workers = args.max_workers or int(stage_settings["hard_negative_review_max_workers"])
    max_workers = max(1, int(max_workers))
    batch_size = max(1, int(args.batch_size))

    llm_backend = build_hard_negative_llm_backend(config_path)
    miner = HardNegativeMiner(
        llm=llm_backend,
        scholar_client=None,  # type: ignore[arg-type]
        scholar_max_results=10,
        download_selected_pdfs=False,
        review_max_workers=max_workers,
        pdf_output_dir=Path(args.pdf_output_dir).expanduser().resolve() if args.pdf_output_dir else None,
    )

    llm_key_count = len(getattr(getattr(llm_backend, "request_manager", None), "_slots", []))
    llm_per_key_concurrency = int(getattr(getattr(llm_backend, "request_manager", None), "per_key_max_concurrent_requests", 1))
    logger.info(
        "stage5_startup scheduler_mode=%s max_workers=%s batch_size=%s llm_keys=%s llm_per_key_concurrency=%s llm_budget=%s",
        args.scheduler_mode,
        max_workers,
        batch_size,
        llm_key_count,
        llm_per_key_concurrency,
        llm_key_count * llm_per_key_concurrency,
    )

    stats = ProgressStats()
    monitor = WorkerMonitor(
        Path(args.monitor_file).expanduser().resolve() if str(args.monitor_file).strip() else None
    )
    stop_event = threading.Event()
    processed_counter = 0
    processed_lock = threading.Lock()

    def can_process_more() -> bool:
        if args.scheduler_mode == "independent_worker":
            return True
        with processed_lock:
            return processed_counter < batch_size

    def claim_next_row() -> Optional[dict]:
        nonlocal processed_counter
        if not can_process_more():
            return None
        try:
            rows = claim_pending_candidates(1, engine=db)
        except Exception as exc:
            logger.exception("stage5_row_claim_failed error=%s", exc)
            return None
        if not rows:
            return None
        with processed_lock:
            processed_counter += 1
        row = rows[0]
        logger.info(
            "stage5_row_claimed id=%s query_key=%s retrieval_rank=%s title=%r processed_counter=%s",
            row.get("id"),
            row.get("query_key"),
            row.get("retrieval_rank"),
            row.get("candidate_title"),
            processed_counter,
        )
        return row

    def process_row(
        worker_id: int,
        row: dict,
        *,
        on_download_stage_complete: Optional[Callable[[], None]] = None,
    ) -> tuple[int, dict[str, Any]]:
        started_at = time.monotonic()
        candidate = _row_to_candidate(row)
        monitor.worker_stage(worker_id, "reviewing", row=row)

        def on_stage_update(stage: str, metadata: dict[str, Any]) -> None:
            if stage == "download":
                monitor.worker_stage(worker_id, "download", row=row)
            elif stage in {"docling", "docling_fallback"}:
                monitor.worker_stage(worker_id, "docling", row=row)
            elif stage == "llm":
                monitor.worker_stage(worker_id, "llm", row=row)

        try:
            review = miner._review_single_candidate(
                str(row["query_text"]),
                candidate,
                on_download_stage_complete=on_download_stage_complete,
                on_stage_update=on_stage_update,
            )
        except Exception as exc:
            monitor.worker_stage(worker_id, "failed", row=row, error=f"review_exception: {exc}")
            logger.exception("stage5_row_failed id=%s stage=review_exception error=%s", row.get("id"), exc)
            return int(row["id"]), {
                "parse_status": "metadata_only",
                "review_status": "failed",
                "error_message": f"review_exception: {exc}",
                "review_payload": None,
                "timings": {"total_row_seconds": time.monotonic() - started_at},
            }

        if not review:
            monitor.worker_stage(worker_id, "failed", row=row, error="response_parse_failed")
            logger.warning("stage5_row_failed id=%s stage=response_parse_failed", row.get("id"))
            return int(row["id"]), {
                "parse_status": "metadata_only",
                "review_status": "failed",
                "error_message": "response_parse_failed",
                "review_payload": None,
                "timings": {"total_row_seconds": time.monotonic() - started_at},
            }

        timings = dict(review.get("timings") or {})
        timings.setdefault("total_row_seconds", time.monotonic() - started_at)
        return int(row["id"]), {
            "parse_status": str(review.get("parse_status") or "metadata_only"),
            "review_status": "completed",
            "review_label": str(review.get("label") or ""),
            "review_reason": str(review.get("reason") or ""),
            "error_message": None,
            "review_payload": {
                "candidate_title": candidate.paper_title,
                "candidate_arxiv_id": candidate.arxiv_id,
                "candidate_pdf_url": candidate.pdf_url,
                "label": review.get("label"),
                "need_pro_review": bool(review.get("need_pro_review", False)),
                "reason": review.get("reason"),
                "parse_status": review.get("parse_status"),
            },
            "timings": timings,
        }

    def persist_row_result(row: dict, candidate_id: int, payload: dict[str, Any]) -> None:
        worker_id = int(payload.get("_worker_id") or 0)
        if worker_id:
            monitor.worker_stage(worker_id, "persisting", row=row)
        db_payload = {key: value for key, value in payload.items() if key not in {"timings", "_worker_id"}}
        try:
            update_candidate_result(candidate_id, engine=db, **db_payload)
        except Exception as exc:
            if worker_id:
                monitor.worker_stage(worker_id, "db_update_failed", row=row, error=str(exc))
            logger.exception("stage5_db_update_failed id=%s error=%s", candidate_id, exc)
            raise
        stats.record_row(payload)
        if worker_id:
            monitor.worker_completed_row(worker_id, payload)
        log_event = "stage5_row_completed" if payload.get("review_status") == "completed" else "stage5_row_failed"
        logger.info(
            "%s id=%s rank=%s review_status=%s label=%s parse_status=%s total_row_seconds=%.3f download_seconds=%.3f docling_wait_seconds=%.3f docling_parse_seconds=%.3f deepseek_wait_seconds=%.3f deepseek_request_seconds=%.3f title=%r error=%s",
            log_event,
            candidate_id,
            row.get("retrieval_rank"),
            payload.get("review_status"),
            payload.get("review_label"),
            payload.get("parse_status"),
            float((payload.get("timings") or {}).get("total_row_seconds", 0.0) or 0.0),
            float((payload.get("timings") or {}).get("download_seconds", 0.0) or 0.0),
            float((payload.get("timings") or {}).get("docling_wait_seconds", 0.0) or 0.0),
            float((payload.get("timings") or {}).get("docling_parse_seconds", 0.0) or 0.0),
            float((payload.get("timings") or {}).get("deepseek_wait_seconds", 0.0) or 0.0),
            float((payload.get("timings") or {}).get("deepseek_request_seconds", 0.0) or 0.0),
            row.get("candidate_title"),
            payload.get("error_message"),
        )

    def worker_loop(worker_id: int, initial_row: dict, startup_event: threading.Event) -> WorkerSummary:
        summary = WorkerSummary(worker_id=worker_id)
        stats.worker_started()
        monitor.worker_started(worker_id, initial_row)
        logger.info("stage5_worker_started worker_id=%s", worker_id)
        row: Optional[dict] = initial_row
        first_row = True
        try:
            while row is not None:
                # Keep trying the gate until it fires — a single failed download
                # shouldn't permanently cap us at one worker.
                callback_event = startup_event if not startup_event.is_set() else None
                callback_invoked = threading.Event()

                def on_download_stage_complete() -> None:
                    if callback_event is not None and not callback_invoked.is_set():
                        callback_invoked.set()
                        callback_event.set()
                        logger.info("stage5_startup_gate_open worker_id=%s row_id=%s", worker_id, row.get("id"))

                candidate_id, payload = process_row(
                    worker_id,
                    row,
                    on_download_stage_complete=on_download_stage_complete if callback_event is not None else None,
                )
                try:
                    payload["_worker_id"] = worker_id
                    persist_row_result(row, candidate_id, payload)
                except Exception:
                    logger.exception("stage5_row_persist_failed worker_id=%s row_id=%s", worker_id, row.get("id"))
                summary.processed_rows += 1
                first_row = False
                row = claim_next_row()
                if row is not None:
                    monitor.worker_stage(worker_id, "claimed", row=row)
                    logger.info("stage5_worker_claimed_next worker_id=%s next_row_id=%s", worker_id, row.get("id"))
        finally:
            stats.worker_exited()
            monitor.worker_exited(worker_id, summary.processed_rows)
            logger.info("stage5_worker_exited worker_id=%s processed_rows=%s", worker_id, summary.processed_rows)
        return summary

    def summary_loop() -> None:
        interval = max(1, int(args.summary_interval))
        while not stop_event.wait(interval):
            counts = load_status_summary(engine=db, canonical=True)
            snapshot = stats.snapshot_interval()
            monitor.update_summary(queue_counts=counts, latest_summary=snapshot)
            monitor.write_snapshot()
            logger.info(
                "stage5_loop_summary pending=%s processing=%s completed=%s failed=%s active_workers=%s rows_in_interval=%s avg_download_seconds=%.3f avg_docling_wait_seconds=%.3f avg_docling_parse_seconds=%.3f avg_deepseek_wait_seconds=%.3f avg_deepseek_request_seconds=%.3f avg_total_row_seconds=%.3f",
                counts.get("pending", 0),
                counts.get("processing", 0),
                counts.get("completed", 0),
                counts.get("failed", 0),
                int(snapshot["active_workers"]),
                int(snapshot["rows_in_interval"]),
                snapshot["avg_download_seconds"],
                snapshot["avg_docling_wait_seconds"],
                snapshot["avg_docling_parse_seconds"],
                snapshot["avg_deepseek_wait_seconds"],
                snapshot["avg_deepseek_request_seconds"],
                snapshot["avg_total_row_seconds"],
            )

    def monitor_loop() -> None:
        interval = max(1, int(args.monitor_interval))
        while not stop_event.wait(interval):
            monitor.write_snapshot()

    first_row = claim_next_row()
    if first_row is None:
        logger.info("stage5_queue_empty")
        print("no pending candidates")
        print(load_status_summary(engine=db))
        return 0

    summary_thread = threading.Thread(target=summary_loop, daemon=True)
    summary_thread.start()
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()

    worker_count = max_workers if args.scheduler_mode == "independent_worker" else min(max_workers, batch_size)
    futures: list[Future[WorkerSummary]] = []

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        next_initial_row: Optional[dict] = first_row
        for worker_id in range(1, worker_count + 1):
            if next_initial_row is None:
                break
            startup_event = threading.Event()
            futures.append(executor.submit(worker_loop, worker_id, next_initial_row, startup_event))
            logger.info("stage5_worker_launch worker_id=%s initial_row_id=%s", worker_id, next_initial_row.get("id"))
            if worker_id < worker_count:
                logger.info("stage5_startup_gate_waiting worker_id=%s", worker_id)
                startup_event.wait()
                next_initial_row = claim_next_row()
            else:
                next_initial_row = None
        logger.info("stage5_startup_target_reached launched_workers=%s", len(futures))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logger.exception("stage5_worker_future_failed error=%s", exc)

    stop_event.set()
    summary_thread.join(timeout=1.0)
    monitor.update_summary(queue_counts=load_status_summary(engine=db, canonical=True), latest_summary=stats.snapshot_interval())
    monitor.write_snapshot()
    monitor_thread.join(timeout=1.0)

    print(load_status_summary(engine=db))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
