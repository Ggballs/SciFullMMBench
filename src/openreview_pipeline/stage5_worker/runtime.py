from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from utils.docling_parse import parse_pdf_text_only

logger = logging.getLogger(__name__)

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

DOCLING_TIMEOUT_SECONDS_DEFAULT = int(os.environ.get("DOCLING_TIMEOUT_SECONDS", "300"))
_DOCLING_POOL_SIZE = max(
    1,
    int(
        os.environ.get(
            "DOCLING_POOL_SIZE",
            os.environ.get("DOCLING_MAX_CONCURRENT", "4"),
        )
    ),
)
_PDF_DOWNLOAD_TIMEOUT_SECONDS = int(os.environ.get("PDF_DOWNLOAD_TIMEOUT_SECONDS", "45"))
_PDF_DOWNLOAD_MAX_RETRIES = int(os.environ.get("PDF_DOWNLOAD_MAX_RETRIES", "4"))
_PDF_DOWNLOAD_RETRY_BACKOFF_SECONDS = float(os.environ.get("PDF_DOWNLOAD_RETRY_BACKOFF_SECONDS", "8"))
_PDF_DOWNLOAD_COOLDOWN_SECONDS = float(os.environ.get("PDF_DOWNLOAD_COOLDOWN_SECONDS", "3"))
_ARXIV_TUNNEL_PORT = int(os.environ.get("ARXIV_TUNNEL_PORT", "0"))
_ARXIV_CONNECT_TO_ARGS: list[str] = []
if _ARXIV_TUNNEL_PORT > 0:
    for host in ("arxiv.org:443", "export.arxiv.org:443"):
        _ARXIV_CONNECT_TO_ARGS.extend(["--connect-to", f"{host}:127.0.0.1:{_ARXIV_TUNNEL_PORT}"])

StageCallback = Callable[[str, dict[str, Any]], None]


def _pdf_proxy_settings() -> dict[str, Any]:
    return {
        "http_proxy": os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or "",
        "https_proxy": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or "",
        "arxiv_tunnel_port": _ARXIV_TUNNEL_PORT,
        "uses_connect_to_tunnel": bool(_ARXIV_CONNECT_TO_ARGS),
    }


class _PdfDownloadGate:
    def __init__(self, cooldown_seconds: float) -> None:
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._condition = threading.Condition()
        self._queue: deque[tuple[int, int]] = deque()
        self._next_ticket = 1
        self._owner_worker_id: Optional[int] = None
        self._owner_title: Optional[str] = None
        self._state = "idle"
        self._cooldown_until = 0.0

    def acquire(self, worker_id: int, paper_title: str) -> None:
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            self._queue.append((ticket, worker_id))
            while True:
                now = time.monotonic()
                cooldown_remaining = max(0.0, self._cooldown_until - now)
                queue_head = self._queue[0] if self._queue else None
                if (
                    queue_head is not None
                    and queue_head[0] == ticket
                    and self._owner_worker_id is None
                    and cooldown_remaining <= 0.0
                ):
                    self._queue.popleft()
                    self._owner_worker_id = worker_id
                    self._owner_title = paper_title
                    self._state = "downloading"
                    return
                self._condition.wait(timeout=cooldown_remaining if cooldown_remaining > 0 else None)

    def release(self, *, success: bool) -> None:
        if success:
            with self._condition:
                self._owner_worker_id = None
                self._owner_title = None
                self._state = "idle"
                self._cooldown_until = 0.0
                self._condition.notify_all()
            return

        with self._condition:
            self._owner_worker_id = None
            self._owner_title = None
            self._state = "cooldown"
            self._cooldown_until = time.monotonic() + self._cooldown_seconds
            self._condition.notify_all()
        time.sleep(self._cooldown_seconds)
        with self._condition:
            if self._state == "cooldown" and time.monotonic() >= self._cooldown_until:
                self._state = "idle"
                self._cooldown_until = 0.0
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            cooldown_remaining = max(0.0, self._cooldown_until - time.monotonic())
            return {
                "state": self._state,
                "owner_worker_id": self._owner_worker_id,
                "owner_title": self._owner_title,
                "queued_workers": [worker_id for _, worker_id in self._queue],
                "queued_worker_count": len(self._queue),
                "cooldown_remaining_seconds": round(cooldown_remaining, 3),
            }


_PDF_DOWNLOAD_GATE = _PdfDownloadGate(cooldown_seconds=_PDF_DOWNLOAD_COOLDOWN_SECONDS)
_DOCLING_INFLIGHT_LOCK = threading.Lock()
_DOCLING_INFLIGHT = 0
_WORKER_DOCLING_STATE_LOCK = threading.Lock()
_WORKER_DOCLING_STATES: dict[int, dict[str, Any]] = {}


@dataclass
class _DoclingPoolSlot:
    slot_id: int
    initialized: bool = False
    busy: bool = False
    current_worker_id: Optional[int] = None
    parse_count: int = 0
    last_parse_seconds: float = 0.0
    last_error: Optional[str] = None


class _DoclingPool:
    def __init__(self, pool_size: int) -> None:
        self._pool_size = max(1, int(pool_size))
        self._condition = threading.Condition()
        self._slots = [_DoclingPoolSlot(slot_id=index + 1) for index in range(self._pool_size)]
        self._wait_queue: deque[tuple[int, int]] = deque()
        self._next_ticket = 1

    def acquire(self, worker_id: int) -> _DoclingPoolSlot:
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            self._wait_queue.append((ticket, worker_id))
            _update_worker_docling_state(worker_id, waiting_for_docling=True)
            while True:
                queue_head = self._wait_queue[0] if self._wait_queue else None
                free_slot = next((slot for slot in self._slots if not slot.busy), None)
                if queue_head is not None and queue_head[0] == ticket and free_slot is not None:
                    self._wait_queue.popleft()
                    free_slot.busy = True
                    free_slot.current_worker_id = worker_id
                    _update_worker_docling_state(
                        worker_id,
                        waiting_for_docling=False,
                        holding_docling_slot=True,
                        docling_slot_id=free_slot.slot_id,
                    )
                    return free_slot
                self._condition.wait()

    def release(self, slot_id: int, *, worker_id: Optional[int], parse_seconds: float, error: Optional[str]) -> None:
        with self._condition:
            slot = self._slots[slot_id - 1]
            slot.busy = False
            slot.current_worker_id = None
            slot.last_parse_seconds = float(parse_seconds or 0.0)
            slot.last_error = error
            if error is None:
                slot.parse_count += 1
            if worker_id is not None:
                _update_worker_docling_state(
                    worker_id,
                    holding_docling_slot=False,
                    docling_slot_id=None,
                )
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            slots = [
                {
                    "slot_id": slot.slot_id,
                    "initialized": slot.initialized,
                    "busy": slot.busy,
                    "current_worker_id": slot.current_worker_id,
                    "parse_count": slot.parse_count,
                    "last_parse_seconds": round(slot.last_parse_seconds, 3),
                    "last_error": slot.last_error,
                }
                for slot in self._slots
            ]
            busy_slots = sum(1 for slot in self._slots if slot.busy)
            initialized_slots = sum(1 for slot in self._slots if slot.initialized)
            queued_workers = [worker_id for _, worker_id in self._wait_queue]
            with _WORKER_DOCLING_STATE_LOCK:
                worker_states = {worker_id: dict(state) for worker_id, state in _WORKER_DOCLING_STATES.items()}
            return {
                "mode": "shared_pool",
                "pool_size": self._pool_size,
                "max_concurrent": self._pool_size,
                "inflight": busy_slots,
                "initialized_slots": initialized_slots,
                "busy_slots": busy_slots,
                "available_slots": self._pool_size - busy_slots,
                "queued_workers": queued_workers,
                "queued_worker_ids": queued_workers,
                "queued_worker_count": len(queued_workers),
                "slots": slots,
                "workers": worker_states,
            }


_DOCLING_POOL = _DoclingPool(pool_size=_DOCLING_POOL_SIZE)


def get_docling_runtime_snapshot() -> dict[str, Any]:
    return _DOCLING_POOL.snapshot()


def get_pdf_download_runtime_snapshot() -> dict[str, Any]:
    return _PDF_DOWNLOAD_GATE.snapshot()


def _ensure_worker_docling_state(worker_id: int) -> dict[str, Any]:
    with _WORKER_DOCLING_STATE_LOCK:
        state = _WORKER_DOCLING_STATES.get(worker_id)
        if state is None:
            state = {
                "worker_id": worker_id,
                "converter_initialized": False,
                "parse_count": 0,
                "last_parse_seconds": 0.0,
                "last_error": None,
                "current_parse": False,
                "waiting_for_docling": False,
                "holding_docling_slot": False,
                "docling_slot_id": None,
            }
            _WORKER_DOCLING_STATES[worker_id] = state
        return state


def _update_worker_docling_state(worker_id: Optional[int], **updates: Any) -> None:
    if worker_id is None:
        return
    with _WORKER_DOCLING_STATE_LOCK:
        state = _WORKER_DOCLING_STATES.get(worker_id)
        if state is None:
            state = {
                "worker_id": worker_id,
                "converter_initialized": False,
                "parse_count": 0,
                "last_parse_seconds": 0.0,
                "last_error": None,
                "current_parse": False,
                "waiting_for_docling": False,
                "holding_docling_slot": False,
                "docling_slot_id": None,
            }
            _WORKER_DOCLING_STATES[worker_id] = state
        parse_count_increment = int(updates.pop("parse_count_increment", 0) or 0)
        state.update(updates)
        if parse_count_increment:
            state["parse_count"] = int(state.get("parse_count", 0)) + parse_count_increment


def _is_retryable_pdf_download_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
    if isinstance(exc, subprocess.TimeoutExpired):
        return True
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = (getattr(exc, "stderr", "") or "").lower()
        if exc.returncode in {6, 7, 28}:
            return True
        if " 403 " in stderr or " 404 " in stderr or " 410 " in stderr:
            return False
        return True
    return isinstance(exc, (TimeoutError, URLError))


def _download_pdf_to_path(pdf_url: str, pdf_path: Path, timeout_seconds: int) -> None:
    curl_path = shutil.which("curl")
    if curl_path:
        cmd = [
            curl_path,
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            str(timeout_seconds),
            *_ARXIV_CONNECT_TO_ARGS,
            "-A",
            "Mozilla/5.0",
            "-H",
            "Accept: application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
            "-o",
            str(pdf_path),
            pdf_url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds + 5)
        if result.returncode != 0:
            stderr_summary = (result.stderr or "").strip()[:300]
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=f"curl stderr: {stderr_summary}" if stderr_summary else result.stderr,
            )
        return

    request = Request(
        pdf_url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as resp:
        pdf_path.write_bytes(resp.read())


class Stage5RuntimeSupport:
    _worker_local = threading.local()

    def bind_worker_context(self, worker_id: int) -> None:
        self._worker_local.worker_id = int(worker_id)
        _ensure_worker_docling_state(int(worker_id))

    def unbind_worker_context(self) -> None:
        for attr in ("worker_id",):
            if hasattr(self._worker_local, attr):
                delattr(self._worker_local, attr)

    def _get_worker_id(self) -> Optional[int]:
        worker_id = getattr(self._worker_local, "worker_id", None)
        return int(worker_id) if worker_id is not None else None

    def _parse_pdf_direct_docling(self, pdf_path: Path, paper_title: str) -> tuple[Optional[str], dict[str, Any]]:
        global _DOCLING_INFLIGHT
        worker_id = self._get_worker_id()
        metrics: dict[str, Any] = {
            "docling_wait_seconds": 0.0,
            "docling_parse_seconds": 0.0,
            "docling_stage": "direct_parse",
        }

        wait_started_at = time.monotonic()
        slot = _DOCLING_POOL.acquire(int(worker_id or 0))
        metrics["docling_wait_seconds"] = time.monotonic() - wait_started_at
        parse_started_at = time.monotonic()
        with _DOCLING_INFLIGHT_LOCK:
            _DOCLING_INFLIGHT += 1
        _update_worker_docling_state(
            worker_id,
            current_parse=True,
            last_error=None,
            docling_slot_id=slot.slot_id,
        )

        parse_error: Optional[str] = None
        try:
            slot.initialized = True
            _update_worker_docling_state(worker_id, converter_initialized=True)

            payload = parse_pdf_text_only(pdf_path, disable_table_structure=False)
            markdown = str(payload.get("markdown") or "") if payload.get("ok") else None
            page_count = int(payload.get("page_count") or 0)
            parse_error = None if payload.get("ok") else str(payload.get("error") or "docling_parse_failed")

            if markdown is None and parse_error and "is not valid" not in parse_error.lower():
                logger.warning(
                    "stage5_docling_retry_without_tables title=%r slot_id=%s error=%s",
                    paper_title,
                    slot.slot_id,
                    parse_error,
                )
                fallback_payload = parse_pdf_text_only(pdf_path, disable_table_structure=True)
                if fallback_payload.get("ok"):
                    markdown = str(fallback_payload.get("markdown") or "")
                    page_count = int(fallback_payload.get("page_count") or 0)
                    parse_error = None
                    metrics["docling_stage"] = "docling_direct_success_no_tables"
                else:
                    parse_error = str(fallback_payload.get("error") or parse_error or "docling_parse_failed")

            if markdown is None:
                raise RuntimeError(parse_error or "docling_parse_failed")

            metrics["docling_parse_seconds"] = time.monotonic() - parse_started_at
            metrics.setdefault("docling_stage", "docling_direct_success")
            _update_worker_docling_state(
                worker_id,
                current_parse=False,
                last_parse_seconds=metrics["docling_parse_seconds"],
                last_error=None,
                parse_count_increment=1,
            )
            logger.info(
                "stage5_docling_direct_success title=%r slot_id=%s chars=%s pages=%s wait_seconds=%.3f parse_seconds=%.3f",
                paper_title,
                slot.slot_id,
                len(markdown),
                page_count,
                metrics["docling_wait_seconds"],
                metrics["docling_parse_seconds"],
            )
            return markdown, metrics
        except Exception as exc:
            parse_error = str(exc)
            metrics["docling_parse_seconds"] = time.monotonic() - parse_started_at
            metrics["docling_stage"] = "docling_direct_failed"
            metrics["docling_error"] = parse_error
            _update_worker_docling_state(
                worker_id,
                current_parse=False,
                last_parse_seconds=metrics["docling_parse_seconds"],
                last_error=parse_error,
            )
            logger.warning(
                "stage5_docling_direct_failed title=%r slot_id=%s parse_seconds=%.3f error=%s",
                paper_title,
                slot.slot_id,
                metrics["docling_parse_seconds"],
                exc,
            )
            return None, metrics
        finally:
            _DOCLING_POOL.release(
                slot.slot_id,
                worker_id=worker_id,
                parse_seconds=metrics["docling_parse_seconds"],
                error=parse_error,
            )
            with _DOCLING_INFLIGHT_LOCK:
                _DOCLING_INFLIGHT = max(0, _DOCLING_INFLIGHT - 1)

    def _download_and_parse_pdf(
        self,
        pdf_url: str,
        paper_title: str,
        *,
        worker_id: Optional[int] = None,
        on_download_stage_complete: Optional[Callable[[], None]] = None,
        on_stage_update: Optional[StageCallback] = None,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "download_seconds": 0.0,
            "docling_wait_seconds": 0.0,
            "docling_parse_seconds": 0.0,
            "docling_stage": "not_started",
        }
        pdf_dir = self.pdf_output_dir or (Path("outputs") / "hard_negative_pdfs")
        pdf_dir = Path(pdf_dir)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        pdf_name = f"{self._pdf_cache_slug(paper_title)}-{self._pdf_url_hash(pdf_url)}.pdf"
        pdf_path = pdf_dir / pdf_name
        startup_gate_opened = False

        def _download_if_needed(*, force_redownload: bool = False) -> Optional[dict[str, Any]]:
            nonlocal startup_gate_opened
            need_download = force_redownload or not pdf_path.exists() or pdf_path.stat().st_size == 0
            if not need_download:
                return None

            download_success = False
            download_started_at = time.monotonic()
            try:
                if on_stage_update is not None:
                    on_stage_update("download_wait", {"pdf_url": pdf_url})
                _PDF_DOWNLOAD_GATE.acquire(int(worker_id or 0), paper_title)
                try:
                    if on_stage_update is not None:
                        on_stage_update("download", {"pdf_url": pdf_url, **_pdf_proxy_settings()})
                    proxy_settings = _pdf_proxy_settings()
                    logger.info(
                        "stage5_pdf_download_begin title=%r pdf_url=%s force_redownload=%s http_proxy=%r https_proxy=%r arxiv_tunnel_port=%s uses_connect_to_tunnel=%s",
                        paper_title,
                        pdf_url,
                        force_redownload,
                        proxy_settings["http_proxy"],
                        proxy_settings["https_proxy"],
                        proxy_settings["arxiv_tunnel_port"],
                        proxy_settings["uses_connect_to_tunnel"],
                    )
                    for attempt in range(1, _PDF_DOWNLOAD_MAX_RETRIES + 1):
                        try:
                            if force_redownload and pdf_path.exists():
                                pdf_path.unlink(missing_ok=True)
                            _download_pdf_to_path(pdf_url, pdf_path, _PDF_DOWNLOAD_TIMEOUT_SECONDS)
                            logger.info(
                                "stage5_pdf_download_success title=%r pdf_url=%s bytes=%s attempt=%s force_redownload=%s",
                                paper_title,
                                pdf_url,
                                pdf_path.stat().st_size,
                                attempt,
                                force_redownload,
                            )
                            download_success = True
                            break
                        except Exception as exc:
                            if pdf_path.exists():
                                pdf_path.unlink(missing_ok=True)
                            retryable = _is_retryable_pdf_download_error(exc)
                            if attempt < _PDF_DOWNLOAD_MAX_RETRIES and retryable:
                                backoff_seconds = _PDF_DOWNLOAD_RETRY_BACKOFF_SECONDS * attempt
                                logger.warning(
                                    "stage5_pdf_download_retry title=%r pdf_url=%s attempt=%s/%s backoff_seconds=%.1f force_redownload=%s error=%s",
                                    paper_title,
                                    pdf_url,
                                    attempt,
                                    _PDF_DOWNLOAD_MAX_RETRIES,
                                    backoff_seconds,
                                    force_redownload,
                                    exc,
                                )
                                time.sleep(backoff_seconds)
                                continue
                            logger.warning(
                                "stage5_pdf_download_failed title=%r pdf_url=%s attempt=%s/%s force_redownload=%s error=%s",
                                paper_title,
                                pdf_url,
                                attempt,
                                _PDF_DOWNLOAD_MAX_RETRIES,
                                force_redownload,
                                exc,
                            )
                            if on_stage_update is not None:
                                on_stage_update(
                                    "download_cooldown",
                                    {
                                        "pdf_url": pdf_url,
                                        "cooldown_seconds": _PDF_DOWNLOAD_COOLDOWN_SECONDS,
                                    },
                                )
                            return {
                                "markdown": None,
                                "metrics": metrics,
                                "failure_stage": "download_failed",
                            }
                finally:
                    _PDF_DOWNLOAD_GATE.release(success=download_success)
            finally:
                metrics["download_seconds"] += time.monotonic() - download_started_at
                try:
                    if download_success and on_download_stage_complete is not None and not startup_gate_opened:
                        startup_gate_opened = True
                        on_download_stage_complete()
                except Exception as exc:
                    logger.warning("stage5_download_stage_callback_failed title=%r error=%s", paper_title, exc)
            return None

        download_result = _download_if_needed(force_redownload=False)
        if download_result is not None:
            return download_result

        if on_stage_update is not None:
            on_stage_update("docling", {"pdf_url": pdf_url})
        markdown, docling_metrics = self._parse_pdf_direct_docling(pdf_path, paper_title)
        metrics.update(docling_metrics)
        if markdown:
            return {"markdown": markdown, "metrics": metrics, "failure_stage": None}

        docling_error = str(docling_metrics.get("docling_error") or "").lower()
        if "is not valid" in docling_error:
            logger.warning(
                "stage5_invalid_cached_pdf title=%r pdf_path=%s action=delete_and_redownload",
                paper_title,
                pdf_path,
            )
            if pdf_path.exists():
                pdf_path.unlink(missing_ok=True)
            download_result = _download_if_needed(force_redownload=True)
            if download_result is not None:
                return download_result
            if on_stage_update is not None:
                on_stage_update("docling", {"pdf_url": pdf_url, "retry": True})
            markdown, docling_metrics = self._parse_pdf_direct_docling(pdf_path, paper_title)
            metrics.update(docling_metrics)
            if markdown:
                return {"markdown": markdown, "metrics": metrics, "failure_stage": None}

        return {"markdown": None, "metrics": metrics, "failure_stage": "docling_parse_failed"}

    def _pdf_cache_slug(self, paper_title: str) -> str:
        raise NotImplementedError

    def _pdf_url_hash(self, pdf_url: str) -> str:
        raise NotImplementedError
