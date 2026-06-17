from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, TextIO

from .project_paths import DEFAULT_LOG_DIR


LOG_DIR_ENV_VARS = ("OPENREVIEW_PIPELINE_LOG_DIR", "SCIFULL_LOG_DIR", "LOG_DIR")
_TEE_MARKER = "_openreview_pipeline_tee"
_CONSOLE_HANDLER_MARKER = "_openreview_pipeline_console_handler"


class TeeStream:
    def __init__(self, original: TextIO, log_file: TextIO) -> None:
        self.original = original
        self.log_file = log_file
        setattr(self, _TEE_MARKER, True)

    def write(self, data: str) -> int:
        self.original.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self.original.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.original.isatty()

    @property
    def encoding(self) -> Optional[str]:
        return getattr(self.original, "encoding", None)

    def fileno(self) -> int:
        return self.original.fileno()


def resolve_log_dir(log_dir: Optional[Path] = None) -> Path:
    raw_log_dir = log_dir or next(
        (os.getenv(name) for name in LOG_DIR_ENV_VARS if os.getenv(name)),
        None,
    )
    return Path(raw_log_dir).expanduser().resolve() if raw_log_dir else DEFAULT_LOG_DIR


def configure_project_logging(
    log_dir: Optional[Path] = None,
    *,
    level: int = logging.INFO,
    capture_prints: bool = True,
) -> Path:
    target_dir = resolve_log_dir(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    log_path = target_dir / "openreview_pipeline.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(getattr(handler, "baseFilename", "")) == log_path
        for handler in root_logger.handlers
    ):
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if not any(getattr(handler, _CONSOLE_HANDLER_MARKER, False) for handler in root_logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        setattr(stream_handler, _CONSOLE_HANDLER_MARKER, True)
        root_logger.addHandler(stream_handler)

    if capture_prints:
        stdout_path = target_dir / "openreview_pipeline.stdout.log"
        if not getattr(sys.stdout, _TEE_MARKER, False):
            stdout_file = stdout_path.open("a", encoding="utf-8", buffering=1)
            sys.stdout = TeeStream(sys.stdout, stdout_file)  # type: ignore[assignment]
        if not getattr(sys.stderr, _TEE_MARKER, False):
            stderr_file = stdout_path.open("a", encoding="utf-8", buffering=1)
            sys.stderr = TeeStream(sys.stderr, stderr_file)  # type: ignore[assignment]

    return target_dir
