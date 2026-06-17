import json
import logging
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .app_logging import configure_project_logging, resolve_log_dir
from .llm import (
    LLMBackend,
    LLMRequestManager,
    MockLLMBackend,
    OpenAICompatibleBackend,
    create_openai_compatible_backend,
    load_llm_config,
)
from .project_paths import (
    CONFIGS_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_LOG_DIR,
    OFFLINE_PROCESS_DIR,
    OUTPUTS_DIR,
    PROMPTS_DIR,
    REPO_ROOT,
    SCRIPTS_DIR,
    TEST_DATA_DIR,
    resolve_prompt_path,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def load_json(path: Path, model: type[T]) -> T:
    logger.info(f"Loading {model.__name__} from {path}")
    with open(path) as f:
        data = json.load(f)
    return model.model_validate(data)


def save_json(path: Path, data: BaseModel) -> None:
    logger.info(f"Saving {type(data).__name__} to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data.model_dump(mode="json"), f, indent=2, ensure_ascii=False)


def load_prompt_template(path: Path) -> str:
    logger.debug(f"Loading prompt template from {path}")
    with open(path) as f:
        return f.read()


def extract_interleaved_pdf_content(*args, **kwargs):
    from openreview_pipeline.pdf_interleaved_extraction import extract_interleaved_pdf_content as _extract

    return _extract(*args, **kwargs)


def write_interleaved_pdf_content(*args, **kwargs):
    from openreview_pipeline.pdf_interleaved_extraction import write_interleaved_pdf_content as _write

    return _write(*args, **kwargs)


__all__ = [
    "CONFIGS_DIR",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_LOG_DIR",
    "LLMBackend",
    "LLMRequestManager",
    "MockLLMBackend",
    "OpenAICompatibleBackend",
    "OFFLINE_PROCESS_DIR",
    "OUTPUTS_DIR",
    "PROMPTS_DIR",
    "REPO_ROOT",
    "SCRIPTS_DIR",
    "TEST_DATA_DIR",
    "create_openai_compatible_backend",
    "configure_project_logging",
    "extract_interleaved_pdf_content",
    "load_json",
    "load_llm_config",
    "load_prompt_template",
    "resolve_log_dir",
    "resolve_prompt_path",
    "save_json",
    "write_interleaved_pdf_content",
]
