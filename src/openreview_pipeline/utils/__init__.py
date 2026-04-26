import json
import logging
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

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
    from .pdf_interleaved_extraction import extract_interleaved_pdf_content as _extract

    return _extract(*args, **kwargs)


def write_interleaved_pdf_content(*args, **kwargs):
    from .pdf_interleaved_extraction import write_interleaved_pdf_content as _write

    return _write(*args, **kwargs)


__all__ = [
    "extract_interleaved_pdf_content",
    "load_json",
    "load_prompt_template",
    "save_json",
    "write_interleaved_pdf_content",
]
