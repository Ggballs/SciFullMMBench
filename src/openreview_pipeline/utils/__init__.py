import json
import logging
from pathlib import Path
from typing import TypeVar, Generic

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
