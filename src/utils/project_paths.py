from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "config.yaml"
OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_LOG_DIR = OUTPUTS_DIR / "logs"
TEST_DATA_DIR = REPO_ROOT / "tests" / "test_data"
SCRIPTS_DIR = REPO_ROOT / "tests" / "scripts"
PACKAGE_ROOT = SRC_ROOT / "openreview_pipeline"
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
OFFLINE_PROCESS_DIR = PACKAGE_ROOT / "offline_process"


def resolve_prompt_path(*parts: str) -> Path:
    return PROMPTS_DIR.joinpath(*parts)
