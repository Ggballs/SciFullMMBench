import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetAdapter(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """Load dataset from path and return DataFrame with 'query' column."""
        pass

    @abstractmethod
    def filter_human(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for human-written queries only."""
        pass

    def _safe_str(self, value: Any) -> str:
        """Convert value to string safely."""
        if value is None:
            return ""
        return str(value).strip()


class LitSearchLoader(DatasetAdapter):
    """Loader for LitSearch dataset format.

    Expected columns (from HuggingFace princeton-nlp/LitSearch):
    - query: the search query text
    - query_set: category区分human vs synthetic:
        - human: 'manual_acl', 'manual_iclr' (246 queries)
        - synthetic: 'inline_acl', 'inline_nonacl' (351 queries)
    - specificity: int (0 or 1)
    - quality: int (1 or 2)
    - corpusids: list of related paper IDs

    The query_set column is used to distinguish human-written queries
    (manual_acl, manual_iclr) from GPT-4 generated queries (inline_acl, inline_nonacl).
    """

    DEFAULT_QUERY_COLUMN = "query"
    DEFAULT_HUMAN_FLAG_COLUMN = "query_set"
    HUMAN_QUERY_SETS = {"manual_acl", "manual_iclr"}
    SYNTHETIC_QUERY_SETS = {"inline_acl", "inline_nonacl"}

    def __init__(
        self,
        query_column: str = DEFAULT_QUERY_COLUMN,
        human_flag_column: Optional[str] = None,
        human_flag_value: Optional[str] = None,
    ):
        self.query_column = query_column
        self.human_flag_column = human_flag_column or self.DEFAULT_HUMAN_FLAG_COLUMN
        self.human_flag_value = human_flag_value
        self._use_hf_loader = False

    def load(self, path: str) -> pd.DataFrame:
        """Load LitSearch dataset from file or HuggingFace."""
        path_obj = Path(path)

        if path_obj.suffix.lower() in (".json", ".jsonl", ".csv", ".tsv"):
            if not path_obj.exists():
                raise FileNotFoundError(f"Dataset path not found: {path}")
            return self._load_from_file(path_obj)
        elif self._is_huggingface_path(path):
            self._use_hf_loader = True
            return self._load_from_huggingface(path)
        else:
            suffix = path_obj.suffix.lower() if path_obj.suffix else "none"
            raise ValueError(f"Unsupported file format or path: {suffix}")

    def _is_huggingface_path(self, path: str) -> bool:
        """Check if path is a HuggingFace dataset identifier."""
        return "/" in path and not Path(path).exists()

    def _load_from_huggingface(self, dataset_id: str) -> pd.DataFrame:
        """Load dataset directly from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets package required for HuggingFace loading: pip install datasets")

        ds = load_dataset("princeton-nlp/LitSearch", "query", split="full")
        df = pd.DataFrame(ds)
        logger.info(f"Loaded LitSearch from HuggingFace: {len(df)} rows")
        return self._normalize_columns(df)

    def _load_from_file(self, path: Path) -> pd.DataFrame:
        """Load from local file."""
        suffix = path.suffix.lower()

        if suffix == ".json":
            return self._load_json(path)
        elif suffix == ".jsonl":
            return self._load_jsonl(path)
        elif suffix in (".csv", ".tsv"):
            sep = "," if suffix == ".csv" else "\t"
            return self._load_csv(path, sep)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_json(self, path: Path) -> pd.DataFrame:
        """Load JSON file, handling various structures."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if "train" in data and isinstance(data["train"], list):
                df = pd.DataFrame(data["train"])
            elif "data" in data and isinstance(data["data"], list):
                df = pd.DataFrame(data["data"])
            else:
                df = pd.DataFrame([data])
        else:
            raise ValueError(f"Unexpected JSON structure in {path}")

        return self._normalize_columns(df)

    def _load_jsonl(self, path: Path) -> pd.DataFrame:
        """Load JSONL file line by line."""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_num} due to JSON error: {e}")
                    continue

        if not records:
            raise ValueError(f"No valid records found in {path}")

        return self._normalize_columns(pd.DataFrame(records))

    def _load_csv(self, path: Path, sep: str = ",") -> pd.DataFrame:
        """Load CSV/TSV file."""
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep=sep, encoding="latin-1")

        return self._normalize_columns(df)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns exist and standardize column names."""
        if self.query_column not in df.columns:
            available = list(df.columns)
            logger.warning(
                f"Query column '{self.query_column}' not found. Available: {available}"
            )
            if "query" in available:
                self.query_column = "query"
            elif "text" in available:
                self.query_column = "text"
            else:
                raise ValueError(
                    f"No suitable query column found. Available: {available}"
                )

        df = df.rename(columns={self.query_column: "query"})

        if self.human_flag_column in df.columns:
            df = df.rename(columns={self.human_flag_column: "human_flag"})

        df["query"] = df["query"].apply(self._safe_str)
        df = df[df["query"].str.len() > 0]

        return df.reset_index(drop=True)

    def filter_human(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for human-written queries based on query_set column.

        Human queries are those where query_set is 'manual_acl' or 'manual_iclr'.
        Synthetic queries are 'inline_acl' or 'inline_nonacl' (GPT-4 generated).
        """
        if "human_flag" not in df.columns:
            if self.human_flag_column in df.columns:
                df = df.rename(columns={self.human_flag_column: "human_flag"})
            else:
                logger.warning(
                    f"Human flag column '{self.human_flag_column}' not found. "
                    f"Returning all queries."
                )
                return df

        human_queries = df[df["human_flag"].isin(self.HUMAN_QUERY_SETS)].copy()

        total = len(df)
        human_count = len(human_queries)
        synthetic_count = total - human_count

        logger.info(
            f"Filtered {human_count}/{total} human queries "
            f"(excluded {synthetic_count} synthetic: inline_acl, inline_nonacl)"
        )

        return human_queries.reset_index(drop=True)


class PasaLoader(DatasetAdapter):
    """Loader for PASA/RealScholarQuery dataset format.

    Expected columns (from CarlanLark/pasa-dataset):
    - question: the search query text (NOT 'query')
    - answer: list of related paper titles
    - answer_arxiv_id: list of arxiv IDs
    - source_meta: metadata dict with published_time
    - qid: query ID

    Note: RealScholarQuery contains only 50 REAL human queries from AI researchers.
    AutoScholarQuery contains synthetic queries and should be excluded.
    """

    DEFAULT_QUERY_COLUMN = "question"
    DEFAULT_HUMAN_FLAG_COLUMN = "is_human"
    DEFAULT_HUMAN_FLAG_VALUE = True

    def __init__(
        self,
        query_column: str = DEFAULT_QUERY_COLUMN,
        human_flag_column: str = DEFAULT_HUMAN_FLAG_COLUMN,
        human_flag_value: Any = DEFAULT_HUMAN_FLAG_VALUE,
    ):
        self.query_column = query_column
        self.human_flag_column = human_flag_column
        self.human_flag_value = human_flag_value

    def load(self, path: str) -> pd.DataFrame:
        """Load PASA dataset from file or HuggingFace.

        Supports:
        - Local files (JSON, JSONL, CSV, TSV)
        - HuggingFace dataset path (e.g., 'CarlanLark/pasa-dataset/RealScholarQuery')
        """
        path_obj = Path(path)

        if path_obj.suffix.lower() in (".json", ".jsonl", ".csv", ".tsv"):
            if not path_obj.exists():
                raise FileNotFoundError(f"Dataset path not found: {path}")
            return self._load_from_file(path_obj)
        elif self._is_huggingface_path(path):
            return self._load_from_huggingface(path)
        else:
            suffix = path_obj.suffix.lower() if path_obj.suffix else "none"
            raise ValueError(f"Unsupported file format or path: {suffix}")

    def _is_huggingface_path(self, path: str) -> bool:
        """Check if path is a HuggingFace dataset identifier."""
        return "/" in path and not Path(path).exists()

    def _load_from_huggingface(self, dataset_id: str) -> pd.DataFrame:
        """Load dataset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets package required for HuggingFace loading: pip install datasets")

        parts = dataset_id.split("/")
        if len(parts) >= 2:
            repo_id = "/".join(parts[:-1])
            config_name = parts[-1]
        else:
            repo_id = dataset_id
            config_name = None

        if config_name:
            ds = load_dataset(repo_id, config_name, split="test")
        else:
            ds = load_dataset(repo_id, split="test")

        df = pd.DataFrame(ds)
        logger.info(f"Loaded PASA from HuggingFace: {len(df)} rows")
        return self._normalize_columns(df)

    def _load_from_file(self, path: Path) -> pd.DataFrame:
        """Load from local file."""
        suffix = path.suffix.lower()

        if suffix == ".json":
            return self._load_json(path)
        elif suffix == ".jsonl":
            return self._load_jsonl(path)
        elif suffix in (".csv", ".tsv"):
            sep = "," if suffix == ".csv" else "\t"
            return self._load_csv(path, sep)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_json(self, path: Path) -> pd.DataFrame:
        """Load JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if "train" in data and isinstance(data["train"], list):
                df = pd.DataFrame(data["train"])
            elif "data" in data and isinstance(data["data"], list):
                df = pd.DataFrame(data["data"])
            elif "queries" in data and isinstance(data["queries"], list):
                df = pd.DataFrame(data["queries"])
            else:
                df = pd.DataFrame([data])
        else:
            raise ValueError(f"Unexpected JSON structure in {path}")

        return self._normalize_columns(df)

    def _load_jsonl(self, path: Path) -> pd.DataFrame:
        """Load JSONL file."""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_num} due to JSON error: {e}")
                    continue

        if not records:
            raise ValueError(f"No valid records found in {path}")

        return self._normalize_columns(pd.DataFrame(records))

    def _load_csv(self, path: Path, sep: str = ",") -> pd.DataFrame:
        """Load CSV/TSV file."""
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep=sep, encoding="latin-1")

        return self._normalize_columns(df)

    def _normalize_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Ensure required columns exist and standardize column names."""
        available = list(df.columns)

        if self.query_column not in df.columns:
            logger.warning(
                f"Query column '{self.query_column}' not found. Available: {available}"
            )
            potential_mappings = {
                "query": ["query", "text", "q", "query_text", "question"],
                "is_human": ["is_human", "human", "is_human_written", "is_real"],
            }

            found_mapping = False
            for standard_name, alternatives in potential_mappings.items():
                for alt in alternatives:
                    if alt in available:
                        df = df.rename(columns={alt: standard_name})
                        self.query_column = standard_name
                        logger.info(f"Mapped column '{alt}' to '{standard_name}'")
                        found_mapping = True
                        break
                if found_mapping:
                    break

            if self.query_column not in df.columns:
                raise ValueError(
                    f"No suitable query column found. Available: {available}"
                )

        df = df.rename(columns={self.query_column: "query"})

        if self.human_flag_column in df.columns:
            df = df.rename(columns={self.human_flag_column: "human_flag"})
        elif "is_human" in df.columns:
            df = df.rename(columns={"is_human": "human_flag"})

        df["query"] = df["query"].apply(self._safe_str)
        df = df[df["query"].str.len() > 0]

        return df.reset_index(drop=True)

    def filter_human(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for human-written queries.

        RealScholarQuery contains only real human queries (no synthetic flag needed).
        AutoScholarQuery contains only synthetic queries (all filtered out).
        If no human flag column exists, assume all queries are human (for RealScholarQuery).
        """
        if "human_flag" not in df.columns:
            logger.info(
                f"No human flag column found. Assuming all {len(df)} queries are human "
                f"(RealScholarQuery style - all real queries)."
            )
            return df

        human_queries = df[df["human_flag"] == self.human_flag_value].copy()
        synthetic_count = len(df) - len(human_queries)
        logger.info(
            f"Filtered {len(human_queries)}/{len(df)} human queries "
            f"(excluded {synthetic_count} synthetic)"
        )

        return human_queries.reset_index(drop=True)


def create_loader(
    dataset_name: str,
    query_column: Optional[str] = None,
    human_flag_column: Optional[str] = None,
    human_flag_value: Any = None,
) -> DatasetAdapter:
    """Factory function to create appropriate dataset loader.

    Args:
        dataset_name: Name of dataset ('litsearch', 'pasa', 'realscholarquery')
        query_column: Optional custom query column name
        human_flag_column: Optional custom human flag column name
        human_flag_value: Optional custom human flag value

    Returns:
        Appropriate DatasetAdapter instance
    """
    dataset_name = dataset_name.lower().strip()

    if dataset_name in ("litsearch", "litsearch"):
        return LitSearchLoader(
            query_column=query_column or LitSearchLoader.DEFAULT_QUERY_COLUMN,
            human_flag_column=human_flag_column or LitSearchLoader.DEFAULT_HUMAN_FLAG_COLUMN,
            human_flag_value=human_flag_value or LitSearchLoader.DEFAULT_HUMAN_FLAG_VALUE,
        )
    elif dataset_name in ("pasa", "realscholarquery", "autoscholarquery"):
        return PasaLoader(
            query_column=query_column or PasaLoader.DEFAULT_QUERY_COLUMN,
            human_flag_column=human_flag_column or PasaLoader.DEFAULT_HUMAN_FLAG_COLUMN,
            human_flag_value=human_flag_value if human_flag_value is not None else PasaLoader.DEFAULT_HUMAN_FLAG_VALUE,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'litsearch' or 'pasa'.")
