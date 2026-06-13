from __future__ import annotations

import hashlib
import json
import logging
import math
import os as _os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import sqlalchemy
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from evaluations.embedding.embeddings import BGEM3Embedder, HttpTextEmbedder

logger = logging.getLogger(__name__)

CANONICAL_VIEWS = ("motivation", "method", "experiment/result")
TABLE_COLUMN_ORDER = [
    ("motivation", 10),
    ("motivation", 20),
    ("method", 10),
    ("method", 20),
    ("experiment/result", 10),
    ("experiment/result", 20),
    ("overall", 10),
    ("overall", 20),
]
GROUP_METRIC_K = 10
FIXED_MODEL_ORDER = [
    "BM25",
    "Qwen3-Embed-8B",
    "BGE-M3",
    "Instructor-XL",
    "GritLM-7B",
    "SPECTER2",
    "SCiNCL",
]


def normalize_source_view(value: str) -> str:
    raw = str(value or "").strip().lower()
    normalized = raw.replace("_", "/")
    if normalized == "experiment":
        normalized = "experiment/result"
    if normalized not in CANONICAL_VIEWS:
        raise ValueError(f"Unsupported source_view: {value!r}")
    return normalized


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return text or "item"


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    query_text: str
    target_paper_id: str
    source_view: str


@dataclass(frozen=True)
class CorpusRecord:
    corpus_paper_id: str
    title: str
    abstract: str
    long_text: Optional[str] = None


@dataclass(frozen=True)
class SearchResult:
    query_id: str
    ranked_paper_ids: list[str]


class DatasetConfig(BaseModel):
    queries_path: str = "data/queries.jsonl"
    corpus_path: str = "data/corpus.jsonl"
    query_id_field: str = "query_id"
    query_text_field: str = "query_text"
    target_paper_id_field: str = "target_paper_id"
    source_view_field: str = "source_view"
    corpus_paper_id_field: str = "paper_id"
    title_field: str = "title"
    abstract_field: str = "abstract"
    long_text_field: Optional[str] = None


class RetrieverConfig(BaseModel):
    name: str
    type: str
    model_id_or_path: str = ""
    device: Optional[str] = None
    service_url: Optional[str] = None
    cache_dir: str = ""
    # Pyserini BM25
    index_dir: Optional[str] = None
    # PostgreSQL dense embedding
    pg_model_name: Optional[str] = None
    query_service_url: Optional[str] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        if value not in {"bm25", "dense_embedding", "pyserini_bm25", "postgres_embedding"}:
            raise ValueError(f"Unsupported retriever type: {value}")
        return value


class EvaluationConfig(BaseModel):
    dataset: DatasetConfig
    output_dir: str = "outputs/per_view_retrieval_eval"
    top_k: list[int] = Field(default_factory=lambda: [10, 20])
    bootstrap_samples: int = 1000
    random_seed: int = 13
    model_lineup: list[RetrieverConfig]

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, values: list[int]) -> list[int]:
        normalized = sorted({int(v) for v in values})
        if not normalized or any(v <= 0 for v in normalized):
            raise ValueError("top_k must contain positive integers.")
        return normalized

    @model_validator(mode="after")
    def validate_model_lineup(self) -> "EvaluationConfig":
        lineup_names = [model.name for model in self.model_lineup]
        for name in lineup_names:
            if name not in FIXED_MODEL_ORDER:
                raise ValueError(
                    f"Model '{name}' is not in the supported model list: {FIXED_MODEL_ORDER}"
                )
        return self


class SentenceTransformerLikeEmbedder:
    def __init__(self, model_id_or_path: str, device: Optional[str] = None) -> None:
        self.model_id_or_path = model_id_or_path
        self.device = device
        self._model = None
        self._tokenizer = None
        self._hf_model = None

    def _load_sentence_transformer(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {"trust_remote_code": True}
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_id_or_path, **kwargs)
        return self._model

    def _load_hf(self):
        if self._tokenizer is None or self._hf_model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id_or_path,
                trust_remote_code=True,
            )
            self._hf_model = AutoModel.from_pretrained(
                self.model_id_or_path,
                trust_remote_code=True,
            )
            if self.device:
                self._hf_model.to(torch.device(self.device))
            self._hf_model.eval()
        return self._tokenizer, self._hf_model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            model = self._load_sentence_transformer()
            vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return [list(map(float, vector)) for vector in vectors]
        except Exception as exc:
            logger.warning(
                "SentenceTransformer load/encode failed for %s, falling back to AutoModel: %s",
                self.model_id_or_path,
                exc,
            )
        import torch

        tokenizer, model = self._load_hf()
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=4096,
        )
        if self.device:
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            masked = token_embeddings * attention_mask
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().numpy().astype(float).tolist()


class BaseRetriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config

    def validate(self) -> None:
        raise NotImplementedError

    def build_index(self, corpus: list[CorpusRecord]) -> None:
        raise NotImplementedError

    def search(self, queries: list[QueryRecord], top_k: int) -> list[SearchResult]:
        raise NotImplementedError


class InProcessBM25Okapi:
    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = float(k1)
        self.b = float(b)
        self.corpus_size = len(tokenized_corpus)
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / max(1, self.corpus_size)
        self.doc_freqs: list[Counter[str]] = []
        self.idf: dict[str, float] = {}
        nd: Counter[str] = Counter()
        for doc in tokenized_corpus:
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            nd.update(frequencies.keys())
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=float)
        for token in query_tokens:
            idf = self.idf.get(token)
            if idf is None:
                continue
            for idx, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(token, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / max(self.avgdl, 1e-9))
                scores[idx] += idf * (tf * (self.k1 + 1) / denom)
        return scores


class BM25Retriever(BaseRetriever):
    def __init__(self, config: RetrieverConfig):
        super().__init__(config)
        self._corpus: list[CorpusRecord] = []
        self._tokenized_corpus: list[list[str]] = []
        self._paper_ids: list[str] = []
        self._bm25 = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+", str(text).lower())

    def validate(self) -> None:
        return None

    def build_index(self, corpus: list[CorpusRecord]) -> None:
        self._corpus = corpus
        self._paper_ids = [record.corpus_paper_id for record in corpus]
        corpus_texts = [build_corpus_text(record) for record in corpus]
        self._tokenized_corpus = [self._tokenize(text) for text in corpus_texts]
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._tokenized_corpus)
        except Exception:
            self._bm25 = InProcessBM25Okapi(self._tokenized_corpus)

    def search(self, queries: list[QueryRecord], top_k: int) -> list[SearchResult]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index has not been built.")
        results: list[SearchResult] = []
        for query in queries:
            query_tokens = self._tokenize(query.query_text)
            scores = np.asarray(self._bm25.get_scores(query_tokens), dtype=float)
            top_indices = np.argsort(-scores)[:top_k]
            ranked = [self._paper_ids[idx] for idx in top_indices.tolist()]
            results.append(SearchResult(query_id=query.query_id, ranked_paper_ids=ranked))
        return results


class DenseEmbeddingRetriever(BaseRetriever):
    def __init__(self, config: RetrieverConfig):
        super().__init__(config)
        self._embedder = None
        self._corpus: list[CorpusRecord] = []
        self._paper_ids: list[str] = []
        self._corpus_embeddings: Optional[np.ndarray] = None
        self._cache_dir = Path(config.cache_dir)

    def _build_embedder(self):
        if self._embedder is not None:
            return self._embedder
        if self.config.name == "BGE-M3":
            if self.config.service_url:
                self._embedder = HttpTextEmbedder(self.config.service_url)
            else:
                self._embedder = BGEM3Embedder(
                    model_path=self.config.model_id_or_path,
                    device=self.config.device or "cpu",
                )
        else:
            self._embedder = SentenceTransformerLikeEmbedder(
                model_id_or_path=self.config.model_id_or_path,
                device=self.config.device,
            )
        return self._embedder

    def validate(self) -> None:
        if self.config.name == "BM25":
            return
        if self.config.name == "BGE-M3":
            if self.config.service_url:
                return
            path = Path(self.config.model_id_or_path)
            if not path.exists():
                raise FileNotFoundError(f"BGE-M3 model path does not exist: {path}")
            return
        self._build_embedder()
        self._embedder.embed_texts(["validation probe"])

    def _corpus_cache_paths(self) -> tuple[Path, Path]:
        return self._cache_dir / "corpus_embeddings.npy", self._cache_dir / "corpus_meta.json"

    def _query_cache_paths(self) -> tuple[Path, Path]:
        return self._cache_dir / "query_embeddings.npy", self._cache_dir / "query_meta.json"

    def build_index(self, corpus: list[CorpusRecord]) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._corpus = corpus
        self._paper_ids = [record.corpus_paper_id for record in corpus]
        corpus_path, meta_path = self._corpus_cache_paths()
        expected_meta = {
            "paper_ids": self._paper_ids,
            "model": self.config.model_id_or_path,
            "service_url": self.config.service_url,
        }
        if corpus_path.exists() and meta_path.exists():
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if cached_meta == expected_meta:
                self._corpus_embeddings = np.load(corpus_path)
                return
        texts = [build_corpus_text(record) for record in corpus]
        embedder = self._build_embedder()
        embeddings = np.asarray(embedder.embed_texts(texts), dtype=np.float32)
        embeddings = l2_normalize(embeddings)
        np.save(corpus_path, embeddings)
        meta_path.write_text(json.dumps(expected_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        self._corpus_embeddings = embeddings

    def _embed_queries(self, queries: list[QueryRecord]) -> np.ndarray:
        query_ids = [query.query_id for query in queries]
        query_path, meta_path = self._query_cache_paths()
        expected_meta = {
            "query_ids": query_ids,
            "model": self.config.model_id_or_path,
            "service_url": self.config.service_url,
        }
        if query_path.exists() and meta_path.exists():
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if cached_meta == expected_meta:
                return np.load(query_path)
        embedder = self._build_embedder()
        embeddings = np.asarray(embedder.embed_texts([query.query_text for query in queries]), dtype=np.float32)
        embeddings = l2_normalize(embeddings)
        np.save(query_path, embeddings)
        meta_path.write_text(json.dumps(expected_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return embeddings

    def search(self, queries: list[QueryRecord], top_k: int) -> list[SearchResult]:
        if self._corpus_embeddings is None:
            raise RuntimeError("Dense index has not been built.")
        query_embeddings = self._embed_queries(queries)
        scores = query_embeddings @ self._corpus_embeddings.T
        results: list[SearchResult] = []
        for index, query in enumerate(queries):
            top_indices = np.argsort(-scores[index])[:top_k]
            ranked = [self._paper_ids[idx] for idx in top_indices.tolist()]
            results.append(SearchResult(query_id=query.query_id, ranked_paper_ids=ranked))
        return results


class PyseriniBM25Retriever(BaseRetriever):
    """BM25 retriever backed by a pre-built Pyserini/Lucene index."""

    def __init__(self, config: RetrieverConfig):
        super().__init__(config)
        self._searcher = None
        self._index_dir = str(config.index_dir or "")
        # pyserini needs these env vars at import time
        _os.environ.setdefault("JAVA_HOME", "/data3/weiyiyang/jdk21")
        _os.environ.setdefault("OPENAI_API_KEY", "dummy-for-bm25")

    def _load_searcher(self):
        from pyserini.search.lucene import LuceneSearcher
        self._searcher = LuceneSearcher(self._index_dir)

    def validate(self) -> None:
        self._load_searcher()
        _ = self._searcher.num_docs

    def build_index(self, corpus: list[CorpusRecord]) -> None:
        if self._searcher is None:
            self._load_searcher()

    def search(self, queries: list[QueryRecord], top_k: int) -> list[SearchResult]:
        if self._searcher is None:
            raise RuntimeError("Pyserini BM25 searcher not initialized.")
        results: list[SearchResult] = []
        for query in queries:
            hits = self._searcher.search(query.query_text, k=top_k)
            ranked = [hit.docid for hit in hits]
            results.append(SearchResult(query_id=query.query_id, ranked_paper_ids=ranked))
        return results


class PostgresEmbeddingRetriever(BaseRetriever):
    """Dense retriever that reads corpus embeddings from PostgreSQL and embeds queries via HTTP."""

    def __init__(self, config: RetrieverConfig):
        super().__init__(config)
        self._pg_model_name = str(config.pg_model_name or "")
        self._query_embedder = HttpTextEmbedder(str(config.query_service_url or ""))
        self._paper_ids: list[str] = []
        self._corpus_embeddings: Optional[np.ndarray] = None

    def validate(self) -> None:
        # Verify PG connection and that embeddings exist
        engine = _pg_get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                sqlalchemy.text(
                    "SELECT COUNT(*) AS cnt FROM paper_text_embeddings "
                    "WHERE model_name = :model"
                ),
                {"model": self._pg_model_name},
            ).fetchone()
            if row is None or row.cnt == 0:
                raise RuntimeError(
                    f"No embeddings found for model '{self._pg_model_name}' "
                    f"in paper_text_embeddings"
                )
            logger.info("PostgresEmbeddingRetriever[%s]: %d corpus embeddings found.",
                         self._pg_model_name, row.cnt)

    def build_index(self, corpus: list[CorpusRecord]) -> None:
        self._paper_ids = [r.corpus_paper_id for r in corpus]
        engine = _pg_get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                sqlalchemy.text(
                    "SELECT paper_id, embedding FROM paper_text_embeddings "
                    "WHERE model_name = :model"
                ),
                {"model": self._pg_model_name},
            ).fetchall()
        # Build ordered array matching corpus paper_ids
        emb_by_paper: dict[str, np.ndarray] = {}
        for row in rows:
            emb_val = row.embedding
            if emb_val is None:
                continue
            if isinstance(emb_val, str):
                values = [float(v) for v in emb_val.strip("[]").split(",")]
            else:
                values = [float(v) for v in emb_val]
            emb_by_paper[row.paper_id] = np.asarray(values, dtype=np.float32)
        emb_list = []
        for pid in self._paper_ids:
            vec = emb_by_paper.get(pid)
            if vec is not None:
                emb_list.append(vec)
            else:
                # Shouldn't happen for a well-built index, but guard with zeros
                dim = len(next(iter(emb_by_paper.values()))) if emb_by_paper else 1
                emb_list.append(np.zeros(dim, dtype=np.float32))
        self._corpus_embeddings = np.stack(emb_list, axis=0)
        self._corpus_embeddings = l2_normalize(self._corpus_embeddings)
        logger.info("PostgresEmbeddingRetriever[%s]: loaded corpus %s",
                     self._pg_model_name, self._corpus_embeddings.shape)

    def search(self, queries: list[QueryRecord], top_k: int) -> list[SearchResult]:
        if self._corpus_embeddings is None:
            raise RuntimeError("Corpus embeddings not loaded.")
        query_texts = [q.query_text for q in queries]
        query_embeddings = self._query_embedder.embed_texts(query_texts)
        query_embeddings = l2_normalize(np.asarray(query_embeddings, dtype=np.float32))
        scores = query_embeddings @ self._corpus_embeddings.T
        results: list[SearchResult] = []
        for idx, query in enumerate(queries):
            top_indices = np.argsort(-scores[idx])[:top_k]
            ranked = [self._paper_ids[i] for i in top_indices.tolist()]
            results.append(SearchResult(query_id=query.query_id, ranked_paper_ids=ranked))
        return results


# Shared PG engine (lazy-init)
_pg_engine = None


def _pg_get_engine():
    global _pg_engine
    if _pg_engine is None:
        db_url = (
            _os.environ.get("GOLDEN_EMBEDDING_DB_URL")
            or _os.environ.get("SCIFULL_GOLDEN_EMBEDDING_DB_URL")
            or _os.environ.get("DATABASE_URL")
            or "postgresql+psycopg://scifull:westlakenlp@localhost:5433/scifullmmbench"
        )
        _pg_engine = sqlalchemy.create_engine(db_url, pool_pre_ping=True)
    return _pg_engine


def load_config(path: Path) -> EvaluationConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return EvaluationConfig.model_validate(data)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_queries(config: DatasetConfig) -> list[QueryRecord]:
    path = Path(config.queries_path).expanduser().resolve()
    rows = read_jsonl(path)
    queries: list[QueryRecord] = []
    for row in rows:
        query_text = str(row[config.query_text_field]).strip()
        target_paper_id = str(row[config.target_paper_id_field]).strip()
        source_view = normalize_source_view(str(row[config.source_view_field]))
        query_id = str(row.get(config.query_id_field) or "")
        if not query_id:
            raw = f"{query_text}\n{target_paper_id}\n{source_view}"
            query_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        queries.append(
            QueryRecord(
                query_id=query_id,
                query_text=query_text,
                target_paper_id=target_paper_id,
                source_view=source_view,
            )
        )
    return queries


def load_corpus(config: DatasetConfig) -> list[CorpusRecord]:
    path = Path(config.corpus_path).expanduser().resolve()
    rows = read_jsonl(path)
    corpus: list[CorpusRecord] = []
    for row in rows:
        corpus.append(
            CorpusRecord(
                corpus_paper_id=str(row[config.corpus_paper_id_field]).strip(),
                title=str(row.get(config.title_field) or "").strip(),
                abstract=str(row.get(config.abstract_field) or "").strip(),
                long_text=(
                    str(row.get(config.long_text_field) or "").strip()
                    if config.long_text_field and row.get(config.long_text_field) is not None
                    else None
                ),
            )
        )
    return corpus


def build_corpus_text(record: CorpusRecord) -> str:
    if record.long_text:
        return record.long_text
    return "\n".join(part for part in [record.title, record.abstract] if part).strip()


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return array / norms


def build_retriever(config: RetrieverConfig) -> BaseRetriever:
    if config.type == "bm25":
        return BM25Retriever(config)
    if config.type == "pyserini_bm25":
        return PyseriniBM25Retriever(config)
    if config.type == "postgres_embedding":
        return PostgresEmbeddingRetriever(config)
    return DenseEmbeddingRetriever(config)


def bootstrap_confidence_interval(values: list[int], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    draws = []
    for _ in range(samples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        draws.append(float(sample.mean()))
    lower = float(np.percentile(draws, 2.5))
    upper = float(np.percentile(draws, 97.5))
    return lower, upper


def kendall_tau_for_rankings(rank_a: dict[str, int], rank_b: dict[str, int]) -> float:
    views = list(CANONICAL_VIEWS)
    concordant = 0
    discordant = 0
    for i, left in enumerate(views):
        for right in views[i + 1 :]:
            a = rank_a[left] - rank_a[right]
            b = rank_b[left] - rank_b[right]
            if a == 0 or b == 0:
                continue
            if a * b > 0:
                concordant += 1
            elif a * b < 0:
                discordant += 1
    denom = concordant + discordant
    return 0.0 if denom == 0 else (concordant - discordant) / denom


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _group_queries_by_paper(queries: list[QueryRecord]) -> dict[str, list[QueryRecord]]:
    grouped: dict[str, list[QueryRecord]] = defaultdict(list)
    for query in queries:
        grouped[query.target_paper_id].append(query)
    return grouped


def _group_metrics_for_model(
    *,
    queries_by_paper: dict[str, list[QueryRecord]],
    query_rows: list[dict[str, Any]],
    top_k: list[int],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, float]]]:
    row_lookup = {str(row["query_id"]): row for row in query_rows}
    per_paper_rows: list[dict[str, Any]] = []
    summary: dict[int, dict[str, float]] = {}
    corpus_fallback_rank = max(top_k) + 1

    for paper_id, paper_queries in sorted(queries_by_paper.items()):
        available_views = sorted({query.source_view for query in paper_queries})
        view_to_queries: dict[str, list[QueryRecord]] = defaultdict(list)
        for query in paper_queries:
            view_to_queries[query.source_view].append(query)

        paper_row: dict[str, Any] = {
            "target_paper_id": paper_id,
            "available_views": available_views,
            "query_count": len(paper_queries),
        }
        best_rank_by_view: dict[str, Optional[int]] = {}
        for view, view_queries in view_to_queries.items():
            ranks = [
                row_lookup[query.query_id].get("target_rank")
                for query in view_queries
                if query.query_id in row_lookup
            ]
            valid_ranks = [int(rank) for rank in ranks if rank is not None]
            best_rank_by_view[view] = min(valid_ranks) if valid_ranks else None
        paper_row["best_rank_by_view"] = best_rank_by_view

        for k in top_k:
            successful_views = {
                view
                for view, rank in best_rank_by_view.items()
                if rank is not None and int(rank) <= k
            }
            any_view = int(bool(successful_views))
            all_view = int(len(successful_views) == len(available_views) and bool(available_views))
            view_coverage = float(len(successful_views) / len(available_views)) if available_views else 0.0
            ranks_for_variance = [
                best_rank_by_view.get(view) if best_rank_by_view.get(view) is not None and int(best_rank_by_view[view]) <= k else k + 1
                for view in available_views
            ]
            rank_variance = float(np.var(ranks_for_variance)) if len(ranks_for_variance) > 1 else 0.0
            paper_row[f"any_view@{k}"] = any_view
            paper_row[f"all_view@{k}"] = all_view
            paper_row[f"view_coverage@{k}"] = view_coverage
            paper_row[f"rank_variance@{k}"] = rank_variance
        per_paper_rows.append(paper_row)

    for k in top_k:
        any_values = [float(row[f"any_view@{k}"]) for row in per_paper_rows]
        all_values = [float(row[f"all_view@{k}"]) for row in per_paper_rows]
        coverage_values = [float(row[f"view_coverage@{k}"]) for row in per_paper_rows]
        variance_values = [float(row[f"rank_variance@{k}"]) for row in per_paper_rows]
        summary[k] = {
            "AnyView": float(np.mean(any_values)) if any_values else 0.0,
            "AllView": float(np.mean(all_values)) if all_values else 0.0,
            "ViewCoverage": float(np.mean(coverage_values)) if coverage_values else 0.0,
            "RankVariance": float(np.mean(variance_values)) if variance_values else 0.0,
        }
    return per_paper_rows, summary


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def create_heatmap(path: Path, matrix: np.ndarray, row_labels: list[str], col_labels: list[str]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=25, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Recall by Model and View")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def create_vote_bar_chart(path: Path, vote_counts: dict[str, int]) -> None:
    import matplotlib.pyplot as plt

    labels = list(CANONICAL_VIEWS)
    values = [vote_counts.get(label, 0) for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#3d6fb6")
    ax.set_ylabel("Vote count")
    ax.set_title("Hardest-View Vote Counts")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_per_view_retrieval_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(config.dataset)
    corpus = load_corpus(config.dataset)
    query_lookup = {query.query_id: query for query in queries}

    max_k = max(config.top_k)
    per_query_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    per_paper_metric_rows: list[dict[str, Any]] = []
    summary_by_model: dict[str, dict[tuple[str, int], float]] = defaultdict(dict)
    ci_by_model: dict[str, dict[tuple[str, int], tuple[float, float]]] = defaultdict(dict)
    group_summary_by_model: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    queries_by_paper = _group_queries_by_paper(queries)

    for model_config in config.model_lineup:
        retriever = build_retriever(model_config)
        logger.info("Validating retriever %s", model_config.name)
        retriever.validate()
        logger.info("Building index for %s", model_config.name)
        retriever.build_index(corpus)
        logger.info("Searching for %s", model_config.name)
        results = retriever.search(queries, max_k)

        grouped_hits: dict[tuple[str, int], list[int]] = defaultdict(list)
        overall_hits: dict[int, list[int]] = defaultdict(list)
        for result in results:
            query = query_lookup[result.query_id]
            hit_positions = {paper_id: idx + 1 for idx, paper_id in enumerate(result.ranked_paper_ids)}
            target_rank = hit_positions.get(query.target_paper_id)
            row = {
                "model": model_config.name,
                "query_id": query.query_id,
                "query_text": query.query_text,
                "target_paper_id": query.target_paper_id,
                "source_view": query.source_view,
                "ranked_paper_ids": result.ranked_paper_ids,
                "target_rank": target_rank,
            }
            for k in config.top_k:
                hit = int(target_rank is not None and target_rank <= k)
                row[f"hit@{k}"] = hit
                grouped_hits[(query.source_view, k)].append(hit)
                overall_hits[k].append(hit)
            per_query_rows.append(row)

        for view in CANONICAL_VIEWS:
            view_count = sum(1 for query in queries if query.source_view == view)
            for k in config.top_k:
                values = grouped_hits[(view, k)]
                recall = float(np.mean(values)) if values else 0.0
                ci = bootstrap_confidence_interval(
                    values,
                    samples=config.bootstrap_samples,
                    seed=config.random_seed + hash((model_config.name, view, k)) % 100000,
                )
                summary_by_model[model_config.name][(view, k)] = recall
                ci_by_model[model_config.name][(view, k)] = ci
                metric_rows.append(
                    {
                        "model": model_config.name,
                        "view": view,
                        "k": k,
                        "query_count": view_count,
                        "hit_count": int(sum(values)),
                        "recall": recall,
                        "ci_low": ci[0],
                        "ci_high": ci[1],
                    }
                )
        for k in config.top_k:
            values = overall_hits[k]
            overall_recall = float(np.mean(values)) if values else 0.0
            summary_by_model[model_config.name][("overall", k)] = overall_recall
            metric_rows.append(
                {
                    "model": model_config.name,
                    "view": "overall",
                    "k": k,
                    "query_count": len(values),
                    "hit_count": int(sum(values)),
                    "recall": float(np.mean(values)) if values else 0.0,
                    "ci_low": bootstrap_confidence_interval(values, config.bootstrap_samples, config.random_seed)[0]
                    if values
                    else 0.0,
                    "ci_high": bootstrap_confidence_interval(values, config.bootstrap_samples, config.random_seed)[1]
                    if values
                    else 0.0,
                }
            )

        model_query_rows = [row for row in per_query_rows if row["model"] == model_config.name]
        paper_rows, group_summary = _group_metrics_for_model(
            queries_by_paper=queries_by_paper,
            query_rows=model_query_rows,
            top_k=config.top_k,
        )
        group_summary_by_model[model_config.name] = group_summary
        for row in paper_rows:
            per_paper_metric_rows.append({"model": model_config.name, **row})

    write_jsonl(output_dir / "per_query_results.jsonl", per_query_rows)
    write_jsonl(output_dir / "per_paper_group_metrics.jsonl", per_paper_metric_rows)

    import csv

    metric_path = output_dir / "per_model_per_view_metrics.csv"
    with metric_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "view", "k", "query_count", "hit_count", "recall", "ci_low", "ci_high"],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    cross_view_path = output_dir / "cross_view_metrics.csv"
    with cross_view_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "k", "AnyView", "AllView", "ViewCoverage", "RankVariance"],
        )
        writer.writeheader()
        for model_name in FIXED_MODEL_ORDER:
            for k in config.top_k:
                writer.writerow(
                    {
                        "model": model_name,
                        "k": k,
                        **group_summary_by_model[model_name][k],
                    }
                )

    def _view_label(view: str) -> str:
        if view == "experiment/result":
            return "Experiment/Result"
        if view == "overall":
            return "Overall"
        return view.title()

    group_metric_headers: list[str] = []
    for k in sorted(config.top_k):
        for metric in ("AnyView", "AllView", "ViewCoverage", "RankVariance"):
            group_metric_headers.append(f"{metric}@{k}")

    main_table_headers = ["Model"] + group_metric_headers + [
        f"{_view_label(view)} Recall@{k}" for view, k in TABLE_COLUMN_ORDER
    ]
    main_table_rows: list[list[str]] = []
    main_table_csv_rows: list[dict[str, Any]] = []
    heatmap_matrix = []
    for model_name in FIXED_MODEL_ORDER:
        row: list[str] = [model_name]
        csv_row: dict[str, Any] = {"Model": model_name}
        for k_val in sorted(config.top_k):
            gs = group_summary_by_model[model_name][k_val]
            for metric in ("AnyView", "AllView", "ViewCoverage", "RankVariance"):
                val = gs[metric]
                row.append(f"{val:.3f}")
                csv_row[f"{metric}@{k_val}"] = val
        heatmap_row = []
        for view, k_val in TABLE_COLUMN_ORDER:
            recall = summary_by_model[model_name].get((view, k_val), 0.0)
            key = f"{_view_label(view)} Recall@{k_val}"
            csv_row[key] = recall
            row.append(f"{recall:.3f}")
            if k_val == 10 and view != "overall":
                heatmap_row.append(recall)
        main_table_rows.append(row)
        main_table_csv_rows.append(csv_row)
        heatmap_matrix.append(heatmap_row)

    main_table_csv = output_dir / "main_table.csv"
    with main_table_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=main_table_headers)
        writer.writeheader()
        writer.writerows(main_table_csv_rows)

    (output_dir / "main_table.md").write_text(
        markdown_table(main_table_headers, main_table_rows),
        encoding="utf-8",
    )

    hardest_vote_counts: dict[int, Counter[str]] = {k: Counter() for k in config.top_k}
    rankings_by_k: dict[int, dict[str, dict[str, int]]] = defaultdict(dict)
    interaction_lines = ["# Model-by-View Interaction Report", ""]
    difficulty_lines = ["# View Difficulty Summary", ""]
    difficulty_lines.append(f"- Group metrics at @{GROUP_METRIC_K} use view-level success: a view counts as retrieved if at least one query from that view hits the target paper within top-{GROUP_METRIC_K}.")
    difficulty_lines.append("")

    for k in config.top_k:
        difficulty_lines.append(f"## Recall@{k}")
        difficulty_lines.append("")
        for model_name in FIXED_MODEL_ORDER:
            recalls = {view: summary_by_model[model_name][(view, k)] for view in CANONICAL_VIEWS}
            min_recall = min(recalls.values())
            hardest = [view for view, recall in recalls.items() if recall == min_recall]
            for view in hardest:
                hardest_vote_counts[k][view] += 1
            ordered = sorted(recalls.items(), key=lambda item: (-item[1], item[0]))
            rankings_by_k[k][model_name] = {view: idx + 1 for idx, (view, _) in enumerate(ordered)}
            difficulty_lines.append(
                f"- `{model_name}`: hardest={', '.join(hardest)} | "
                + ", ".join(f"{view}={recalls[view]:.3f}" for view in CANONICAL_VIEWS)
            )
        difficulty_lines.append("")

    for k in config.top_k:
        vote_counts = hardest_vote_counts[k]
        difficulty_lines.append(f"### Hardest-view votes @ {k}")
        for view in CANONICAL_VIEWS:
            difficulty_lines.append(f"- `{view}`: {vote_counts.get(view, 0)}")
        taus = []
        model_names = list(FIXED_MODEL_ORDER)
        for i, left_model in enumerate(model_names):
            for right_model in model_names[i + 1 :]:
                taus.append(
                    kendall_tau_for_rankings(rankings_by_k[k][left_model], rankings_by_k[k][right_model])
                )
        avg_tau = float(np.mean(taus)) if taus else 0.0
        difficulty_lines.append(f"- average Kendall-style concordance @ {k}: {avg_tau:.3f}")
        difficulty_lines.append("")

    for model_name in FIXED_MODEL_ORDER:
        interaction_lines.append(f"## {model_name}")
        recalls = {view: summary_by_model[model_name][(view, 10)] for view in CANONICAL_VIEWS}
        mean_recall = float(np.mean(list(recalls.values())))
        group_metrics = group_summary_by_model[model_name][GROUP_METRIC_K]
        interaction_lines.append(
            "- cross-view metrics: "
            f"AnyView@{GROUP_METRIC_K}={group_metrics['AnyView']:.3f}, "
            f"AllView@{GROUP_METRIC_K}={group_metrics['AllView']:.3f}, "
            f"ViewCoverage@{GROUP_METRIC_K}={group_metrics['ViewCoverage']:.3f}, "
            f"RankVariance@{GROUP_METRIC_K}={group_metrics['RankVariance']:.3f}"
        )
        for view in CANONICAL_VIEWS:
            delta = recalls[view] - mean_recall
            interaction_lines.append(
                f"- `{view}`: recall@10={recalls[view]:.3f}, deviation_from_model_mean={delta:+.3f}"
            )
        interaction_lines.append("")

    (output_dir / "view_difficulty_summary.md").write_text("\n".join(difficulty_lines), encoding="utf-8")
    (output_dir / "interaction_summary.md").write_text("\n".join(interaction_lines), encoding="utf-8")

    create_heatmap(
        output_dir / "recall_heatmap.png",
        np.asarray(heatmap_matrix, dtype=float),
        FIXED_MODEL_ORDER,
        ["Motivation", "Method", "Experiment/Result"],
    )
    create_vote_bar_chart(
        output_dir / "hardest_view_votes.png",
        dict(hardest_vote_counts[10]),
    )

    return {
        "output_dir": str(output_dir),
        "query_count": len(queries),
        "corpus_count": len(corpus),
        "models": FIXED_MODEL_ORDER,
        "group_metric_k": GROUP_METRIC_K,
    }
