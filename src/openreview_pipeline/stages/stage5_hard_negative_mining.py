import json
import hashlib
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from openreview_pipeline.llm import LLMBackend
from openreview_pipeline.schemas.schemas_queries import GeneratedQueriesDataset, RetrievalQuery
from openreview_pipeline.utils import load_json, load_prompt_template, save_json

logger = logging.getLogger(__name__)
PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-pro"

_DOCLING_MAX_CONCURRENT = max(1, int(os.environ.get("DOCLING_MAX_CONCURRENT", "4")))
_DOCLING_SEMAPHORE = threading.BoundedSemaphore(_DOCLING_MAX_CONCURRENT)
_DOCLING_TIMEOUT_SECONDS = int(os.environ.get("DOCLING_TIMEOUT_SECONDS", "300"))
_PDF_DOWNLOAD_TIMEOUT_SECONDS = int(os.environ.get("PDF_DOWNLOAD_TIMEOUT_SECONDS", "45"))
_PDF_DOWNLOAD_MAX_RETRIES = int(os.environ.get("PDF_DOWNLOAD_MAX_RETRIES", "4"))
_PDF_DOWNLOAD_RETRY_BACKOFF_SECONDS = float(os.environ.get("PDF_DOWNLOAD_RETRY_BACKOFF_SECONDS", "8"))
_PDF_DOWNLOAD_MAX_CONCURRENT = max(1, int(os.environ.get("PDF_DOWNLOAD_MAX_CONCURRENT", "3")))
_PDF_DOWNLOAD_SEMAPHORE = threading.BoundedSemaphore(_PDF_DOWNLOAD_MAX_CONCURRENT)
_ARXIV_TUNNEL_PORT = int(os.environ.get("ARXIV_TUNNEL_PORT", "0"))
_ARXIV_CONNECT_TO_ARGS: list[str] = []
if _ARXIV_TUNNEL_PORT > 0:
    for host in ("arxiv.org:443", "export.arxiv.org:443"):
        _ARXIV_CONNECT_TO_ARGS.extend(["--connect-to", f"{host}:127.0.0.1:{_ARXIV_TUNNEL_PORT}"])

StageCallback = Callable[[str, dict[str, Any]], None]


def _clean_api_tokens(raw_tokens: Any) -> List[str]:
    if raw_tokens is None:
        return []
    if isinstance(raw_tokens, str):
        raw_tokens = re.split(r"[,\s]+", raw_tokens)
    if not isinstance(raw_tokens, list):
        return []
    return [str(token).strip() for token in raw_tokens if str(token).strip()]


def _deepseek_tokens_from_env() -> List[str]:
    return (
        _clean_api_tokens(os.environ.get("SCIFULL_HARD_NEGATIVE_LLM_API_TOKENS"))
        or _clean_api_tokens(os.environ.get("DEEPSEEK_API_KEYS"))
        or _clean_api_tokens(os.environ.get("DEEPSEEK_API_KEY"))
    )


def _semantic_scholar_tokens_from_env() -> List[str]:
    return (
        _clean_api_tokens(os.environ.get("SCIFULL_SEMANTIC_SCHOLAR_API_KEYS"))
        or _clean_api_tokens(os.environ.get("SEMANTIC_SCHOLAR_API_KEYS"))
        or _clean_api_tokens(os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
    )


def _is_retryable_pdf_download_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
    if isinstance(exc, subprocess.TimeoutExpired):
        return True
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = (getattr(exc, "stderr", "") or "").lower()
        # curl exit code 6 = DNS failure, 7 = failed to connect, 28 = timeout
        # These are always retryable.
        if exc.returncode in {6, 7, 28}:
            return True
        # curl exit code 22 = HTTP error (--fail). Parse stderr for status code.
        if " 403 " in stderr or " 404 " in stderr or " 410 " in stderr:
            return False
        # Any other curl HTTP error (429, 500, 502, 503, 504, etc.) is retryable.
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
                result.returncode, cmd,
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


def resolve_hard_negative_llm_settings(
    config: dict[str, Any],
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, object]:
    stages_config = config.get("stages", {})
    if not isinstance(stages_config, dict):
        stages_config = {}
    stage_config = stages_config.get("hard_negative_mining", {})
    if not isinstance(stage_config, dict):
        stage_config = {}
    stage_llm_config = stage_config.get("llm", {})
    if not isinstance(stage_llm_config, dict):
        stage_llm_config = {}

    global_llm_config = config.get("llm", {})
    if not isinstance(global_llm_config, dict):
        global_llm_config = {}

    api_tokens = _clean_api_tokens(stage_llm_config.get("api_tokens")) or _deepseek_tokens_from_env()
    if not api_tokens:
        raise ValueError(
            "Missing DeepSeek keys for hard-negative mining. Configure "
            "stages.hard_negative_mining.llm.api_tokens, DEEPSEEK_API_KEYS, "
            "DEEPSEEK_API_KEY, or SCIFULL_HARD_NEGATIVE_LLM_API_TOKENS."
        )

    settings = {
        "base_url": stage_llm_config.get("base_url") or DEEPSEEK_BASE_URL,
        "model": stage_llm_config.get("model") or DEEPSEEK_MODEL,
        "api_tokens": api_tokens,
        "per_key_request_interval_seconds": stage_llm_config.get(
            "per_key_request_interval_seconds",
            global_llm_config.get("per_key_request_interval_seconds", 0.0),
        ),
        "per_key_max_concurrent_requests": stage_llm_config.get(
            "per_key_max_concurrent_requests",
            global_llm_config.get("per_key_max_concurrent_requests", 1),
        ),
        "max_retries": stage_llm_config.get("max_retries", global_llm_config.get("max_retries", 3)),
        "retry_backoff_seconds": stage_llm_config.get(
            "retry_backoff_seconds",
            global_llm_config.get("retry_backoff_seconds", 8.0),
        ),
        "max_tokens": stage_llm_config.get("max_tokens", global_llm_config.get("max_tokens", 4096)),
        "temperature": stage_llm_config.get("temperature", global_llm_config.get("temperature", 0.0)),
        "seed": stage_llm_config.get("seed", global_llm_config.get("seed")),
    }

    if base_url:
        settings["base_url"] = base_url
    if model:
        settings["model"] = model

    return settings


class ScholarCandidatePaper(BaseModel):
    paper_title: str
    arxiv_id: Optional[str] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    authors: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    pdf_path: Optional[str] = None
    pdf_download_status: Optional[str] = None
    pdf_download_error: Optional[str] = None
    citations: Optional[int] = None
    source: str = "google_scholar"


class HardNegativePaper(BaseModel):
    paper_title: str
    arxiv_id: Optional[str] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    authors: List[str] = Field(default_factory=list)
    hard_negative_reason: str = ""
    source_query: str = ""
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    pdf_path: Optional[str] = None
    pdf_download_status: Optional[str] = None
    pdf_download_error: Optional[str] = None
    citations: Optional[int] = None


class PositivePaper(BaseModel):
    paper_title: str
    arxiv_id: Optional[str] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    authors: List[str] = Field(default_factory=list)
    positive_reason: str = ""
    source_query: str = ""
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    pdf_path: Optional[str] = None
    pdf_download_status: Optional[str] = None
    pdf_download_error: Optional[str] = None
    citations: Optional[int] = None


class HardNegativeMiningResult(BaseModel):
    paper_id: str
    paper_title: str
    query: str
    query_type: str = "IR"
    source_view: str
    is_multimodal: bool = False
    related_bullet_indice: Optional[int] = None
    related_bullet_justification: Optional[str] = None
    multimodal_rationale: Optional[str] = None
    hard_negatives: List[HardNegativePaper]
    positives: List[PositivePaper] = Field(default_factory=list)
    keywords_extracted: List[str]
    search_queries_used: List[str] = Field(default_factory=list)
    retrieved_candidates: int = 0
    mining_method: str = "google_scholar_real_search"


class HardNegativeMiningDataset(BaseModel):
    results: List[HardNegativeMiningResult]
    total_queries: int
    total_mined: int
    total_hard_negatives: int
    total_positives: int
    mined_at: datetime = Field(default_factory=datetime.now)


class GoogleScholarClient(ABC):
    @abstractmethod
    def search(self, query: str, limit: int) -> List[ScholarCandidatePaper]:
        raise NotImplementedError


@dataclass
class SearchClientRuntimeConfig:
    provider_name: str
    min_interval_seconds: float = 0.0
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff_seconds: float = 3.0
    retry_backoff_multiplier: float = 2.0
    cache_dir: Optional[Path] = None



class CachedRateLimitedSearchClient(GoogleScholarClient):
    def __init__(
        self,
        delegate: GoogleScholarClient,
        config: SearchClientRuntimeConfig,
    ):
        self._delegate = delegate
        self._config = config
        self._lock = threading.Lock()
        self._next_allowed_request_ts = 0.0

        if self._config.cache_dir is not None:
            self._config.cache_dir.mkdir(parents=True, exist_ok=True)

    def search(self, query: str, limit: int) -> List[ScholarCandidatePaper]:
        cache_path = self._cache_path(query, limit)
        if cache_path is not None and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    cached = [ScholarCandidatePaper.model_validate(item) for item in payload]
                    logger.info(
                        "Search cache hit for provider=%s query=%r limit=%s (%s results)",
                        self._config.provider_name,
                        query,
                        limit,
                        len(cached),
                    )
                    return cached
            except Exception as exc:
                logger.warning("Failed to read search cache %s: %s", cache_path, exc)

        max_attempts = max(1, int(self._config.max_retries) + 1)
        backoff_seconds = max(0.0, float(self._config.retry_backoff_seconds))
        last_error: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            self._wait_for_turn()
            try:
                results = self._delegate.search(query, limit)
                self._save_cache(cache_path, results)
                return results
            except Exception as exc:
                last_error = exc
                if attempt >= max_attempts:
                    break
                sleep_seconds = backoff_seconds * (
                    max(1.0, float(self._config.retry_backoff_multiplier)) ** (attempt - 1)
                )
                logger.warning(
                    "Search failed for provider=%s query=%r attempt=%s/%s: %s; retrying in %.1fs",
                    self._config.provider_name,
                    query,
                    attempt,
                    max_attempts,
                    exc,
                    sleep_seconds,
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

        raise RuntimeError(
            f"{self._config.provider_name} search failed after {max_attempts} attempts: {last_error}"
        ) from last_error

    def _wait_for_turn(self) -> None:
        min_interval = max(0.0, float(self._config.min_interval_seconds))
        if min_interval <= 0:
            return

        while True:
            with self._lock:
                now = time.monotonic()
                wait_seconds = self._next_allowed_request_ts - now
                if wait_seconds <= 0:
                    self._next_allowed_request_ts = now + min_interval
                    return
            time.sleep(min(wait_seconds, 0.5))

    def _cache_path(self, query: str, limit: int) -> Optional[Path]:
        if self._config.cache_dir is None:
            return None
        key = hashlib.sha1(
            f"{self._config.provider_name}\n{int(limit)}\n{query.strip()}".encode("utf-8")
        ).hexdigest()
        return self._config.cache_dir / f"{key}.json"

    def _save_cache(self, cache_path: Optional[Path], results: List[ScholarCandidatePaper]) -> None:
        if cache_path is None:
            return
        try:
            cache_path.write_text(
                json.dumps([item.model_dump(mode="json") for item in results], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to write search cache %s: %s", cache_path, exc)


class SerpApiGoogleScholarClient(GoogleScholarClient):
    def __init__(self, api_key: str, language: str = "en", timeout_seconds: float = 30.0):
        self.api_key = api_key
        self.language = language
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    def search(self, query: str, limit: int) -> List[ScholarCandidatePaper]:
        params = {
            "engine": "google_scholar",
            "q": query,
            "hl": self.language,
            "num": max(1, min(int(limit), 20)),
            "api_key": self.api_key,
        }
        url = f"https://serpapi.com/search.json?{urlencode(params)}"
        try:
            with urlopen(url, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"SerpAPI Google Scholar search failed: {exc}") from exc

        results: List[ScholarCandidatePaper] = []
        for item in payload.get("organic_results", [])[:limit]:
            if not isinstance(item, dict):
                continue

            publication_info = item.get("publication_info") or {}
            authors = []
            if isinstance(publication_info, dict):
                raw_authors = publication_info.get("authors") or []
                if isinstance(raw_authors, list):
                    authors = [
                        author.get("name", "").strip()
                        for author in raw_authors
                        if isinstance(author, dict) and author.get("name")
                    ]
                summary = publication_info.get("summary") or ""
            else:
                summary = str(publication_info)

            year = _extract_year(summary)
            results.append(
                ScholarCandidatePaper(
                    paper_title=str(item.get("title", "")).strip(),
                    arxiv_id=_extract_arxiv_id(
                        item.get("link"),
                        item.get("resources"),
                        item.get("publication_info"),
                    ),
                    abstract=str(item.get("snippet", "")).strip() or None,
                    venue=_extract_venue(summary),
                    year=year,
                    authors=authors,
                    url=item.get("link"),
                    pdf_url=(
                        _extract_pdf_url(item.get("resources"), item.get("link"))
                        or _build_arxiv_pdf_url(
                            _extract_arxiv_id(
                                item.get("link"),
                                item.get("resources"),
                                item.get("publication_info"),
                            )
                        )
                    ),
                    citations=_extract_serpapi_citations(item),
                    source="google_scholar_serpapi",
                )
            )
        return [paper for paper in results if paper.paper_title]


class ScholarlyGoogleScholarClient(GoogleScholarClient):
    def __init__(self, http_proxy: Optional[str] = None, https_proxy: Optional[str] = None):
        self._http_proxy = http_proxy or os.environ.get("SCHOLARLY_HTTP_PROXY")
        self._https_proxy = https_proxy or os.environ.get("SCHOLARLY_HTTPS_PROXY")

    def search(self, query: str, limit: int) -> List[ScholarCandidatePaper]:
        try:
            from scholarly import scholarly
        except ImportError as exc:
            raise RuntimeError(
                "scholarly is not installed. Install the scholar extra or configure SerpAPI."
            ) from exc

        saved_http = os.environ.get("HTTP_PROXY")
        saved_https = os.environ.get("HTTPS_PROXY")
        saved_all = os.environ.get("ALL_PROXY")
        try:
            if self._http_proxy:
                os.environ["HTTP_PROXY"] = self._http_proxy
                os.environ["HTTPS_PROXY"] = self._https_proxy
                os.environ["ALL_PROXY"] = self._https_proxy
                logger.info("scholarly using proxy: %s", self._http_proxy)

            results: List[ScholarCandidatePaper] = []
            search_iter = scholarly.search_pubs(query)
            for _ in range(max(1, int(limit))):
                pub = next(search_iter, None)
                if pub is None:
                    break
                parsed = _parse_scholarly_publication(pub)
                if parsed is not None:
                    results.append(parsed)
            return results
        except Exception as exc:
            raise RuntimeError(f"scholarly Google Scholar search failed: {exc}") from exc
        finally:
            if saved_http is None:
                os.environ.pop("HTTP_PROXY", None)
            else:
                os.environ["HTTP_PROXY"] = saved_http
            if saved_https is None:
                os.environ.pop("HTTPS_PROXY", None)
            else:
                os.environ["HTTPS_PROXY"] = saved_https
            if saved_all is None:
                os.environ.pop("ALL_PROXY", None)
            else:
                os.environ["ALL_PROXY"] = saved_all


class ArxivSearchClient(GoogleScholarClient):
    _API_URL = "http://export.arxiv.org/api/query"
    _USER_AGENT = "SciFullMMBench arXiv search script"

    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    def search(self, query: str, limit: int) -> List[ScholarCandidatePaper]:
        max_results = max(1, min(int(limit), 20))
        params = {
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{self._API_URL}?{urlencode(params)}"
        request = Request(
            url,
            headers={"User-Agent": self._USER_AGENT},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"arXiv API search failed: {exc}") from exc

        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise RuntimeError(f"arXiv API returned invalid XML: {exc}") from exc

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        results: List[ScholarCandidatePaper] = []
        for entry in root.findall("atom:entry", ns):
            title = _xml_text(entry.find("atom:title", ns))
            if not title:
                continue

            summary = _xml_text(entry.find("atom:summary", ns)) or None
            published = _xml_text(entry.find("atom:published", ns))
            year = _extract_year(published)
            authors = [
                _xml_text(author.find("atom:name", ns))
                for author in entry.findall("atom:author", ns)
                if _xml_text(author.find("atom:name", ns))
            ]

            entry_id = _xml_text(entry.find("atom:id", ns))
            pdf_url = None
            abs_url = entry_id or None
            for link in entry.findall("atom:link", ns):
                href = (link.attrib.get("href") or "").strip()
                title_attr = (link.attrib.get("title") or "").strip().lower()
                link_type = (link.attrib.get("type") or "").strip().lower()
                if title_attr == "pdf" or link_type == "application/pdf":
                    pdf_url = href
                    break

            arxiv_id = _extract_arxiv_id(entry_id, pdf_url)
            results.append(
                ScholarCandidatePaper(
                    paper_title=title,
                    arxiv_id=arxiv_id,
                    abstract=summary,
                    venue="arXiv",
                    year=year,
                    authors=authors,
                    url=abs_url,
                    pdf_url=pdf_url or _build_arxiv_pdf_url(arxiv_id),
                    citations=None,
                    source="arxiv_api",
                )
            )
        return results


class SemanticScholarSearchClient(GoogleScholarClient):
    _API_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    _MAX_PAGE_SIZE = 1000

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        api_keys: Optional[List[str]] = None,
    ):
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.api_keys = [token for token in (api_keys or _semantic_scholar_tokens_from_env()) if token]

    def search(self, query: str, limit: int) -> List[ScholarCandidatePaper]:
        base_params = {
            "query": query,
            "year": "2018-",
            "fieldsOfStudy": "Computer Science",
            "fields": ",".join(
                [
                    "title",
                    "year",
                    "venue",
                    "url",
                    "externalIds",
                    "openAccessPdf",
                    "citationCount",
                ]
            ),
        }
        auth_headers = [{}]
        if self.api_keys:
            auth_headers = [
                {"Authorization": f"Bearer {api_key}"}
                for api_key in self.api_keys
            ]

        payload_items: Optional[List[dict[str, Any]]] = None
        last_error: Optional[Exception] = None
        for headers in auth_headers:
            try:
                accumulated: List[dict[str, Any]] = []
                token: Optional[str] = None
                target_count = max(1, min(int(limit), 20)) * 20
                max_pages = max(1, min(10, (target_count + self._MAX_PAGE_SIZE - 1) // self._MAX_PAGE_SIZE))

                for _ in range(max_pages):
                    params = dict(base_params)
                    if token:
                        params["token"] = token
                    url = f"{self._API_URL}?{urlencode(params)}"
                    request = Request(
                        url,
                        headers={
                            "User-Agent": "SciFullMMBench/1.0 (semantic scholar retrieval)",
                            "Accept": "application/json",
                            **headers,
                        },
                    )
                    with urlopen(request, timeout=self.timeout_seconds) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    items = payload.get("data", [])
                    if isinstance(items, list):
                        accumulated.extend(item for item in items if isinstance(item, dict))
                    token = payload.get("token")
                    if len(accumulated) >= target_count or not token:
                        break

                payload_items = accumulated
                break
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if isinstance(exc, HTTPError) and exc.code == 429 and headers:
                    logger.warning("Semantic Scholar key hit 429; trying next key.")
                    continue
                if headers:
                    logger.warning("Semantic Scholar request failed with current key: %s", exc)
                    continue
                break

        if payload_items is None:
            raise RuntimeError(f"Semantic Scholar bulk search failed: {last_error}") from last_error

        results: List[ScholarCandidatePaper] = []
        for item in payload_items:
            title = str(item.get("title", "")).strip()
            if not title:
                continue

            external_ids = item.get("externalIds") or {}
            paper_url = str(item.get("url", "")).strip() or None
            open_access_pdf = item.get("openAccessPdf") if isinstance(item.get("openAccessPdf"), dict) else {}
            venue = str(item.get("venue", "")).strip() or None
            if not venue or venue.lower() != "arxiv.org":
                continue
            arxiv_id = _extract_arxiv_id(external_ids, paper_url, open_access_pdf)
            pdf_url = _build_arxiv_pdf_url(arxiv_id) if arxiv_id else (str(open_access_pdf.get("url", "")).strip() or None)
            if not pdf_url:
                continue

            results.append(
                ScholarCandidatePaper(
                    paper_title=title,
                    arxiv_id=arxiv_id,
                    abstract=None,
                    venue=venue,
                    year=_safe_int(item.get("year")),
                    authors=[],
                    url=paper_url,
                    pdf_url=pdf_url,
                    citations=_safe_int(item.get("citationCount")),
                    source="semantic_scholar_bulk_api",
                )
            )
            if len(results) >= max(1, min(int(limit), 20)):
                break
        return results


def _extract_year(text: Any) -> Optional[int]:
    if text is None:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", str(text))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _extract_venue(summary: str) -> Optional[str]:
    if not summary:
        return None
    parts = [part.strip() for part in summary.split(" - ") if part.strip()]
    if len(parts) < 2:
        return None
    venue_part = parts[-1]
    venue = re.sub(r"\b(19|20)\d{2}\b", "", venue_part).strip(" ,")
    return venue or None


def _xml_text(node: Optional[ET.Element]) -> str:
    if node is None or node.text is None:
        return ""
    return re.sub(r"\s+", " ", node.text).strip()


def _extract_arxiv_id(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                found = _extract_arxiv_id(item)
                if found:
                    return found
            continue
        if isinstance(value, dict):
            for item in value.values():
                found = _extract_arxiv_id(item)
                if found:
                    return found
            continue

        match = re.search(r"arxiv\.org/(abs|pdf)/([0-9]+\.[0-9]+)", str(value), re.IGNORECASE)
        if match:
            return match.group(2)
        match = re.search(r"\b([0-9]{4}\.[0-9]{4,5})(v\d+)?\b", str(value))
        if match:
            return match.group(1)
    return None


def _build_arxiv_pdf_url(arxiv_id: Optional[str]) -> Optional[str]:
    if not arxiv_id:
        return None
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def _extract_pdf_url(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                found = _extract_pdf_url(item)
                if found:
                    return found
            continue
        if isinstance(value, dict):
            for item in value.values():
                found = _extract_pdf_url(item)
                if found:
                    return found
            continue

        text = str(value).strip()
        if re.search(r"https?://\S+\.pdf(\?.*)?$", text, re.IGNORECASE):
            return text
    return None


def _extract_serpapi_citations(item: dict[str, Any]) -> Optional[int]:
    inline_links = item.get("inline_links") or {}
    cited_by = inline_links.get("cited_by") if isinstance(inline_links, dict) else None
    total = cited_by.get("total") if isinstance(cited_by, dict) else None
    try:
        return int(total) if total is not None else None
    except (TypeError, ValueError):
        return None


def _parse_scholarly_publication(pub: Any) -> Optional[ScholarCandidatePaper]:
    data = pub if isinstance(pub, dict) else getattr(pub, "__dict__", {})
    if not isinstance(data, dict):
        return None

    bib = data.get("bib") if isinstance(data.get("bib"), dict) else {}
    title = str(bib.get("title", "")).strip() or str(data.get("title", "")).strip()
    if not title:
        return None

    authors = []
    raw_authors = bib.get("author")
    if isinstance(raw_authors, str):
        authors = [part.strip() for part in raw_authors.split(" and ") if part.strip()]
    elif isinstance(raw_authors, list):
        authors = [str(part).strip() for part in raw_authors if str(part).strip()]

    year = None
    try:
        if bib.get("pub_year"):
            year = int(str(bib.get("pub_year")))
    except (TypeError, ValueError):
        year = None

    return ScholarCandidatePaper(
        paper_title=title,
        arxiv_id=_extract_arxiv_id(
            data.get("pub_url"),
            data.get("eprint_url"),
            bib.get("url"),
        ),
        abstract=str(bib.get("abstract", "")).strip() or None,
        venue=(
            str(bib.get("venue", "")).strip()
            or str(bib.get("journal", "")).strip()
            or None
        ),
        year=year,
        authors=authors,
        url=data.get("pub_url") or bib.get("url"),
        pdf_url=(
            _extract_pdf_url(
                data.get("eprint_url"),
                data.get("pub_url"),
                bib.get("url"),
            )
            or _build_arxiv_pdf_url(
                _extract_arxiv_id(
                    data.get("pub_url"),
                    data.get("eprint_url"),
                    bib.get("url"),
                )
            )
        ),
        citations=_safe_int(data.get("num_citations")),
        source="google_scholar_scholarly",
    )


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _slugify(text: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        slug = "item"
    return slug[:max_length].rstrip("-") or "item"


class HardNegativeMiner:
    def __init__(
        self,
        llm: LLMBackend,
        scholar_client: GoogleScholarClient,
        scholar_max_results: int = 10,
        download_selected_pdfs: bool = False,
        pdf_output_dir: Optional[Path] = None,
        review_max_workers: int = 1,
        *,
        docling_timeout_seconds: int = _DOCLING_TIMEOUT_SECONDS,
    ):
        self.llm = llm
        self.scholar_client = scholar_client
        self.scholar_max_results = max(3, int(scholar_max_results))
        self.download_selected_pdfs = download_selected_pdfs
        self.pdf_output_dir = pdf_output_dir.resolve() if pdf_output_dir else None
        self.review_max_workers = max(1, int(review_max_workers))
        self._docling_timeout = int(docling_timeout_seconds)
        self._search_query_attempts = 0
        self._search_query_failures = 0
        self._search_failed_queries: dict[str, int] = {}

    def _load_prompt(self, filename: str) -> str:
        return load_prompt_template(PROMPT_DIR / filename)

    def _render_prompt(self, filename: str, **replacements: str) -> str:
        prompt = self._load_prompt(filename)
        for key, value in replacements.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", value)
        return prompt

    def extract_keywords(self, query: str) -> List[str]:
        prompt = self._render_prompt(
            "hard_negative_extract_keywords.txt",
            query=query,
        )
        response = self.llm.generate(prompt)
        try:
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                keywords = json.loads(json_match.group())
                if isinstance(keywords, list):
                    return [str(k).strip() for k in keywords if str(k).strip()][:5]
        except Exception as exc:
            logger.warning("Failed to parse keywords from response: %s", exc)
        return []

    def _build_search_queries(self, query: str, keywords: List[str]) -> List[str]:
        compact_keywords: List[str] = []
        for keyword in keywords[:2]:
            normalized = " ".join(str(keyword).split()).strip()
            if not normalized:
                continue
            compact = " ".join(normalized.split()[:4]).strip()
            if compact:
                compact_keywords.append(compact)
        if not compact_keywords:
            logger.warning("No keywords extracted for query; skipping retrieval query build.")
            return []
        return [" ".join(compact_keywords[:2]).strip()]

    def _build_fallback_search_queries(self, keywords: List[str]) -> List[str]:
        fallback_queries: List[str] = []
        for keyword in keywords[:2]:
            normalized = " ".join(str(keyword).split()).strip()
            if not normalized:
                continue
            compact = " ".join(normalized.split()[:4]).strip()
            if compact and compact not in fallback_queries:
                fallback_queries.append(compact)
        return fallback_queries

    def _search_google_scholar(self, search_queries: List[str]) -> List[ScholarCandidatePaper]:
        unique_candidates: dict[str, ScholarCandidatePaper] = {}
        per_query_limit = max(3, self.scholar_max_results // max(1, len(search_queries)))

        for search_query in search_queries:
            self._search_query_attempts += 1
            try:
                candidates = self.scholar_client.search(search_query, per_query_limit)
            except Exception as exc:
                self._search_query_failures += 1
                self._search_failed_queries[search_query] = self._search_failed_queries.get(search_query, 0) + 1
                logger.warning("Google Scholar search failed for '%s': %s", search_query, exc)
                continue

            for candidate in candidates:
                normalized_title = re.sub(r"\s+", " ", candidate.paper_title.strip().lower())
                if normalized_title and normalized_title not in unique_candidates:
                    unique_candidates[normalized_title] = candidate

        return list(unique_candidates.values())[: self.scholar_max_results]

    def _merge_candidates(
        self,
        primary: List[ScholarCandidatePaper],
        fallback: List[ScholarCandidatePaper],
    ) -> List[ScholarCandidatePaper]:
        merged: List[ScholarCandidatePaper] = []
        seen_titles: set[str] = set()
        for candidate in primary + fallback:
            normalized_title = re.sub(r"\s+", " ", candidate.paper_title.strip().lower())
            if not normalized_title or normalized_title in seen_titles:
                continue
            seen_titles.add(normalized_title)
            merged.append(candidate)
            if len(merged) >= self.scholar_max_results:
                break
        return merged

    def retrieve_candidates_for_query(
        self,
        query_text: str,
        keywords: List[str],
    ) -> tuple[List[str], List[ScholarCandidatePaper]]:
        search_queries = self._build_search_queries(query_text, keywords)
        candidates = self._search_google_scholar(search_queries)
        for fallback_query in self._build_fallback_search_queries(keywords):
            if fallback_query in search_queries or len(candidates) >= self.scholar_max_results:
                continue
            logger.info(
                "Primary search returned %s candidates for query '%s'; supplementing with fallback phrase '%s'.",
                len(candidates),
                query_text,
                fallback_query,
            )
            fallback_candidates = self._search_google_scholar([fallback_query])
            if fallback_candidates:
                candidates = self._merge_candidates(candidates, fallback_candidates)
                search_queries = [*search_queries, fallback_query]
        return search_queries, candidates

    def _paper_identifier(self, paper_title: str, arxiv_id: Optional[str]) -> str:
        if arxiv_id:
            return arxiv_id.replace("/", "_")
        return hashlib.sha1(paper_title.encode("utf-8")).hexdigest()[:12]

    def _paper_output_dir(self, query: str, category: str) -> Path:
        base_dir = self.pdf_output_dir or (Path("outputs") / "hard_negative_pdfs")
        query_key = f"{_slugify(query, 48)}-{hashlib.sha1(query.encode('utf-8')).hexdigest()[:10]}"
        return base_dir / query_key / category

    def _download_pdf_for_selected_paper(
        self,
        paper: Union[ScholarCandidatePaper, HardNegativePaper, PositivePaper],
        *,
        query: str,
        category: str,
    ) -> None:
        pdf_url = paper.pdf_url or _build_arxiv_pdf_url(paper.arxiv_id) or _extract_pdf_url(paper.url)
        paper.pdf_url = pdf_url
        if not self.download_selected_pdfs:
            paper.pdf_path = None
            paper.pdf_download_status = "url_only" if pdf_url else "unavailable"
            paper.pdf_download_error = None if pdf_url else "No downloadable PDF URL found"
            return

        if not pdf_url:
            paper.pdf_download_status = "unavailable"
            paper.pdf_download_error = "No downloadable PDF URL found"
            return

        output_dir = self._paper_output_dir(query, category)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self._paper_identifier(paper.paper_title, paper.arxiv_id)}_{_slugify(paper.paper_title, 80)}.pdf"
        output_path = output_dir / filename

        if output_path.exists() and output_path.stat().st_size > 0:
            paper.pdf_path = str(output_path)
            paper.pdf_download_status = "downloaded"
            paper.pdf_download_error = None
            return

        request = Request(
            pdf_url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urlopen(request, timeout=60) as response:
                payload = response.read()
                content_type = response.headers.get("Content-Type", "")
        except Exception as exc:
            paper.pdf_download_status = "failed"
            paper.pdf_download_error = str(exc)
            return

        if not payload:
            paper.pdf_download_status = "failed"
            paper.pdf_download_error = "Empty response body"
            return

        if not payload.startswith(b"%PDF") and "pdf" not in content_type.lower():
            paper.pdf_download_status = "failed"
            paper.pdf_download_error = f"Non-PDF response (Content-Type: {content_type or 'unknown'})"
            return

        output_path.write_bytes(payload)
        paper.pdf_path = str(output_path)
        paper.pdf_download_status = "downloaded"
        paper.pdf_download_error = None

    def _candidate_to_hard_negative(
        self,
        candidate: ScholarCandidatePaper,
        query: str,
        reason: str,
    ) -> HardNegativePaper:
        return HardNegativePaper(
            paper_title=candidate.paper_title,
            arxiv_id=candidate.arxiv_id,
            abstract=candidate.abstract,
            venue=candidate.venue,
            year=candidate.year,
            authors=candidate.authors,
            hard_negative_reason=reason,
            source_query=query,
            url=candidate.url,
            pdf_url=candidate.pdf_url,
            pdf_path=candidate.pdf_path,
            pdf_download_status=candidate.pdf_download_status,
            pdf_download_error=candidate.pdf_download_error,
            citations=candidate.citations,
        )

    def _candidate_to_positive(
        self,
        candidate: ScholarCandidatePaper,
        query: str,
        reason: str,
    ) -> PositivePaper:
        return PositivePaper(
            paper_title=candidate.paper_title,
            arxiv_id=candidate.arxiv_id,
            abstract=candidate.abstract,
            venue=candidate.venue,
            year=candidate.year,
            authors=candidate.authors,
            positive_reason=reason,
            source_query=query,
            url=candidate.url,
            pdf_url=candidate.pdf_url,
            pdf_path=candidate.pdf_path,
            pdf_download_status=candidate.pdf_download_status,
            pdf_download_error=candidate.pdf_download_error,
            citations=candidate.citations,
        )

    def _parse_pdf_direct_docling(self, pdf_path: Path, paper_title: str) -> tuple[Optional[str], dict[str, Any]]:
        """Parse a PDF using Docling directly in-process, with a semaphore to limit concurrency."""
        metrics: dict[str, Any] = {
            "docling_wait_seconds": 0.0,
            "docling_parse_seconds": 0.0,
            "docling_stage": "direct_parse",
        }

        wait_started_at = time.monotonic()
        with _DOCLING_SEMAPHORE:
            metrics["docling_wait_seconds"] = time.monotonic() - wait_started_at
            parse_started_at = time.monotonic()

            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption

                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_table_structure = True
                pipeline_options.generate_page_images = False
                pipeline_options.generate_picture_images = False
                pipeline_options.generate_table_images = False
                pipeline_options.do_formula_enrichment = False
                if hasattr(pipeline_options, "do_ocr"):
                    pipeline_options.do_ocr = False

                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                    }
                )
                result = converter.convert(str(pdf_path))
                doc = result.document
                markdown = doc.export_to_markdown()
                page_count = len(doc.pages)

                metrics["docling_parse_seconds"] = time.monotonic() - parse_started_at
                metrics["docling_stage"] = "docling_direct_success"
                logger.info(
                    "stage5_docling_direct_success title=%r chars=%s pages=%s wait_seconds=%.3f parse_seconds=%.3f",
                    paper_title, len(markdown), page_count,
                    metrics["docling_wait_seconds"], metrics["docling_parse_seconds"],
                )
                return str(markdown), metrics
            except Exception as exc:
                metrics["docling_parse_seconds"] = time.monotonic() - parse_started_at
                metrics["docling_stage"] = "docling_direct_failed"
                logger.warning(
                    "stage5_docling_direct_failed title=%r parse_seconds=%.3f error=%s",
                    paper_title, metrics["docling_parse_seconds"], exc,
                )
                return None, metrics

    def _download_and_parse_pdf(
        self,
        pdf_url: str,
        paper_title: str,
        *,
        on_download_stage_complete: Optional[Callable[[], None]] = None,
        on_stage_update: Optional[StageCallback] = None,
    ) -> dict[str, Any]:
        """Download a PDF and parse with Docling directly in-process."""
        metrics: dict[str, Any] = {
            "download_seconds": 0.0,
            "docling_wait_seconds": 0.0,
            "docling_parse_seconds": 0.0,
            "docling_stage": "not_started",
        }
        pdf_dir = self.pdf_output_dir or (Path("outputs") / "hard_negative_pdfs")
        pdf_dir = Path(pdf_dir)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        pdf_name = f"{_slugify(paper_title, 80)}-{hashlib.sha1(pdf_url.encode('utf-8')).hexdigest()[:10]}.pdf"
        pdf_path = pdf_dir / pdf_name
        download_success = False

        download_started_at = time.monotonic()
        try:
            if not pdf_path.exists() or pdf_path.stat().st_size == 0:
                if on_stage_update is not None:
                    on_stage_update("download", {"pdf_url": pdf_url})
                logger.info("stage5_pdf_download_begin title=%r pdf_url=%s", paper_title, pdf_url)
                with _PDF_DOWNLOAD_SEMAPHORE:
                    for attempt in range(1, _PDF_DOWNLOAD_MAX_RETRIES + 1):
                        try:
                            _download_pdf_to_path(pdf_url, pdf_path, _PDF_DOWNLOAD_TIMEOUT_SECONDS)
                            logger.info(
                                "stage5_pdf_download_success title=%r pdf_url=%s bytes=%s attempt=%s",
                                paper_title,
                                pdf_url,
                                pdf_path.stat().st_size,
                                attempt,
                            )
                            download_success = True
                            break
                        except Exception as exc:
                            retryable = _is_retryable_pdf_download_error(exc)
                            if attempt < _PDF_DOWNLOAD_MAX_RETRIES and retryable:
                                backoff_seconds = _PDF_DOWNLOAD_RETRY_BACKOFF_SECONDS * attempt
                                logger.warning(
                                    "stage5_pdf_download_retry title=%r pdf_url=%s attempt=%s/%s backoff_seconds=%.1f error=%s",
                                    paper_title,
                                    pdf_url,
                                    attempt,
                                    _PDF_DOWNLOAD_MAX_RETRIES,
                                    backoff_seconds,
                                    exc,
                                )
                                time.sleep(backoff_seconds)
                                continue
                            logger.warning(
                                "stage5_pdf_download_failed title=%r pdf_url=%s attempt=%s/%s error=%s",
                                paper_title,
                                pdf_url,
                                attempt,
                                _PDF_DOWNLOAD_MAX_RETRIES,
                                exc,
                            )
                            return {
                                "markdown": None,
                                "metrics": metrics,
                                "failure_stage": "download_failed",
                            }
            else:
                download_success = True
        finally:
            metrics["download_seconds"] = time.monotonic() - download_started_at
            try:
                if download_success and on_download_stage_complete is not None:
                    on_download_stage_complete()
            except Exception as exc:
                logger.warning("stage5_download_stage_callback_failed title=%r error=%s", paper_title, exc)

        if on_stage_update is not None:
            on_stage_update("docling", {"pdf_url": pdf_url})
        markdown, docling_metrics = self._parse_pdf_direct_docling(pdf_path, paper_title)
        metrics.update(docling_metrics)
        if markdown:
            return {
                "markdown": markdown,
                "metrics": metrics,
                "failure_stage": None,
            }
        return {"markdown": None, "metrics": metrics, "failure_stage": "docling_parse_failed"}

    def _review_single_candidate(
        self,
        query: str,
        candidate: ScholarCandidatePaper,
        *,
        on_download_stage_complete: Optional[Callable[[], None]] = None,
        on_stage_update: Optional[StageCallback] = None,
    ) -> Optional[dict[str, Any]]:
        started_at = time.monotonic()
        authors = ", ".join(candidate.authors[:6]) if candidate.authors else "Unknown authors"
        metadata_only_evidence = (
            "No full PDF is attached. Base the decision only on the title, venue, year, "
            "authors, and abstract/snippet above. If that evidence is not strong enough, "
            "return ignored."
        )
        docling_evidence_template = (
            "The full text of the candidate paper (converted from PDF by Docling) follows:\n\n"
            "<paper_content>\n{markdown}\n</paper_content>\n\n"
            "Base the decision on the full paper content above."
        )

        pdf_url = candidate.pdf_url or _build_arxiv_pdf_url(candidate.arxiv_id) or _extract_pdf_url(candidate.url)
        candidate.pdf_url = pdf_url

        if pdf_url:
            pdf_result = self._download_and_parse_pdf(
                pdf_url,
                candidate.paper_title,
                on_download_stage_complete=on_download_stage_complete,
                on_stage_update=on_stage_update,
            )
            markdown = pdf_result["markdown"]
            timings = dict(pdf_result["metrics"])
            if markdown:
                markdown_truncated = markdown[:60000]
                evidence = docling_evidence_template.format(markdown=markdown_truncated)
                parse_status = "parsed"
            else:
                logger.info("stage5_docling_metadata_fallback title=%r", candidate.paper_title)
                evidence = metadata_only_evidence
                parse_status = "metadata_only"
        else:
            try:
                if on_download_stage_complete is not None:
                    on_download_stage_complete()
            except Exception as exc:
                logger.warning("stage5_download_stage_callback_failed title=%r error=%s", candidate.paper_title, exc)
            logger.info("stage5_no_pdf_metadata_only title=%r", candidate.paper_title)
            evidence = metadata_only_evidence
            parse_status = "no_pdf"
            timings = {
                "download_seconds": 0.0,
                "docling_wait_seconds": 0.0,
                "docling_parse_seconds": 0.0,
                "docling_stage": "no_pdf",
            }

        prompt = self._render_prompt(
            "hard_negative_review_candidate.txt",
            query=query,
            paper_title=candidate.paper_title,
            authors=authors,
            year=str(candidate.year) if candidate.year is not None else "Unknown",
            venue=candidate.venue or "Unknown venue",
            abstract=candidate.abstract or "No abstract/snippet available",
            paper_evidence=evidence,
        )
        if on_stage_update is not None:
            on_stage_update("llm", {"parse_status": parse_status})
        llm_started_at = time.monotonic()
        response = self.llm.generate(prompt)
        llm_request_seconds = time.monotonic() - llm_started_at
        llm_wait_seconds = 0.0
        llm_token = None
        request_manager = getattr(self.llm, "request_manager", None)
        if request_manager is not None and hasattr(request_manager, "get_last_call_metadata"):
            metadata = request_manager.get_last_call_metadata()
            if metadata is not None:
                llm_wait_seconds = float(metadata.wait_seconds)
                llm_request_seconds = float(metadata.request_seconds)
                llm_token = metadata.masked_token
        logger.info(
            "stage5_llm_request_complete title=%r wait_seconds=%.3f request_seconds=%.3f token=%s",
            candidate.paper_title,
            llm_wait_seconds,
            llm_request_seconds,
            llm_token,
        )

        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            parsed = json.loads(json_match.group()) if json_match else {}
        except Exception as exc:
            logger.warning("stage5_response_parse_failed title=%r error=%s", candidate.paper_title, exc)
            return None

        if not isinstance(parsed, dict):
            return None

        label = str(parsed.get("label", "")).strip().lower()
        if label not in {"positive", "hard_negative", "ignored"}:
            return None

        reason = str(parsed.get("reason", "")).strip()
        need_pro_review = parsed.get("need_pro_review", False)
        if isinstance(need_pro_review, str):
            need_pro_review = need_pro_review.strip().lower() in {"1", "true", "yes", "y"}
        else:
            need_pro_review = bool(need_pro_review)
        return {
            "label": label,
            "reason": reason,
            "need_pro_review": need_pro_review,
            "candidate": candidate,
            "parse_status": parse_status,
            "timings": {
                **timings,
                "deepseek_wait_seconds": llm_wait_seconds,
                "deepseek_request_seconds": llm_request_seconds,
                "total_row_seconds": time.monotonic() - started_at,
            },
        }

    def _download_selected_pdfs(
        self,
        query: str,
        papers: List[Union[HardNegativePaper, PositivePaper]],
        category: str,
    ) -> None:
        for paper in papers:
            self._download_pdf_for_selected_paper(paper, query=query, category=category)

    def _review_candidates(
        self,
        query: str,
        candidates: List[ScholarCandidatePaper],
    ) -> tuple[List[HardNegativePaper], List[PositivePaper]]:
        if not candidates:
            return [], []

        hard_negative_indices: List[int] = []
        hard_negative_by_index: dict[int, HardNegativePaper] = {}
        positives: List[PositivePaper] = []
        max_workers = min(self.review_max_workers, len(candidates))
        candidate_index_map = {id(candidate): index for index, candidate in enumerate(candidates)}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._review_single_candidate, query, candidate): candidate
                for candidate in candidates
            }
            for future in as_completed(futures):
                candidate = futures[future]
                try:
                    review = future.result()
                except Exception as exc:
                    logger.warning("Candidate review failed for '%s': %s", candidate.paper_title, exc)
                    continue

                if not review:
                    continue

                label = review["label"]
                reason = review["reason"]
                if label == "positive":
                    positives.append(self._candidate_to_positive(candidate, query, reason))
                elif label == "hard_negative":
                    candidate_index = candidate_index_map[id(candidate)]
                    hard_negative_indices.append(candidate_index)
                    hard_negative_by_index[candidate_index] = self._candidate_to_hard_negative(
                        candidate,
                        query,
                        reason,
                    )

        hard_negatives = [
            hard_negative_by_index[index]
            for index in sorted(hard_negative_indices)[:5]
        ]
        return hard_negatives, positives

    def mine_for_query(
        self,
        paper_id: str,
        paper_title: str,
        query: RetrievalQuery,
    ) -> HardNegativeMiningResult:
        logger.debug("Mining hard negatives for query: %s...", query.query_text[:50])

        keywords = self.extract_keywords(query.query_text)
        logger.debug("Extracted keywords: %s", keywords)

        search_queries, candidates = self.retrieve_candidates_for_query(query.query_text, keywords)
        hard_negatives, positives = self._review_candidates(query.query_text, candidates)
        self._download_selected_pdfs(query.query_text, hard_negatives, "hard_negatives")
        self._download_selected_pdfs(query.query_text, positives, "positives")

        return HardNegativeMiningResult(
            paper_id=paper_id,
            paper_title=paper_title,
            query=query.query_text,
            query_type=query.query_type,
            source_view=query.source_view,
            is_multimodal=query.is_multimodal,
            related_bullet_indice=query.related_bullet_indice,
            related_bullet_justification=query.related_bullet_justification,
            multimodal_rationale=query.multimodal_rationale,
            hard_negatives=hard_negatives,
            positives=positives,
            keywords_extracted=keywords,
            search_queries_used=search_queries,
            retrieved_candidates=len(candidates),
            mining_method="google_scholar_real_search",
        )

    def _result_key(self, result: HardNegativeMiningResult) -> tuple[str, str, str, str]:
        return (result.paper_id, result.query, result.source_view, result.query_type)

    def _query_key(self, paper_id: str, query: RetrievalQuery) -> tuple[str, str, str, str]:
        return (paper_id, query.query_text, query.source_view, query.query_type)

    def apply(
        self,
        dataset: GeneratedQueriesDataset,
        checkpoint_path: Optional[Path] = None,
    ) -> HardNegativeMiningDataset:
        logger.info("Mining hard negatives for %s queries", dataset.total_queries)

        all_results = []
        completed_keys = set()
        if checkpoint_path and checkpoint_path.exists():
            try:
                checkpoint = load_json(checkpoint_path, HardNegativeMiningDataset)
                expected_keys = {
                    self._query_key(paper.paper_id, query)
                    for paper in dataset.papers_queries
                    for query in paper.queries_by_view
                }
                all_results = [
                    result
                    for result in checkpoint.results
                    if self._result_key(result) in expected_keys
                ]
                completed_keys = {self._result_key(result) for result in all_results}
                if completed_keys:
                    logger.info(
                        "Loaded %s existing hard-negative mining results from %s",
                        len(completed_keys),
                        checkpoint_path,
                    )
            except Exception as exc:
                logger.warning("Could not load hard-negative mining checkpoint %s: %s", checkpoint_path, exc)

        work_items = [
            (paper.paper_id, paper.paper_title, query)
            for paper in dataset.papers_queries
            for query in paper.queries_by_view
            if self._query_key(paper.paper_id, query) not in completed_keys
        ]

        progress = tqdm(
            work_items,
            total=len(work_items),
            desc="Mining hard negatives",
            unit="query",
            dynamic_ncols=True,
        )
        for paper_id, paper_title, query in progress:
            result = self.mine_for_query(paper_id, paper_title, query)
            all_results.append(result)
            if checkpoint_path:
                save_json(checkpoint_path, self._build_dataset(all_results))
            total_hard_negatives = sum(len(item.hard_negatives) for item in all_results)
            total_positives = sum(len(item.positives) for item in all_results)
            progress.set_postfix_str(
                f"hard_negatives={total_hard_negatives}, positives={total_positives}"
            )
        progress.close()

        results_by_key = {self._result_key(result): result for result in all_results}
        all_results = [
            results_by_key[self._query_key(paper.paper_id, query)]
            for paper in dataset.papers_queries
            for query in paper.queries_by_view
            if self._query_key(paper.paper_id, query) in results_by_key
        ]
        result_dataset = self._build_dataset(all_results)
        logger.info(
            "Hard-negative-mining stage success: %s/%s queries (%.1f%%), %s with hard negatives.",
            result_dataset.total_queries,
            dataset.total_queries,
            (result_dataset.total_queries / dataset.total_queries * 100) if dataset.total_queries else 100.0,
            result_dataset.total_mined,
        )
        logger.info(
            "Search summary: attempts=%s, failures=%s, success_rate=%.1f%%",
            self._search_query_attempts,
            self._search_query_failures,
            (
                (self._search_query_attempts - self._search_query_failures) / self._search_query_attempts * 100.0
                if self._search_query_attempts
                else 100.0
            ),
        )
        if self._search_failed_queries:
            failed_items = sorted(
                self._search_failed_queries.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            preview = ", ".join([f"{query!r}:{count}" for query, count in failed_items[:10]])
            logger.warning("Top failed search queries (count): %s", preview)
        return result_dataset

    def _build_dataset(self, results: List[HardNegativeMiningResult]) -> HardNegativeMiningDataset:
        return HardNegativeMiningDataset(
            results=results,
            total_queries=len(results),
            total_mined=len([r for r in results if r.hard_negatives]),
            total_hard_negatives=sum(len(r.hard_negatives) for r in results),
            total_positives=sum(len(r.positives) for r in results),
        )

    def run(self, input_path: Path, output_path: Path) -> None:
        logger.info("Running hard-negative-mining stage: %s -> %s", input_path, output_path)
        if self.pdf_output_dir is None:
            self.pdf_output_dir = output_path.parent / f"{output_path.stem}_pdfs"
        dataset = load_json(input_path, GeneratedQueriesDataset)
        result = self.apply(dataset, checkpoint_path=output_path)
        save_json(output_path, result)


def build_google_scholar_client(
    provider: str,
    *,
    serpapi_api_key: str = "",
    semantic_scholar_api_keys: Optional[List[str]] = None,
    language: str = "en",
    scholarly_http_proxy: Optional[str] = None,
    scholarly_https_proxy: Optional[str] = None,
    timeout_seconds: float = 30.0,
    min_interval_seconds: Optional[float] = None,
    max_retries: int = 3,
    retry_backoff_seconds: Optional[float] = None,
    retry_backoff_multiplier: Optional[float] = None,
    cache_dir: Optional[Path] = None,
) -> GoogleScholarClient:
    normalized = (provider or "").strip().lower()
    runtime_config = _default_search_runtime_config(
        normalized,
        timeout_seconds=timeout_seconds,
        min_interval_seconds=min_interval_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        retry_backoff_multiplier=retry_backoff_multiplier,
        cache_dir=cache_dir,
    )
    if normalized == "serpapi":
        if not serpapi_api_key:
            raise ValueError(
                "search.provider is 'serpapi' but no search.serpapi_api_key was configured."
            )
        return CachedRateLimitedSearchClient(
            SerpApiGoogleScholarClient(
                api_key=serpapi_api_key,
                language=language,
                timeout_seconds=runtime_config.timeout_seconds,
            ),
            runtime_config,
        )
    if normalized == "scholarly":
        return CachedRateLimitedSearchClient(
            ScholarlyGoogleScholarClient(
                http_proxy=scholarly_http_proxy,
                https_proxy=scholarly_https_proxy,
            ),
            runtime_config,
        )
    if normalized == "arxiv":
        return CachedRateLimitedSearchClient(
            ArxivSearchClient(timeout_seconds=runtime_config.timeout_seconds),
            runtime_config,
        )
    if normalized in {"semantic_scholar", "semanticscholar", "s2"}:
        return CachedRateLimitedSearchClient(
            SemanticScholarSearchClient(
                timeout_seconds=runtime_config.timeout_seconds,
                api_keys=semantic_scholar_api_keys,
            ),
            runtime_config,
        )
    raise ValueError(f"Unsupported Google Scholar provider: {provider}")


def _default_search_runtime_config(
    provider_name: str,
    *,
    timeout_seconds: float,
    min_interval_seconds: Optional[float],
    max_retries: int,
    retry_backoff_seconds: Optional[float],
    retry_backoff_multiplier: Optional[float],
    cache_dir: Optional[Path],
) -> SearchClientRuntimeConfig:
    normalized = provider_name.strip().lower()
    provider_defaults: dict[str, dict[str, float]] = {
        "arxiv": {
            "min_interval_seconds": 3.5,
            "retry_backoff_seconds": 6.0,
            "retry_backoff_multiplier": 2.0,
        },
        "semantic_scholar": {
            "min_interval_seconds": 1.0,
            "retry_backoff_seconds": 4.0,
            "retry_backoff_multiplier": 2.0,
        },
        "semanticscholar": {
            "min_interval_seconds": 1.0,
            "retry_backoff_seconds": 4.0,
            "retry_backoff_multiplier": 2.0,
        },
        "s2": {
            "min_interval_seconds": 1.0,
            "retry_backoff_seconds": 4.0,
            "retry_backoff_multiplier": 2.0,
        },
        "scholarly": {
            "min_interval_seconds": 2.0,
            "retry_backoff_seconds": 5.0,
            "retry_backoff_multiplier": 2.0,
        },
    }
    defaults = provider_defaults.get(normalized, {})
    return SearchClientRuntimeConfig(
        provider_name=normalized or "unknown",
        min_interval_seconds=max(
            float(defaults.get("min_interval_seconds", 0.0)),
            float(0.0 if min_interval_seconds is None else min_interval_seconds),
        ),
        timeout_seconds=max(1.0, float(timeout_seconds)),
        max_retries=max(0, int(max_retries)),
        retry_backoff_seconds=max(
            float(defaults.get("retry_backoff_seconds", 3.0)),
            float(0.0 if retry_backoff_seconds is None else retry_backoff_seconds),
        ),
        retry_backoff_multiplier=max(
            float(defaults.get("retry_backoff_multiplier", 2.0)),
            float(1.0 if retry_backoff_multiplier is None else retry_backoff_multiplier),
        ),
        cache_dir=cache_dir,
    )
