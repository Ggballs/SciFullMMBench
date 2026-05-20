import json
import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union
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


class SerpApiGoogleScholarClient(GoogleScholarClient):
    def __init__(self, api_key: str, language: str = "en"):
        self.api_key = api_key
        self.language = language

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
            with urlopen(url, timeout=30) as response:
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
    def search(self, query: str, limit: int) -> List[ScholarCandidatePaper]:
        try:
            from scholarly import scholarly
        except ImportError as exc:
            raise RuntimeError(
                "scholarly is not installed. Install the scholar extra or configure SerpAPI."
            ) from exc

        results: List[ScholarCandidatePaper] = []
        try:
            search_iter = scholarly.search_pubs(query)
            for _ in range(max(1, int(limit))):
                pub = next(search_iter, None)
                if pub is None:
                    break
                parsed = _parse_scholarly_publication(pub)
                if parsed is not None:
                    results.append(parsed)
        except Exception as exc:
            raise RuntimeError(f"scholarly Google Scholar search failed: {exc}") from exc

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
    ):
        self.llm = llm
        self.scholar_client = scholar_client
        self.scholar_max_results = max(3, int(scholar_max_results))
        self.download_selected_pdfs = download_selected_pdfs
        self.pdf_output_dir = pdf_output_dir.resolve() if pdf_output_dir else None
        self.review_max_workers = max(1, int(review_max_workers))

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
        variants = [query.strip()]
        keyword_query = " ".join(keywords[:4]).strip()
        if keyword_query and keyword_query.lower() != query.strip().lower():
            variants.append(keyword_query)
        return [variant for variant in variants if variant]

    def _search_google_scholar(self, search_queries: List[str]) -> List[ScholarCandidatePaper]:
        unique_candidates: dict[str, ScholarCandidatePaper] = {}
        per_query_limit = max(3, self.scholar_max_results // max(1, len(search_queries)))

        for search_query in search_queries:
            try:
                candidates = self.scholar_client.search(search_query, per_query_limit)
            except Exception as exc:
                logger.warning("Google Scholar search failed for '%s': %s", search_query, exc)
                continue

            for candidate in candidates:
                normalized_title = re.sub(r"\s+", " ", candidate.paper_title.strip().lower())
                if normalized_title and normalized_title not in unique_candidates:
                    unique_candidates[normalized_title] = candidate

        return list(unique_candidates.values())[: self.scholar_max_results]

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

    def _review_single_candidate(
        self,
        query: str,
        candidate: ScholarCandidatePaper,
    ) -> Optional[dict[str, Any]]:
        authors = ", ".join(candidate.authors[:6]) if candidate.authors else "Unknown authors"
        prompt = self._render_prompt(
            "hard_negative_review_candidate.txt",
            query=query,
            paper_title=candidate.paper_title,
            authors=authors,
            year=str(candidate.year) if candidate.year is not None else "Unknown",
            venue=candidate.venue or "Unknown venue",
            abstract=candidate.abstract or "No abstract/snippet available",
            paper_evidence=(
                "The full candidate paper PDF is attached separately. Base the decision on that file."
            ),
        )

        pdf_url = candidate.pdf_url or _build_arxiv_pdf_url(candidate.arxiv_id) or _extract_pdf_url(candidate.url)
        candidate.pdf_url = pdf_url
        if not pdf_url:
            logger.debug("Skipping candidate without usable remote PDF URL: %s", candidate.paper_title)
            return None

        try:
            response = self.llm.generate_with_pdf_url(prompt, pdf_url)
        except Exception as exc:
            logger.warning(
                "Remote PDF review failed for '%s' via %s: %s",
                candidate.paper_title,
                pdf_url,
                exc,
            )
            return None

        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            parsed = json.loads(json_match.group()) if json_match else {}
        except Exception as exc:
            logger.warning("Failed to parse review response for '%s': %s", candidate.paper_title, exc)
            return None

        if not isinstance(parsed, dict):
            return None

        label = str(parsed.get("label", "")).strip().lower()
        if label not in {"positive", "hard_negative", "ignored"}:
            return None

        reason = str(parsed.get("reason", "")).strip()
        return {
            "label": label,
            "reason": reason,
            "candidate": candidate,
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

        search_queries = self._build_search_queries(query.query_text, keywords)
        candidates = self._search_google_scholar(search_queries)
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
    language: str = "en",
) -> GoogleScholarClient:
    normalized = (provider or "").strip().lower()
    if normalized == "serpapi":
        if not serpapi_api_key:
            raise ValueError(
                "search.provider is 'serpapi' but no search.serpapi_api_key was configured."
            )
        return SerpApiGoogleScholarClient(api_key=serpapi_api_key, language=language)
    if normalized == "scholarly":
        return ScholarlyGoogleScholarClient()
    raise ValueError(f"Unsupported Google Scholar provider: {provider}")
