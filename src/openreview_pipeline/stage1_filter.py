import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List

import yaml
from tqdm.auto import tqdm

from openreview_pipeline.schemas import DownloadedPapersDataset
from openreview_pipeline.schemas.schemas_filter import FilteredPapersDataset, FilterResult, FilterRuleResult
from utils import load_json, save_json

logger = logging.getLogger(__name__)

DEFAULT_RULES_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "rules.yaml"
MULTIMODAL_REF_PATTERN = re.compile(
    r"\b(?P<kind>fig(?:ure)?s?|tables?|eq(?:uation)?s?|alg(?:orithm)?s?)"
    r"\.?\s*(?:\(?\s*)"
    r"(?P<label>[A-Za-z]?\d+(?:\.\d+)?(?:\s*(?:-|–|—|to|and|,)\s*[A-Za-z]?\d+(?:\.\d+)?)*)"
    r"\s*\)?",
    flags=re.IGNORECASE,
)
MEANINGFUL_DISCUSSION_TERMS = {
    "ablation",
    "analysis",
    "analyze",
    "baseline",
    "comparison",
    "compare",
    "concern",
    "contradict",
    "evidence",
    "experiment",
    "fail",
    "improve",
    "limitation",
    "missing",
    "performance",
    "result",
    "show",
    "support",
    "trend",
    "unclear",
    "weak",
    "worse",
}
KIND_ALIASES = {
    "fig": "Figure",
    "figs": "Figure",
    "figure": "Figure",
    "figures": "Figure",
    "table": "Table",
    "tables": "Table",
    "eq": "Equation",
    "eqs": "Equation",
    "equation": "Equation",
    "equations": "Equation",
    "alg": "Algorithm",
    "algs": "Algorithm",
    "algorithm": "Algorithm",
    "algorithms": "Algorithm",
}


@dataclass(frozen=True)
class _OpenReviewTextEntry:
    source_ref: str
    text: str


@dataclass
class _MultimodalEvidenceSnippet:
    source_ref: str
    text: str
    multimodal_refs: list[str]
    meaningful: bool


@dataclass
class _MultimodalEvidenceGroup:
    multimodal_ref: str
    snippets: list[_MultimodalEvidenceSnippet] = field(default_factory=list)

    @property
    def meaningful_snippets(self) -> list[_MultimodalEvidenceSnippet]:
        return [snippet for snippet in self.snippets if snippet.meaningful]


def _content_value(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value", "")
    if value is None:
        return ""
    return str(value).strip()


def _entry_iter(items: Iterable[Any], section: str, entry_prefix: str) -> Iterable[_OpenReviewTextEntry]:
    for idx, item in enumerate(items or [], 1):
        content = getattr(item, "content", None)
        number = getattr(item, "number", idx)
        if content is None and isinstance(item, dict):
            content = item.get("content", {})
            number = item.get("number", idx)
        if not isinstance(content, dict):
            continue
        for key, value in content.items():
            text = _content_value(value)
            if text:
                yield _OpenReviewTextEntry(
                    source_ref=f"{section}/{entry_prefix} {number}/{key}",
                    text=text,
                )


def _iter_openreview_discussion_entries(paper_meta: Any) -> list[_OpenReviewTextEntry]:
    if isinstance(paper_meta, dict):
        reviews = paper_meta.get("reviews", [])
        rebuttals = paper_meta.get("rebuttals", [])
        comments = paper_meta.get("comments", [])
    else:
        reviews = getattr(paper_meta, "reviews", [])
        rebuttals = getattr(paper_meta, "rebuttals", [])
        comments = getattr(paper_meta, "comments", [])

    entries: list[_OpenReviewTextEntry] = []
    entries.extend(_entry_iter(reviews, "Reviews", "Review"))
    entries.extend(_entry_iter(rebuttals, "Rebuttals", "Rebuttal"))
    entries.extend(_entry_iter(comments, "Comments", "Comment"))
    return entries


def _normalize_multimodal_ref(kind: str, label: str) -> str:
    normalized_kind = KIND_ALIASES.get(kind.strip().lower().rstrip("."), kind.strip().title())
    label = re.sub(r"\s+", "", label.strip())
    return f"{normalized_kind} {label}"


def _expand_ref_labels(raw_label: str) -> list[str]:
    raw_label = str(raw_label or "").strip()
    if not raw_label:
        return []
    if re.search(r"\s*(?:,|and)\s*", raw_label, flags=re.IGNORECASE):
        labels = re.split(r"\s*(?:,|and)\s*", raw_label, flags=re.IGNORECASE)
        return [label.strip() for label in labels if label.strip()]
    return [raw_label]


def _extract_multimodal_refs(text: str) -> list[str]:
    refs: list[str] = []
    for match in MULTIMODAL_REF_PATTERN.finditer(text or ""):
        kind = match.group("kind")
        raw_label = match.group("label")
        if not kind or not raw_label:
            continue
        for label in _expand_ref_labels(raw_label):
            normalized = _normalize_multimodal_ref(kind, label)
            if normalized not in refs:
                refs.append(normalized)
    return refs


def _is_meaningful_multimodal_discussion(text: str) -> bool:
    lowered = f" {str(text or '').lower()} "
    return any(re.search(r"\b" + re.escape(term) + r"\b", lowered) for term in MEANINGFUL_DISCUSSION_TERMS)


def _extract_multimodal_evidence_snippets(paper_meta: Any) -> list[_MultimodalEvidenceSnippet]:
    snippets: list[_MultimodalEvidenceSnippet] = []
    for entry in _iter_openreview_discussion_entries(paper_meta):
        refs = _extract_multimodal_refs(entry.text)
        if not refs:
            continue
        snippets.append(
            _MultimodalEvidenceSnippet(
                source_ref=entry.source_ref,
                text=entry.text,
                multimodal_refs=refs,
                meaningful=_is_meaningful_multimodal_discussion(entry.text),
            )
        )
    return snippets


def _group_multimodal_evidence(
    snippets: Iterable[_MultimodalEvidenceSnippet],
    *,
    meaningful_only: bool = False,
) -> list[_MultimodalEvidenceGroup]:
    groups: dict[str, _MultimodalEvidenceGroup] = {}
    for snippet in snippets:
        if meaningful_only and not snippet.meaningful:
            continue
        for ref in snippet.multimodal_refs:
            groups.setdefault(ref, _MultimodalEvidenceGroup(multimodal_ref=ref)).snippets.append(snippet)
    return list(groups.values())


def _build_multimodal_filter_diagnostics(
    paper_meta: Any,
    *,
    min_meaningful_snippets: int = 2,
) -> dict[str, Any]:
    snippets = _extract_multimodal_evidence_snippets(paper_meta)
    meaningful = [snippet for snippet in snippets if snippet.meaningful]
    groups = _group_multimodal_evidence(meaningful)
    refs = sorted({ref for snippet in meaningful for ref in snippet.multimodal_refs})
    source_refs = [snippet.source_ref for snippet in meaningful]
    passed = len(meaningful) >= int(min_meaningful_snippets)
    reason = (
        f"Found {len(meaningful)} meaningful multimodal discussion snippets "
        f"across {len(refs)} concrete evidence refs."
        if passed
        else f"Only {len(meaningful)} meaningful multimodal discussion snippets found; "
        f"requires {int(min_meaningful_snippets)}."
    )
    return {
        "passed": passed,
        "meaningful_snippet_count": len(meaningful),
        "total_evidence_snippet_count": len(snippets),
        "matched_evidence_refs": refs,
        "source_refs": source_refs,
        "groups": [
            {
                "multimodal_ref": group.multimodal_ref,
                "source_refs": [snippet.source_ref for snippet in group.meaningful_snippets],
                "snippet_count": len(group.meaningful_snippets),
            }
            for group in groups
        ],
        "reason": reason,
    }


class RuleConfig:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_RULES_PATH
        self._config = self._load_config()

    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._default_config()
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _default_config(self) -> Dict:
        return {
            "accepted": {
                "accept_keywords": ["accept", "accepted", "oral", "poster", "spotlight", "strong accept"],
                "reject_keywords": ["reject", "rejected", "strong reject", "not accept"],
            },
            "similar_paper": {
                "citation_threshold": 0.05,
                "max_citations_per_word": 0.05,
                "citation_patterns": ["et al", "arxiv", "paper~", "proceedings"],
            },
            "multimodal_info": {
                "min_multimodal_mentions": 2,
                "multimodal_keywords": ["figure", "fig", "table", "diagram", "chart", "equation", "formula", "plot"],
                "content_sources": ["abstract", "review", "rebuttal", "comments"],
            },
        }

    @property
    def accept_keywords(self) -> List[str]:
        return self._config.get("accepted", {}).get("accept_keywords", [])

    @property
    def reject_keywords(self) -> List[str]:
        return self._config.get("accepted", {}).get("reject_keywords", [])

    @property
    def citation_threshold(self) -> float:
        return self._config.get("similar_paper", {}).get("citation_threshold", 0.05)

    @property
    def citation_patterns(self) -> List[str]:
        return self._config.get("similar_paper", {}).get("citation_patterns", [])

    @property
    def min_multimodal_mentions(self) -> int:
        multimodal_config = self._config.get("multimodal_info", {})
        return multimodal_config.get(
            "min_meaningful_discussions",
            multimodal_config.get("min_multimodal_mentions", 2),
        )

    @property
    def multimodal_keywords(self) -> List[str]:
        return self._config.get("multimodal_info", {}).get("multimodal_keywords", [])

    @property
    def content_sources(self) -> List[str]:
        return self._config.get("multimodal_info", {}).get("content_sources", [])


class RuleBasedFilter:
    def __init__(self, config_path: Optional[Path] = None, rules_config: Optional[RuleConfig] = None, limit: Optional[int] = None):
        self.config = rules_config or RuleConfig(config_path)
        self.limit = limit

    def check_accepted(self, paper: Dict) -> bool:
        decision = paper.get("decision")
        if decision:
            decision_content = decision.get("content", {})
            decision_value = decision_content.get("decision", "")
            if isinstance(decision_value, dict):
                decision_value = decision_value.get("value", "")
            decision_value = str(decision_value).lower()

            has_accept = any(kw in decision_value for kw in self.config.accept_keywords)
            has_reject = any(kw in decision_value for kw in self.config.reject_keywords)

            if has_reject and not has_accept:
                return False
            return has_accept

        title = paper.get("paper", {}).get("title", "").lower()
        abstract = paper.get("paper", {}).get("abstract", "").lower()
        venue = paper.get("paper", {}).get("venue", "").lower()
        text = f"{title} {abstract} {venue}"

        has_accept = any(kw in text for kw in self.config.accept_keywords)
        has_reject = any(kw in text for kw in self.config.reject_keywords)

        if has_reject and not has_accept:
            return False
        return has_accept

    def check_similar_paper(self, paper: Dict) -> bool:
        abstract = paper.get("paper", {}).get("abstract", "").lower()
        title = paper.get("paper", {}).get("title", "").lower()
        text = f"{title} {abstract}"

        word_count = len(text.split())
        if word_count == 0:
            return False

        citation_count = 0
        for pattern in self.config.citation_patterns:
            citation_count += len(re.findall(re.escape(pattern), text))

        citation_ratio = citation_count / word_count
        return citation_ratio >= self.config.citation_threshold

    def analyze_multimodal_info(self, paper: Dict) -> Dict:
        return _build_multimodal_filter_diagnostics(
            paper,
            min_meaningful_snippets=self.config.min_multimodal_mentions,
        )

    def check_multimodal_info(self, paper: Dict) -> bool:
        return bool(self.analyze_multimodal_info(paper).get("passed"))

    def apply(
        self,
        dataset: DownloadedPapersDataset,
        checkpoint_path: Optional[Path] = None,
    ) -> FilteredPapersDataset:
        total = dataset.total_count
        limit = self.limit if self.limit else total
        logger.info(f"Applying filter to {min(limit, total)} of {total} papers (limit={self.limit})")

        results = []
        passed_count = 0
        target_papers = list(dataset.papers[:limit])
        target_ids = {paper.paper.id for paper in target_papers}
        completed_ids = set()
        if checkpoint_path and checkpoint_path.exists():
            try:
                checkpoint = load_json(checkpoint_path, FilteredPapersDataset)
                results = [
                    result
                    for result in checkpoint.results
                    if result.paper.paper.id in target_ids
                ]
                completed_ids = {result.paper.paper.id for result in results}
                passed_count = sum(1 for result in results if result.passed)
                if completed_ids:
                    logger.info(
                        "Loaded %s existing filter results from %s",
                        len(completed_ids),
                        checkpoint_path,
                    )
            except Exception as exc:
                logger.warning("Could not load filter checkpoint %s: %s", checkpoint_path, exc)

        work_items = [paper for paper in target_papers if paper.paper.id not in completed_ids]

        progress = tqdm(
            work_items,
            total=len(work_items),
            desc="Filtering papers",
            unit="paper",
            dynamic_ncols=True,
        )
        for paper_meta in progress:
            try:
                paper_dict = paper_meta.model_dump()

                accepted = self.check_accepted(paper_dict)
                similar_paper = self.check_similar_paper(paper_dict)
                multimodal_diagnostics = self.analyze_multimodal_info(paper_dict)
                multimodal_info = bool(multimodal_diagnostics.get("passed"))

                passed = accepted and not similar_paper and multimodal_info

                if passed:
                    passed_count += 1
                progress.set_postfix_str(f"passed={passed_count}")

                results.append(
                    FilterResult(
                        paper=paper_meta,
                        passed=passed,
                        details=FilterRuleResult(
                            accepted=accepted,
                            similar_paper=similar_paper,
                            multimodal_info=multimodal_info,
                            multimodal_evidence_refs=multimodal_diagnostics.get("matched_evidence_refs", []),
                            multimodal_source_refs=multimodal_diagnostics.get("source_refs", []),
                            meaningful_multimodal_snippet_count=int(
                                multimodal_diagnostics.get("meaningful_snippet_count", 0)
                            ),
                            total_multimodal_snippet_count=int(
                                multimodal_diagnostics.get("total_evidence_snippet_count", 0)
                            ),
                            multimodal_evidence_groups=multimodal_diagnostics.get("groups", []),
                            multimodal_evidence_reason=multimodal_diagnostics.get("reason"),
                        ),
                    )
                )
                if checkpoint_path:
                    save_json(
                        checkpoint_path,
                        FilteredPapersDataset(
                            results=results,
                            total_input=len(results),
                            total_passed=sum(1 for result in results if result.passed),
                            total_filtered=sum(1 for result in results if not result.passed),
                        ),
                    )
            except Exception as e:
                logger.error(f"Error filtering paper {paper_meta.paper.id}: {e}")
                import traceback
                logger.error(f"Stack trace: {traceback.format_exc()}")
                results.append(
                    FilterResult(
                        paper=paper_meta,
                        passed=False,
                        details=FilterRuleResult(
                            accepted=False,
                            similar_paper=False,
                            multimodal_info=False,
                        ),
                    )
                )
                if checkpoint_path:
                    save_json(
                        checkpoint_path,
                        FilteredPapersDataset(
                            results=results,
                            total_input=len(results),
                            total_passed=sum(1 for result in results if result.passed),
                            total_filtered=sum(1 for result in results if not result.passed),
                        ),
                    )
        progress.close()

        logger.info(
            "Filter stage success: %s/%s papers passed (%.1f%%).",
            passed_count,
            len(results),
            (passed_count / len(results) * 100) if results else 100.0,
        )

        return FilteredPapersDataset(
            results=results,
            total_input=len(results),
            total_passed=passed_count,
            total_filtered=len(results) - passed_count,
        )

    def run(self, input_path: Path, output_path: Path) -> None:
        logger.info(f"Running filter stage: {input_path} -> {output_path}")
        dataset = load_json(input_path, DownloadedPapersDataset)
        result = self.apply(dataset, checkpoint_path=output_path)
        save_json(output_path, result)
