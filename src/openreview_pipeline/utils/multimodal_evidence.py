from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


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
class OpenReviewTextEntry:
    source_ref: str
    text: str


@dataclass
class MultimodalEvidenceSnippet:
    source_ref: str
    text: str
    multimodal_refs: list[str]
    meaningful: bool


@dataclass
class MultimodalEvidenceGroup:
    multimodal_ref: str
    snippets: list[MultimodalEvidenceSnippet] = field(default_factory=list)

    @property
    def meaningful_snippets(self) -> list[MultimodalEvidenceSnippet]:
        return [snippet for snippet in self.snippets if snippet.meaningful]


def _content_value(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value", "")
    if value is None:
        return ""
    return str(value).strip()


def _entry_iter(items: Iterable[Any], section: str, entry_prefix: str) -> Iterable[OpenReviewTextEntry]:
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
                yield OpenReviewTextEntry(
                    source_ref=f"{section}/{entry_prefix} {number}/{key}",
                    text=text,
                )


def iter_openreview_discussion_entries(paper_meta: Any) -> list[OpenReviewTextEntry]:
    if isinstance(paper_meta, dict):
        reviews = paper_meta.get("reviews", [])
        rebuttals = paper_meta.get("rebuttals", [])
        comments = paper_meta.get("comments", [])
    else:
        reviews = getattr(paper_meta, "reviews", [])
        rebuttals = getattr(paper_meta, "rebuttals", [])
        comments = getattr(paper_meta, "comments", [])

    entries = []
    entries.extend(_entry_iter(reviews, "Reviews", "Review"))
    entries.extend(_entry_iter(rebuttals, "Rebuttals", "Rebuttal"))
    entries.extend(_entry_iter(comments, "Comments", "Comment"))
    return entries


def normalize_multimodal_ref(kind: str, label: str) -> str:
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


def extract_multimodal_refs(text: str) -> list[str]:
    refs = []
    for match in MULTIMODAL_REF_PATTERN.finditer(text or ""):
        kind = match.group("kind")
        raw_label = match.group("label")
        if not kind or not raw_label:
            continue
        for label in _expand_ref_labels(raw_label):
            normalized = normalize_multimodal_ref(kind, label)
            if normalized not in refs:
                refs.append(normalized)
    return refs


def is_meaningful_multimodal_discussion(text: str) -> bool:
    lowered = f" {str(text or '').lower()} "
    return any(re.search(r"\b" + re.escape(term) + r"\b", lowered) for term in MEANINGFUL_DISCUSSION_TERMS)


def extract_multimodal_evidence_snippets(
    paper_meta: Any,
) -> list[MultimodalEvidenceSnippet]:
    snippets = []
    for entry in iter_openreview_discussion_entries(paper_meta):
        refs = extract_multimodal_refs(entry.text)
        if not refs:
            continue
        snippets.append(
            MultimodalEvidenceSnippet(
                source_ref=entry.source_ref,
                text=entry.text,
                multimodal_refs=refs,
                meaningful=is_meaningful_multimodal_discussion(entry.text),
            )
        )
    return snippets


def group_multimodal_evidence(
    snippets: Iterable[MultimodalEvidenceSnippet],
    *,
    meaningful_only: bool = False,
) -> list[MultimodalEvidenceGroup]:
    groups: dict[str, MultimodalEvidenceGroup] = {}
    for snippet in snippets:
        if meaningful_only and not snippet.meaningful:
            continue
        for ref in snippet.multimodal_refs:
            groups.setdefault(ref, MultimodalEvidenceGroup(multimodal_ref=ref)).snippets.append(snippet)
    return list(groups.values())


def build_multimodal_filter_diagnostics(
    paper_meta: Any,
    *,
    min_meaningful_snippets: int = 2,
) -> dict[str, Any]:
    snippets = extract_multimodal_evidence_snippets(paper_meta)
    meaningful = [snippet for snippet in snippets if snippet.meaningful]
    groups = group_multimodal_evidence(meaningful)
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
