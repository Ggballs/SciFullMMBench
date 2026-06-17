from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.llm_base import create_openai_compatible_backend, load_llm_config

EXPERIMENT_RESULT_LABEL = "experiment/result"
VIEW_LABELS = ("motivation", "method", EXPERIMENT_RESULT_LABEL)
VALID_LABELS = set(VIEW_LABELS) | {"unclear"}
DEFAULT_MAX_TARGET_PAPERS = 3
DEFAULT_MAX_TARGET_CONTEXT_CHARS = 1800
DEFAULT_GOLDEN_QUERY_PATHS = (
    Path("tests/test_data/query_analysis/IR-type/golden_litsearch-human-queries_quality2_with_targets.jsonl"),
    Path("tests/test_data/query_analysis/IR-type/golden_pasa-realscholarquery_with_targets.jsonl"),
    Path("tests/test_data/query_analysis/QA-type/golden_queries.json"),
)


@dataclass(frozen=True)
class QueryItem:
    query_id: str
    query_type: str
    source_file: str
    row_index: int
    query: str
    target_papers: list[dict[str, str]]


@dataclass(frozen=True)
class QueryClassification:
    query_id: str
    query_type: str
    source_file: str
    row_index: int
    query: str
    target_papers: list[dict[str, str]]
    labels: list[str]
    primary_label: str
    ambiguous: bool
    confidence: float
    rationale: str


@dataclass(frozen=True)
class QueryConsensus:
    query_id: str
    query_type: str
    source_file: str
    row_index: int
    query: str
    target_papers: list[dict[str, str]]
    call1_labels: list[str]
    call1_primary_label: str
    call1_ambiguous: bool
    call1_confidence: float
    call1_rationale: str
    call2_labels: list[str]
    call2_primary_label: str
    call2_ambiguous: bool
    call2_confidence: float
    call2_rationale: str
    agreement_status: str
    bucket: str
    labels: list[str]
    primary_label: str
    ambiguous: bool
    confidence: float
    rationale: str


def discover_golden_query_paths(input_root: Path) -> list[Path]:
    paths = []
    for relative_path in DEFAULT_GOLDEN_QUERY_PATHS:
        path = (REPO_ROOT / relative_path).resolve()
        try:
            path.relative_to(input_root)
        except ValueError:
            continue
        if path.is_file():
            paths.append(path)
    return paths


def _query_type_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "qa-type" in parts:
        return "QA"
    if "ir-type" in parts:
        return "IR"
    return "unknown"


def _safe_id_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value or "query"


def _extract_query_from_row(row: dict[str, Any], query_column: str) -> Optional[str]:
    value = row.get(query_column)
    if value is None and query_column != "query":
        value = row.get("query")
    if value is None:
        return None
    query = str(value).strip()
    return query or None


def _parse_json_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _stringify_target_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _clean_context_text(value: str) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_target_papers(value: Any) -> list[dict[str, str]]:
    value = _parse_json_like(value)
    if value is None or value == "":
        return []
    if isinstance(value, dict):
        value = [value]
    elif not isinstance(value, list):
        value = [{"title": str(value).strip()}]

    papers: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            paper = {
                "title": _stringify_target_value(
                    item.get("title")
                    or item.get("paper_title")
                    or item.get("name")
                    or item.get("answer")
                    or item.get("title")
                ),
                "url": _stringify_target_value(item.get("url") or item.get("link")),
                "score": _stringify_target_value(item.get("score")),
                "paper_urls": _stringify_target_value(item.get("paper_urls") or item.get("paper_url")),
                "source_id": _stringify_target_value(
                    item.get("source_id")
                    or item.get("corpusid")
                    or item.get("corpus_id")
                    or item.get("arxiv_id")
                    or item.get("paper_id")
                ),
                "abstract": _stringify_target_value(item.get("abstract") or item.get("body")),
            }
        else:
            paper = {"title": str(item).strip(), "url": "", "source_id": "", "abstract": ""}
        if any(paper.values()):
            papers.append(paper)
    return papers


def _extract_target_papers_from_row(row: dict[str, Any], target_papers_column: str) -> list[dict[str, str]]:
    candidate_keys = [
        target_papers_column,
        "target_papers",
        "target_paper",
        "targets",
        "answer",
        "answers",
        "qualifying_answers",
        "paper_title",
        "title",
    ]
    for key in candidate_keys:
        if key in row and row[key] not in (None, ""):
            return _normalize_target_papers(row[key])
    return []


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        return rows
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for key in ("queries", "data", "items", "examples", "results"):
                candidate = value.get(key)
                if isinstance(candidate, list):
                    return [item for item in candidate if isinstance(item, dict)]
            return [value]
        return []
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported query file extension: {path.suffix}")


def load_queries(paths: Iterable[Path], query_column: str, target_papers_column: str) -> list[QueryItem]:
    queries: list[QueryItem] = []
    for path in paths:
        query_type = _query_type_from_path(path)
        source_stem = _safe_id_part(path.stem)
        for row_index, row in enumerate(_load_rows(path), 1):
            query = _extract_query_from_row(row, query_column=query_column)
            if not query:
                continue
            queries.append(
                QueryItem(
                    query_id=f"{query_type}_{source_stem}_{row_index:04d}",
                    query_type=query_type,
                    source_file=str(path),
                    row_index=row_index,
                    query=query,
                    target_papers=_extract_target_papers_from_row(
                        row,
                        target_papers_column=target_papers_column,
                    ),
                )
            )
    return queries


def _format_target_papers(target_papers: list[dict[str, str]]) -> str:
    if not target_papers:
        return "No reference answer context provided."
    lines = []
    for index, paper in enumerate(target_papers[:DEFAULT_MAX_TARGET_PAPERS], 1):
        parts = [f"Reference answer {index}"]
        title = paper.get("title", "").strip()
        url = paper.get("url", "").strip()
        score = paper.get("score", "").strip()
        paper_urls = paper.get("paper_urls", "").strip()
        source_id = paper.get("source_id", "").strip()
        abstract = _clean_context_text(paper.get("abstract", ""))
        if title:
            parts.append(f"title={title}")
        if score:
            parts.append(f"score={score}")
        if paper_urls:
            parts.append(f"paper_urls={paper_urls}")
        if source_id:
            parts.append(f"source_id={source_id}")
        if url:
            parts.append(f"url={url}")
        if abstract:
            parts.append(f"answer_body={abstract[:DEFAULT_MAX_TARGET_CONTEXT_CHARS]}")
        lines.append("; ".join(parts))
    if len(target_papers) > DEFAULT_MAX_TARGET_PAPERS:
        lines.append(f"{len(target_papers) - DEFAULT_MAX_TARGET_PAPERS} additional reference answers omitted.")
    return "\n".join(lines)


def _format_query_item(index: int, item: QueryItem) -> str:
    context_label = "Reference answer context" if item.query_type == "QA" else "Target paper context"
    return (
        f"{index}. Query: {item.query}\n"
        f"   {context_label}:\n"
        f"   {_format_target_papers(item.target_papers).replace(chr(10), chr(10) + '   ')}"
    )


def build_classification_prompt(queries: list[QueryItem], prompt_variant: int = 1) -> str:
    numbered = "\n\n".join(_format_query_item(idx, item) for idx, item in enumerate(queries, 1))
    variant_instruction = (
        "Classify each query directly from the codebook and boundary rules."
        if prompt_variant == 1
        else "Independently re-check each query against the same codebook and boundary rules. Focus on the answer-query connection, not surface keywords."
    )
    return f"""
You are an expert research QA annotator. Classify each academic QA query into one or more labels:

M (Motivation)
Me (Method)
E (Experiment/Result)

Use lowercase JSON label values: "motivation", "method", "experiment/result".

Assign multiple labels only when multiple aspects of cited paper content are genuinely needed to explain how the answer addresses the query.

## Pass Instruction
{variant_instruction}

## Label Definitions
M — Motivation:
Assign motivation only when the answer uses the cited paper’s own motivation-side content to address the query. motivation is  WHY a research problem, assumption, metric, limitation, or approach matters in cited paper. This includes problem framing, conceptual justification, research gaps, goals, hypotheses, contribution claims, or reasons something is flawed or needed.

Me — Method:
Assign method when the answer uses the cited paper’s method-side content to address the query. method is HOW an approach works or should be done in cited paper. This includes models, algorithms, procedures, frameworks, pipelines, dataset construction, training/inference steps, raining and inference procedure.

E — Experiment/Result:
Assign experiment/result when the answer uses what the cited paper established, found, proved, measured, compared, analyzed, or empirically observed. experiment/result is WHAT was empirically tested, measured, compared, found, or analyzed in cited paper. This includes evaluation setup, benchmarks, datasets used for testing, baselines, ablations, empirical comparisons, measured performance, observed behavior, qualitative or quantitative analysis findings.

## Core QA Reasoning Principle

Do not label based on the surface wording of the query.
Do not label based on the rhetorical role of the citation in the answer.

Instead, label based on which view of the cited paper’s content is used by the answer to address the query.

In other words, ask:
What kind of content from the cited paper does the answer rely on?
Not:
What kind of explanation is the answer giving?


For each QA item:
1. Read the query to understand what the user is asking.
2. Read the qualifying answer as reference evidence, especially the section where using the cited papers.
3. Identify the cited paper or papers.
For each cited paper, determine which content view from that paper is being used:
- motivation-side content?
- method-side content?
- result-side content?
4. Assign only the labels corresponding to those cited-paper content views.


Return valid JSON only with this shape:
{{
  "results": [
    {{
      "index": 1,
      "labels": ["motivation"],
      "primary_label": "motivation",
      "ambiguous": false,
      "confidence": 0.85,
      "rationale": "Paper name — label
Explain exactly what content from the cited paper is used by the answer and why that content belongs to this label.
Paper name — label
Explain exactly what content from the cited paper is used by the answer and why that content belongs to this label.."
    }}
  ]
}}

Queries:
{numbered}
"""


def _normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    if label in {
        "experiment",
        "experiment_result",
        "embedding_result",
        "experimental_result",
        "experimental_results",
        "result",
        "results",
        "experiments",
        "outcome",
        "outcomes",
        "finding",
        "findings",
    }:
        return EXPERIMENT_RESULT_LABEL
    if label in {"dataset", "method_dataset", "approach", "algorithm", "implementation"}:
        return "method"
    if label in {"contribution", "problem", "gap", "novelty", "goal"}:
        return "motivation"
    return label if label in VALID_LABELS else "unclear"


def _normalize_labels(value: Any) -> list[str]:
    raw_values: list[Any]
    if isinstance(value, list):
        raw_values = value
    elif isinstance(value, str):
        raw_values = re.split(r"[,;/|]+|\band\b", value)
    else:
        raw_values = [value]

    labels: list[str] = []
    for raw_label in raw_values:
        label = _normalize_label(raw_label)
        if label not in labels:
            labels.append(label)

    content_labels = [label for label in labels if label in VIEW_LABELS]
    if content_labels:
        return content_labels
    return ["unclear"]


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 1.0 < confidence <= 100.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    return default


def normalize_classifications(
    queries: list[QueryItem],
    raw_response: dict[str, Any],
) -> list[QueryClassification]:
    raw_items = raw_response.get("results", [])
    if not isinstance(raw_items, list):
        raw_items = []

    by_index: dict[int, dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        by_index[index] = item

    classifications: list[QueryClassification] = []
    for index, query in enumerate(queries, 1):
        item = by_index.get(index, {})
        labels = _normalize_labels(item.get("labels", item.get("label", item.get("view"))))
        primary_label = _normalize_label(item.get("primary_label", item.get("primary", labels[0])))
        if primary_label not in labels:
            primary_label = labels[0]
        ambiguous = _normalize_bool(item.get("ambiguous"), default=len(labels) > 1)
        classifications.append(
            QueryClassification(
                query_id=query.query_id,
                query_type=query.query_type,
                source_file=query.source_file,
                row_index=query.row_index,
                query=query.query,
                target_papers=query.target_papers,
                labels=labels,
                primary_label=primary_label,
                ambiguous=ambiguous,
                confidence=_normalize_confidence(item.get("confidence")),
                rationale=str(item.get("rationale", "")).strip(),
            )
        )
    return classifications


def _extract_json_object_text(content: str) -> str:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in LLM response: {text[:300]}")

    in_string = False
    escaped = False
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError(f"Unclosed JSON object in LLM response: {text[:300]}")


def _parse_llm_json_response(content: str) -> dict[str, Any]:
    json_text = _extract_json_object_text(content)
    try:
        value = json.loads(json_text)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", json_text)
        value = json.loads(repaired)
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON root is not an object")
    return value


def generate_json_response(backend: Any, prompt: str, max_attempts: int = 3) -> dict[str, Any]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, max_attempts) + 1):
        repair_instruction = ""
        if attempt > 1:
            repair_instruction = (
                "\n\nYour previous response was not valid JSON. Return only a single valid JSON object "
                "matching the requested schema. Use double quotes for every string and no markdown fences."
            )
        content = backend.generate(prompt + repair_instruction)
        try:
            return _parse_llm_json_response(content)
        except Exception as exc:
            last_exc = exc
            print(f"Could not parse LLM JSON response on attempt {attempt}/{max_attempts}: {exc}", flush=True)
    raise ValueError(f"Could not parse LLM JSON response after {max_attempts} attempts") from last_exc


def batched(items: list[QueryItem], batch_size: int) -> Iterable[list[QueryItem]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def infer_max_concurrent_requests(backend: Any) -> int:
    request_manager = getattr(backend, "request_manager", None)
    slots = getattr(request_manager, "_slots", None)
    per_key = int(getattr(request_manager, "per_key_max_concurrent_requests", 1) or 1)
    if isinstance(slots, list) and slots:
        return max(1, len(slots) * max(1, per_key))
    return 1


def classify_queries(
    queries: list[QueryItem],
    backend: Any,
    batch_size: int,
    prompt_variant: int = 1,
    max_concurrent_requests: int = 1,
) -> list[QueryClassification]:
    batches = list(batched(queries, max(1, batch_size)))
    if not batches:
        return []
    print(
        f"Starting LLM call pass {prompt_variant}: "
        f"{len(queries)} queries, {len(batches)} batches, "
        f"batch_size={max(1, batch_size)}, max_concurrent_requests={max_concurrent_requests}",
        flush=True,
    )

    def classify_batch(batch_index: int, batch: list[QueryItem]) -> tuple[int, list[QueryClassification]]:
        raw_response = generate_json_response(backend, build_classification_prompt(batch, prompt_variant=prompt_variant))
        return batch_index, normalize_classifications(batch, raw_response)

    max_workers = max(1, min(int(max_concurrent_requests), len(batches)))
    if max_workers == 1:
        classifications: list[QueryClassification] = []
        for index, batch in enumerate(batches):
            _, batch_classifications = classify_batch(index, batch)
            classifications.extend(batch_classifications)
            print(
                f"Finished pass {prompt_variant} batch {index + 1}/{len(batches)}",
                flush=True,
            )
        return classifications

    by_batch_index: dict[int, list[QueryClassification]] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(classify_batch, batch_index, batch): batch_index
            for batch_index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_index, batch_classifications = future.result()
            by_batch_index[batch_index] = batch_classifications
            completed += 1
            print(
                f"Finished pass {prompt_variant} batch {completed}/{len(batches)}",
                flush=True,
            )

    classifications = []
    for batch_index in range(len(batches)):
        classifications.extend(by_batch_index[batch_index])
    return classifications


def label_set_key(labels: list[str]) -> tuple[str, ...]:
    content_labels = [label for label in labels if label in VIEW_LABELS]
    if content_labels:
        return tuple(sorted(set(content_labels)))
    return ("unclear",)


def _classifications_by_id(
    classifications: list[QueryClassification],
) -> dict[str, QueryClassification]:
    return {item.query_id: item for item in classifications}


def _bucket_for_pair(
    left: QueryClassification,
    right: QueryClassification,
    high_confidence: float,
) -> tuple[str, str]:
    if label_set_key(left.labels) != label_set_key(right.labels):
        return "conflict", "conflict"
    if left.confidence >= high_confidence and right.confidence >= high_confidence:
        return "match", "high_confidence"
    return "match", "low_confidence"


def build_consensus(
    queries: list[QueryItem],
    call1: list[QueryClassification],
    call2: list[QueryClassification],
    high_confidence: float,
) -> list[QueryConsensus]:
    call1_by_id = _classifications_by_id(call1)
    call2_by_id = _classifications_by_id(call2)
    consensus: list[QueryConsensus] = []
    for query in queries:
        left = call1_by_id[query.query_id]
        right = call2_by_id[query.query_id]
        agreement_status, bucket = _bucket_for_pair(left, right, high_confidence)
        labels = list(left.labels)
        primary_label = left.primary_label
        if bucket == "conflict":
            union = sorted(set(left.labels) | set(right.labels))
            labels = [label for label in union if label in VIEW_LABELS] or ["unclear"]
            primary_label = left.primary_label
        if primary_label not in labels:
            primary_label = labels[0]
        confidence = min(left.confidence, right.confidence)
        rationale = (
            f"Call 1: {left.rationale} | Call 2: {right.rationale}"
            if bucket == "conflict"
            else left.rationale
        )
        consensus.append(
            QueryConsensus(
                query_id=query.query_id,
                query_type=query.query_type,
                source_file=query.source_file,
                row_index=query.row_index,
                query=query.query,
                target_papers=query.target_papers,
                call1_labels=left.labels,
                call1_primary_label=left.primary_label,
                call1_ambiguous=left.ambiguous,
                call1_confidence=left.confidence,
                call1_rationale=left.rationale,
                call2_labels=right.labels,
                call2_primary_label=right.primary_label,
                call2_ambiguous=right.ambiguous,
                call2_confidence=right.confidence,
                call2_rationale=right.rationale,
                agreement_status=agreement_status,
                bucket=bucket,
                labels=labels,
                primary_label=primary_label,
                ambiguous=left.ambiguous or right.ambiguous or bucket == "conflict",
                confidence=confidence,
                rationale=rationale,
            )
        )
    return consensus


def distribution(classifications: list[QueryClassification]) -> dict[str, Any]:
    primary_counts = Counter(item.primary_label for item in classifications)
    multilabel_counts = Counter(label for item in classifications for label in item.labels)
    type_counts = Counter(item.query_type for item in classifications)
    total = len(classifications)
    return {
        "total_queries": total,
        "query_types": dict(sorted(type_counts.items())),
        "ambiguous": sum(1 for item in classifications if item.ambiguous),
        "primary_labels": {
            label: {
                "count": int(primary_counts.get(label, 0)),
                "ratio": (primary_counts.get(label, 0) / total) if total else 0.0,
            }
            for label in sorted(VALID_LABELS)
        },
        "multi_label_counts": {
            label: int(multilabel_counts.get(label, 0))
            for label in sorted(VALID_LABELS)
        },
    }


def consensus_distribution(consensus: list[QueryConsensus]) -> dict[str, Any]:
    primary_counts = Counter(item.primary_label for item in consensus)
    multilabel_counts = Counter(label for item in consensus for label in item.labels)
    type_counts = Counter(item.query_type for item in consensus)
    bucket_counts = Counter(item.bucket for item in consensus)
    agreement_counts = Counter(item.agreement_status for item in consensus)
    total = len(consensus)
    return {
        "total_queries": total,
        "query_types": dict(sorted(type_counts.items())),
        "buckets": dict(sorted(bucket_counts.items())),
        "agreement_status": dict(sorted(agreement_counts.items())),
        "ambiguous": sum(1 for item in consensus if item.ambiguous),
        "primary_labels": {
            label: {
                "count": int(primary_counts.get(label, 0)),
                "ratio": (primary_counts.get(label, 0) / total) if total else 0.0,
            }
            for label in sorted(VALID_LABELS)
        },
        "multi_label_counts": {
            label: int(multilabel_counts.get(label, 0))
            for label in sorted(VALID_LABELS)
        },
    }


def classification_to_row(item: QueryClassification) -> dict[str, Any]:
    row = asdict(item)
    row["labels"] = "|".join(item.labels)
    row["target_papers"] = json.dumps(item.target_papers, ensure_ascii=False)
    return row


def consensus_to_row(item: QueryConsensus) -> dict[str, Any]:
    row = asdict(item)
    for key in ("call1_labels", "call2_labels", "labels"):
        row[key] = "|".join(row[key])
    row["target_papers"] = json.dumps(item.target_papers, ensure_ascii=False)
    return row


def review_row(item: QueryClassification | QueryConsensus) -> dict[str, Any]:
    if isinstance(item, QueryConsensus):
        llm_rationale = item.rationale
        return {
            "query_id": item.query_id,
            "query_type": item.query_type,
            "source_file": item.source_file,
            "row_index": item.row_index,
            "query": item.query,
            "triage_bucket": item.bucket,
            "agreement_status": item.agreement_status,
            "llm_primary_label": item.primary_label,
            "llm_labels": "|".join(item.labels),
            "llm_ambiguous": item.ambiguous,
            "llm_confidence": item.confidence,
            "llm_rationale": llm_rationale,
            "call1_primary_label": item.call1_primary_label,
            "call1_labels": "|".join(item.call1_labels),
            "call1_confidence": item.call1_confidence,
            "call1_rationale": item.call1_rationale,
            "call2_primary_label": item.call2_primary_label,
            "call2_labels": "|".join(item.call2_labels),
            "call2_confidence": item.call2_confidence,
            "call2_rationale": item.call2_rationale,
            "human_motivation": "",
            "human_method": "",
            "human_experiment": "",
            "human_ambiguous": "",
            "human_primary_label": "",
            "human_confidence": "",
            "human_decision": "",
            "human_notes": "",
            "target_papers": json.dumps(item.target_papers, ensure_ascii=False),
        }
    return {
        "query_id": item.query_id,
        "query_type": item.query_type,
        "source_file": item.source_file,
        "row_index": item.row_index,
        "query": item.query,
        "llm_primary_label": item.primary_label,
        "llm_labels": "|".join(item.labels),
        "llm_ambiguous": item.ambiguous,
        "llm_confidence": item.confidence,
        "llm_rationale": item.rationale,
        "human_motivation": "",
        "human_method": "",
        "human_experiment": "",
        "human_ambiguous": "",
        "human_primary_label": "",
        "human_confidence": "",
        "human_decision": "",
        "human_notes": "",
        "target_papers": json.dumps(item.target_papers, ensure_ascii=False),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_backend(config_path: Path):
    llm_config = load_llm_config(config_path)
    return create_openai_compatible_backend(
        base_url=str(llm_config["base_url"]),
        api_tokens=[str(token) for token in llm_config["api_tokens"]],
        model=str(llm_config["model"]),
        max_tokens=int(llm_config.get("max_tokens", 4096)),
        temperature=float(llm_config.get("temperature", 0.0)),
        seed=llm_config.get("seed"),
        embedding_model=llm_config.get("embedding_model"),
        per_key_request_interval_seconds=float(llm_config.get("per_key_request_interval_seconds", 0.0)),
        per_key_max_concurrent_requests=int(llm_config.get("per_key_max_concurrent_requests", 1)),
        max_retries=int(llm_config.get("max_retries", 3)),
        retry_backoff_seconds=float(llm_config.get("retry_backoff_seconds", 8.0)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify golden QA/IR queries into motivation/method/experiment views with target paper context."
    )
    parser.add_argument(
        "--input-root",
        default="tests/test_data/query_analysis",
        help="Root containing QA-type and IR-type golden query files.",
    )
    parser.add_argument(
        "--golden-queries",
        nargs="*",
        default=None,
        help=(
            "Optional explicit JSONL/JSON/CSV golden query files. Defaults to the two "
            "IR with-target files plus QA-type/golden_queries.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/query_analysis/golden_view_classification",
        help="Directory for classification and human-review outputs.",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml.")
    parser.add_argument("--query-column", default="query", help="Column/key containing query text.")
    parser.add_argument(
        "--target-papers-column",
        default="target_papers",
        help="Column/key containing target paper metadata. Supports JSON list/dict strings in CSV.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Queries per LLM request.")
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=0,
        help="Parallel LLM requests per call pass. 0 uses one request slot per configured API key.",
    )
    parser.add_argument(
        "--num-calls",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of independent LLM calls per query.",
    )
    parser.add_argument(
        "--high-confidence",
        type=float,
        default=0.8,
        help="Threshold for both calls to enter the high-confidence accepted bucket.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.golden_queries:
        paths = [Path(path).expanduser().resolve() for path in args.golden_queries]
    else:
        paths = discover_golden_query_paths(input_root)
    if not paths:
        raise ValueError(f"No golden query files found under {input_root}")

    queries = load_queries(
        paths,
        query_column=args.query_column,
        target_papers_column=args.target_papers_column,
    )
    if not queries:
        raise ValueError("No queries were loaded from the requested golden files.")

    backend = build_backend(Path(args.config).expanduser().resolve())
    max_concurrent_requests = (
        infer_max_concurrent_requests(backend)
        if int(args.max_concurrent_requests) <= 0
        else int(args.max_concurrent_requests)
    )
    call1 = classify_queries(
        queries,
        backend,
        batch_size=args.batch_size,
        prompt_variant=1,
        max_concurrent_requests=max_concurrent_requests,
    )
    call1_rows = [classification_to_row(item) for item in call1]
    write_jsonl(output_dir / "golden_query_view_call1_classifications.jsonl", call1_rows)
    write_csv(output_dir / "golden_query_view_call1_classifications.csv", call1_rows)

    if args.num_calls == 1:
        write_jsonl(output_dir / "golden_query_view_classifications.jsonl", call1_rows)
        write_csv(output_dir / "golden_query_view_classifications.csv", call1_rows)
        write_json(output_dir / "golden_query_view_distribution.json", distribution(call1))
        write_csv(
            output_dir / "golden_query_view_review_sample.csv",
            [review_row(item) for item in call1],
        )
        print(f"Loaded {len(queries)} queries from {len(paths)} files.")
        print(f"Saved one-call classifications and full review file to {output_dir}")
        return 0

    call2 = classify_queries(
        queries,
        backend,
        batch_size=args.batch_size,
        prompt_variant=2,
        max_concurrent_requests=max_concurrent_requests,
    )
    call2_rows = [classification_to_row(item) for item in call2]
    consensus = build_consensus(
        queries,
        call1,
        call2,
        high_confidence=max(0.0, min(1.0, float(args.high_confidence))),
    )
    consensus_rows = [consensus_to_row(item) for item in consensus]

    write_jsonl(output_dir / "golden_query_view_call2_classifications.jsonl", call2_rows)
    write_csv(output_dir / "golden_query_view_call2_classifications.csv", call2_rows)
    write_jsonl(output_dir / "golden_query_view_consensus.jsonl", consensus_rows)
    write_csv(output_dir / "golden_query_view_consensus.csv", consensus_rows)
    write_json(output_dir / "golden_query_view_distribution.json", consensus_distribution(consensus))
    write_csv(
        output_dir / "golden_query_view_review_sample.csv",
        [review_row(item) for item in consensus],
    )

    print(f"Loaded {len(queries)} queries from {len(paths)} files.")
    print(f"Saved two-call classifications, triage buckets, and full review file to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
