"""LLM-as-a-judge style metrics for academic search queries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


SPECIFICITY_CALIBRATION_SCALE = (
    "1=severely under-specified, 2=under-specified, 3=well-calibrated (ideal), "
    "4=over-specified, 5=pathologically over-specified"
)
LEXICAL_NATURALISM_SCALE = (
    "1=keyword dump / fragmented, 2=stilted or awkward, 3=natural researcher register (ideal), "
    "4=over-formalized / essay-register, 5=synthetic fluent / LLM-polished prose"
)
FIT_SCALE = "0-1 closeness to ideal centered score of 3, computed as 1 - abs(score - 3) / 2"


def load_llm_config(config_path: Path) -> Dict[str, Any]:
    """Load OpenAI-compatible LLM settings from config.yaml."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    llm_config = config.get("llm", {})
    result = {
        "base_url": llm_config.get("base_url", ""),
        "api_token": llm_config.get("api_token", ""),
        "model": llm_config.get("model", "gpt-4o-mini"),
        "seed": llm_config.get("seed"),
    }
    if not result["base_url"] or not result["api_token"]:
        raise ValueError(
            f"Missing llm.base_url or llm.api_token in {config_path}; "
            "LLM-as-a-judge metrics require a configured API."
        )
    return result


def _normalize_judge_mode(judge_mode: str) -> str:
    normalized = str(judge_mode).strip().lower()
    if normalized not in {"batch", "single_query"}:
        raise ValueError("judge_mode must be 'batch' or 'single_query'")
    return normalized


def _build_batches(queries: List[str], batch_size: int, judge_mode: str) -> List[tuple[int, List[str]]]:
    effective_batch_size = 1 if _normalize_judge_mode(judge_mode) == "single_query" else max(1, batch_size)
    return [
        (start, queries[start:start + effective_batch_size])
        for start in range(0, len(queries), effective_batch_size)
    ]


def _score_batches(
    queries: List[str],
    base_url: str,
    api_token: str,
    model: str,
    batch_size: int,
    judge_mode: str,
    max_concurrency: int,
    seed: Optional[int],
    prompt_builder: Any,
    result_normalizer: Any,
    log_label: str,
) -> List[Dict[str, Any]]:
    from openreview_pipeline.llm import OpenAICompatibleBackend

    judge_mode = _normalize_judge_mode(judge_mode)
    max_concurrency = max(1, int(max_concurrency))
    batches = _build_batches(queries, batch_size=batch_size, judge_mode=judge_mode)

    def _score_single_batch(start: int, batch: List[str]) -> List[Dict[str, Any]]:
        backend = OpenAICompatibleBackend(
            base_url=base_url,
            api_token=api_token,
            model=model,
            temperature=0.0,
            seed=seed,
        )
        logger.info(
            "LLM %s scoring queries %s-%s/%s (%s)",
            log_label,
            start + 1,
            start + len(batch),
            len(queries),
            judge_mode,
        )
        raw_results = backend.generate_json(prompt_builder(batch))
        batch_results = result_normalizer(batch, raw_results)
        for item in batch_results:
            item["index"] = start + item["index"]
        return batch_results

    if max_concurrency == 1 or len(batches) <= 1:
        per_query: List[Dict[str, Any]] = []
        for start, batch in batches:
            per_query.extend(_score_single_batch(start, batch))
        return per_query

    per_query = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(_score_single_batch, start, batch): start
            for start, batch in batches
        }
        for future in as_completed(futures):
            per_query.extend(future.result())

    per_query.sort(key=lambda item: item.get("index", 0))
    return per_query


def build_query_judge_prompt(queries: List[str]) -> str:
    """Build a batch prompt for style-only academic query scoring."""
    numbered_queries = "\n".join(
        f"{idx}. {query}" for idx, query in enumerate(queries, start=1)
    )
    return f"""
You are an expert judge of academic paper-search queries.

You will score each query on two metrics: Specificity Calibration and Lexical Naturalism.
Read the metric definitions and scoring rubrics carefully before scoring.

Your Task
Given a query, score it on each metric independently. For each metric, output:
Score (integer)
Rationale (2-4 sentences explaining the score with reference to specific textual evidence in the query)

Judge only style and human-likeness.
Do NOT judge retrieval correctness, paper relevance, or whether the paper actually exists.
Focus only on:
1. Specificity Calibration
2. Lexical Naturalism

Important:
- Both metrics use a bipolar 1-5 scale.
- For BOTH metrics, score 3 is ideal.
- Score 1 and score 5 are opposite failure modes.
- Do not assume that a higher score is better.
- The examples are illustrative anchors, not templates to match literally.

Metric 1. Specificity Calibration
Definition:
Is the level of detail calibrated like a human-written researcher query?

What to judge:
- whether the query gives enough detail to guide retrieval
- whether it avoids being too broad
- whether it avoids becoming so detailed that it looks like reconstruction of a known paper rather than genuine search

Interpret this as a centered scale:
1 = far too broad
2 = somewhat too broad
3 = well calibrated
4 = somewhat too specific
5 = pathologically over-specific

Score 1 — Severely Under-specified
Description:
The query is so broad that it cannot guide retrieval in a meaningful way. It names a topic or domain but gives no useful filtering criterion.
Markers:
- almost any paper in the field would count
- no operational narrowing condition
- reads like a topic label or keyword area
Example:
"Give me papers about large language models."

Score 2 — Under-specified
Description:
The query identifies a real research direction, but lacks enough detail to distinguish the papers the researcher actually wants from a large amount of related work.
Markers:
- one broad topic is present
- some filtering is implied, but still too weak
- method, task, or setting is mentioned, but not enough to focus results well
Example:
"Give me papers about how to rank search results by the use of LLM."

Score 3 — Well-calibrated
Description:
The query is specific enough to constrain retrieval meaningfully, but not so narrow that it presupposes the answer. It reflects genuine uncertainty: the researcher knows what kind of paper they want, but not which paper.
Markers:
- multiple meaningful constraints interact
- the query implies a category of papers, not just one known paper
- results would need evaluation, not mere lookup
Example:
"Are there any large-scale and open-source text simplification datasets dealing with long passages?"

Score 4 — Over-specified
Description:
The query contains enough constraints that it starts to look like external reconstruction of a particular paper or a tiny handful of papers, rather than open-ended search.
Markers:
- several highly specific constraints
- unlikely combination of details
- reads more like remembered paper content than a normal search
Example:
"Find the NLP paper that focuses on dialogue generation and introduces advancements in the augmentation of one-to-many or one-to-one dialogue data by conducting augmentation within the semantic space."

Score 5 — Pathologically Over-specified
Description:
The query encodes so many specific details, exclusions, or methodological requirements that it reads like a structured filter over already-known papers, not a real search under uncertainty.
Markers:
- multiple detailed technical requirements across several dimensions
- exclusions or filtering clauses that feel review-like
- reads like a paraphrased abstract, requirements spec, or literature-review entry
Example:
"I am looking for the paper that builds a multimodal foundation model with visual, audio, and audio-visual pretraining data, excludes survey papers, and evaluates long-form audiovisual understanding across multiple benchmarks."

Metric 2. Lexical Naturalism
Definition:
Is the vocabulary, phrasing, and register consistent with how researchers actually write queries?

What to judge:
- whether the query sounds like something a real researcher would type
- whether the syntax is natural and query-like
- whether the register is appropriately academic without becoming essay-like or LLM-polished

Interpret this as a centered scale:
1 = keyword dump / fragmented
2 = awkward or stilted
3 = natural researcher register
4 = over-formalized / essay-like
5 = synthetic fluent / LLM-polished

Score 1 — Keyword Dump / Machine-like Fragmentation
Description:
The query is just a string of terms with no real syntactic structure. It reads like extracted keywords or metadata, not natural language.
Markers:
- no real sentence structure
- no connective logic
- noun phrases only
- may resemble SEO tags or structured search fragments
Example:
"LLM agent reinforcement learning reward shaping training evaluation benchmark"

Score 2 — Stilted / Over-compressed Natural Language
Description:
The query uses natural language, but it is awkward, compressed, or slightly malformed in ways that make it feel clunky rather than fluent.
Markers:
- missing articles or prepositions
- awkward conversational filler
- grammatically off or compressed phrasing
- recognizably human, but not well-formed
Example:
"Do you know some papers about using reward shaping methods to train large language model agent."

Score 3 — Natural Researcher Register
Description:
The query reads like something a researcher would actually type or say. The vocabulary is appropriately technical without being performatively formal.
Markers:
- direct and purposive
- fluent but not polished
- technical terms used naturally
- feels like someone searching while doing work
Example:
"Are there any papers that build dense retrievers with mixture-of-experts architecture where each expert is responsible for different types of queries?"

Score 4 — Over-formalized / Essay-register
Description:
The query is grammatically correct and technically fluent, but written in a register that is too formal or composed for search behavior. It sounds like prose, not a query.
Markers:
- full sentence or multi-sentence structure
- polite instruction wording
- requirement-spec style
- reads like writing rather than searching
Example:
"I am looking for research papers on the construction of multimodal foundation models that support both visual and audio inputs. These models should be pre-trained on large-scale datasets, including visual, audio, and audio-visual data. Please exclude survey papers."

Score 5 — Synthetic Fluency / LLM-polished Prose
Description:
The query is too clean, complete, and balanced to feel like real human search behavior. It reads like an LLM completing a prompt to write an idealized search query.
Markers:
- perfectly balanced clause structure
- every dimension of the topic is explicitly accounted for
- unusually complete, polished, and symmetric
- lacks the roughness typical of real human search phrasing
Example:
"Please provide scholarly works demonstrating that smaller, carefully curated pre-training datasets can yield superior large language models relative to larger corpora."

Keep the two metrics distinct:
- Specificity Calibration = whether the amount of detail is calibrated like a human query
- Lexical Naturalism = whether the wording and register sound like a human query

A query can score:
- high on one metric and low on the other
- differently on the two metrics for different reasons

Queries:
{numbered_queries}

Return JSON only in this exact format:
{{
  "results": [
    {{
      "query": "...",
      "metric_1_specificity_calibration": {{
        "score": 3,
        "rationale": "2-4 sentences citing specific textual evidence from the query."
      }},
      "metric_2_lexical_naturalism": {{
        "score": 2,
        "rationale": "2-4 sentences citing specific textual evidence from the query."
      }}
    }}
  ]
}}
""".strip()


def build_semantic_constraint_prompt(queries: List[str]) -> str:
    """Build a batch prompt for LLM semantic constraint counting."""
    numbered_queries = "\n".join(
        f"{idx}. {query}" for idx, query in enumerate(queries, start=1)
    )
    return f"""
You are evaluating academic paper-search queries.

For each query, identify semantic retrieval constraints. A constraint is any condition
that narrows what paper should be retrieved, even if it is not expressed with words
like "for" or "with".

Count constraints semantically, not by keyword. Examples of constraints:
- task or setting constraints
- method/model constraints
- dataset/benchmark constraints
- comparison/result constraints
- scope/exclusion constraints
- visual/multimodal constraints

Use a conservative count:
- 0 = generic query with no real narrowing condition
- 1 = one main narrowing condition
- 2 = two distinct narrowing conditions
- 3 = three distinct narrowing conditions
- 4 = four distinct narrowing conditions

Output the exact semantic constraint count as a non-negative integer: 0, 1, 2, 3, 4, ...

Counting rule:
- Count distinct retrieval-narrowing conditions, not words.
- Do not split one idea into multiple counts.
- Count a phrase only if removing it would materially broaden the search.

Examples:
- 0: "papers about large language models"
  Reason: topic only, no real retrieval filter.
- 1: "papers on RLHF for hallucination reduction"
  Reason: one main narrowing condition around hallucination reduction.
- 2: "papers on dense retrieval for biomedical question answering"
  Reason: method constraint + task/domain constraint.
- 3: "papers on dense retrieval for biomedical question answering on low-resource datasets"
  Reason: method + task/domain + dataset/scope constraint.
- 4: "papers on dense retrieval for biomedical question answering on low-resource datasets with zero-shot evaluation"
  Reason: method + task/domain + dataset/scope + evaluation setting.

Queries:
{numbered_queries}

Return valid JSON only in this exact shape:
{{
  "results": [
    {{
      "index": 1,
      "semantic_constraint_count": 2
    }}
  ]
}}
""".strip()


def _parse_score_1_to_5(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= parsed <= 5:
        return parsed
    return None


def _centered_fit(score: Optional[int], ideal: int = 3) -> Optional[float]:
    if score is None:
        return None
    return max(0.0, 1.0 - (abs(score - ideal) / 2.0))


def _summary_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def normalize_judge_results(
    queries: List[str],
    raw_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Normalize judge output into one result per query."""
    raw_items = raw_results.get("results", [])
    if not isinstance(raw_items, list):
        raw_items = []

    normalized = []
    for index, query in enumerate(queries, start=1):
        item = raw_items[index - 1] if index - 1 < len(raw_items) and isinstance(raw_items[index - 1], dict) else {}

        specificity_obj = item.get("metric_1_specificity_calibration", {})
        if not isinstance(specificity_obj, dict):
            specificity_obj = {}

        lexical_obj = item.get("metric_2_lexical_naturalism", {})
        if not isinstance(lexical_obj, dict):
            lexical_obj = {}

        specificity_score = _parse_score_1_to_5(specificity_obj.get("score"))
        lexical_score = _parse_score_1_to_5(lexical_obj.get("score"))

        normalized.append(
            {
                "index": index,
                "query": query,
                "specificity_calibration_score": specificity_score,
                "specificity_calibration_rationale": str(
                    specificity_obj.get("rationale", "")
                ).strip(),
                "specificity_calibration_fit": _centered_fit(specificity_score),
                "lexical_naturalism_score": lexical_score,
                "lexical_naturalism_rationale": str(
                    lexical_obj.get("rationale", "")
                ).strip(),
                "lexical_naturalism_fit": _centered_fit(lexical_score),
            }
        )

    return normalized


def summarize_judge_results(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize per-query LLM judge results into style metric shapes."""
    if not per_query:
        return {
            "qualitative_metrics": {
                "method": "llm_as_judge",
            },
            "llm_judge": {
                "method": "llm_as_judge",
                "per_query": [],
            },
        }

    specificity_scores = [
        float(item["specificity_calibration_score"])
        for item in per_query
        if item.get("specificity_calibration_score") is not None
    ]
    specificity_fits = [
        float(item["specificity_calibration_fit"])
        for item in per_query
        if item.get("specificity_calibration_fit") is not None
    ]
    lexical_scores = [
        float(item["lexical_naturalism_score"])
        for item in per_query
        if item.get("lexical_naturalism_score") is not None
    ]
    lexical_fits = [
        float(item["lexical_naturalism_fit"])
        for item in per_query
        if item.get("lexical_naturalism_fit") is not None
    ]

    qualitative_metrics = {
        "method": "llm_as_judge",
        "specificity_calibration": {
            "scale": SPECIFICITY_CALIBRATION_SCALE,
            "ideal_score": 3,
            **_summary_stats(specificity_scores),
        },
        "specificity_calibration_fit": {
            "scale": FIT_SCALE,
            "ideal_score": 1.0,
            **_summary_stats(specificity_fits),
        },
        "lexical_naturalism": {
            "scale": LEXICAL_NATURALISM_SCALE,
            "ideal_score": 3,
            **_summary_stats(lexical_scores),
        },
        "lexical_naturalism_fit": {
            "scale": FIT_SCALE,
            "ideal_score": 1.0,
            **_summary_stats(lexical_fits),
        },
    }

    return {
        "qualitative_metrics": qualitative_metrics,
        "llm_judge": {
            "method": "llm_as_judge",
            "per_query": per_query,
        },
    }


def normalize_semantic_constraint_results(
    queries: List[str],
    raw_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Normalize semantic constraint output into one result per query."""
    by_index: Dict[int, Dict[str, Any]] = {}
    for item in raw_results.get("results", []):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        by_index[index] = item

    normalized = []
    for index, query in enumerate(queries, start=1):
        item = by_index.get(index, {})
        try:
            count = int(item.get("semantic_constraint_count", 0))
        except (TypeError, ValueError):
            count = 0
        count = max(0, count)

        normalized.append(
            {
                "index": index,
                "query": query,
                "has_constraint": count > 0,
                "semantic_constraint_count": count,
            }
        )

    return normalized


def summarize_semantic_constraints(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize per-query semantic constraint judgments."""
    if not per_query:
        empty_constraint_stats = {
            "method": "llm_as_judge",
            "scale": "exact non-negative integer semantic constraint count",
            "constraints_per_query": {},
            "queries_with_constraints": 0,
            "constraint_ratio": 0,
        }
        return {
            "constraint_count": empty_constraint_stats,
            "semantic_constraint_analysis": {
                "method": "llm_as_judge",
                "scale": empty_constraint_stats["scale"],
                "semantic_constraints_per_query": {},
                "queries_with_constraints": 0,
                "constraint_ratio": 0,
                "per_query": [],
            },
        }

    counts = [item["semantic_constraint_count"] for item in per_query]
    queries_with_constraints = sum(1 for item in per_query if item["has_constraint"])
    stats = {
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts)),
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "median": float(np.median(counts)),
    }
    scale = "exact non-negative integer semantic constraint count"

    return {
        "constraint_count": {
            "method": "llm_as_judge",
            "scale": scale,
            "constraints_per_query": stats,
            "queries_with_constraints": queries_with_constraints,
            "constraint_ratio": queries_with_constraints / len(per_query),
        },
        "semantic_constraint_analysis": {
            "method": "llm_as_judge",
            "scale": scale,
            "semantic_constraints_per_query": stats,
            "queries_with_constraints": queries_with_constraints,
            "constraint_ratio": queries_with_constraints / len(per_query),
            "per_query": per_query,
        },
    }


def compute_llm_judged_metrics(
    queries: List[str],
    base_url: str,
    api_token: str,
    model: str,
    batch_size: int = 25,
    judge_mode: str = "batch",
    max_concurrency: int = 1,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run LLM-as-a-judge scoring and return metric dictionaries."""
    per_query = _score_batches(
        queries=queries,
        base_url=base_url,
        api_token=api_token,
        model=model,
        batch_size=batch_size,
        judge_mode=judge_mode,
        max_concurrency=max_concurrency,
        seed=seed,
        prompt_builder=build_query_judge_prompt,
        result_normalizer=normalize_judge_results,
        log_label="judge",
    )
    return summarize_judge_results(per_query)


def compute_llm_semantic_constraint_metrics(
    queries: List[str],
    base_url: str,
    api_token: str,
    model: str,
    batch_size: int = 25,
    judge_mode: str = "batch",
    max_concurrency: int = 1,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run LLM semantic constraint counting and return metric dictionaries."""
    per_query = _score_batches(
        queries=queries,
        base_url=base_url,
        api_token=api_token,
        model=model,
        batch_size=batch_size,
        judge_mode=judge_mode,
        max_concurrency=max_concurrency,
        seed=seed,
        prompt_builder=build_semantic_constraint_prompt,
        result_normalizer=normalize_semantic_constraint_results,
        log_label="semantic constraint",
    )
    return summarize_semantic_constraints(per_query)
