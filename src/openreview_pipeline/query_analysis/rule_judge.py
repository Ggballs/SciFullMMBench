from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


QUESTION_TEMPLATES = [
    (r"^is there (a |any )?(paper|work|method|model|dataset|tool|study|approach|technique|system|decoder-|knowledge graph|vision-language|backdoor|multimodal|audio|video|numerical reasoning)?( that | which | )?", "Is there a paper/method that/which..."),
    (r"^is there any (paper|work|method|model|dataset|tool|study|approach|technique|system) ", "Is there a paper/method that/which..."),
    (r"^is there such a? (paper|work|method|model|dataset|tool|study|approach|technique|system)?( that | which | )?", "Is there a paper/method that/which..."),
    (r"^are there (any |)?(([a-z0-9-]+ )*)benchmark papers? ", "Is there a paper/method that/which..."),
    (r"^are there (any |)?(([a-z0-9-]+ )*)(benchmark|benchmarks|dataset|datasets|study|studies|paper|papers|work|works|method|methods|model|models|approach|approaches) ", "Is there a paper/method that/which..."),
    (r"^are there (any |)?(sequential |papers? |methods? |models? |approaches? )", "Is there a paper/method that/which..."),
    (r"^are there any papers? (that |which |on |about )?", "Is there a paper/method that/which..."),
    (r"^any papers? (on |about |regarding |that |which )?", "Is there a paper/method that/which..."),
    (r"^which (([a-z0-9-]+ )*)(paper|papers|study|studies|benchmark|benchmarks|dataset|datasets|method|methods|model|models|work|works)( first |proposes |introduces |presents |shows |finds |found |uses |is |are |was |were |about |on |)", "What [open-source/latest/first] method..."),
    (r"^which paper(s| research| vision-language| backdoor| numerical| knowledge graph|)?( first |proposes |introduces |presents |is | |about |on |)", "What [open-source/latest/first] method..."),
    (r"^which (is |are |was |were |)?the (first |)?", "What [open-source/latest/first] method..."),
    (r"^which (knowledge graph|vision-language|multimodal|backdoor|audio|video|benchmark|study|method|model|approach|work|dataset|technique|system|decoder|numerical reasoning|numeral|paper|papers|research) (first |proposes |introduces |presents |shows |finds |found |uses |is |was |focuses |developed |can |published |)", "What [open-source/latest/first] method..."),
    (r"^which numerical reasoning paper", "What [open-source/latest/first] method..."),
    (r"^what('s| is| are| was| were|') ", "What [open-source/latest/first] method..."),
    (r"^what (open-source |latest |recent )?(([a-z0-9-]+ )*)(paper|work|method|model|dataset|benchmark|study|papers|studies|benchmarks|datasets) ", "What [open-source/latest/first] method..."),
    (r"^what (limitations|ability|capabilities) (do|does|are|is)? ?", "What [open-source/latest/first] method..."),
    (r"^give me (all |any |some |)?papers? (that |which |on |about |showing |demonstrating |)", "Give me papers [that/on/about]..."),
    (r"^give me all (visual |)?(LLM|model|papers?|multimodal |)", "Give me papers [that/on/about]..."),
    (r"^show me (all |some |cutting edge |)?(research |papers? |studies? |work |)", "Give me papers [that/on/about]..."),
    (r"^show me (research |papers? |studies? |work ) (on |about |regarding )?", "Give me papers [that/on/about]..."),
    (r"^provide (me with |)(all |any )?papers? ", "Give me papers [that/on/about]..."),
    (r"^provide (me with |)(all |any )?(papers?|research|studies) ", "Give me papers [that/on/about]..."),
    (r"^list (all |any )?papers? ", "Give me papers [that/on/about]..."),
    (r"^find (me |all |any )?papers? ", "Give me papers [that/on/about]..."),
    (r"^all papers (on |about |regarding |)", "Give me papers [that/on/about]..."),
    (r"^papers? (that |which |on |about |proposing |presenting |introducing )?", "Give me papers [that/on/about]..."),
    (r"^research (on |about |regarding )?", "Give me papers [that/on/about]..."),
    (r"^search for (papers? |research |)", "Give me papers [that/on/about]..."),
    (r"^how to ", "How to [achieve/do]..."),
    (r"^how does ", "How does [X work]..."),
    (r"^how do ", "How do [X work]..."),
    (r"^how can ", "How can [we/X]..."),
    (r"^why ", "Why [does/is X]..."),
    (r"^can ", "Can [we/X]..."),
    (r"^can i ", "Can I..."),
    (r"^could ", "Could [we/X]..."),
    (r"^help me (search |find |)", "Help me search/find..."),
    (r"^i am looking for ", "I am looking for..."),
    (r"^i need (papers? |research |studies? ) (on |about |regarding )?", "I need papers on/about..."),
    (r"^i would like (to |)", "I would like to..."),
    (r"^using ", "Using [X for Y]..."),
    (r"^does ", "Does [X work/affect]..."),
    (r"^do ", "Do [X/they]..."),
    (r"^if one (would like |wants |)", "If one would like/wants..."),
    (r"^who ", "Who [proposed/first]..."),
    (r"^when ", "When [was X proposed]..."),
    (r"^where ", "Where [is X from]..."),
    (r"^papers?( on| about| regarding| using| with| for)", "Papers [on/about]..."),
    (r"^[A-Z][a-z]+( [A-Z][a-z]+)+", "Paper title/name format"),
]


def detect_question_template(query: str) -> Tuple[Optional[str], str]:
    query_lower = query.lower().strip()
    for pattern_regex, template_name in QUESTION_TEMPLATES:
        if re.match(pattern_regex, query_lower):
            return template_name, pattern_regex
    return None, "other"


def compute_length_stats(queries: List[str]) -> Dict[str, Any]:
    if not queries:
        return {"char_length": {}, "token_length": {}, "total_queries": 0}
    char_lengths = [len(q) for q in queries]
    word_counts = [len(q.split()) for q in queries]
    return {
        "char_length": {"mean": float(np.mean(char_lengths)), "std": float(np.std(char_lengths)), "min": int(np.min(char_lengths)), "max": int(np.max(char_lengths)), "median": float(np.median(char_lengths)), "p25": float(np.percentile(char_lengths, 25)), "p75": float(np.percentile(char_lengths, 75))},
        "token_length": {"mean": float(np.mean(word_counts)), "std": float(np.std(word_counts)), "min": int(np.min(word_counts)), "max": int(np.max(word_counts)), "median": float(np.median(word_counts)), "p25": float(np.percentile(word_counts, 25)), "p75": float(np.percentile(word_counts, 75))},
        "total_queries": len(queries),
    }


def compute_question_template_distribution(queries: List[str]) -> Dict[str, Any]:
    if not queries:
        return {"template_distribution": {}, "template_ratios": {}, "unmatched_count": 0, "unmatched_ratio": 0.0, "template_examples": {}, "total_templates": 0}
    template_counts: Counter[str] = Counter()
    template_examples: Dict[str, List[str]] = {}
    unmatched_count = 0
    for query in queries:
        template_name, _ = detect_question_template(query)
        if template_name is None:
            unmatched_count += 1
            continue
        template_counts[template_name] += 1
        template_examples.setdefault(template_name, [])
        if len(template_examples[template_name]) < 3:
            template_examples[template_name].append(query)
    total = len(queries)
    return {
        "template_distribution": dict(template_counts),
        "template_ratios": {name: count / total for name, count in template_counts.items()},
        "unmatched_count": unmatched_count,
        "unmatched_ratio": unmatched_count / total if total else 0.0,
        "template_examples": template_examples,
        "total_templates": len(template_counts),
    }


def extract_representative_examples(queries: List[str], n: int = 15, diversity: bool = True) -> List[Dict[str, str]]:
    if not queries:
        return []
    examples = [{"query": query, "template": detect_question_template(query)[0] or "other"} for query in queries]
    if not diversity:
        return examples[:n]
    buckets: Dict[str, List[Dict[str, str]]] = {}
    for example in examples:
        buckets.setdefault(example["template"], []).append(example)
    result: List[Dict[str, str]] = []
    per_template = max(1, n // len(buckets))
    for bucket in buckets.values():
        result.extend(bucket[:per_template])
    return result[:n]


def analyze_query(query: str) -> Dict[str, Any]:
    template_name, pattern = detect_question_template(query)
    return {"index": 1, "query": query, "char_length": len(query), "token_length": len(query.split()), "template": template_name or "other", "matched_pattern": pattern, "matched_template": template_name is not None}


def analyze_queries(queries: List[str]) -> Dict[str, Any]:
    per_query = []
    for index, query in enumerate(queries, start=1):
        row = analyze_query(query)
        row["index"] = index
        per_query.append(row)
    return {
        "per_query": per_query,
        "length_stats": compute_length_stats(queries),
        "question_templates": compute_question_template_distribution(queries),
        "representative_examples": extract_representative_examples(queries, n=20, diversity=True),
    }


def compute_all_metrics(queries: List[str]) -> Dict[str, Any]:
    results = analyze_queries(queries)
    return {"length_stats": results["length_stats"], "question_templates": results["question_templates"]}
