import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below", "between",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "don", "now", "i", "me", "my", "myself", "we", "our", "ours",
    "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "these", "those", "am", "about", "against", "because", "until",
    "while", "if", "both", "up", "down", "out", "off", "over",
}

CONNECTOR_WORDS = {
    "for", "with", "without", "using", "based", "through", "between",
    "among", "against", "towards", "from", "into", "during", "before", "after",
}

QUESTION_TEMPLATES = [
    (r"^is there (a |any )?(paper|work|method|model|dataset|tool|study|approach|technique|system|decoder-|knowledge graph|vision-language|backdoor|multimodal|audio|video|numerical reasoning)?( that | which | )?", "Is there a paper/method that/which..."),
    (r"^is there any (paper|work|method|model|dataset|tool|study|approach|technique|system) ", "Is there a paper/method that/which..."),
    (r"^is there such a? (paper|work|method|model|dataset|tool|study|approach|technique|system)?( that | which | )?", "Is there a paper/method that/which..."),
    (r"^are there (any |)?(sequential |papers? |methods? |models? |approaches? )", "Is there a paper/method that/which..."),
    (r"^are there any papers? (that |which |on |about )?", "Is there a paper/method that/which..."),
    (r"^any papers? (on |about |regarding |that |which )?", "Is there a paper/method that/which..."),
    (r"^which paper(s| research| vision-language| backdoor| numerical| knowledge graph|)?( first |proposes |introduces |presents |is | |about |on |)", "What [open-source/latest/first] method..."),
    (r"^which (is |are |was |were |)?the (first |)?", "What [open-source/latest/first] method..."),
    (r"^which (knowledge graph|vision-language|multimodal|backdoor|audio|video|method|model|approach|work|dataset|technique|system|decoder|numerical reasoning|numeral|paper|papers|research) (first |proposes |introduces |presents |is |was |focuses |developed |can |published |)", "What [open-source/latest/first] method..."),
    (r"^which numerical reasoning paper", "What [open-source/latest/first] method..."),
    (r"^what('s| is| are| was| were|') ", "What [open-source/latest/first] method..."),
    (r"^what (open-source |latest |recent )?(paper|work|method|model|dataset|papers|studies) ", "What [open-source/latest/first] method..."),
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

CONSTRAINT_PATTERNS = [
    (r"\bfor\b", "for"),
    (r"\bwith\b", "with"),
    (r"\busing\b", "using"),
    (r"\bbased on\b", "based on"),
    (r"\bwithout\b", "without"),
    (r"\bunder\b", "under"),
    (r"\babout\b", "about"),
    (r"\bregarding\b", "regarding"),
    (r"\babout\b", "about"),
    (r"\bduring\b", "during"),
    (r"\bthrough\b", "through"),
]

METHOD_KEYWORDS = {
    "transformer", "attention", "neural", "network", "model", "learning", "training",
    "bert", "gpt", "llm", "clip", "diffusion", "gan", "vae", "rl", "reinforcement",
    "supervised", "unsupervised", "semi-supervised", "self-supervised", "contrastive",
    "encoder", "decoder", "architecture", "layer", "embedding", "token", "attention",
    "multimodal", "vision", "language", "audio", "video", "image", "text",
}

DATASET_KEYWORDS = {
    "imagenet", "coco", "vqa", "squad", "glue", "superglue", "arxiv", "paper",
    "dataset", "benchmark", "training", "test", "validation", "data",
}


def detect_question_template(query: str) -> Tuple[Optional[str], str]:
    """Detect the question template of a query.

    Args:
        query: Query string

    Returns:
        Tuple of (template_name, matched_pattern)
    """
    query_lower = query.lower().strip()

    for pattern_regex, template_name in QUESTION_TEMPLATES:
        if re.match(pattern_regex, query_lower):
            return template_name, pattern_regex

    return None, "other"


def compute_length_stats(queries: List[str]) -> Dict[str, Any]:
    """Compute length statistics for queries.

    Args:
        queries: List of query strings

    Returns:
        Dictionary with length statistics
    """
    if not queries:
        return {}

    char_lengths = [len(q) for q in queries]
    word_counts = [len(q.split()) for q in queries]

    return {
        "char_length": {
            "mean": float(np.mean(char_lengths)),
            "std": float(np.std(char_lengths)),
            "min": int(np.min(char_lengths)),
            "max": int(np.max(char_lengths)),
            "median": float(np.median(char_lengths)),
            "p25": float(np.percentile(char_lengths, 25)),
            "p75": float(np.percentile(char_lengths, 75)),
        },
        "token_length": {
            "mean": float(np.mean(word_counts)),
            "std": float(np.std(word_counts)),
            "min": int(np.min(word_counts)),
            "max": int(np.max(word_counts)),
            "median": float(np.median(word_counts)),
            "p25": float(np.percentile(word_counts, 25)),
            "p75": float(np.percentile(word_counts, 75)),
        },
        "total_queries": len(queries),
    }


def compute_token_variety(queries: List[str]) -> Dict[str, Any]:
    """Compute token variety/lexical richness metrics.

    Args:
        queries: List of query strings

    Returns:
        Dictionary with token variety statistics
    """
    if not queries:
        return {}

    type_token_ratios = []
    unique_token_counts = []
    total_token_counts = []

    for q in queries:
        words = q.lower().split()
        words_set = set(words)
        total_token_counts.append(len(words))
        unique_token_counts.append(len(words_set))
        if len(words) > 0:
            type_token_ratios.append(len(words_set) / len(words))
        else:
            type_token_ratios.append(0)

    return {
        "type_token_ratio": {
            "mean": float(np.mean(type_token_ratios)),
            "std": float(np.std(type_token_ratios)),
            "min": float(np.min(type_token_ratios)),
            "max": float(np.max(type_token_ratios)),
            "median": float(np.median(type_token_ratios)),
        },
        "unique_tokens_per_query": {
            "mean": float(np.mean(unique_token_counts)),
            "std": float(np.std(unique_token_counts)),
            "min": int(np.min(unique_token_counts)),
            "max": int(np.max(unique_token_counts)),
        },
        "avg_query_length": float(np.mean(total_token_counts)),
    }


def compute_constraint_count(queries: List[str]) -> Dict[str, Any]:
    """Count explicit constraints in queries.

    Args:
        queries: List of query strings

    Returns:
        Dictionary with constraint statistics
    """
    if not queries:
        return {}

    constraint_counts = []
    constraint_types_used = Counter()
    queries_with_constraints = 0

    for q in queries:
        q_lower = q.lower()
        count = 0
        used_types = set()

        for pattern, constraint_name in CONSTRAINT_PATTERNS:
            matches = re.findall(pattern, q_lower)
            if matches:
                count += len(matches)
                used_types.add(constraint_name)
                constraint_types_used[constraint_name] += len(matches)

        constraint_counts.append(count)
        if count > 0:
            queries_with_constraints += 1

    return {
        "constraints_per_query": {
            "mean": float(np.mean(constraint_counts)),
            "std": float(np.std(constraint_counts)),
            "min": int(np.min(constraint_counts)),
            "max": int(np.max(constraint_counts)),
            "median": float(np.median(constraint_counts)),
        },
        "queries_with_constraints": queries_with_constraints,
        "constraint_ratio": queries_with_constraints / len(queries) if queries else 0,
        "constraint_type_distribution": dict(constraint_types_used),
    }


def compute_method_dataset_mentions(queries: List[str]) -> Dict[str, Any]:
    """Count method and dataset mentions in queries.

    Args:
        queries: List of query strings

    Returns:
        Dictionary with method/dataset mention statistics
    """
    if not queries:
        return {}

    method_mentions = 0
    dataset_mentions = 0
    queries_with_methods = 0
    queries_with_datasets = 0

    for q in queries:
        q_lower = q.lower()
        words = set(q_lower.split())

        has_method = any(m in words or m in q_lower for m in METHOD_KEYWORDS)
        has_dataset = any(d in words or d in q_lower for d in DATASET_KEYWORDS)

        if has_method:
            method_mentions += 1
            queries_with_methods += 1
        if has_dataset:
            dataset_mentions += 1
            queries_with_datasets += 1

    total = len(queries)

    return {
        "method_mention_ratio": method_mentions / total if total > 0 else 0,
        "dataset_mention_ratio": dataset_mentions / total if total > 0 else 0,
        "queries_with_methods": queries_with_methods,
        "queries_with_datasets": queries_with_datasets,
    }


def compute_question_template_distribution(queries: List[str]) -> Dict[str, Any]:
    """Compute question template distribution.

    Args:
        queries: List of query strings

    Returns:
        Dictionary with template statistics
    """
    if not queries:
        return {}

    template_counts = Counter()
    template_examples = {}
    unmatched_count = 0

    for q in queries:
        template_name, _ = detect_question_template(q)
        if template_name:
            template_counts[template_name] += 1
            if template_name not in template_examples:
                template_examples[template_name] = []
            if len(template_examples[template_name]) < 3:
                template_examples[template_name].append(q)
        else:
            unmatched_count += 1

    total = len(queries)

    return {
        "template_distribution": dict(template_counts),
        "template_ratios": {
            t: c / total for t, c in template_counts.items()
        },
        "unmatched_count": unmatched_count,
        "unmatched_ratio": unmatched_count / total if total > 0 else 0,
        "template_examples": template_examples,
        "total_templates": len(template_counts),
    }


def compute_qualitative_metrics(queries: List[str]) -> Dict[str, Any]:
    """Compute qualitative metrics (specificity, naturalness, academic tone).

    Args:
        queries: List of query strings

    Returns:
        Dictionary with qualitative metrics
    """
    if not queries:
        return {}

    specificity_scores = []
    naturalness_scores = []
    academic_tone_scores = []

    for q in queries:
        words = q.lower().split()
        q_stripped = q.strip()

        specificity = 0.5
        naturalness = 0.5
        academic_tone = 0.5

        specificity += len(words) / 100
        if len(words) > 10:
            specificity += 0.2
        if re.search(r'\d+', q):
            specificity += 0.1
        if re.search(r'\bfor\b.*\busing\b|\bwith\b.*\busing\b', q_lower := q.lower()):
            specificity += 0.15
        specificity = min(1.0, specificity)

        naturalness = 0.5
        if q_stripped.endswith('?'):
            naturalness += 0.2
        if not q_stripped.startswith(('we', 'this paper', 'our')):
            naturalness += 0.2
        if 'in this paper' not in q.lower() and 'in our paper' not in q.lower():
            naturalness += 0.1
        naturalness = min(1.0, naturalness)

        academic_tone = 0.5
        if any(w in words for w in ['paper', 'research', 'study', 'method', 'approach', 'model']):
            academic_tone += 0.2
        if not any(w in words for w in ['hello', 'hi', 'hey', 'please help', 'thanks']):
            academic_tone += 0.2
        if '?' in q_stripped or 'whether' in q.lower():
            academic_tone += 0.1
        academic_tone = min(1.0, academic_tone)

        specificity_scores.append(specificity)
        naturalness_scores.append(naturalness)
        academic_tone_scores.append(academic_tone)

    return {
        "specificity": {
            "mean": float(np.mean(specificity_scores)),
            "std": float(np.std(specificity_scores)),
            "min": float(np.min(specificity_scores)),
            "max": float(np.max(specificity_scores)),
            "median": float(np.median(specificity_scores)),
        },
        "naturalness": {
            "mean": float(np.mean(naturalness_scores)),
            "std": float(np.std(naturalness_scores)),
            "min": float(np.min(naturalness_scores)),
            "max": float(np.max(naturalness_scores)),
            "median": float(np.median(naturalness_scores)),
        },
        "academic_tone": {
            "mean": float(np.mean(academic_tone_scores)),
            "std": float(np.std(academic_tone_scores)),
            "min": float(np.min(academic_tone_scores)),
            "max": float(np.max(academic_tone_scores)),
            "median": float(np.median(academic_tone_scores)),
        },
    }


def compute_all_metrics(queries: List[str]) -> Dict[str, Any]:
    """Compute all quantitative and qualitative metrics.

    Args:
        queries: List of query strings

    Returns:
        Dictionary with all computed metrics
    """
    return {
        "length_stats": compute_length_stats(queries),
        "token_variety": compute_token_variety(queries),
        "constraint_count": compute_constraint_count(queries),
        "method_dataset_mentions": compute_method_dataset_mentions(queries),
        "question_templates": compute_question_template_distribution(queries),
        "qualitative_metrics": compute_qualitative_metrics(queries),
    }


def extract_representative_examples(
    queries: List[str],
    n: int = 15,
    diversity: bool = True
) -> List[Dict[str, str]]:
    """Extract representative examples with template labels.

    Args:
        queries: List of query strings
        n: Number of examples to return
        diversity: Whether to prefer diverse examples

    Returns:
        List of dicts with 'query' and 'template' keys
    """
    if not queries:
        return []

    examples = []
    for q in queries:
        template_name, _ = detect_question_template(q)
        examples.append({
            "query": q,
            "template": template_name or "other"
        })

    if not diversity:
        return examples[:n]

    template_buckets = {}
    for ex in examples:
        t = ex["template"]
        if t not in template_buckets:
            template_buckets[t] = []
        template_buckets[t].append(ex)

    result = []
    per_template = max(1, n // len(template_buckets))
    for template, bucket in template_buckets.items():
        result.extend(bucket[:per_template])

    return result[:n]
