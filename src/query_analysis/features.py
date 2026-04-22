import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
def compute_all_metrics(queries: List[str]) -> Dict[str, Any]:
    """Compute local non-LLM metrics.

    Args:
        queries: List of query strings

    Returns:
        Dictionary with computed local metrics
    """
    return {
        "length_stats": compute_length_stats(queries),
        "question_templates": compute_question_template_distribution(queries),
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
