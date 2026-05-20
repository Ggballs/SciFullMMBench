from __future__ import annotations

import csv
import html
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


IR_CONSENSUS_PATH = Path(
    "outputs/query_analysis/golden_view_classification_with_targets/final_human_consensus_annotations.csv"
)
QA_CONSENSUS_PATH = Path(
    "outputs/query_analysis/golden_view_classification_with_targets_qa_filtered/final_human_consensus_annotations.csv"
)
DEFAULT_OUTPUT_PATH = Path("outputs/query_analysis/golden_retrieval_icl_examples.json")

VALID_VIEW_LABELS = {"motivation", "method", "experiment/result"}
IMPORTABLE_DECISIONS = {"accept", "fix"}
NOTE_TITLE_OVERRIDES = {
    "Pollard (1981)": "Strong Consistency of K-Means Clustering",
    "(10.1186/1471-2288-14-137)": "Modern modelling techniques are data hungry: a simulation study for predicting dichotomous endpoints",
    "Wainer & Cawley": "Nested cross-validation when selecting classifiers is overzealous for most practical applications",
    "Cawley & Talbot": "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation",
    "Jain et al": "Parallelizing Stochastic Gradient Descent for Least Squares Regression: Mini-batching, Averaging, and Model Misspecification",
    "Bagnall & Cawley": "On the Use of Default Parameter Settings in the Empirical Evaluation of Classification Algorithms",
    "Gneiting and Resin (2023)": "Regression Diagnostics meets Forecast Evaluation: Conditional Calibration, Reliability Diagrams, and Coefficient of Determination",
    "Chen et al., REDQ": "Randomized Ensembled Double Q-Learning: Learning Fast Without a Model",
    "Hiraoka et al., DroQ": "Dropout Q-Functions for Doubly Efficient Reinforcement Learning",
    "Stanley, D'Ambrosio, and Gauci": "A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks",
    "Stanley, D\u2019Ambrosio, and Gauci": "A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks",
    "Paper: arXiv:2410.00179": "Evaluating the fairness of task-adaptive pretraining on unlabeled test data before few-shot text classification",
}


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return normalize_whitespace(" ".join(self.parts))


@dataclass(frozen=True)
class GoldenRetrievalICLExample:
    example_id: str
    query_id: str
    query_type: str
    view_label: str
    query: str
    target_papers: list[str]
    answer_original_content: str
    answer_tldr: str
    human_view_note: str
    indexing_content: str
    retrieval_content: str


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_view_label(label: str) -> str:
    normalized = normalize_whitespace(label).lower()
    if normalized in {"experiment", "experiments", "result", "results", "experiment/result"}:
        return "experiment/result"
    return normalized


def parse_labels(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_labels = value
    else:
        text = str(value or "").strip()
        if not text:
            raw_labels = []
        else:
            try:
                parsed = json.loads(text)
                raw_labels = parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                raw_labels = re.split(r"[|,]", text)
    labels: list[str] = []
    for label in raw_labels:
        normalized = normalize_view_label(str(label))
        if normalized in VALID_VIEW_LABELS and normalized not in labels:
            labels.append(normalized)
    return labels


def parse_target_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows = value
    else:
        text = str(value or "").strip()
        if not text:
            return []
        try:
            rows = json.loads(text)
        except json.JSONDecodeError:
            return []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def clean_html_text(value: str) -> str:
    text = html.unescape(str(value or ""))
    if "<" not in text or ">" not in text:
        return normalize_whitespace(text)
    parser = _HTMLTextExtractor()
    parser.feed(text)
    return parser.text()


def summarize_answer_text(text: str, *, max_chars: int = 700) -> str:
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    selected: list[str] = []
    total = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        next_total = total + len(sentence) + (1 if selected else 0)
        if selected and next_total > max_chars:
            break
        selected.append(sentence)
        total = next_total
    summary = " ".join(selected).strip() or text[:max_chars].strip()
    return summary.rstrip(" ,;:") + ("..." if len(summary) < len(text) else "")


_NOTE_MARKER_RE = re.compile(
    r"\s+[\u2014-]\s+(?P<label>motivation|method|experiment/result|experiment)\b[.:]?",
    flags=re.IGNORECASE,
)


def _section_start_before_marker(note: str, marker_start: int) -> int:
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'\u201c])", note[:marker_start]):
        prefix = note[max(0, match.start() - 12) : match.start()].lower()
        if prefix.endswith(" et al") or prefix.endswith(" vs") or prefix.endswith(" vs."):
            continue
        start = match.end()
    return start


def split_notes_by_view(note: str) -> dict[str, str]:
    note = normalize_whitespace(note)
    if not note:
        return {}
    matches = list(_NOTE_MARKER_RE.finditer(note))
    sections: dict[str, list[str]] = {}
    for idx, match in enumerate(matches):
        label = normalize_view_label(match.group("label"))
        start = _section_start_before_marker(note, match.start())
        end = (
            _section_start_before_marker(note, matches[idx + 1].start())
            if idx + 1 < len(matches)
            else len(note)
        )
        section = normalize_whitespace(note[start:end])
        if section:
            sections.setdefault(label, []).append(section)
    return {label: " ".join(parts) for label, parts in sections.items()}


def qa_view_notes_from_final_note(note: str) -> list[tuple[str, str]]:
    sections = split_notes_by_view(note)
    return [(label, section) for label, section in sections.items() if section]


def extract_titles_from_note(note: str) -> list[str]:
    titles: list[str] = []
    for pattern in (
        r"[\u201c\"]([^\u201d\"]{8,220})[\u201d\"]",
        r"[\u2018']([^\u2019']{8,220})[\u2019']",
    ):
        for match in re.finditer(pattern, note):
            title = normalize_whitespace(match.group(1))
            if title and title not in titles:
                titles.append(title)
    return titles


def extract_citation_mentions_from_note(note: str) -> list[str]:
    mentions: list[str] = []
    note = normalize_whitespace(note)
    for match in _NOTE_MARKER_RE.finditer(note):
        start = _section_start_before_marker(note, match.start())
        citation = normalize_whitespace(note[start : match.start()].strip(" .;:"))
        if citation and citation not in mentions:
            mentions.append(citation)
    return mentions


def _title_from_unquoted_mention(mention: str) -> Optional[str]:
    mention = normalize_whitespace(re.sub(r"\([^)]*10\.\d{4,9}/[^)]*\)", "", mention))
    if not mention or re.search(r"\bet al\.?\b", mention, flags=re.I):
        return None
    if re.search(r"\breferences?\b", mention, flags=re.I):
        return None
    if re.fullmatch(r"[A-Z][A-Za-z-]+(?:\s*,?\s*[A-Z][A-Za-z-]+){0,4}(?:\s*\(\d{4}\))?", mention):
        return None
    if len(mention.split()) >= 4 or ":" in mention:
        return mention.strip(" ,.;:")
    return None


def _title_from_crossref_query(query: str, timeout: float) -> Optional[str]:
    params = urllib.parse.urlencode({"query.bibliographic": query, "rows": "1"})
    url = f"https://api.crossref.org/works?{params}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    items = payload.get("message", {}).get("items", [])
    if items:
        titles = items[0].get("title", [])
        if titles:
            return normalize_whitespace(str(titles[0]))
    return None


def _paper_urls_from_target(target: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("paper_urls", "url"):
        value = target.get(key)
        if not value:
            continue
        if isinstance(value, list):
            values.extend(value)
            continue
        text = str(value).strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                values.extend(parsed)
            else:
                values.append(parsed)
        except json.JSONDecodeError:
            values.append(text)
    return [str(value).strip() for value in values if str(value).strip()]


def _title_from_crossref(doi: str, timeout: float) -> Optional[str]:
    encoded = urllib.parse.quote(doi.strip().removeprefix("doi:"), safe="")
    url = f"https://api.crossref.org/works/{encoded}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    titles = payload.get("message", {}).get("title", [])
    if titles:
        return normalize_whitespace(str(titles[0]))
    return None


def _title_from_arxiv(arxiv_id: str, timeout: float) -> Optional[str]:
    url = "https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(arxiv_id)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    root = ET.fromstring(payload)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    title = root.findtext("atom:entry/atom:title", namespaces=namespace)
    return normalize_whitespace(title or "") or None


def resolve_title_from_url(value: str, *, timeout: float = 8.0) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9.]+)(?:v\d+)?", text, flags=re.I)
    if arxiv_match:
        return _title_from_arxiv(arxiv_match.group(1), timeout)
    doi_match = re.search(r"(10\.\d{4,9}/[^\s\"'<>]+)", text, flags=re.I)
    if doi_match:
        return _title_from_crossref(doi_match.group(1).rstrip("].,;)"), timeout)
    return None


def collect_target_titles(
    *,
    query_type: str,
    targets: list[dict[str, Any]],
    note: str,
    resolve_web: bool = False,
) -> tuple[list[str], list[str]]:
    titles: list[str] = []
    unresolved: list[str] = []
    note = normalize_whitespace(note)

    if query_type == "QA":
        for mention in extract_citation_mentions_from_note(note):
            exact_titles = extract_titles_from_note(mention)
            for exact_title in exact_titles:
                if exact_title not in titles:
                    titles.append(exact_title)
            if any(title in mention for title in exact_titles):
                continue
            title = NOTE_TITLE_OVERRIDES.get(mention) or ""
            if not title and resolve_web:
                arxiv_match = re.search(r"arxiv:([0-9.]+)(?:v\d+)?", mention, flags=re.I)
                if arxiv_match:
                    try:
                        title = _title_from_arxiv(arxiv_match.group(1), 8.0) or ""
                    except Exception:
                        title = ""
            if not title:
                title = _title_from_unquoted_mention(mention) or ""
            if not title and resolve_web:
                doi_match = re.search(r"(10\.\d{4,9}/[^\s\"'<>),]+)", mention, flags=re.I)
                try:
                    if doi_match:
                        title = _title_from_crossref(doi_match.group(1), 8.0) or ""
                    else:
                        title = _title_from_crossref_query(mention, 8.0) or ""
                except Exception:
                    title = ""
            if title:
                if title not in titles:
                    titles.append(title)
            else:
                unresolved.append(mention)
        return titles, unresolved

    for target in targets:
        title = normalize_whitespace(str(target.get("title") or ""))
        if not title and resolve_web:
            for url in _paper_urls_from_target(target):
                try:
                    title = resolve_title_from_url(url) or ""
                except Exception:
                    title = ""
                if title:
                    break
        if title:
            if title not in titles:
                titles.append(title)
        else:
            unresolved.extend(_paper_urls_from_target(target) or ["missing target title"])
    return titles, unresolved


def make_example_id(query_id: str, view_label: str) -> str:
    slug = view_label.replace("/", "_").replace(" ", "_")
    return f"{query_id}__{slug}"


def format_labeled_content(parts: Iterable[tuple[str, str]]) -> str:
    lines = [
        f"{label}: {normalize_whitespace(value)}"
        for label, value in parts
        if normalize_whitespace(value)
    ]
    return "\n".join(lines)


def build_examples_from_csv_paths(
    csv_paths: Iterable[Path],
    *,
    resolve_web_titles: bool = False,
    answer_tldr_generator: Optional[Callable[[str], str]] = None,
) -> tuple[list[GoldenRetrievalICLExample], dict[str, Any]]:
    examples: list[GoldenRetrievalICLExample] = []
    report: dict[str, Any] = {
        "source_rows": 0,
        "eligible_rows": 0,
        "expanded_rows": 0,
        "skipped_decisions": {},
        "by_type_view": {},
        "note_mismatches": [],
        "unresolved_titles": [],
    }

    for csv_path in csv_paths:
        with Path(csv_path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                report["source_rows"] += 1
                decision = normalize_whitespace(row.get("final_decision", "")).lower()
                if decision not in IMPORTABLE_DECISIONS:
                    skipped = report["skipped_decisions"]
                    skipped[decision or "blank"] = skipped.get(decision or "blank", 0) + 1
                    continue

                query_type = normalize_whitespace(row.get("query_type", "")).upper()
                query = normalize_whitespace(row.get("query", ""))
                final_note = normalize_whitespace(row.get("final_notes", ""))
                final_labels = parse_labels(row.get("final_labels"))
                note_sections = qa_view_notes_from_final_note(final_note) if query_type == "QA" else []
                expanded_sections = note_sections or [(label, "") for label in final_labels]
                if query_type not in {"IR", "QA"} or not query or not expanded_sections:
                    continue

                report["eligible_rows"] += 1
                query_id = normalize_whitespace(row.get("query_id", ""))
                targets = parse_target_rows(row.get("target_papers"))
                answer_original_content = ""
                answer_tldr = ""
                if query_type == "QA":
                    answer_original_content = clean_html_text(" ".join(str(t.get("abstract") or "") for t in targets))
                    if answer_tldr_generator and answer_original_content:
                        answer_tldr = normalize_whitespace(answer_tldr_generator(answer_original_content))

                if query_type == "QA":
                    note_labels = [label for label, _ in expanded_sections]
                    if sorted(note_labels) != sorted(final_labels):
                        report["note_mismatches"].append(
                            {
                                "query_id": query_id,
                                "final_labels": final_labels,
                                "note_labels": note_labels,
                            }
                        )

                for view_label, section_note in expanded_sections:
                    human_view_note = section_note if query_type == "QA" else ""
                    titles, unresolved = collect_target_titles(
                        query_type=query_type,
                        targets=targets,
                        note=human_view_note,
                        resolve_web=resolve_web_titles,
                    )
                    title_text = "; ".join(titles)
                    indexing_content = format_labeled_content(
                        [
                            ("Query", query),
                            ("Target papers", title_text),
                        ]
                    )
                    if query_type == "QA":
                        retrieval_content = format_labeled_content(
                            [
                                ("Query", query),
                                ("Answer TLDR", answer_tldr),
                                ("Target papers", title_text),
                                ("Human view note", human_view_note),
                            ]
                        )
                    else:
                        retrieval_content = format_labeled_content([("Query", query)])
                    examples.append(
                        GoldenRetrievalICLExample(
                            example_id=make_example_id(query_id, view_label),
                            query_id=query_id,
                            query_type=query_type,
                            view_label=view_label,
                            query=query,
                            target_papers=titles,
                            answer_original_content=answer_original_content,
                            answer_tldr=answer_tldr,
                            human_view_note=human_view_note,
                            indexing_content=indexing_content,
                            retrieval_content=retrieval_content,
                        )
                    )
                    key = f"{query_type}:{view_label}"
                    report["by_type_view"][key] = report["by_type_view"].get(key, 0) + 1
                    report["expanded_rows"] += 1
                    if unresolved:
                        report["unresolved_titles"].append(
                            {
                                "query_id": query_id,
                                "view_label": view_label,
                                "values": unresolved,
                            }
                        )

    return examples, report


def write_examples_json(
    examples: list[GoldenRetrievalICLExample],
    output_path: Path,
    *,
    report: Optional[dict[str, Any]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(example) for example in examples], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if report is not None:
        report_path = output_path.with_name(output_path.stem + "_report.json")
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
