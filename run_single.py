import yaml
import json
import logging
import re
from pathlib import Path
from datetime import datetime

from openreview_pipeline.llm import OpenAICompatibleBackend
from openreview_pipeline.stages import Summarizer, QueryGenerator, QueryFilter
from openreview_pipeline.schemas.schemas_pipeline import PipelineOutput, PipelinePaper
from openreview_pipeline.utils import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_PAPER_ID = "XZNXSM4rHG"
OUTPUT_DIR = Path("data/iclr_2026")
VENUE = "ICLR.cc"
YEAR = 2026
LIMIT = 10

with open("config.yaml") as f:
    config = yaml.safe_load(f)

llm_config = config.get("llm", {})
openreview_creds = config.get("openreview", {})

backend = OpenAICompatibleBackend(
    base_url=llm_config["base_url"],
    api_token=llm_config["api_token"],
    model=llm_config["model"],
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

filter_path = OUTPUT_DIR / "01_filtered_10.json"
summarized_path = OUTPUT_DIR / "02_summarized_10.json"
queries_path = OUTPUT_DIR / "03_queries_10.json"
filtered_path = OUTPUT_DIR / "04_filtered_10.json"


def check_accepted(paper_data: dict) -> bool:
    decision = paper_data.get("decision")
    if not decision:
        return False
    content = decision.get("content", {})
    decision_value = content.get("decision", "")
    if isinstance(decision_value, dict):
        decision_value = decision_value.get("value", "")
    return "accept" in str(decision_value).lower()


def check_similar_paper(paper_data: dict) -> bool:
    abstract = paper_data.get("paper", {}).get("abstract", "")
    title = paper_data.get("paper", {}).get("title", "")

    similar_patterns = [
        r"\d+ similar papers?",
        r"related work.*similar",
        r"benchmarks? similar",
        r"similar to \w+",
        r"same as previous",
    ]

    text = abstract + " " + title
    for pattern in similar_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def check_multimodal_info(paper_data: dict) -> bool:
    abstract = paper_data.get("paper", {}).get("abstract", "")
    keywords = paper_data.get("paper", {}).get("keywords", [])

    multimodal_keywords = [
        "figure", "table", "diagram", "chart", "equation",
        "image", "video", "audio", "multimodal", "visual",
        "fig", "graph", "plot", "illustration"
    ]

    text = abstract.lower()
    keyword_text = " ".join(keywords).lower()
    combined = text + " " + keyword_text

    found_keywords = [kw for kw in multimodal_keywords if kw in combined]
    return len(found_keywords) >= 2


logger.info(f"=== Fetching {LIMIT} papers including MRMR ===")
from openreview_pipeline.stages.stage0_download import OpenReviewAPIDownloader, note_to_dict, classify_note

downloader = OpenReviewAPIDownloader(
    venue=VENUE,
    year=YEAR,
    username=openreview_creds.get("username"),
    password=openreview_creds.get("password"),
    token=openreview_creds.get("token"),
)

client = downloader._get_client()

invitation = f"{VENUE}/{YEAR}/Conference/-/Submission"
logger.info(f"Fetching submissions from {invitation}")
submissions = list(client.get_all_notes(invitation=invitation))
logger.info(f"Total submissions: {len(submissions)}")

collected = []
for sub in submissions:
    nd = note_to_dict(sub)
    title = nd["content"].get("title", "")
    if TARGET_PAPER_ID in [sub.id, str(sub.number)] or "MRMR" in title:
        collected.append(sub)
        logger.info(f"Found MRMR paper: {sub.id}")

if len(collected) < LIMIT:
    for sub in submissions:
        if sub not in collected:
            collected.append(sub)
            if len(collected) >= LIMIT:
                break

logger.info(f"Collected {len(collected)} papers")

results = []
passed_count = 0

for sub in collected:
    try:
        logger.info(f"Processing: {sub.id}")
        logger.info(f"Title: {note_to_dict(sub)['content'].get('title', '')[:60]}")

        reviews = []
        comments = []
        rebuttals = []
        decisions = []

        forum_notes = downloader._collect_forum_notes(client, sub)

        for note_dict in forum_notes:
            kind = classify_note(note_dict)
            note_content = note_dict.get("content", {})
            inv = note_dict.get("invitation") or (note_dict.get("invitations", [""])[0] if note_dict.get("invitations") else "")

            if kind == "review":
                reviews.append({
                    "id": note_dict["id"],
                    "paper_id": sub.id,
                    "invitation": inv,
                    "content": downloader._build_review_content(note_content),
                    "number": note_dict.get("number", 0),
                    "cdate": note_dict.get("cdate"),
                    "tcdate": note_dict.get("tcdate"),
                })
            elif kind == "rebuttal":
                rebuttals.append({
                    "id": note_dict["id"],
                    "paper_id": sub.id,
                    "invitation": inv,
                    "content": downloader._build_review_content(note_content),
                    "number": note_dict.get("number", 0),
                    "cdate": note_dict.get("cdate"),
                    "tcdate": note_dict.get("tcdate"),
                })
            elif kind == "comment":
                comments.append({
                    "id": note_dict["id"],
                    "paper_id": sub.id,
                    "invitation": inv,
                    "content": downloader._build_review_content(note_content),
                    "number": note_dict.get("number", 0),
                    "cdate": note_dict.get("cdate"),
                    "tcdate": note_dict.get("tcdate"),
                })
            elif kind == "decision":
                decisions.append({
                    "id": note_dict["id"],
                    "paper_id": sub.id,
                    "invitation": inv,
                    "content": downloader._build_review_content(note_content),
                    "number": note_dict.get("number", 0),
                })

        nd = note_to_dict(sub)
        paper_content = nd["content"]

        paper_data = {
            "paper": {
                "id": sub.id,
                "title": paper_content.get("title", ""),
                "abstract": paper_content.get("abstract", ""),
                "authors": paper_content.get("authors", []),
                "venue": VENUE,
                "year": YEAR,
                "keywords": paper_content.get("keywords", []),
                "venueid": paper_content.get("venueid", ""),
                "submission_number": sub.number,
            },
            "reviews": reviews,
            "comments": comments,
            "rebuttals": rebuttals,
            "decision": decisions[0] if decisions else None,
        }

        accepted = check_accepted(paper_data)
        similar_paper = check_similar_paper(paper_data)
        multimodal_info = check_multimodal_info(paper_data)

        passed = accepted and not similar_paper and multimodal_info

        if passed:
            passed_count += 1

        logger.info(f"  Reviews: {len(reviews)}, Comments: {len(comments)}, Rebuttals: {len(rebuttals)}, Decisions: {len(decisions)}")
        logger.info(f"  Filter: accepted={accepted}, similar={similar_paper}, multimodal={multimodal_info}, passed={passed}")

        results.append({
            "paper": paper_data,
            "passed": passed,
            "details": {
                "accepted": accepted,
                "similar_paper": similar_paper,
                "multimodal_info": multimodal_info,
            }
        })

    except Exception as e:
        logger.error(f"Error processing {sub.id}: {e}")
        continue

filtered_data = {
    "results": results,
    "total_input": len(results),
    "total_passed": passed_count,
    "total_filtered": len(results) - passed_count,
    "filtered_at": datetime.now().isoformat()
}

with open(filter_path, "w") as f:
    json.dump(filtered_data, f, indent=2)
logger.info(f"Saved filtered data to {filter_path}")
logger.info(f"Filter result: {passed_count}/{len(results)} passed")

passed_papers = [r for r in results if r["passed"]]
logger.info(f"\n=== Stage 2: Summarizing {len(passed_papers)} passed papers ===")
summarizer = Summarizer(llm=backend, llm_limit=len(passed_papers))
summarizer.run(filter_path, summarized_path)

with open(summarized_path) as f:
    summarized = json.load(f)
logger.info(f"Summarized {len(summarized['summaries'])} papers")

logger.info(f"\n=== Stage 3: Generating queries ===")
generator = QueryGenerator(llm=backend)
generator.run(summarized_path, queries_path)

with open(queries_path) as f:
    queries = json.load(f)
queries_by_paper = {}
for pq in queries["papers_queries"]:
    queries_by_paper[pq["paper_id"]] = pq["queries_by_view"]
logger.info(f"Generated {queries['total_queries']} queries")

logger.info(f"\n=== Stage 4: Filtering queries ===")
query_filter = QueryFilter(llm=backend)
query_filter.run(queries_path, filtered_path)

with open(filtered_path) as f:
    filtered_queries = json.load(f)
filtered_by_query = {}
for fq in filtered_queries["results"]:
    filtered_by_query[fq["original_query"]] = fq
passed_queries = sum(1 for fq in filtered_queries["results"] if fq.get("verdict") == "Keep")
logger.info(f"Filtered: {passed_queries}/{len(filtered_queries['results'])} passed (Keep)")

logger.info(f"\n=== Building combined output ===")
combined = PipelineOutput(
    venue=VENUE,
    year=YEAR,
    generated_at=datetime.now(),
)

for r in passed_papers:
    paper_data = r["paper"]
    pid = paper_data["paper"]["id"]
    sp = {}
    for s in summarized.get("summaries", []):
        if s["paper_id"] == pid:
            sp = s
            break
    pqs = queries_by_paper.get(pid, [])

    p = PipelinePaper(
        paper_id=pid,
        paper_title=paper_data["paper"]["title"],
        abstract=paper_data["paper"]["abstract"],
        authors=paper_data["paper"].get("authors", []),
        venue=paper_data["paper"].get("venue", ""),
        year=paper_data["paper"].get("year", 0),
        keywords=paper_data["paper"].get("keywords", []),
        reviews=paper_data.get("reviews", []),
        comments=paper_data.get("comments", []),
        rebuttals=paper_data.get("rebuttals", []),
        decision=paper_data.get("decision"),
    )
    p.passed = r["passed"]
    p.filter_reasons = r["details"]
    p.summary = sp
    p.queries = pqs

    for q in p.queries:
        if q["query_text"] in filtered_by_query:
            fq = filtered_by_query[q["query_text"]]
            dims = fq.get("dimensions", {})
            p.filtered_queries.append({
                "query_text": fq["original_query"],
                "is_multimodal": fq.get("is_multimodal", False),
                "source_view": fq.get("source_view", ""),
                "dimensions": {
                    "full_paper_reliance": dims.get("full_paper_reliance", "FAIL"),
                    "authenticity": dims.get("authenticity", "FAIL"),
                    "relevance": dims.get("relevance", "FAIL"),
                    "difficulty": dims.get("difficulty", "TOO_HARD"),
                    "false_negative_risk": dims.get("false_negative_risk", "HIGH"),
                },
                "reasoning": fq.get("reasoning", ""),
                "verdict": fq.get("verdict", "Hard Reject"),
                "revised_query": fq.get("revised_query"),
            })

    combined.papers.append(p)

combined.total_papers = len(combined.papers)
combined.total_passed = passed_count
combined.total_queries = sum(len(p.queries) for p in combined.papers)
combined.total_queries_passed = sum(1 for p in combined.papers for fq in p.filtered_queries if fq.get("verdict") == "Keep")

output_path = OUTPUT_DIR / "pipeline_output_10.json"
save_json(output_path, combined)

logger.info(f"\n=== Complete! ===")
logger.info(f"Output: {output_path}")
logger.info(f"Papers processed: {len(results)}")
logger.info(f"Papers passed filter: {combined.total_papers}")
logger.info(f"Queries: {combined.total_queries} generated, {combined.total_queries_passed} passed")