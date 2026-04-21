import yaml
import json
import logging
from pathlib import Path
from datetime import datetime

from openreview_pipeline.llm import OpenAICompatibleBackend
from openreview_pipeline.stages import DatasetDownloader, RuleBasedFilter, Summarizer, QueryGenerator, QueryFilter
from openreview_pipeline.schemas.schemas_pipeline import PipelineOutput, PipelinePaper
from openreview_pipeline.utils import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/iclr_2026")
FILTER_LIMIT = 10
LLM_LIMIT = 10
VENUE = "ICLR.cc"
YEAR = 2026

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

download_path = OUTPUT_DIR / "00_downloaded.json"
filter_path = OUTPUT_DIR / "01_filtered.json"
summarized_path = OUTPUT_DIR / "02_summarized.json"
queries_path = OUTPUT_DIR / "03_queries.json"
filtered_path = OUTPUT_DIR / "04_filtered_queries.json"

logger.info("=== Stage 0: Downloading papers from {} {} ===".format(VENUE, YEAR))
downloader = DatasetDownloader(venue=VENUE, year_threshold=YEAR, output_dir=str(OUTPUT_DIR))
downloader.set_openreview_credentials(
    username=openreview_creds.get("username"),
    password=openreview_creds.get("password"),
    token=openreview_creds.get("token"),
)
downloader.run(download_path)

logger.info("=== Stage 1: Filtering papers (limit: {}) ===".format(FILTER_LIMIT))
filter_stage = RuleBasedFilter(limit=FILTER_LIMIT)
filter_stage.run(download_path, filter_path)

with open(filter_path) as f:
    filtered = json.load(f)
logger.info("Filter: {}/{} papers passed".format(filtered["total_passed"], filtered["total_input"]))

logger.info("\n=== Stage 2: Summarizing papers (LLM limit: {}) ===".format(LLM_LIMIT))
summarizer = Summarizer(llm=backend, llm_limit=LLM_LIMIT)
summarizer.run(filter_path, summarized_path)

with open(summarized_path) as f:
    summarized = json.load(f)
summarized_papers = {s["paper_id"]: s for s in summarized["summaries"]}
logger.info("Summarized {} papers".format(len(summarized_papers)))

logger.info("\n=== Stage 3: Generating queries ===")
generator = QueryGenerator(llm=backend)
generator.run(summarized_path, queries_path)

with open(queries_path) as f:
    queries = json.load(f)
queries_by_paper = {}
for pq in queries["papers_queries"]:
    queries_by_paper[pq["paper_id"]] = pq["queries_by_view"]
logger.info("Generated queries for {} papers".format(len(queries_by_paper)))

logger.info("\n=== Stage 4: Filtering queries ===")
query_filter = QueryFilter(llm=backend)
query_filter.run(queries_path, filtered_path)

with open(filtered_path) as f:
    filtered_queries = json.load(f)
filtered_by_query = {}
for fq in filtered_queries["results"]:
    filtered_by_query[fq["original_query"]] = fq
passed_queries = sum(1 for fq in filtered_queries["results"] if fq.get("verdict") == "Keep")
logger.info("Filtered queries: {}/{} passed (Keep)".format(passed_queries, len(filtered_queries["results"])))

logger.info("\n=== Building combined output ===")
with open(download_path) as f:
    downloaded = json.load(f)
downloaded_papers = {p["paper"]["id"]: p for p in downloaded["papers"]}

filtered_papers = {}
for r in filtered["results"]:
    paper_info = r.get("paper", {})
    paper_data = paper_info.get("paper", {})
    pid = paper_data.get("id")
    if pid:
        filtered_papers[pid] = {
            "passed": r.get("passed", False),
            "reasons": paper_info.get("details", {}),
            "original": r.get("paper", {}),
        }

combined = PipelineOutput(
    venue=VENUE,
    year=YEAR,
    generated_at=datetime.now(),
)

summarized_paper_ids = set(summarized_papers.keys())
combined.papers = []
for pid in summarized_paper_ids:
    dp = downloaded_papers.get(pid, {})
    fp = filtered_papers.get(pid, {})
    sp = summarized_papers.get(pid, {})
    pqs = queries_by_paper.get(pid, [])

    original_paper = dp.get("paper", {})
    p = PipelinePaper(
        paper_id=pid,
        paper_title=original_paper.get("title", ""),
        abstract=original_paper.get("abstract", ""),
        authors=original_paper.get("authors", []),
        venue=original_paper.get("venue", ""),
        year=original_paper.get("year", 0),
        keywords=original_paper.get("keywords", []),
        reviews=dp.get("reviews", []),
        comments=dp.get("comments", []),
        rebuttals=dp.get("rebuttals", []),
        decision=dp.get("decision"),
    )

    p.passed = fp.get("passed", False)
    p.filter_reasons = fp.get("reasons", {})
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
combined.total_passed = len(summarized_paper_ids)
combined.total_queries = sum(len(p.queries) for p in combined.papers)
combined.total_queries_passed = sum(1 for p in combined.papers for fq in p.filtered_queries if fq.get("verdict") == "Keep")

output_path = OUTPUT_DIR / "pipeline_output.json"
save_json(output_path, combined)

logger.info("\n=== Pipeline Complete! ===")
logger.info("Combined output: {}".format(output_path))
logger.info("Summary:")
logger.info("  Papers filtered: {}/{}".format(FILTER_LIMIT, filtered["total_input"]))
logger.info("  Papers passed filter: {}".format(filtered["total_passed"]))
logger.info("  Papers summarized: {}".format(combined.total_papers))
logger.info("  Total queries: {}".format(combined.total_queries))
logger.info("  Passed queries (Keep): {}".format(combined.total_queries_passed))