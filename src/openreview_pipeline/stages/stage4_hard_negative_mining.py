import logging
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

from openreview_pipeline.llm import LLMBackend
from openreview_pipeline.schemas.schemas_queries import FilteredQueriesDataset, FilteredQuery
from openreview_pipeline.utils import load_json, save_json

logger = logging.getLogger(__name__)


class HardNegativePaper(BaseModel):
    paper_title: str
    arxiv_id: Optional[str] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    authors: List[str] = Field(default_factory=list)
    relevance_score: float = 0.0
    hard_negative_reason: str = ""
    source_query: str = ""


class HardNegativeMiningResult(BaseModel):
    query: str
    source_view: str
    hard_negatives: List[HardNegativePaper]
    keywords_extracted: List[str]
    mining_method: str = "gpt_keyword_google_scholar"


class HardNegativeMiningDataset(BaseModel):
    results: List[HardNegativeMiningResult]
    total_queries: int
    total_mined: int
    total_hard_negatives: int
    mined_at: datetime = Field(default_factory=datetime.now)


class HardNegativeMiner:
    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def extract_keywords(self, query: str) -> List[str]:
        prompt = f"""Given the following retrieval query for a scientific paper, extract 3-5 key search keywords that could be used to find related papers on Google Scholar or arXiv.

Query: {query}

Extract the most important technical keywords, methods, or topics that would help find related but DISTINCT papers (not the same paper). Focus on:
- Specific methods or techniques mentioned
- Domain-specific terminology
- Task or problem types
- Dataset names if mentioned

Output a JSON array of strings (keywords only, no explanation):
["keyword1", "keyword2", "keyword3"]
"""
        response = self.llm.generate(prompt)
        try:
            import json
            import re
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                keywords = json.loads(json_match.group())
                if isinstance(keywords, list):
                    return [k for k in keywords if isinstance(k, str)][:5]
        except Exception as e:
            logger.warning(f"Failed to parse keywords from response: {e}")
        return []

    def search_google_scholar(self, keywords: List[str], query: str) -> List[HardNegativePaper]:
        results = []
        search_query = " ".join(keywords[:3])
        prompt = f"""You are searching Google Scholar for papers related to this query:
Original Query: {query}
Search Terms: {search_query}

Find 3-5 papers that are:
1. Related to the topic/technique in the original query
2. But NOT the same paper being queried about
3. Could serve as HARD NEGATIVES (plausible matches but ultimately irrelevant)

For each paper, provide:
- Title
- arXiv ID (if available)
- A brief abstract or description
- Why it might seem relevant but is actually not the right match

Output a JSON array of objects with this structure:
[
  {{
    "paper_title": "Paper Title",
    "arxiv_id": "2301.12345" or null,
    "abstract": "Brief abstract...",
    "relevance_score": 0.7,
    "hard_negative_reason": "Why this is a hard negative..."
  }}
]

If no suitable papers found, return an empty array []. Only return actual papers, not placeholders.
"""
        response = self.llm.generate(prompt)
        try:
            import json
            import re
            json_match = re.search(r'\[[\s\S]*?\]', response, re.DOTALL)
            if json_match:
                papers = json.loads(json_match.group())
                if isinstance(papers, list):
                    for p in papers:
                        if isinstance(p, dict) and "paper_title" in p:
                            results.append(HardNegativePaper(
                                paper_title=p.get("paper_title", "Unknown"),
                                arxiv_id=p.get("arxiv_id"),
                                abstract=p.get("abstract"),
                                relevance_score=p.get("relevance_score", 0.5),
                                hard_negative_reason=p.get("hard_negative_reason", ""),
                                source_query=query,
                            ))
        except Exception as e:
            logger.warning(f"Failed to parse Google Scholar results: {e}")
        return results

    def mine_for_query(self, query: FilteredQuery) -> HardNegativeMiningResult:
        logger.debug(f"Mining hard negatives for query: {query.original_query[:50]}...")

        keywords = self.extract_keywords(query.original_query)
        logger.debug(f"Extracted keywords: {keywords}")

        hard_negatives = []
        if keywords:
            hard_negatives = self.search_google_scholar(keywords, query.original_query)

        return HardNegativeMiningResult(
            query=query.original_query,
            source_view=query.source_view,
            hard_negatives=hard_negatives,
            keywords_extracted=keywords,
            mining_method="gpt_keyword_google_scholar",
        )

    def apply(self, dataset: FilteredQueriesDataset) -> HardNegativeMiningDataset:
        logger.info(f"Mining hard negatives for {len(dataset.results)} queries")

        all_results = []
        total_hard_negatives = 0

        for query in dataset.results:
            result = self.mine_for_query(query)
            all_results.append(result)
            total_hard_negatives += len(result.hard_negatives)

        logger.info(f"Hard negative mining complete: {total_hard_negatives} hard negatives for {len(all_results)} queries")

        return HardNegativeMiningDataset(
            results=all_results,
            total_queries=len(all_results),
            total_mined=len([r for r in all_results if r.hard_negatives]),
            total_hard_negatives=total_hard_negatives,
        )

    def run(self, input_path: Path, output_path: Path) -> None:
        logger.info(f"Running hard-negative-mining stage: {input_path} -> {output_path}")
        dataset = load_json(input_path, FilteredQueriesDataset)
        result = self.apply(dataset)
        save_json(output_path, result)