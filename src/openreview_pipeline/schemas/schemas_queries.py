from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


class RetrievalQuery(BaseModel):
    query_text: str
    is_multimodal: bool = False
    source_view: str


class GeneratedQueriesForPaper(BaseModel):
    paper_id: str
    paper_title: str
    queries_by_view: List[RetrievalQuery]
    generated_at: datetime = Field(default_factory=datetime.now)


class GeneratedQueriesDataset(BaseModel):
    papers_queries: List[GeneratedQueriesForPaper]
    total_papers: int
    total_queries: int
    generated_at: datetime = Field(default_factory=datetime.now)


class QueryDimensions(BaseModel):
    full_paper_reliance: str = Field(description="PASS | FAIL")
    authenticity: str = Field(description="PASS | FAIL")
    relevance: str = Field(description="PASS | FAIL")
    difficulty: str = Field(description="PASS | TOO_EASY | TOO_HARD")
    false_negative_risk: str = Field(description="LOW | HIGH")


class FilteredQuery(BaseModel):
    original_query: str
    is_multimodal: bool = False
    source_view: str
    dimensions: QueryDimensions
    reasoning: str
    verdict: str = Field(description="Keep | Revise | Hard Reject")
    revised_query: Optional[str] = None


class FilteredQueriesDataset(BaseModel):
    results: List[FilteredQuery]
    total_input: int
    total_passed: int
    total_filtered: int
    filtered_at: datetime = Field(default_factory=datetime.now)