from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


class PipelinePaper(BaseModel):
    paper_id: str
    paper_title: str
    abstract: str = ""
    authors: List[str] = Field(default_factory=list)
    venue: str = ""
    year: int = 0
    keywords: List[str] = Field(default_factory=list)

    reviews: List[dict] = Field(default_factory=list)
    comments: List[dict] = Field(default_factory=list)
    rebuttals: List[dict] = Field(default_factory=list)
    decision: Optional[dict] = None

    passed: bool = False
    filter_reasons: dict = Field(default_factory=dict)

    summary: Optional[dict] = None

    queries: List[dict] = Field(default_factory=list)
    filtered_queries: List[dict] = Field(default_factory=list)


class PipelineOutput(BaseModel):
    venue: str = ""
    year: int = 0
    total_papers: int = 0
    total_passed: int = 0
    total_queries: int = 0
    total_queries_passed: int = 0

    papers: List[PipelinePaper] = Field(default_factory=list)

    generated_at: datetime = Field(default_factory=datetime.now)