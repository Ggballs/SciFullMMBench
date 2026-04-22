from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, List, Any


class RetrievalQuery(BaseModel):
    query_text: str
    is_multimodal: bool = False
    source_view: str
    related_bullet_indices: List[int] = Field(default_factory=list)
    related_bullet_justification: Optional[str] = None

    @field_validator("related_bullet_indices", mode="before")
    @classmethod
    def normalize_related_bullet_indices(cls, value: Any) -> List[int]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]

        normalized: List[int] = []
        for item in value:
            try:
                index = int(item)
            except (TypeError, ValueError):
                continue
            if index > 0 and index not in normalized:
                normalized.append(index)
        return normalized

    @field_validator("related_bullet_justification", mode="before")
    @classmethod
    def normalize_related_bullet_justification(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


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
