from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from .schemas import OpenReviewPaperWithMetadata


class FilterRuleResult(BaseModel):
    accepted: bool
    similar_paper: bool
    multimodal_info: bool


class FilterResult(BaseModel):
    paper: OpenReviewPaperWithMetadata
    passed: bool
    details: FilterRuleResult
    filtered_at: datetime = Field(default_factory=datetime.now)


class FilteredPapersDataset(BaseModel):
    results: List[FilterResult]
    total_input: int
    total_passed: int
    total_filtered: int
    filtered_at: datetime = Field(default_factory=datetime.now)
