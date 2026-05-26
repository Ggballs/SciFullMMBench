from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from .schemas import OpenReviewPaperWithMetadata


class FilterRuleResult(BaseModel):
    accepted: bool
    similar_paper: bool
    multimodal_info: bool
    multimodal_evidence_refs: List[str] = Field(default_factory=list)
    multimodal_source_refs: List[str] = Field(default_factory=list)
    meaningful_multimodal_snippet_count: int = 0
    total_multimodal_snippet_count: int = 0
    multimodal_evidence_groups: List[Dict[str, Any]] = Field(default_factory=list)
    multimodal_evidence_reason: Optional[str] = None


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
