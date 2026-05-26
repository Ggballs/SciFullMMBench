from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class PipelineQuery(BaseModel):
    query_text: str
    query_type: str = "IR"
    is_multimodal: bool = False
    source_view: str = ""
    related_bullet_indice: Optional[int] = None
    related_bullet_justification: Optional[str] = None
    multimodal_rationale: Optional[str] = None
    retrieved_golden_queries: list[dict[str, Any]] = Field(default_factory=list)
    hard_negative_context: Optional[dict[str, Any]] = None
    query_analysis: Optional[dict[str, Any]] = None


class PipelinePaper(BaseModel):
    paper_id: str
    paper_title: str = ""
    paper_dir: Optional[str] = None
    openreview: dict[str, Any] = Field(default_factory=dict)
    filter_status: Optional[dict[str, Any]] = None
    summary_views: list[dict[str, Any]] = Field(default_factory=list)
    queries: list[PipelineQuery] = Field(default_factory=list)


class PipelineOutput(BaseModel):
    artifact_type: str = "final_pipeline_output"
    generated_at: datetime = Field(default_factory=datetime.now)
    paths: dict[str, Optional[str]] = Field(default_factory=dict)
    dataset_overview: dict[str, Any] = Field(default_factory=dict)
    query_analysis_summary: dict[str, Any] = Field(default_factory=dict)
    stage5_summary: dict[str, Any] = Field(default_factory=dict)
    papers: list[PipelinePaper] = Field(default_factory=list)
