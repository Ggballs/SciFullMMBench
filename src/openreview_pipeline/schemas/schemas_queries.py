from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .schemas_summarize import ViewBulletPoints


class RetrievalQuery(BaseModel):
    query_text: str
    query_type: str = "IR"
    is_multimodal: bool = False
    source_view: str
    related_bullet_indice: Optional[int] = None
    related_bullet_justification: Optional[str] = None
    multimodal_rationale: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_related_bullet_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "related_bullet_indice" in data:
            return data

        legacy_keys = (
            "related_bullet_indices",
            "related_bulletpoint_indices",
            "related_bullet_point_indices",
            "bullet_indice",
            "bullet_indices",
            "bullet_point_indices",
        )
        for key in legacy_keys:
            if key in data:
                normalized = dict(data)
                normalized["related_bullet_indice"] = data.get(key)
                return normalized
        return data

    @field_validator("related_bullet_indice", mode="before")
    @classmethod
    def normalize_related_bullet_indice(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        if not isinstance(value, list):
            value = [value]

        for item in value:
            try:
                index = int(item)
            except (TypeError, ValueError):
                continue
            if index > 0:
                return index
        return None

    @field_validator("related_bullet_justification", mode="before")
    @classmethod
    def normalize_related_bullet_justification(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @model_validator(mode="after")
    def clear_non_multimodal_rationale(self) -> "RetrievalQuery":
        if not self.is_multimodal:
            self.multimodal_rationale = None
        return self

    @field_validator("multimodal_rationale", mode="before")
    @classmethod
    def normalize_multimodal_rationale(cls, value: Any) -> Optional[str]:
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


class RetrievalEvaluation(BaseModel):
    full_paper_reliance: str = Field(description="PASS | FAIL")
    false_negative_risk: Optional[str] = Field(default=None, description="LOW | HIGH")
    reasoning: str = ""


class RuleBasedStyleEvaluation(BaseModel):
    char_length: int
    token_length: int
    question_template: str
    matched_pattern: str
    matched_template: bool


class LLMStyleEvaluation(BaseModel):
    specificity_calibration_score: Optional[int] = None
    specificity_calibration_rationale: str = ""
    lexical_naturalism_score: Optional[int] = None
    lexical_naturalism_rationale: str = ""
    semantic_constraint_count: int = 0
    semantic_constraint_rationale: str = ""


class StyleEvaluation(BaseModel):
    rule_based: RuleBasedStyleEvaluation
    llm_based: LLMStyleEvaluation


class QueryHardNegativeContext(BaseModel):
    hard_negatives: List[Dict[str, Any]] = Field(default_factory=list)
    positives: List[Dict[str, Any]] = Field(default_factory=list)
    keywords_extracted: List[str] = Field(default_factory=list)
    search_queries_used: List[str] = Field(default_factory=list)
    retrieved_candidates: int = 0
    mining_method: Optional[str] = None


class QueryAnalysisEntry(BaseModel):
    query_text: str
    query_type: str = "IR"
    source_view: str
    is_multimodal: bool = False
    related_bullet_indice: Optional[int] = None
    related_bullet_justification: Optional[str] = None
    multimodal_rationale: Optional[str] = None
    hard_negative_context: Optional[QueryHardNegativeContext] = None
    retrieval_evaluation: RetrievalEvaluation
    style_evaluation: StyleEvaluation
    decision: str = Field(description="Keep | Hard Reject")


class PaperQueryAnalysis(BaseModel):
    paper_id: str
    paper_title: str
    abstract: Optional[str] = None
    pdf_url: Optional[str] = None
    openreview_url: Optional[str] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    authors: List[str] = Field(default_factory=list)
    summary_views: List[ViewBulletPoints] = Field(default_factory=list)
    queries: List[QueryAnalysisEntry] = Field(default_factory=list)


class QueryAnalysisDataset(BaseModel):
    papers: List[PaperQueryAnalysis]
    total_papers: int
    total_queries: int
    dataset_summary: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.now)
