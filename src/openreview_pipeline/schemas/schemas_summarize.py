from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from typing import Optional, List, Any


class BulletPoint(BaseModel):
    index: int
    text: str


class ViewBulletPoints(BaseModel):
    view_name: str
    summary: Optional[str] = None
    bullet_points: List[BulletPoint] = Field(default_factory=list)

    @field_validator("bullet_points", mode="before")
    @classmethod
    def normalize_bullet_points(cls, value: Any) -> List[dict]:
        if not value:
            return []
        if not isinstance(value, list):
            return []

        normalized: List[dict] = []
        for idx, item in enumerate(value, 1):
            if isinstance(item, dict):
                text = item.get("text") or item.get("bullet_point") or item.get("content") or ""
            else:
                text = str(item)
            text = text.strip()
            if text:
                normalized.append({"index": idx, "text": text})
        return normalized

    @model_validator(mode="after")
    def assign_indices(self) -> "ViewBulletPoints":
        self.bullet_points = [
            BulletPoint(index=idx, text=bullet.text.strip())
            for idx, bullet in enumerate(self.bullet_points, 1)
            if bullet.text.strip()
        ]
        if self.summary is not None:
            self.summary = self.summary.strip() or None
        return self


class PaperSummary(BaseModel):
    paper_id: str
    paper_title: str
    views: List[ViewBulletPoints]
    generated_at: datetime = Field(default_factory=datetime.now)


class SummarizedPapersDataset(BaseModel):
    summaries: List[PaperSummary]
    total_papers: int
    generated_at: datetime = Field(default_factory=datetime.now)
