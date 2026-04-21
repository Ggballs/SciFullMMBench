from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any


class ViewBulletPoints(BaseModel):
    view_name: str
    bullet_points: List[str]


class PaperSummary(BaseModel):
    paper_id: str
    paper_title: str
    views: List[ViewBulletPoints]
    raw_summary: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.now)


class SummarizedPapersDataset(BaseModel):
    summaries: List[PaperSummary]
    total_papers: int
    generated_at: datetime = Field(default_factory=datetime.now)
