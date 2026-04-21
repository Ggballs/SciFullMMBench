from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any


class OpenReviewPaper(BaseModel):
    id: str
    title: str
    abstract: str
    authors: List[str]
    venue: str
    year: int
    pdf_url: Optional[str] = None
    cdate: Optional[int] = None
    odate: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    num_citations: Optional[int] = None
    rating: Optional[int] = None
    tags: List[str] = Field(default_factory=list)
    venueid: Optional[str] = None
    submission_number: Optional[int] = None


class ReviewContent(BaseModel):
    title: Optional[str] = None
    summary_of_impact: Optional[str] = None
    strength: Optional[str] = None
    weaknesses: Optional[str] = None
    questions_and_concerns: Optional[str] = None
    limitations: Optional[str] = None
    rating: Optional[str] = None
    confidence: Optional[str] = None
    should_accept: Optional[str] = None
    should_reject: Optional[str] = None
    ethics_review_complete: Optional[str] = None


class Review(BaseModel):
    id: str
    paper_id: str
    invitation: str
    content: Dict[str, Any]
    number: int
    cdate: Optional[int] = None
    tcdate: Optional[int] = None


class Rebuttal(BaseModel):
    id: str
    paper_id: str
    invitation: str
    content: Dict[str, Any]
    number: int
    cdate: Optional[int] = None
    tcdate: Optional[int] = None


class Comment(BaseModel):
    id: str
    paper_id: str
    invitation: str
    content: Dict[str, Any]
    number: int
    cdate: Optional[int] = None
    tcdate: Optional[int] = None


class Decision(BaseModel):
    id: str
    paper_id: str
    invitation: str
    content: Dict[str, Any]
    number: int
    cdate: Optional[int] = None


class OpenReviewPaperWithMetadata(BaseModel):
    paper: OpenReviewPaper
    reviews: List[Review] = Field(default_factory=list)
    rebuttals: List[Rebuttal] = Field(default_factory=list)
    comments: List[Comment] = Field(default_factory=list)
    decision: Optional[Decision] = None
    downloaded_at: datetime = Field(default_factory=datetime.now)
    source: str = "openreview"


class DownloadedPapersDataset(BaseModel):
    papers: List[OpenReviewPaperWithMetadata]
    downloaded_at: datetime = Field(default_factory=datetime.now)
    total_count: int
