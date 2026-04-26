import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DEFAULT_YEAR_THRESHOLD = 2021
OPENREVIEW_BASEURL = "https://api2.openreview.net"

REVIEW_FIELD_MAPPING = {
    "summary": "review",
    "strengths": "pros",
    "weaknesses": "cons",
    "soundness": "quality",
    "presentation": "clarity",
    "contribution": "originality",
}


def try_import_openreview():
    try:
        import openreview

        return openreview
    except ImportError:
        logger.warning("openreview package not installed. Install with: pip install openreview")
        return None


def unwrap_value(value: Any) -> Any:
    """Return OpenReview's wrapped `{"value": ...}` payload as a plain value."""
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def normalize_content(content: Any) -> Dict[str, Any]:
    """Normalize a note content payload into a plain dictionary."""
    if not isinstance(content, dict):
        return {}
    return {key: unwrap_value(value) for key, value in content.items()}


def note_to_dict(note: Any) -> Dict[str, Any]:
    """Convert an OpenReview note object or raw dict into a normalized dict."""
    if isinstance(note, dict):
        invitation = note.get("invitation")
        invitations = note.get("invitations")
        content = note.get("content", {}) or {}
        return {
            "id": note.get("id"),
            "invitation": invitation,
            "invitations": invitations,
            "number": note.get("number"),
            "forum": note.get("forum"),
            "replyto": note.get("replyto"),
            "signatures": note.get("signatures"),
            "readers": note.get("readers"),
            "writers": note.get("writers"),
            "cdate": note.get("cdate"),
            "tcdate": note.get("tcdate"),
            "content": normalize_content(content),
        }

    invitation = getattr(note, "invitation", None)
    invitations = getattr(note, "invitations", None)
    content = getattr(note, "content", {}) or {}
    return {
        "id": getattr(note, "id", None),
        "invitation": invitation,
        "invitations": invitations,
        "number": getattr(note, "number", None),
        "forum": getattr(note, "forum", None),
        "replyto": getattr(note, "replyto", None),
        "signatures": getattr(note, "signatures", None),
        "readers": getattr(note, "readers", None),
        "writers": getattr(note, "writers", None),
        "cdate": getattr(note, "cdate", None),
        "tcdate": getattr(note, "tcdate", None),
        "content": normalize_content(content),
    }


def classify_note(note_dict: Dict[str, Any]) -> str:
    """Classify a forum note based on invitation names and content keys.

    The ordering matters. For example, an invitation like
    "Response_to_Review" contains the word "review" but should still be
    treated as a rebuttal/response.
    """
    invitation = str(note_dict.get("invitation") or "").lower()
    invitations = " ".join(str(item).lower() for item in (note_dict.get("invitations") or []))
    content_keys = {str(key).lower() for key in (note_dict.get("content") or {}).keys()}
    combined_text = f"{invitation} {invitations}"

    if "decision" in combined_text or "decision" in content_keys:
        return "decision"
    if "meta_review" in combined_text or "metareview" in combined_text:
        return "meta_review"
    if (
        "rebuttal" in combined_text
        or "author_response" in combined_text
        or "response" in combined_text
    ):
        return "rebuttal"
    if "official_review" in combined_text:
        return "review"
    if "review" in combined_text and "meta" not in combined_text:
        return "review"
    if "comment" in combined_text:
        return "comment"
    return "other"


class OpenReviewAPIDownloader:
    def __init__(
        self,
        venue: str = "ICLR.cc",
        year: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        self.venue = venue
        self.year = year or datetime.now().year
        self.username = username
        self.password = password
        self.token = token
        self.output_dir = Path(output_dir) if output_dir else None
        self.client = None

    def _get_client(self):
        openreview = try_import_openreview()
        if openreview is None:
            return None

        try:
            from openreview.api import OpenReviewClient as ClientClass
        except ImportError:
            ClientClass = openreview.Client

        if self.token:
            return ClientClass(baseurl=OPENREVIEW_BASEURL, token=self.token)
        if self.username and self.password:
            return ClientClass(
                baseurl=OPENREVIEW_BASEURL,
                username=self.username,
                password=self.password,
            )

        logger.warning("No OpenReview credentials provided. Set username/password or token.")
        return None

    def _get_blind_submission_invitation(self) -> str:
        return f"{self.venue}/{self.year}/Conference/-/Submission"

    def _extract_authors(self, note: Any) -> List[str]:
        content = getattr(note, "content", None)
        authors = content.get("authors", []) if content is not None else note.get("authors", [])
        return authors if isinstance(authors, list) else []

    def _extract_keywords(self, note: Any) -> List[str]:
        content = getattr(note, "content", None)
        keywords = content.get("keywords", []) if content is not None else note.get("keywords", [])

        if isinstance(keywords, list):
            return keywords
        if isinstance(keywords, str):
            return [item.strip() for item in keywords.split(",") if item.strip()]
        return []

    def _normalize_review_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize content and apply lightweight field aliases."""
        raw_content = normalize_content(content)
        normalized = {
            REVIEW_FIELD_MAPPING.get(key, key): value
            for key, value in raw_content.items()
            if value is not None
        }
        return normalized

    def _is_accepted(self, decision_content: Dict[str, Any]) -> bool:
        decision_value = decision_content.get("decision", "")
        if isinstance(decision_value, dict):
            decision_value = decision_value.get("value", "")
        return "accept" in str(decision_value).lower()

    def _fetch_forum_note_candidates(self, client: Any, submission: Any) -> List[Any]:
        """Return all known raw notes in the submission forum.

        We combine direct replies from submission details with a full forum
        query because different OpenReview configurations may expose one or the
        other more reliably.
        """
        details = getattr(submission, "details", {}) or {}
        direct_replies = details.get("directReplies", []) if isinstance(details, dict) else []

        try:
            forum_notes = list(client.get_all_notes(forum=submission.id))
        except Exception as exc:
            logger.warning("forum fetch failed for %s: %s", submission.id, exc)
            forum_notes = []

        return list(direct_replies) + forum_notes

    def _collect_forum_notes(self, client: Any, submission: Any) -> List[Dict[str, Any]]:
        """Collect every note in the submission forum, including nested replies.

        This intentionally keeps replies to reviews, comments, and rebuttals.
        Filtering to only `replyto == submission.id` would drop author responses
        nested under a review thread.
        """
        merged_notes: List[Dict[str, Any]] = []
        seen_note_ids = set()

        for raw_note in self._fetch_forum_note_candidates(client, submission):
            note_dict = note_to_dict(raw_note)
            note_id = note_dict.get("id")

            if not note_id or note_id == submission.id or note_id in seen_note_ids:
                continue

            seen_note_ids.add(note_id)
            merged_notes.append(note_dict)

        merged_notes.sort(
            key=lambda note: (
                note.get("cdate") is None,
                note.get("cdate") if note.get("cdate") is not None else float("inf"),
                note.get("tcdate") if note.get("tcdate") is not None else float("inf"),
                str(note.get("number") or ""),
                str(note.get("id") or ""),
            )
        )
        return merged_notes

    def _build_paper(self, submission: Any):
        from openreview_pipeline.schemas import OpenReviewPaper

        content = getattr(submission, "content", {}) or {}
        title = content.get("title", "")
        abstract = content.get("abstract", "")
        venueid = content.get("venueid")
        paper_id = submission.id

        paper_data = {
            "id": paper_id,
            "title": unwrap_value(title) if isinstance(title, dict) else str(title),
            "abstract": unwrap_value(abstract) if isinstance(abstract, dict) else str(abstract),
            "authors": self._extract_authors(submission),
            "venue": self.venue,
            "year": self.year,
            "pdf_url": f"https://openreview.net/pdf?id={paper_id}",
            "keywords": self._extract_keywords(submission),
            "venueid": unwrap_value(venueid) if isinstance(venueid, dict) else venueid,
            "submission_number": getattr(submission, "number", None),
        }
        return OpenReviewPaper(**paper_data)

    def _split_forum_notes(self, submission_id: str, forum_notes: Iterable[Dict[str, Any]]) -> Dict[str, List[Any]]:
        from openreview_pipeline.schemas import Comment, Decision, Rebuttal, Review

        buckets: Dict[str, List[Any]] = {
            "reviews": [],
            "rebuttals": [],
            "comments": [],
            "meta_reviews": [],
            "decisions": [],
            "others": [],
        }

        def _safe_invitation(note: Dict[str, Any]) -> str:
            invitation = note.get("invitation")
            if isinstance(invitation, str) and invitation:
                return invitation

            invitations = note.get("invitations")
            if isinstance(invitations, list):
                for item in invitations:
                    if isinstance(item, str) and item:
                        return item

            return "unknown_invitation"

        def _safe_number(value: Any) -> int:
            if isinstance(value, int):
                return value
            try:
                return int(value)
            except Exception:
                return 0

        for note_idx, note_dict in enumerate(forum_notes):
            note_kind = classify_note(note_dict)
            content = self._normalize_review_content(note_dict.get("content", {}))
            note_id = note_dict.get("id") or f"{submission_id}_{note_kind}_{note_idx}"
            common_kwargs = {
                "id": str(note_id),
                "paper_id": submission_id,
                "invitation": _safe_invitation(note_dict),
                "content": content,
                "number": _safe_number(note_dict.get("number", 0)),
            }

            if note_kind == "review":
                buckets["reviews"].append(
                    Review(
                        **common_kwargs,
                        cdate=note_dict.get("cdate"),
                        tcdate=note_dict.get("tcdate"),
                    )
                )
            elif note_kind == "rebuttal":
                buckets["rebuttals"].append(
                    Rebuttal(
                        **common_kwargs,
                        cdate=note_dict.get("cdate"),
                        tcdate=note_dict.get("tcdate"),
                    )
                )
            elif note_kind == "comment":
                buckets["comments"].append(
                    Comment(
                        **common_kwargs,
                        cdate=note_dict.get("cdate"),
                        tcdate=note_dict.get("tcdate"),
                    )
                )
            elif note_kind == "decision":
                buckets["decisions"].append(Decision(**common_kwargs))
            elif note_kind == "meta_review":
                buckets["meta_reviews"].append(note_dict)
            else:
                buckets["others"].append(note_dict)

        return buckets

    def _should_keep_paper(
        self,
        note_buckets: Dict[str, List[Any]],
        accepted_only: bool,
    ) -> Tuple[bool, str]:
        if not note_buckets["reviews"]:
            return False, "no_reviews"

        if not note_buckets["decisions"]:
            return False, "no_decision"

        decision = note_buckets["decisions"][0]
        if accepted_only and not self._is_accepted(decision.content):
            return False, "not_accepted"

        return True, "ok"

    def fetch_papers(self, limit: Optional[int] = None, accepted_only: bool = True) -> List[dict]:
        from openreview_pipeline.schemas import OpenReviewPaperWithMetadata

        client = self._get_client()
        if client is None:
            logger.error("Cannot connect to OpenReview API. Check credentials.")
            return []

        invitation = self._get_blind_submission_invitation()
        logger.info("Fetching submissions from %s", invitation)

        try:
            submissions = list(client.get_all_notes(invitation=invitation))
            logger.info("Total submissions found: %s", len(submissions))
        except Exception as exc:
            logger.error("Failed to fetch submissions: %s", exc)
            return []

        papers = []
        skip_counts = {"no_reviews": 0, "no_decision": 0, "not_accepted": 0}

        progress = tqdm(
            submissions,
            total=len(submissions),
            desc="Downloading papers",
            unit="paper",
            dynamic_ncols=True,
        )
        for submission in progress:
            try:
                paper = self._build_paper(submission)
                forum_notes = self._collect_forum_notes(client, submission)
                note_buckets = self._split_forum_notes(submission.id, forum_notes)

                keep_paper, reason = self._should_keep_paper(note_buckets, accepted_only=accepted_only)
                if not keep_paper:
                    skip_counts[reason] += 1
                    continue

                paper_with_metadata = OpenReviewPaperWithMetadata(
                    paper=paper,
                    reviews=note_buckets["reviews"],
                    rebuttals=note_buckets["rebuttals"],
                    comments=note_buckets["comments"],
                    decision=note_buckets["decisions"][0],
                )
                papers.append(paper_with_metadata)
                progress.set_postfix_str(f"kept={len(papers)}")

                if limit is not None and len(papers) >= limit:
                    logger.info("Reached limit of %s papers", limit)
                    break

            except Exception as exc:
                logger.warning("Failed to process submission %s: %s", submission.id, exc)
        progress.close()

        logger.info(
            "Fetched %s papers (skipped: %s no reviews, %s no decision, %s not accepted)",
            len(papers),
            skip_counts["no_reviews"],
            skip_counts["no_decision"],
            skip_counts["not_accepted"],
        )
        return papers

    def fetch_paper_by_forum_id(self, forum_id: str) -> List[dict]:
        from openreview_pipeline.schemas import OpenReviewPaperWithMetadata

        client = self._get_client()
        if client is None:
            logger.error("Cannot connect to OpenReview API. Check credentials.")
            return []

        try:
            submission = client.get_note(forum_id)
        except TypeError:
            submission = client.get_note(id=forum_id)
        except Exception as exc:
            logger.error("Failed to fetch forum %s: %s", forum_id, exc)
            return []

        try:
            paper = self._build_paper(submission)
            forum_notes = self._collect_forum_notes(client, submission)
            note_buckets = self._split_forum_notes(submission.id, forum_notes)
            paper_with_metadata = OpenReviewPaperWithMetadata(
                paper=paper,
                reviews=note_buckets["reviews"],
                rebuttals=note_buckets["rebuttals"],
                comments=note_buckets["comments"],
                decision=note_buckets["decisions"][0] if note_buckets["decisions"] else None,
            )
            return [paper_with_metadata]
        except Exception as exc:
            logger.error("Failed to process forum %s: %s", forum_id, exc)
            return []


class DatasetDownloader:
    def __init__(
        self,
        venue: str = "ICLR",
        year_threshold: int = DEFAULT_YEAR_THRESHOLD,
        output_dir: Optional[str] = None,
    ):
        self.venue = venue.replace(".", ".")
        self.year_threshold = year_threshold
        self.output_dir = Path(output_dir) if output_dir else None
        self._openreview_downloader = None

    def set_openreview_credentials(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
    ):
        venue_id = self.venue.replace(" ", "_")
        self._openreview_downloader = OpenReviewAPIDownloader(
            venue=venue_id,
            year=self.year_threshold,
            username=username,
            password=password,
            token=token,
            output_dir=self.output_dir,
        )

    def fetch_recent_papers(
        self,
        limit: Optional[int] = None,
        forum_id: Optional[str] = None,
    ) -> List:
        from openreview_pipeline.schemas import OpenReviewPaper, OpenReviewPaperWithMetadata

        if self._openreview_downloader:
            if forum_id:
                return self._openreview_downloader.fetch_paper_by_forum_id(forum_id)
            return self._openreview_downloader.fetch_papers(limit=limit, accepted_only=True)

        logger.warning("OpenReview credentials not set. Using stub data.")
        logger.info("Would fetch papers from %s from year %s", self.venue, self.year_threshold)

        if forum_id:
            return [
                OpenReviewPaperWithMetadata(
                    paper=OpenReviewPaper(
                        id=forum_id,
                        title=f"Sample Paper {forum_id}",
                        abstract=f"This is a sample abstract for paper {forum_id}. " * 5,
                        authors=[f"Author {author_idx}" for author_idx in range(3)],
                        venue=self.venue,
                        year=self.year_threshold,
                        pdf_url=f"https://openreview.net/pdf?id={forum_id}",
                        keywords=["AI", "ML"],
                        venueid=f"{self.venue}/{self.year_threshold}",
                        submission_number=1,
                    )
                )
            ]

        papers = []
        count = 0
        for year in range(datetime.now().year, self.year_threshold - 1, -1):
            for index in range(10):
                if limit is not None and count >= limit:
                    return papers

                papers.append(
                    OpenReviewPaperWithMetadata(
                        paper=OpenReviewPaper(
                            id=f"paper_{year}_{index}",
                            title=f"Sample Paper {year}-{index}",
                            abstract=f"This is a sample abstract for paper {year}-{index}. " * 5,
                            authors=[f"Author {author_idx}" for author_idx in range(3)],
                            venue=self.venue,
                            year=year,
                            keywords=["AI", "ML"],
                            venueid=f"{self.venue}/{year}",
                            submission_number=index,
                        )
                    )
                )
                count += 1

        return papers

    def run(
        self,
        output_path: Path,
        limit: Optional[int] = None,
        forum_id: Optional[str] = None,
    ) -> None:
        from openreview_pipeline.schemas import DownloadedPapersDataset
        from openreview_pipeline.utils import save_json

        papers = self.fetch_recent_papers(limit=limit, forum_id=forum_id)
        dataset = DownloadedPapersDataset(papers=papers, total_count=len(papers))
        save_json(output_path, dataset)
