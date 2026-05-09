#!/usr/bin/env python3
"""Download and filter paper-grounded forum posts.

This script is intentionally standalone. It does not implement intent
recognition yet; it only collects normalized records and writes filtered JSONL.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


CROSSVALIDATED_START_DATE = datetime(2020, 1, 1, tzinfo=timezone.utc)
CROSSVALIDATED_QUESTION_SCORE_MIN = 5
CROSSVALIDATED_ANSWER_SCORE_MIN = 5
CROSSVALIDATED_PAPER_LINK_RE = re.compile(
    r"https?://[^\s<>()\"']*(?:arxiv(?:\.org)?|doi\.org|dx\.doi\.org|aclanthology\.org)[^\s<>()\"']*|"
    r"\bdoi\s*:\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+|\b10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    re.IGNORECASE,
)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_MAX_RETRIES = 3
CROSSVALIDATED_DUMP_DATE = "20260331"
CROSSVALIDATED_ARCHIVE_URL = (
    f"https://archive.org/download/stackexchange_{CROSSVALIDATED_DUMP_DATE}/"
    f"stackexchange_{CROSSVALIDATED_DUMP_DATE}/stats.stackexchange.com.7z"
)
CROSSVALIDATED_ARCHIVE_NAME = f"stats.stackexchange.com.{CROSSVALIDATED_DUMP_DATE}.7z"
CROSSVALIDATED_EXTRACT_DIR_NAME = f"stats.stackexchange.com.{CROSSVALIDATED_DUMP_DATE}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/filter paper-grounded Cross Validated posts."
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "data"),
        help="Output data directory. Defaults to offline-process/intent-recognition/data.",
    )
    parser.add_argument(
        "--crossvalidated-posts",
        type=Path,
        default=None,
        help="Path to Cross Validated Posts.xml.",
    )
    parser.add_argument(
        "--download-crossvalidated-archive",
        action="store_true",
        help="Download the small Cross Validated .7z archive into data/raw/dumps. Extraction is not performed.",
    )
    parser.add_argument(
        "--extract-crossvalidated",
        action="store_true",
        help="Extract the Cross Validated archive with bsdtar after download.",
    )
    parser.add_argument(
        "--process-crossvalidated",
        action="store_true",
        help="Download/extract Cross Validated if needed, filter Posts.xml, then export retained queries.",
    )
    return parser.parse_args()


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_utc_datetime(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def format_utc_timestamp(value: float | int | None) -> str | None:
    if value is None:
        return None
    return format_utc_datetime(datetime.fromtimestamp(float(value), tz=timezone.utc))


def parse_score(value: str | int | None) -> int:
    try:
        return int(value) if value is not None else 0
    except ValueError:
        return 0


def extract_urls(text: str, pattern: re.Pattern[str]) -> list[str]:
    return sorted({match.group(0).rstrip(".,;") for match in pattern.finditer(text or "")})


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(records)


def print_stage(message: str) -> None:
    print(f"[stage] {message}", flush=True)


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024.0 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TB"


def parse_content_length(response: BinaryIO) -> int | None:
    value = response.headers.get("Content-Length")
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def print_download_progress(name: str, downloaded: int, total: int | None) -> None:
    if total:
        percent = min(100.0, downloaded / total * 100)
        message = f"\r[download] {name}: {percent:5.1f}% ({format_bytes(downloaded)} / {format_bytes(total)})"
    else:
        message = f"\r[download] {name}: {format_bytes(downloaded)}"
    print(message, end="", flush=True)


def download_file(url: str, target: Path, *, name: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.with_suffix(target.suffix + ".part")
    if target.exists():
        print_stage(f"{name} archive already exists: {target}")
        return target

    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        existing_size = part_path.stat().st_size if part_path.exists() else 0
        headers = {"User-Agent": "SciFullMMBench offline filter"}
        mode = "wb"
        if existing_size:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"
            print_stage(f"resuming {name} archive from {format_bytes(existing_size)}")
        else:
            print_stage(f"downloading {name} archive")

        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=60) as response:
                status = getattr(response, "status", 200)
                if existing_size and status == 200:
                    existing_size = 0
                    mode = "wb"
                response_length = parse_content_length(response)
                total = existing_size + response_length if response_length is not None else None
                downloaded = existing_size
                last_progress = 0.0
                with part_path.open(mode) as handle:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_progress >= 0.5:
                            print_download_progress(name, downloaded, total)
                            last_progress = now
                print_download_progress(name, downloaded, total)
                print()
                if total is not None and downloaded < total:
                    raise RuntimeError(
                        f"{name} download incomplete: got {format_bytes(downloaded)} "
                        f"of {format_bytes(total)}"
                    )
                part_path.replace(target)
                print_stage(f"saved {name} archive to {target}")
                return target
        except (OSError, URLError, RuntimeError) as exc:
            if attempt >= DOWNLOAD_MAX_RETRIES:
                raise RuntimeError(
                    f"Could not finish downloading {name} after {DOWNLOAD_MAX_RETRIES} attempts. "
                    f"Partial data is kept at {part_path}; rerun the same command to resume. "
                    f"Last error: {exc}"
                ) from exc
            print_stage(f"{name} download attempt {attempt} failed: {exc}; retrying")
            time.sleep(2 * attempt)

    raise RuntimeError(f"unreachable download failure for {name}")


def strip_html(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def build_query_text(record: dict[str, Any]) -> str:
    title = strip_html(str(record.get("title", "")))
    body = strip_html(str(record.get("body", "")))
    return f"{title}\n\n{body}".strip()


def extract_query_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    queries = []
    for record in records:
        queries.append(
            {
                "source": record.get("source", ""),
                "id": record.get("id", ""),
                "query": build_query_text(record),
                "title": strip_html(str(record.get("title", ""))),
                "body": strip_html(str(record.get("body", ""))),
                "score": record.get("score", 0),
                "created_at": record.get("created_at"),
                "created_date": record.get("created_date"),
                "tags": record.get("tags", []),
                "url": record.get("url", ""),
                "qualifying_answers": record.get("qualifying_answers", []),
            }
        )
    return queries


def crossvalidated_archive_path(output_dir: Path) -> Path:
    return output_dir / "raw" / "dumps" / CROSSVALIDATED_ARCHIVE_NAME


def crossvalidated_extract_dir(output_dir: Path) -> Path:
    return output_dir / "raw" / "dumps" / CROSSVALIDATED_EXTRACT_DIR_NAME


def find_crossvalidated_posts_xml(output_dir: Path) -> Path | None:
    extract_dir = crossvalidated_extract_dir(output_dir)
    candidates = list(extract_dir.rglob("Posts.xml")) if extract_dir.exists() else []
    return candidates[0] if candidates else None


def extract_crossvalidated_archive(output_dir: Path) -> Path:
    archive_path = crossvalidated_archive_path(output_dir)
    posts_xml = find_crossvalidated_posts_xml(output_dir)
    if posts_xml is not None:
        print_stage(f"Cross Validated Posts.xml already extracted: {posts_xml}")
        return posts_xml
    if not archive_path.exists():
        raise RuntimeError(f"Cross Validated archive not found: {archive_path}")
    bsdtar = shutil.which("bsdtar")
    if not bsdtar:
        raise RuntimeError("bsdtar is required to extract .7z archives on this machine.")
    extract_dir = crossvalidated_extract_dir(output_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    print_stage(f"extracting Cross Validated archive to {extract_dir}")
    subprocess.run([bsdtar, "-xf", str(archive_path), "-C", str(extract_dir)], check=True)
    posts_xml = find_crossvalidated_posts_xml(output_dir)
    if posts_xml is None:
        raise RuntimeError(f"Extraction finished but Posts.xml was not found under {extract_dir}")
    return posts_xml


def parse_crossvalidated_tags(raw_tags: str | None) -> list[str]:
    if not raw_tags:
        return []
    if raw_tags.startswith("|"):
        return [tag for tag in raw_tags.strip("|").split("|") if tag]
    return re.findall(r"<([^>]+)>", raw_tags)


def crossvalidated_question_candidate(
    attrs: dict[str, str],
) -> dict[str, Any] | None:
    if attrs.get("PostTypeId") != "1":
        return None
    created = parse_datetime(attrs.get("CreationDate"))
    if created is None or created < CROSSVALIDATED_START_DATE:
        return None
    score = parse_score(attrs.get("Score"))
    if score <= CROSSVALIDATED_QUESTION_SCORE_MIN:
        return None
    title = attrs.get("Title", "")
    if not strip_html(title).rstrip().endswith("?"):
        return None
    tags = parse_crossvalidated_tags(attrs.get("Tags"))
    post_id = attrs.get("Id", "")
    return {
        "source": "crossvalidated",
        "id": post_id,
        "title": title,
        "body": attrs.get("Body", ""),
        "score": score,
        "created_utc": created.timestamp(),
        "created_at": format_utc_datetime(created),
        "created_date": created.date().isoformat(),
        "tags": tags,
        "flair": None,
        "url": f"https://stats.stackexchange.com/questions/{post_id}",
        "qualifying_answers": [],
    }


def qualifying_crossvalidated_answer(attrs: dict[str, str], parent_id: str) -> dict[str, Any] | None:
    if attrs.get("PostTypeId") != "2" or attrs.get("ParentId") != parent_id:
        return None
    score = parse_score(attrs.get("Score"))
    if score <= CROSSVALIDATED_ANSWER_SCORE_MIN:
        return None
    body = attrs.get("Body", "")
    paper_urls = extract_urls(body, CROSSVALIDATED_PAPER_LINK_RE)
    if not paper_urls:
        return None
    created = parse_datetime(attrs.get("CreationDate"))
    return {
        "id": attrs.get("Id", ""),
        "body": body,
        "score": score,
        "created_utc": created.timestamp() if created else None,
        "created_at": format_utc_datetime(created) if created else None,
        "created_date": created.date().isoformat() if created else None,
        "url": "",
        "paper_urls": paper_urls,
    }


def iter_post_rows(posts_xml: Path) -> Iterable[dict[str, str]]:
    for _, elem in ET.iterparse(posts_xml, events=("end",)):
        if elem.tag == "row":
            yield dict(elem.attrib)
        elem.clear()


def filter_crossvalidated_posts(posts_xml: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions: dict[str, dict[str, Any]] = {}
    raw_candidates: list[dict[str, Any]] = []
    rows_seen = 0

    for attrs in iter_post_rows(posts_xml):
        rows_seen += 1
        if rows_seen % 100000 == 0:
            print_stage(f"crossvalidated: scanned {rows_seen} rows for candidate questions")
        candidate = crossvalidated_question_candidate(attrs)
        if candidate:
            questions[candidate["id"]] = candidate
            raw_candidates.append(dict(candidate))

    if not questions:
        return raw_candidates, []

    rows_seen = 0
    for attrs in iter_post_rows(posts_xml):
        rows_seen += 1
        if rows_seen % 100000 == 0:
            print_stage(f"crossvalidated: scanned {rows_seen} rows for qualifying answers")
        parent_id = attrs.get("ParentId", "")
        if parent_id not in questions:
            continue
        answer = qualifying_crossvalidated_answer(attrs, parent_id)
        if answer:
            answer["url"] = f"{questions[parent_id]['url']}#{answer['id']}"
            questions[parent_id]["qualifying_answers"].append(answer)

    filtered = [question for question in questions.values() if question["qualifying_answers"]]
    return raw_candidates, filtered


def download_crossvalidated_archive(output_dir: Path) -> Path:
    dump_dir = output_dir / "raw" / "dumps"
    dump_dir.mkdir(parents=True, exist_ok=True)
    target = dump_dir / CROSSVALIDATED_ARCHIVE_NAME
    return download_file(CROSSVALIDATED_ARCHIVE_URL, target, name="crossvalidated")


def latest_iso(current: str | None, candidate: str | None) -> str | None:
    if not candidate:
        return current
    if current is None:
        return candidate
    return max(current, candidate)


def summarize_filtered_qa_times(filtered_path: Path) -> dict[str, Any]:
    if not filtered_path.exists():
        return {}
    latest_question_created_at: str | None = None
    latest_answer_created_at: str | None = None
    question_count = 0
    answer_count = 0
    for record in read_jsonl(filtered_path):
        question_count += 1
        latest_question_created_at = latest_iso(
            latest_question_created_at,
            record.get("created_at") or format_utc_timestamp(record.get("created_utc")),
        )
        for answer in record.get("qualifying_answers", []):
            answer_count += 1
            latest_answer_created_at = latest_iso(
                latest_answer_created_at,
                answer.get("created_at") or format_utc_timestamp(answer.get("created_utc")),
            )
    latest_qa_created_at = latest_iso(latest_question_created_at, latest_answer_created_at)
    return {
        "qa_question_count": question_count,
        "qa_answer_count": answer_count,
        "qa_latest_question_created_at": latest_question_created_at,
        "qa_latest_answer_created_at": latest_answer_created_at,
        "qa_latest_created_at": latest_qa_created_at,
    }


def write_report(output_dir: Path, counts: dict[str, Any]) -> None:
    report_dir = output_dir.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "filter_summary.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["# Filter Summary", ""]
    for key, value in sorted(counts.items()):
        lines.append(f"- {key}: {value}")
    (report_dir / "filter_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.process_crossvalidated:
        args.download_crossvalidated_archive = True
        args.extract_crossvalidated = True

    output_dir = Path(args.output_dir).expanduser().resolve()
    raw_dir = output_dir / "raw"
    filtered_path = output_dir / "filtered" / "paper_grounded_posts.jsonl"
    queries_path = output_dir / "filtered" / "queries.json"
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    will_write_filtered = args.crossvalidated_posts is not None or args.extract_crossvalidated
    if will_write_filtered:
        filtered_path.write_text("", encoding="utf-8")

    counts: dict[str, Any] = {}

    if args.download_crossvalidated_archive:
        downloaded = download_crossvalidated_archive(output_dir)
        counts["downloaded_crossvalidated_archive"] = int(downloaded.exists())

    if args.extract_crossvalidated:
        args.crossvalidated_posts = extract_crossvalidated_archive(output_dir)

    if args.crossvalidated_posts:
        raw, filtered = filter_crossvalidated_posts(args.crossvalidated_posts)
        counts["crossvalidated_raw"] = write_jsonl(raw_dir / "crossvalidated_questions.jsonl", raw)
        counts["crossvalidated_filtered"] = write_jsonl(filtered_path, filtered)
        counts["queries_json_total"] = write_json(queries_path, extract_query_records(filtered))
        counts.update(summarize_filtered_qa_times(filtered_path))

    counts["filtered_total"] = sum(
        value for key, value in counts.items()
        if key.endswith("_filtered") and isinstance(value, int)
    )
    if counts["filtered_total"] == 0 and isinstance(counts.get("qa_question_count"), int):
        counts["filtered_total"] = counts["qa_question_count"]
    write_report(output_dir, counts)
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
