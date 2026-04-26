from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any, Iterable, Optional

from PIL import Image
from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "pdf2interleaved"
DEFAULT_RENDER_DPI = 450

CAPTION_RE = re.compile(r"^(Figure|Fig\.?|Table)\s+([A-Za-z]?\d+)\s*([:.\-])\s*(.+)$", re.IGNORECASE)
SECTION_HEADING_RE = re.compile(r"^(?:\d+|[A-Z])(?:\.\d+)*\s+[A-Z]")
TOP_LEVEL_MAIN_HEADING_RE = re.compile(r"^\d+\s+[A-Z]")
TOP_LEVEL_APPENDIX_HEADING_RE = re.compile(r"^[A-Z]\s+[A-Z]")
WORD_RE = re.compile(r"\w+")

HEADING_REPLACEMENTS = {
    "RELATEDWORK": "RELATED WORK",
    "DATASETCONSTRUCTION": "DATASET CONSTRUCTION",
    "ANNOTATIONGUIDELINE": "ANNOTATION GUIDELINE",
    "ANNOTATIONINTERFACE": "ANNOTATION INTERFACE",
    "ANNOTATORBIOGRAPHY": "ANNOTATOR BIOGRAPHY",
    "DATASETCONSTRUCTIONPROMPTS": "DATASET CONSTRUCTION PROMPTS",
    "THEOREMDATABASECONSTRUCTION": "THEOREM DATABASE CONSTRUCTION",
    "WIKIPEDIACONTENTPROCESSINGPIPELINE": "WIKIPEDIA CONTENT PROCESSING PIPELINE",
    "ANALYSISDETAILS": "ANALYSIS DETAILS",
    "TEST-TIMESCALING": "TEST-TIME SCALING",
    "THEUSE": "THE USE",
    "OFLARGE": "OF LARGE",
    "LARGELANGUAGEMODELS": "LARGE LANGUAGE MODELS",
    "DETAILEDRESULTS": "DETAILED RESULTS",
    "EXPERIMENTDETAILS": "EXPERIMENT DETAILS",
    "DATAEXAMPLES": "DATA EXAMPLES",
}


@dataclass
class TextFragment:
    text: str
    x: float
    y: float
    font_size: float


@dataclass
class TextLine:
    text: str
    x_min: float
    x_max: float
    y: float
    font_size: float


def _default_output_dir(pdf_path: Path) -> Path:
    return DEFAULT_OUTPUT_ROOT / pdf_path.stem


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _estimate_text_width(text: str, font_size: float) -> float:
    compact = text.replace(" ", "")
    return max(font_size * 0.55 * max(len(compact), 1), font_size)


def _join_lines_compact(lines: list[str]) -> str:
    pieces: list[str] = []
    for line in lines:
        normalized = _normalize_whitespace(line)
        if not normalized:
            continue
        if pieces and pieces[-1].endswith("-"):
            pieces[-1] = pieces[-1][:-1] + normalized
        else:
            pieces.append(normalized)
    return " ".join(pieces).strip()


def _line_bbox(line: TextLine) -> tuple[float, float, float, float]:
    top = line.y + max(line.font_size, 8.0)
    bottom = max(line.y - 2.0, 0.0)
    return (line.x_min, bottom, line.x_max, top)


def _merge_bbox(boxes: Iterable[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    boxes = list(boxes)
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _looks_like_heading(text: str) -> bool:
    stripped = _normalize_whitespace(text)
    if not stripped:
        return False
    if stripped in {"ABSTRACT", "REFERENCES", "APPENDIX CONTENTS"}:
        return True
    return bool(SECTION_HEADING_RE.match(stripped))


def _looks_like_appendix_page(page_text: str, top_lines: list[TextLine]) -> bool:
    normalized = _normalize_whitespace(page_text).lower()
    if "appendix contents" in normalized:
        return True
    if "back to appendix table of contents" in normalized:
        return True
    top_text = " ".join(line.text for line in top_lines[:5]).upper()
    if top_text.startswith("APPENDIX") or re.match(r"^[A-Z]\s+[A-Z]{4,}", top_text):
        return True
    return False


def _looks_like_equation_line(text: str) -> bool:
    stripped = _normalize_whitespace(text)
    if len(stripped) < 4:
        return False
    if _looks_like_heading(stripped):
        return False
    if "<image" in stripped.lower():
        return False

    words = re.findall(r"[A-Za-z]+", stripped)
    word_count = len(words)
    symbol_count = len(re.findall(r"[=≤≥≈∑∫√±×÷^_{}]|[α-ωΑ-Ω]", stripped))
    digit_count = len(re.findall(r"\d", stripped))
    prose_markers = {
        "asking",
        "considered",
        "define",
        "defined",
        "document",
        "follows",
        "query",
        "question",
        "relevant",
        "represented",
        "sequence",
        "where",
    }
    lowered_words = {word.lower() for word in words}

    if re.search(r"\(\d+\)$", stripped):
        return True
    if word_count > 10:
        return False
    if lowered_words & prose_markers and word_count > 5:
        return False

    has_assignment = bool(re.search(r"[A-Za-z0-9_)\]}]\s*=\s*[^=]", stripped))
    has_operator_chain = bool(re.search(r"\d+\s*[+*/^]\s*\d+", stripped))
    has_symbolic_structure = bool(re.search(r"[A-Za-z0-9_]+\([^)]*\)", stripped) and re.search(r"[=+*/^]", stripped))
    has_math_symbol = bool(re.search(r"[=≤≥≈∑∫√]", stripped))

    if has_assignment and (symbol_count >= 2 or digit_count >= 2):
        return True
    if has_operator_chain:
        return True
    if has_symbolic_structure and word_count <= 8:
        return True
    if has_math_symbol and symbol_count >= 3 and word_count <= 6:
        return True
    return False


def _contains_institution_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ["university", "institute", "college", "school", "academy", "department", "center", "centre", "hospital", "laboratory", "lab"])


def _looks_like_visual_artifact_text(text: str) -> bool:
    lowered = text.lower()
    if "<image" in lowered:
        return True
    if any(marker in lowered for marker in ["positive docs", "negative docs", "annotated web pages", "multimodal collections"]):
        return True
    if lowered.count("task:") + lowered.count("query:") >= 2:
        return True
    words = WORD_RE.findall(text)
    if len(words) <= 8 and text.count("-") >= 2:
        return True
    return False


def _looks_like_noise_text(text: str) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return True
    if lowered.startswith("published as a conference paper"):
        return True
    if lowered == "appendix contents":
        return True
    if lowered == "back to appendix table of contents":
        return True
    if re.fullmatch(r"\d+", lowered):
        return True
    return False


def _looks_like_references_page(page_lines: list[TextLine]) -> bool:
    normalized_lines = [_normalize_whitespace(line.text) for line in page_lines if _normalize_whitespace(line.text)]
    if not normalized_lines:
        return False
    first_line = normalized_lines[0].lower()
    if first_line.startswith("references"):
        return True
    if any(_is_top_level_heading(text, "maintext") for text in normalized_lines[:10]):
        return False

    url_lines = sum(1 for text in normalized_lines if "urlhttp" in text.lower() or "urlhttps" in text.lower())
    venue_lines = sum(
        1
        for text in normalized_lines
        if any(marker in text.lower() for marker in ["proceedings", "conference", "arxiv", "openreview.net", "aclanthology.org"])
    )
    authorish_lines = sum(
        1
        for text in normalized_lines
        if re.match(r"^[A-Z][A-Za-z'`.-]+(?: [A-Z][A-Za-z'`.-]+){0,4}\.", text)
        or re.match(r"^[A-Z][A-Za-z'`.-]+, [A-Z]", text)
    )
    year_lines = sum(1 for text in normalized_lines if re.search(r"\b20\d{2}\b", text))

    return (url_lines >= 2 and venue_lines >= 3 and year_lines >= 4) or (url_lines >= 3 and authorish_lines >= 4)


def _is_top_level_heading(text: str, container_name: str) -> bool:
    stripped = _normalize_whitespace(text)
    if container_name == "maintext":
        return bool(TOP_LEVEL_MAIN_HEADING_RE.match(stripped)) and not bool(re.match(r"^\d+\.\d+", stripped))
    return bool(TOP_LEVEL_APPENDIX_HEADING_RE.match(stripped)) and not bool(re.match(r"^[A-Z]\.\d+", stripped))


def _normalize_heading_name(text: str) -> str:
    stripped = _normalize_whitespace(text)
    stripped = re.sub(r"^(?:\d+|[A-Z])\s+", "", stripped)
    for src, dst in HEADING_REPLACEMENTS.items():
        stripped = stripped.replace(src, dst)
    stripped = re.sub(r"\s+", " ", stripped).strip(" .:-")
    return stripped.title() if stripped else "Untitled Section"


def _is_back_matter_heading(text: str) -> bool:
    normalized = _normalize_heading_name(text).lower()
    return normalized in {"references", "bibliography"}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "section"


def extract_text_lines_from_page(page) -> list[TextLine]:
    fragments: list[TextFragment] = []

    def visitor(text, _cm, tm, _font_dict, font_size):
        cleaned = text.replace("\x00", " ").strip()
        if not cleaned:
            return
        fragments.append(
            TextFragment(
                text=cleaned,
                x=float(tm[4]),
                y=float(tm[5]),
                font_size=float(font_size or 0.0),
            )
        )

    page.extract_text(visitor_text=visitor)
    if not fragments:
        return []

    rows: list[list[TextFragment]] = []
    for fragment in sorted(fragments, key=lambda item: (-item.y, item.x)):
        if rows and abs(rows[-1][0].y - fragment.y) <= 2.5:
            rows[-1].append(fragment)
        else:
            rows.append([fragment])

    lines: list[TextLine] = []
    for row in rows:
        ordered = sorted(row, key=lambda item: item.x)
        parts: list[str] = []
        x_min = ordered[0].x
        x_max = ordered[0].x
        total_font = 0.0
        last_end: Optional[float] = None
        for fragment in ordered:
            text = _normalize_whitespace(fragment.text)
            if not text:
                continue
            est_width = _estimate_text_width(text, fragment.font_size or 8.0)
            if last_end is not None and fragment.x - last_end > max(fragment.font_size * 0.3, 4.0):
                parts.append(" ")
            parts.append(text)
            last_end = fragment.x + est_width
            x_max = max(x_max, last_end)
            total_font += fragment.font_size or 8.0
        merged = _normalize_whitespace("".join(parts))
        if merged:
            lines.append(
                TextLine(
                    text=merged,
                    x_min=x_min,
                    x_max=x_max,
                    y=ordered[0].y,
                    font_size=total_font / max(len(ordered), 1),
                )
            )
    return lines


def _group_caption_lines(lines: list[TextLine], start_idx: int) -> tuple[list[int], str, str]:
    start_line = lines[start_idx]
    match = CAPTION_RE.match(start_line.text)
    if not match:
        return [], "", ""

    indices = [start_idx]
    caption_type = "table" if match.group(1).lower().startswith("table") else "figure"
    caption_parts = [start_line.text]
    base_x = start_line.x_min
    last_y = start_line.y

    for idx in range(start_idx + 1, len(lines)):
        line = lines[idx]
        if CAPTION_RE.match(line.text) or _looks_like_heading(line.text):
            break
        if abs(last_y - line.y) > 18:
            break
        if abs(line.x_min - base_x) > 40 and caption_type == "table":
            break
        if len(indices) >= 4:
            break
        indices.append(idx)
        caption_parts.append(line.text)
        last_y = line.y
        if len(" ".join(caption_parts)) >= 320:
            break
        if line.text.endswith(".") and len(" ".join(caption_parts)) > 50:
            break

    return indices, caption_type, _normalize_whitespace(" ".join(caption_parts))


def _extract_page_blocks(lines: list[TextLine]) -> tuple[list[dict[str, Any]], set[int]]:
    blocks: list[dict[str, Any]] = []
    used_indices: set[int] = set()

    idx = 0
    while idx < len(lines):
        if idx in used_indices:
            idx += 1
            continue

        caption_indices, caption_type, caption_text = _group_caption_lines(lines, idx)
        if caption_indices:
            used_indices.update(caption_indices)
            caption_bbox = _merge_bbox(_line_bbox(lines[i]) for i in caption_indices)

            if caption_type == "table":
                table_indices: list[int] = []
                last_y = lines[caption_indices[-1]].y
                for j in range(caption_indices[-1] + 1, len(lines)):
                    if j in used_indices:
                        break
                    line = lines[j]
                    if CAPTION_RE.match(line.text) or _looks_like_heading(line.text):
                        break
                    if last_y - line.y > 20 and table_indices:
                        break
                    table_indices.append(j)
                    used_indices.add(j)
                    last_y = line.y
                table_text = "\n".join(lines[j].text for j in table_indices)
                content_bbox = _merge_bbox(_line_bbox(lines[j]) for j in table_indices) if table_indices else caption_bbox
                blocks.append(
                    {
                        "block_type": "table",
                        "y": max(caption_bbox[3], content_bbox[3]),
                        "caption": caption_text,
                        "text": table_text,
                        "caption_line_indices": caption_indices,
                        "content_line_indices": table_indices,
                        "caption_bbox_pdf": caption_bbox,
                        "content_bbox_pdf": content_bbox,
                        "bbox_pdf": _merge_bbox([caption_bbox, content_bbox]),
                    }
                )
            else:
                figure_indices: list[int] = []
                last_y = lines[caption_indices[0]].y
                initial_gap_limit = 160.0
                continuation_gap_limit = 18.0
                for j in range(caption_indices[0] - 1, -1, -1):
                    line = lines[j]
                    if CAPTION_RE.match(line.text) or _looks_like_heading(line.text):
                        break
                    if _looks_like_noise_text(line.text):
                        continue
                    if line.text.lower().startswith("published as a conference paper"):
                        continue
                    if line.x_max > 2000:
                        continue
                    gap = line.y - last_y
                    if gap > (initial_gap_limit if not figure_indices else continuation_gap_limit):
                        break
                    figure_indices.append(j)
                    last_y = line.y

                content_bbox = _merge_bbox(_line_bbox(lines[j]) for j in figure_indices) if figure_indices else caption_bbox
                blocks.append(
                    {
                        "block_type": "figure",
                        "y": caption_bbox[3],
                        "caption": caption_text,
                        "caption_line_indices": caption_indices,
                        "content_line_indices": list(reversed(figure_indices)),
                        "caption_bbox_pdf": caption_bbox,
                        "content_bbox_pdf": content_bbox,
                        "bbox_pdf": caption_bbox,
                    }
                )
            idx = caption_indices[-1] + 1
            continue

        if _looks_like_equation_line(lines[idx].text):
            equation_indices = [idx]
            used_indices.add(idx)
            last_y = lines[idx].y
            for j in range(idx + 1, len(lines)):
                if j in used_indices:
                    break
                candidate = lines[j]
                if not _looks_like_equation_line(candidate.text):
                    break
                if last_y - candidate.y > 18:
                    break
                equation_indices.append(j)
                used_indices.add(j)
                last_y = candidate.y
            blocks.append(
                {
                    "block_type": "equation",
                    "y": max(lines[j].y for j in equation_indices),
                    "text": "\n".join(lines[j].text for j in equation_indices),
                    "line_indices": equation_indices,
                    "bbox_pdf": _merge_bbox(_line_bbox(lines[j]) for j in equation_indices),
                }
            )
            idx = equation_indices[-1] + 1
            continue

        idx += 1

    return blocks, used_indices


def _extract_text_blocks(lines: list[TextLine], used_indices: set[int]) -> list[dict[str, Any]]:
    text_blocks: list[dict[str, Any]] = []
    current_indices: list[int] = []
    last_y: Optional[float] = None

    def flush() -> None:
        nonlocal current_indices, last_y
        if not current_indices:
            return
        block_type = "heading" if len(current_indices) == 1 and _looks_like_heading(lines[current_indices[0]].text) else "text"
        block_text = _normalize_whitespace(" ".join(lines[idx].text for idx in current_indices))
        if _looks_like_noise_text(block_text):
            current_indices = []
            last_y = None
            return
        if block_type == "text" and _looks_like_visual_artifact_text(block_text):
            current_indices = []
            last_y = None
            return
        text_blocks.append(
            {
                "block_type": block_type,
                "y": max(lines[idx].y for idx in current_indices),
                "text": block_text,
                "line_indices": list(current_indices),
                "bbox_pdf": _merge_bbox(_line_bbox(lines[idx]) for idx in current_indices),
            }
        )
        current_indices = []
        last_y = None

    for idx, line in enumerate(lines):
        if idx in used_indices or not _normalize_whitespace(line.text):
            flush()
            continue
        if _looks_like_heading(line.text):
            flush()
            current_indices = [idx]
            flush()
            continue
        if last_y is not None and last_y - line.y > 18:
            flush()
        current_indices.append(idx)
        last_y = line.y
    flush()
    return text_blocks


def detect_appendix_start_page(reader: PdfReader, page_lines: dict[int, list[TextLine]]) -> Optional[int]:
    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if _looks_like_appendix_page(text, page_lines.get(page_idx, [])):
            return page_idx
    return None


def _detect_render_backend() -> str:
    if which("pdftocairo"):
        return "pdftocairo"
    if which("pdftoppm"):
        return "pdftoppm"
    if which("mutool"):
        return "mutool"
    if which("gs"):
        return "gs"
    if which("qlmanage"):
        return "qlmanage"
    if which("sips"):
        return "sips"
    raise RuntimeError(
        "No supported PDF rasterizer found. Install one of: pdftocairo, pdftoppm, mutool, gs, qlmanage, or sips."
    )


def _render_single_page_png(
    pdf_path: Path,
    page_number: int,
    output_path: Path,
    *,
    backend: str,
    dpi: int = DEFAULT_RENDER_DPI,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_base = output_path.with_suffix("")

    def render_with_sips() -> Path:
        reader = PdfReader(str(pdf_path))
        with tempfile.TemporaryDirectory(prefix=f"pdf_page_{page_number:03d}_") as tmpdir:
            single_page_pdf = Path(tmpdir) / f"page_{page_number:03d}.pdf"
            writer = PdfWriter()
            writer.add_page(reader.pages[page_number - 1])
            with single_page_pdf.open("wb") as handle:
                writer.write(handle)
            subprocess.run(
                [
                    "sips",
                    "-s",
                    "format",
                    "png",
                    str(single_page_pdf),
                    "--out",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            page = reader.pages[page_number - 1]
            target_width = max(1, round(float(page.mediabox.width) * dpi / 72.0))
            target_height = max(1, round(float(page.mediabox.height) * dpi / 72.0))
            with Image.open(output_path) as rendered:
                if rendered.size != (target_width, target_height):
                    rendered = rendered.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    rendered.save(output_path)
        return output_path

    if backend == "pdftocairo":
        subprocess.run(
            [
                "pdftocairo",
                "-png",
                "-singlefile",
                "-r",
                str(dpi),
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                str(pdf_path),
                str(output_base),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return output_path

    if backend == "pdftoppm":
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-singlefile",
                "-r",
                str(dpi),
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                str(pdf_path),
                str(output_base),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return output_path

    if backend == "mutool":
        subprocess.run(
            [
                "mutool",
                "draw",
                "-q",
                "-r",
                str(dpi),
                "-F",
                "png",
                "-o",
                str(output_path),
                str(pdf_path),
                str(page_number),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return output_path

    if backend == "gs":
        subprocess.run(
            [
                "gs",
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=png16m",
                f"-r{dpi}",
                f"-dFirstPage={page_number}",
                f"-dLastPage={page_number}",
                f"-sOutputFile={output_path}",
                str(pdf_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return output_path

    if backend == "qlmanage":
        reader = PdfReader(str(pdf_path))
        with tempfile.TemporaryDirectory(prefix=f"pdf_page_{page_number:03d}_") as tmpdir:
            temp_dir = Path(tmpdir)
            single_page_pdf = temp_dir / f"page_{page_number:03d}.pdf"
            thumb_dir = temp_dir / "thumbs"
            thumb_dir.mkdir(parents=True, exist_ok=True)
            writer = PdfWriter()
            writer.add_page(reader.pages[page_number - 1])
            with single_page_pdf.open("wb") as handle:
                writer.write(handle)

            page = reader.pages[page_number - 1]
            target_width = max(1, round(float(page.mediabox.width) * dpi / 72.0))
            target_height = max(1, round(float(page.mediabox.height) * dpi / 72.0))
            preview_size = max(target_width, target_height)

            try:
                subprocess.run(
                    [
                        "qlmanage",
                        "-t",
                        "-s",
                        str(preview_size),
                        "-o",
                        str(thumb_dir),
                        str(single_page_pdf),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                logger.warning("qlmanage render failed for %s page %s; falling back to sips: %s", pdf_path, page_number, exc)
                return render_with_sips()

            rendered_candidates = sorted(thumb_dir.glob("*.png"))
            if not rendered_candidates:
                raise RuntimeError(f"qlmanage did not produce a PNG for {single_page_pdf}")

            rendered_path = rendered_candidates[0]
            shutil.copyfile(rendered_path, output_path)
        return output_path

    if backend == "sips":
        return render_with_sips()

    raise RuntimeError(f"Unsupported rasterizer backend: {backend}")


def _pdf_bbox_to_pixel_bbox(
    bbox_pdf: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    left, bottom, right, top = bbox_pdf
    x0 = int(max(0, min(image_width, round((left / page_width) * image_width))))
    x1 = int(max(0, min(image_width, round((right / page_width) * image_width))))
    y0 = int(max(0, min(image_height, round(((page_height - top) / page_height) * image_height))))
    y1 = int(max(0, min(image_height, round(((page_height - bottom) / page_height) * image_height))))
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _bbox_overlap_ratio(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    return inter_area / max(area_a, 1.0)


def _expand_bbox(
    bbox_pdf: tuple[float, float, float, float],
    *,
    pad_left: float = 0.0,
    pad_bottom: float = 0.0,
    pad_right: float = 0.0,
    pad_top: float = 0.0,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    left, bottom, right, top = bbox_pdf
    return (
        max(0.0, left - pad_left),
        max(0.0, bottom - pad_bottom),
        min(page_width, right + pad_right),
        min(page_height, top + pad_top),
    )


def _caption_column_bbox(caption_bbox: tuple[float, float, float, float], page_width: float) -> tuple[float, float]:
    left, _, right, _ = caption_bbox
    center = (left + right) / 2.0
    if right - left >= page_width * 0.55:
        return (36.0, page_width - 36.0)
    if left <= (page_width / 2.0) - 18.0 and right >= (page_width / 2.0) + 18.0:
        return (36.0, page_width - 36.0)
    if center < page_width / 2.0:
        return (36.0, (page_width / 2.0) - 18.0)
    return ((page_width / 2.0) + 18.0, page_width - 36.0)


def _non_white_mask(image: Image.Image, threshold: int = 245) -> Image.Image:
    grayscale = image.convert("L")
    return grayscale.point(lambda value: 255 if value < threshold else 0, mode="L")


def _find_non_empty_runs(counts: list[int], *, min_count: int, max_gap: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: Optional[int] = None
    gap = 0
    last_positive: Optional[int] = None

    for idx, count in enumerate(counts):
        if count >= min_count:
            if start is None:
                start = idx
            last_positive = idx
            gap = 0
            continue

        if start is None:
            continue
        gap += 1
        if gap > max_gap:
            runs.append((start, (last_positive if last_positive is not None else idx) + 1))
            start = None
            gap = 0
            last_positive = None

    if start is not None:
        end = (last_positive if last_positive is not None else len(counts) - 1) + 1
        runs.append((start, end))
    return runs


def _select_run_nearest_edge(
    runs: list[tuple[int, int]],
    *,
    prefer_start: bool,
    min_span: int,
) -> Optional[tuple[int, int]]:
    eligible = [(start, end) for start, end in runs if end - start >= min_span]
    if not eligible:
        return None
    if prefer_start:
        return min(eligible, key=lambda item: item[0])
    return max(eligible, key=lambda item: item[1])


def _find_visual_content_bbox(
    image: Image.Image,
    search_bbox_px: tuple[int, int, int, int],
    *,
    prefer_near_top: bool,
) -> Optional[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = search_bbox_px
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None

    region = image.crop((x0, y0, x1, y1))
    mask = _non_white_mask(region)
    width, height = mask.size
    pixels = mask.load()

    row_counts = [sum(1 for x in range(width) if pixels[x, y]) for y in range(height)]
    row_runs = _find_non_empty_runs(
        row_counts,
        min_count=max(12, width // 120),
        max_gap=max(4, height // 150),
    )
    row_run = _select_run_nearest_edge(
        row_runs,
        prefer_start=prefer_near_top,
        min_span=max(18, height // 40),
    )
    if row_run is None:
        return None

    row_start, row_end = row_run
    col_counts = [sum(1 for y in range(row_start, row_end) if pixels[x, y]) for x in range(width)]
    col_runs = _find_non_empty_runs(
        col_counts,
        min_count=max(8, (row_end - row_start) // 40),
        max_gap=max(6, width // 150),
    )
    if not col_runs:
        return None

    left = min(start for start, _ in col_runs)
    right = max(end for _, end in col_runs)
    pad_x = max(8, width // 80)
    pad_y = max(8, height // 80)
    return (
        max(x0, x0 + left - pad_x),
        max(y0, y0 + row_start - pad_y),
        min(x1, x0 + right + pad_x),
        min(y1, y0 + row_end + pad_y),
    )


def _trim_isolated_top_run(mask: Image.Image, inner_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    width, height = mask.size
    pixels = mask.load()
    row_counts = [sum(1 for x in range(width) if pixels[x, y]) for y in range(height)]
    row_runs = _find_non_empty_runs(
        row_counts,
        min_count=max(12, width // 120),
        max_gap=max(4, height // 150),
    )
    if len(row_runs) < 2:
        return inner_bbox

    first_start, first_end = row_runs[0]
    second_start, _ = row_runs[1]
    first_height = first_end - first_start
    if first_start > 2:
        return inner_bbox
    if first_height > max(12, height // 25):
        return inner_bbox
    if second_start - first_end < max(20, first_height * 3):
        return inner_bbox

    left, _, right, bottom = inner_bbox
    return (left, max(second_start - 4, 0), right, bottom)


def _neighbor_vertical_limits(
    page_blocks: list[dict[str, Any]],
    block_idx: int,
    *,
    page_height: float,
) -> tuple[float, float]:
    upper_limit = page_height - 24.0
    lower_limit = 24.0

    for candidate in reversed(page_blocks[:block_idx]):
        bbox = candidate.get("caption_bbox_pdf") or candidate.get("content_bbox_pdf") or candidate.get("bbox_pdf")
        if bbox:
            upper_limit = min(upper_limit, bbox[1])
            break

    for candidate in page_blocks[block_idx + 1 :]:
        bbox = candidate.get("caption_bbox_pdf") or candidate.get("content_bbox_pdf") or candidate.get("bbox_pdf")
        if bbox:
            lower_limit = max(lower_limit, bbox[3])
            break

    return upper_limit, lower_limit


def _resolve_pdffigures_json(pdf_path: Path, explicit_path: Optional[Path | str]) -> Optional[Path]:
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        return candidate if candidate.is_file() else None
    candidates = [
        pdf_path.with_suffix(".pdffigures.json"),
        pdf_path.with_suffix(".json"),
        pdf_path.parent / f"{pdf_path.stem}.pdffigures.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _read_pdffigures_payload(pdffigures_json_path: Optional[Path]) -> Optional[dict[str, Any]]:
    if pdffigures_json_path is None or not pdffigures_json_path.is_file():
        return None
    with pdffigures_json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _normalize_pdffigures_bbox(raw_bbox: dict[str, Any], page_height: float) -> Optional[tuple[float, float, float, float]]:
    if not isinstance(raw_bbox, dict):
        return None

    if all(key in raw_bbox for key in ("x1", "y1", "x2", "y2")):
        x1 = float(raw_bbox["x1"])
        y1 = float(raw_bbox["y1"])
        x2 = float(raw_bbox["x2"])
        y2 = float(raw_bbox["y2"])
        left = min(x1, x2)
        right = max(x1, x2)
        top_from_top = min(y1, y2)
        bottom_from_top = max(y1, y2)
        return (left, page_height - bottom_from_top, right, page_height - top_from_top)

    if all(key in raw_bbox for key in ("left", "top", "width", "height")):
        left = float(raw_bbox["left"])
        top_from_top = float(raw_bbox["top"])
        width = float(raw_bbox["width"])
        height = float(raw_bbox["height"])
        return (left, page_height - (top_from_top + height), left + width, page_height - top_from_top)

    return None


def _load_pdffigures_regions(
    pdffigures_payload: Optional[dict[str, Any]],
    page_heights: dict[int, float],
) -> dict[int, list[dict[str, Any]]]:
    if not pdffigures_payload:
        return {}

    items = pdffigures_payload.get("figures") or pdffigures_payload.get("regions") or []
    by_page: dict[int, list[dict[str, Any]]] = {}
    if not isinstance(items, list):
        return by_page

    for item in items:
        if not isinstance(item, dict):
            continue
        page_number = int(item.get("page", item.get("pageNumber", 0)) or 0)
        if page_number <= 0 or page_number not in page_heights:
            continue

        raw_bbox = item.get("regionBoundary") or item.get("figureBoundary") or item.get("boundary")
        bbox_pdf = _normalize_pdffigures_bbox(raw_bbox, page_heights[page_number])
        if bbox_pdf is None:
            continue

        caption = _normalize_whitespace(
            str(item.get("caption") or item.get("captionText") or item.get("name") or "")
        )
        figure_type = str(item.get("figureType") or item.get("type") or "figure").lower()
        block_type = "table" if "table" in figure_type else "figure"
        by_page.setdefault(page_number, []).append(
            {
                "block_type": block_type,
                "y": bbox_pdf[3],
                "caption": caption,
                "bbox_pdf": bbox_pdf,
                "source": "pdffigures2",
            }
        )
    return by_page


def _filter_blocks_against_visual_regions(
    blocks: list[dict[str, Any]],
    visual_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not visual_regions:
        return blocks
    region_bboxes = [region["bbox_pdf"] for region in visual_regions]
    filtered: list[dict[str, Any]] = []
    for block in blocks:
        if block["block_type"] in {"figure", "table"}:
            continue
        bbox = block.get("bbox_pdf")
        if bbox and any(_bbox_overlap_ratio(bbox, region_bbox) >= 0.35 for region_bbox in region_bboxes):
            continue
        filtered.append(block)
    return filtered


def _crop_visual_block(
    page_png_path: Path,
    block: dict[str, Any],
    page_blocks: list[dict[str, Any]],
    block_idx: int,
    page_width: float,
    page_height: float,
    output_path: Path,
) -> Optional[Path]:
    with Image.open(page_png_path) as image:
        image_width, image_height = image.size
        output_path.parent.mkdir(parents=True, exist_ok=True)

        block_type = block["block_type"]
        if block_type == "equation":
            bbox_pdf = _expand_bbox(
                block["bbox_pdf"],
                pad_left=10.0,
                pad_bottom=6.0,
                pad_right=10.0,
                pad_top=6.0,
                page_width=page_width,
                page_height=page_height,
            )
        else:
            caption_bbox = block.get("caption_bbox_pdf") or block["bbox_pdf"]
            if block.get("source") == "pdffigures2":
                pad = 8.0 if block_type == "figure" else 4.0
                bbox_pdf = _expand_bbox(
                    block["bbox_pdf"],
                    pad_left=pad,
                    pad_bottom=pad,
                    pad_right=pad,
                    pad_top=pad,
                    page_width=page_width,
                    page_height=page_height,
                )
            elif block_type == "figure":
                search_left, search_right = _caption_column_bbox(caption_bbox, page_width)
                content_bbox = block.get("content_bbox_pdf")
                upper_limit = min(
                    page_height - 24.0,
                    ((content_bbox[3] + 4.0) if content_bbox else (page_height - 24.0)),
                )
                search_bbox_pdf = (
                    search_left,
                    min(page_height, caption_bbox[3] + 6.0),
                    search_right,
                    min(page_height - 24.0, max(caption_bbox[3] + 20.0, upper_limit)),
                )
                search_bbox_px = _pdf_bbox_to_pixel_bbox(
                    search_bbox_pdf,
                    page_width=page_width,
                    page_height=page_height,
                    image_width=image_width,
                    image_height=image_height,
                )
                region = image.crop(search_bbox_px)
                mask = _non_white_mask(region)
                inner_bbox = mask.getbbox()
                refined_bbox_px = None
                if inner_bbox is not None:
                    inner_bbox = _trim_isolated_top_run(mask, inner_bbox)
                    inner_left, inner_top, inner_right, inner_bottom = inner_bbox
                    pad_x = max(8, region.size[0] // 80)
                    pad_y = max(8, region.size[1] // 80)
                    refined_bbox_px = (
                        max(search_bbox_px[0], search_bbox_px[0] + inner_left - pad_x),
                        max(search_bbox_px[1], search_bbox_px[1] + inner_top - pad_y),
                        min(search_bbox_px[2], search_bbox_px[0] + inner_right + pad_x),
                        min(search_bbox_px[3], search_bbox_px[1] + inner_bottom + pad_y),
                    )
                elif content_bbox is None:
                    refined_bbox_px = _find_visual_content_bbox(
                        image,
                        search_bbox_px,
                        prefer_near_top=False,
                    )
                if refined_bbox_px is not None:
                    x0, y0, x1, y1 = refined_bbox_px
                    if x1 - x0 >= 20 and y1 - y0 >= 20:
                        cropped = image.crop((x0, y0, x1, y1))
                        if "A" in cropped.getbands():
                            background = Image.new("RGBA", cropped.size, (255, 255, 255, 255))
                            cropped = Image.alpha_composite(background, cropped).convert("RGB")
                        elif cropped.mode != "RGB":
                            cropped = cropped.convert("RGB")
                        cropped.save(output_path)
                        return output_path
                bbox_pdf = (
                    search_left,
                    min(page_height, caption_bbox[3] + 6.0),
                    search_right,
                    min(page_height - 24.0, max(caption_bbox[3] + 20.0, upper_limit - 6.0)),
                )
            else:
                table_bbox = block.get("content_bbox_pdf") or block["bbox_pdf"]
                table_width = table_bbox[2] - table_bbox[0]
                if table_width >= page_width * 0.65:
                    table_bbox = (
                        36.0,
                        table_bbox[1],
                        page_width - 36.0,
                        table_bbox[3],
                    )
                bbox_pdf = _expand_bbox(
                    table_bbox,
                    pad_left=8.0,
                    pad_bottom=6.0,
                    pad_right=8.0,
                    pad_top=8.0,
                    page_width=page_width,
                    page_height=page_height,
                )

        x0, y0, x1, y1 = _pdf_bbox_to_pixel_bbox(
            bbox_pdf,
            page_width=page_width,
            page_height=page_height,
            image_width=image_width,
            image_height=image_height,
        )
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None
        cropped = image.crop((x0, y0, x1, y1))
        if "A" in cropped.getbands():
            background = Image.new("RGBA", cropped.size, (255, 255, 255, 255))
            cropped = Image.alpha_composite(background, cropped).convert("RGB")
        elif cropped.mode != "RGB":
            cropped = cropped.convert("RGB")
        cropped.save(output_path)
        return output_path


def _extract_document_metadata(
    page_lines: dict[int, list[TextLine]],
    appendix_start_page: Optional[int],
) -> dict[str, Any]:
    page1_lines = [_normalize_whitespace(line.text) for line in page_lines.get(1, []) if _normalize_whitespace(line.text)]
    if page1_lines and page1_lines[0].lower().startswith("published as a conference paper"):
        page1_lines = page1_lines[1:]

    abstract_idx = next((idx for idx, text in enumerate(page1_lines) if text.upper() == "ABSTRACT"), None)
    intro_idx = next((idx for idx, text in enumerate(page1_lines) if _is_top_level_heading(text, "maintext")), None)

    title_lines: list[str] = []
    body_start = 0
    for idx, line in enumerate(page1_lines):
        if line.upper() == "ABSTRACT":
            break
        if "∗" in line or "*" in line:
            break
        single_letter_tokens = sum(1 for token in line.split() if len(token) == 1 and token.isupper())
        if single_letter_tokens >= 3:
            break
        letters = [char for char in line if char.isalpha()]
        uppercase_ratio = (sum(char.isupper() for char in letters) / len(letters)) if letters else 0.0
        if idx < 4 and uppercase_ratio >= 0.7:
            title_lines.append(line)
            body_start = idx + 1
            continue
        break

    between_title_and_abstract = page1_lines[body_start:abstract_idx] if abstract_idx is not None else []
    authors = [
        line
        for line in between_title_and_abstract
        if not _contains_institution_keyword(line)
        and sum(1 for token in line.split() if re.search(r"[a-z]", token)) >= 2
    ]
    institutions = [
        line
        for line in between_title_and_abstract
        if _contains_institution_keyword(line)
    ]
    abstract_lines = page1_lines[(abstract_idx + 1 if abstract_idx is not None else len(page1_lines)):intro_idx] if intro_idx else []

    return {
        "title": _join_lines_compact(title_lines),
        "authors": authors,
        "institutions": institutions,
        "abstract": _join_lines_compact(abstract_lines),
    }


def _block_payload(block: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "page_number",
        "container_section",
        "subsection_name",
        "subsection_path",
        "block_type",
        "text",
        "caption",
        "asset_path",
        "bbox_pdf",
        "line_indices",
        "caption_line_indices",
        "content_line_indices",
    }
    return {key: value for key, value in block.items() if key in allowed}


def _build_section_subsections(
    pages: list[dict[str, Any]],
    container_name: str,
) -> list[dict[str, Any]]:
    subsections: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    seen_paths: dict[str, int] = {}
    reached_back_matter = False

    def start_subsection(heading_text: str, page_number: int) -> dict[str, Any]:
        nonlocal current
        section_name = _normalize_heading_name(heading_text)
        base_path = f"{container_name}/{section_name}"
        count = seen_paths.get(base_path, 0) + 1
        seen_paths[base_path] = count
        section_path = base_path if count == 1 else f"{base_path} ({count})"
        current = {
            "section_name": section_name,
            "section_path": section_path,
            "section_slug": f"{container_name}/{_slugify(section_path)}",
            "heading": heading_text,
            "start_page": page_number,
            "end_page": page_number,
            "blocks": [],
        }
        subsections.append(current)
        return current

    for page in pages:
        if reached_back_matter:
            break
        page_number = page["page_number"]
        for block in page["blocks"]:
            if container_name == "maintext" and block["block_type"] == "heading" and _is_back_matter_heading(block["text"]):
                reached_back_matter = True
                break

            if block["block_type"] == "heading" and _is_top_level_heading(block["text"], container_name):
                current = start_subsection(block["text"], page_number)
                block["subsection_name"] = current["section_name"]
                block["subsection_path"] = current["section_path"]
                current["blocks"].append(_block_payload(block))
                continue

            if current is None:
                continue

            current["end_page"] = page_number
            block["subsection_name"] = current["section_name"]
            block["subsection_path"] = current["section_path"]
            current["blocks"].append(_block_payload(block))

    return subsections


def _build_markdown_for_container(container_name: str, subsections: list[dict[str, Any]], base_dir: Path) -> str:
    parts = [f"# {container_name.title()}"]
    for subsection in subsections:
        parts.append(f"\n## {subsection['section_name']}")
        for block in subsection["blocks"]:
            block_type = block["block_type"]
            if block_type == "heading":
                if block["text"] != subsection["heading"]:
                    parts.append(f"\n### {block['text']}")
                continue
            if block_type == "text":
                parts.append(f"\n{block['text']}")
                continue
            if block_type == "equation":
                parts.append("\n```text")
                parts.append(block["text"])
                parts.append("```")
                continue

            title = block.get("caption") or f"{block_type.title()} block"
            parts.append(f"\n### {title}")
            if block.get("asset_path"):
                parts.append(f"![{title}]({block['asset_path']})")
            if block_type == "table" and block.get("text"):
                parts.append("\n```text")
                parts.append(block["text"])
                parts.append("```")
    return "\n".join(parts).strip() + "\n"


def _build_combined_markdown(
    metadata: dict[str, Any],
    sections: dict[str, dict[str, Any]],
) -> str:
    parts = ["# Metadata"]
    if metadata.get("title"):
        parts.append(f"\n## Title\n{metadata['title']}")
    if metadata.get("authors"):
        parts.append("\n## Authors")
        parts.extend(f"- {item}" for item in metadata["authors"])
    if metadata.get("institutions"):
        parts.append("\n## Institutions")
        parts.extend(f"- {item}" for item in metadata["institutions"])
    if metadata.get("abstract"):
        parts.append(f"\n## Abstract\n{metadata['abstract']}")
    for container_name in ["maintext", "appendix"]:
        markdown = sections[container_name]["interleaved_markdown"].strip()
        if markdown:
            parts.append("\n" + markdown)
    return "\n".join(parts).strip() + "\n"


def extract_interleaved_pdf_content(
    pdf_path: Path | str,
    output_dir: Optional[Path | str] = None,
    *,
    pdffigures_json_path: Optional[Path | str] = None,
    render_backend: Optional[str] = None,
    render_dpi: int = DEFAULT_RENDER_DPI,
) -> dict[str, Any]:
    pdf_path = Path(pdf_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else _default_output_dir(pdf_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir, ignore_errors=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    page_lines = {page_number: extract_text_lines_from_page(page) for page_number, page in enumerate(reader.pages, start=1)}
    page_heights = {page_number: float(page.mediabox.height) for page_number, page in enumerate(reader.pages, start=1)}
    effective_render_backend = render_backend or _detect_render_backend()
    effective_pdffigures_json_path = _resolve_pdffigures_json(pdf_path, pdffigures_json_path)
    pdffigures_payload = _read_pdffigures_payload(effective_pdffigures_json_path)
    pdffigures_regions = _load_pdffigures_regions(pdffigures_payload, page_heights)
    appendix_start_page = detect_appendix_start_page(reader, page_lines)
    metadata = _extract_document_metadata(page_lines, appendix_start_page)

    raw_pages: dict[str, list[dict[str, Any]]] = {"maintext": [], "appendix": []}
    asset_inventory: list[dict[str, Any]] = []
    references_start_page: Optional[int] = None

    with tempfile.TemporaryDirectory(prefix="pdf2interleaved_pages_") as tmpdir:
        temp_render_dir = Path(tmpdir)
        for page_number, page in enumerate(reader.pages, start=1):
            lines = page_lines[page_number]
            page_width = float(page.mediabox.width)
            page_height = float(page.mediabox.height)
            structural_blocks, used_indices = _extract_page_blocks(lines)
            text_blocks = _extract_text_blocks(lines, used_indices)
            page_pdffigures_regions = pdffigures_regions.get(page_number, [])
            if page_pdffigures_regions:
                structural_blocks = _filter_blocks_against_visual_regions(structural_blocks, page_pdffigures_regions)
                text_blocks = _filter_blocks_against_visual_regions(text_blocks, page_pdffigures_regions)
            all_blocks = sorted(
                structural_blocks + text_blocks + page_pdffigures_regions,
                key=lambda block: (-float(block["y"]), 0 if block["block_type"] == "heading" else 1),
            )

            container_name = "appendix" if appendix_start_page and page_number >= appendix_start_page else "maintext"
            if container_name == "maintext" and references_start_page is None and _looks_like_references_page(lines):
                references_start_page = page_number
            if container_name == "maintext" and references_start_page is not None and page_number >= references_start_page:
                continue
            page_png_path: Optional[Path] = None
            if any(block["block_type"] in {"figure", "table", "equation"} for block in all_blocks):
                page_png_path = _render_single_page_png(
                    pdf_path,
                    page_number=page_number,
                    output_path=temp_render_dir / f"page_{page_number:03d}.png",
                    backend=effective_render_backend,
                    dpi=render_dpi,
                )

            for block_idx, block in enumerate(all_blocks, start=1):
                block["page_number"] = page_number
                block["container_section"] = container_name
                if page_png_path and block["block_type"] in {"figure", "table", "equation"}:
                    crop_path = assets_dir / f"page_{page_number:03d}_{block['block_type']}_{block_idx:02d}.png"
                    cropped = _crop_visual_block(
                        page_png_path=page_png_path,
                        block=block,
                        page_blocks=all_blocks,
                        block_idx=block_idx - 1,
                        page_width=page_width,
                        page_height=page_height,
                        output_path=crop_path,
                    )
                    if cropped is not None:
                        relative_asset_path = cropped.relative_to(output_dir).as_posix()
                        block["asset_path"] = relative_asset_path
                        asset_inventory.append(
                            {
                                "page_number": page_number,
                                "container_section": container_name,
                                "block_index": block_idx,
                                "block_type": block["block_type"],
                                "caption": block.get("caption"),
                                "asset_path": relative_asset_path,
                            }
                        )

            raw_pages[container_name].append(
                {
                    "page_number": page_number,
                    "blocks": all_blocks,
                }
            )

    if appendix_start_page:
        raw_pages["appendix"] = [
            page
            for page in raw_pages["appendix"]
            if not any(
                block.get("block_type") == "text" and block.get("text") == "Appendix Contents"
                for block in page["blocks"]
            )
        ]

    sections = {
        "maintext": {
            "subsections": _build_section_subsections(raw_pages["maintext"], "maintext"),
        },
        "appendix": {
            "subsections": _build_section_subsections(raw_pages["appendix"], "appendix"),
        },
    }

    for container_name in ["maintext", "appendix"]:
        sections[container_name]["subsection_count"] = len(sections[container_name]["subsections"])
        sections[container_name]["interleaved_markdown"] = _build_markdown_for_container(
            container_name,
            sections[container_name]["subsections"],
            base_dir=output_dir,
        )

    combined_markdown = _build_combined_markdown(metadata, sections)

    return {
        "pdf_path": str(pdf_path),
        "output_dir": str(output_dir),
        "page_count": len(reader.pages),
        "appendix_start_page": appendix_start_page,
        "processing": {
            "render_backend": effective_render_backend,
            "render_dpi": render_dpi,
            "pdffigures_json_path": str(effective_pdffigures_json_path) if effective_pdffigures_json_path else None,
            "uses_pdffigures_regions": bool(pdffigures_regions),
            "references_start_page": references_start_page,
        },
        "metadata": metadata,
        "sections": sections,
        "assets": asset_inventory,
        "combined_interleaved_markdown": combined_markdown,
    }


def write_interleaved_pdf_content(
    pdf_path: Path | str,
    output_dir: Optional[Path | str] = None,
    json_name: str = "interleaved_content.json",
    *,
    pdffigures_json_path: Optional[Path | str] = None,
    render_backend: Optional[str] = None,
    render_dpi: int = DEFAULT_RENDER_DPI,
) -> Path:
    result = extract_interleaved_pdf_content(
        pdf_path=pdf_path,
        output_dir=output_dir,
        pdffigures_json_path=pdffigures_json_path,
        render_backend=render_backend,
        render_dpi=render_dpi,
    )
    resolved_output_dir = Path(result["output_dir"])
    output_path = resolved_output_dir / json_name
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    output_path.with_suffix(".md").write_text(result["combined_interleaved_markdown"], encoding="utf-8")
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract interleaved multimodal content from a paper PDF.")
    parser.add_argument("pdf_path", help="Path to the paper PDF.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Directory for extracted assets and outputs. Defaults to outputs/pdf2interleaved/<pdf_stem>.",
    )
    parser.add_argument(
        "--json-name",
        default="interleaved_content.json",
        help="Filename for the structured JSON output.",
    )
    parser.add_argument(
        "--pdffigures-json",
        default=None,
        help="Optional PDFFigures2 JSON path. When available, exact figure/table regions are preferred over heuristics.",
    )
    parser.add_argument(
        "--render-backend",
        default=None,
        help="Optional rasterizer override: pdftocairo, pdftoppm, mutool, gs, or sips.",
    )
    parser.add_argument(
        "--render-dpi",
        type=int,
        default=DEFAULT_RENDER_DPI,
        help="DPI for page rasterization.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    output_path = write_interleaved_pdf_content(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
        json_name=args.json_name,
        pdffigures_json_path=args.pdffigures_json,
        render_backend=args.render_backend,
        render_dpi=args.render_dpi,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
