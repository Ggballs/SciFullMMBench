from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any
from utils.project_paths import TEST_DATA_DIR

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

INSTITUTION_KEYWORDS = (
    "university",
    "institute",
    "college",
    "school",
    "academy",
    "department",
    "center",
    "centre",
    "laboratory",
    "lab",
    "hospital",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse a paper PDF with Docling and export a paper-oriented interleaved JSON."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default=str(TEST_DATA_DIR / "pdf_extraction" / "mrmr.pdf"),
        help="Path to the PDF to parse.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/docling_test",
        help="Directory for exported Docling outputs.",
    )
    parser.add_argument(
        "--formula-enrichment",
        action="store_true",
        help="Enable Docling formula enrichment when the model is available locally.",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=3.0,
        help="Rasterization scale for page, figure, and table image exports.",
    )
    return parser.parse_args()


def normalize_whitespace(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def is_institution_text(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in INSTITUTION_KEYWORDS)


def is_references_heading(text: str) -> bool:
    normalized = normalize_whitespace(text).upper()
    return normalized in {"REFERENCES", "REFERENCE"}


def is_appendix_heading(text: str) -> bool:
    normalized = normalize_whitespace(text).upper()
    if normalized.startswith("APPENDIX"):
        return True
    return bool(re.match(r"^[A-Z]\s+[A-Z].*", normalized))


def is_visual_child(item: Any) -> bool:
    parent = str(getattr(item, "parent", "") or "")
    return parent.startswith("cref='#/pictures/") or parent.startswith("cref='#/tables/")


def bbox_to_list(prov: Any) -> list[float]:
    bbox = prov.bbox
    return [float(bbox.l), float(bbox.b), float(bbox.r), float(bbox.t)]


def build_converter(enable_formula_enrichment: bool, image_scale: float) -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True
    pipeline_options.do_formula_enrichment = enable_formula_enrichment
    pipeline_options.images_scale = image_scale

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def build_text_only_converter(*, disable_table_structure: bool) -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # Limit internal thread pools to prevent thread explosion under concurrency.
    # Configurable via env vars for multi-process deployments:
    #   TORCH_NUM_THREADS (default 2)   — torch intra-op parallelism
    #   TORCH_INTEROP_THREADS (default 1) — torch inter-op parallelism
    #   ONNX_NUM_THREADS (default 1)    — onnxruntime thread count
    # Also sets OMP_NUM_THREADS / MKL_NUM_THREADS if not already set.
    import os as _os
    try:
        import torch
        _torch_threads = int(_os.environ.get("TORCH_NUM_THREADS", "2"))
        torch.set_num_threads(_torch_threads)
        torch.set_num_interop_threads(int(_os.environ.get("TORCH_INTEROP_THREADS", "1")))
    except Exception:
        pass
    try:
        import onnxruntime
        _onnx_threads = int(_os.environ.get("ONNX_NUM_THREADS", "1"))
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = _onnx_threads
        opts.inter_op_num_threads = _onnx_threads
    except Exception:
        pass
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if _var not in _os.environ:
            _os.environ[_var] = _os.environ.get("TORCH_NUM_THREADS", "2")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = not disable_table_structure
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.generate_table_images = False
    pipeline_options.do_formula_enrichment = False
    if hasattr(pipeline_options, "do_ocr"):
        pipeline_options.do_ocr = False

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


import threading

_CONVERTER_LOCK = threading.Lock()
_CONVERTER_POOL: list[Any] = []
_CONVERTER_POOL_SIZE = 0

def _get_converter_from_pool(disable_table_structure: bool) -> Any:
    global _CONVERTER_POOL_SIZE
    if _CONVERTER_POOL_SIZE == 0:
        import os
        _CONVERTER_POOL_SIZE = int(os.environ.get("DOCLING_POOL_SIZE", "8"))
    with _CONVERTER_LOCK:
        if _CONVERTER_POOL:
            return _CONVERTER_POOL.pop()
    return build_text_only_converter(disable_table_structure=disable_table_structure)

def _return_converter_to_pool(converter: Any) -> None:
    with _CONVERTER_LOCK:
        if len(_CONVERTER_POOL) < _CONVERTER_POOL_SIZE:
            _CONVERTER_POOL.append(converter)

def parse_pdf_text_only(pdf_path: str | Path, *, disable_table_structure: bool = False, converter: Any = None) -> dict[str, Any]:
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.is_file():
        return {
            "ok": False,
            "error": f"pdf_not_found: {pdf_path}",
            "markdown": "",
            "page_count": 0,
            "table_structure_enabled": not disable_table_structure,
        }

    own_converter = False
    if converter is None:
        converter = build_text_only_converter(disable_table_structure=disable_table_structure)
        own_converter = True
    try:
        result = converter.convert(str(pdf_path))
        document = result.document
        return {
            "ok": True,
            "markdown": document.export_to_markdown() or "",
            "page_count": len(document.pages) if document.pages else 0,
            "table_structure_enabled": not disable_table_structure,
        }
    except Exception as exc:  # pragma: no cover - runtime helper
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "markdown": "",
            "page_count": 0,
            "table_structure_enabled": not disable_table_structure,
        }


def export_pictures(doc: Any, assets_dir: Path, output_dir: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    picture_map: dict[str, dict[str, Any]] = {}
    asset_inventory: list[dict[str, Any]] = []
    page_counters: dict[int, int] = {}

    for picture in doc.pictures:
        prov = picture.prov[0] if picture.prov else None
        page_no = int(prov.page_no) if prov else 0
        page_counters[page_no] = page_counters.get(page_no, 0) + 1
        image = picture.get_image(doc)
        asset_path: str | None = None
        if image is not None:
            path = assets_dir / f"page_{page_no:03d}_figure_{page_counters[page_no]:02d}.png"
            image.save(path)
            asset_path = path.relative_to(output_dir).as_posix()

        payload = {
            "self_ref": picture.self_ref,
            "block_type": "figure",
            "caption": normalize_whitespace(picture.caption_text(doc)),
            "page_number": page_no if prov else None,
            "bbox_pdf": bbox_to_list(prov) if prov else None,
            "asset_path": asset_path,
        }
        picture_map[picture.self_ref] = payload
        if asset_path:
            asset_inventory.append(
                {
                    "page_number": page_no,
                    "block_type": "figure",
                    "caption": payload["caption"],
                    "asset_path": asset_path,
                }
            )
    return picture_map, asset_inventory


def export_tables(doc: Any, assets_dir: Path, output_dir: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    table_map: dict[str, dict[str, Any]] = {}
    asset_inventory: list[dict[str, Any]] = []
    page_counters: dict[int, int] = {}

    for table in doc.tables:
        prov = table.prov[0] if table.prov else None
        page_no = int(prov.page_no) if prov else 0
        page_counters[page_no] = page_counters.get(page_no, 0) + 1
        stem = f"page_{page_no:03d}_table_{page_counters[page_no]:02d}"
        image = table.get_image(doc)
        dataframe = table.export_to_dataframe(doc)

        image_asset_path: str | None = None
        if image is not None:
            image_path = assets_dir / f"{stem}.png"
            image.save(image_path)
            image_asset_path = image_path.relative_to(output_dir).as_posix()

        csv_path = assets_dir / f"{stem}.csv"
        md_path = assets_dir / f"{stem}.md"
        html_path = assets_dir / f"{stem}.html"
        dataframe.to_csv(csv_path, index=False)
        md_path.write_text(table.export_to_markdown(doc), encoding="utf-8")
        html_path.write_text(table.export_to_html(doc), encoding="utf-8")

        payload = {
            "self_ref": table.self_ref,
            "block_type": "table",
            "caption": normalize_whitespace(table.caption_text(doc)),
            "text": normalize_whitespace(table.export_to_markdown(doc)),
            "page_number": page_no if prov else None,
            "bbox_pdf": bbox_to_list(prov) if prov else None,
            "asset_path": image_asset_path,
            "csv_path": csv_path.relative_to(output_dir).as_posix(),
            "markdown_path": md_path.relative_to(output_dir).as_posix(),
            "html_path": html_path.relative_to(output_dir).as_posix(),
            "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
        }
        table_map[table.self_ref] = payload
        if image_asset_path:
            asset_inventory.append(
                {
                    "page_number": page_no,
                    "block_type": "table",
                    "caption": payload["caption"],
                    "asset_path": image_asset_path,
                }
            )
    return table_map, asset_inventory


def collect_formulas(doc: Any) -> list[dict[str, Any]]:
    from docling_core.types.doc.document import FormulaItem

    formulas: list[dict[str, Any]] = []
    for item, _level in doc.iterate_items():
        if not isinstance(item, FormulaItem):
            continue
        prov = item.prov[0] if item.prov else None
        formulas.append(
            {
                "self_ref": item.self_ref,
                "block_type": "equation",
                "text": normalize_whitespace(item.text),
                "orig": normalize_whitespace(item.orig),
                "page_number": int(prov.page_no) if prov else None,
                "bbox_pdf": bbox_to_list(prov) if prov else None,
            }
        )
    return formulas


def looks_like_noise_text(text: str) -> bool:
    stripped = normalize_whitespace(text)
    if not stripped:
        return True
    if stripped in {"…", ".", ",", ":", ";", "[", "]", "(", ")", "Web", "]."}:
        return True
    if len(stripped) <= 2 and not any(char.isalnum() for char in stripped):
        return True
    return False


def extract_metadata(items: list[Any]) -> tuple[dict[str, Any], str | None]:
    from docling_core.types.doc.document import ListItem, SectionHeaderItem, TextItem

    title: str | None = None
    abstract_heading_ref: str | None = None
    authors: list[str] = []
    institutions: list[str] = []
    abstract_parts: list[str] = []
    in_abstract = False

    for item in items:
        text = normalize_whitespace(getattr(item, "text", None))
        if not text:
            continue

        if isinstance(item, SectionHeaderItem):
            upper = text.upper()
            if title is None and upper != "ABSTRACT":
                title = text
                continue
            if upper == "ABSTRACT":
                in_abstract = True
                abstract_heading_ref = item.self_ref
                continue
            if in_abstract:
                break

        if in_abstract and isinstance(item, (TextItem, ListItem)):
            if is_visual_child(item):
                continue
            if getattr(item, "label", None) == "footnote":
                continue
            abstract_parts.append(text)
            continue

        if title is not None and not in_abstract and isinstance(item, (TextItem, ListItem)):
            if is_institution_text(text):
                institutions.append(text)
            else:
                authors.append(text)

    metadata = {
        "title": title,
        "authors": authors,
        "institutions": institutions,
        "abstract": re.sub(r"\s+\d+$", "", normalize_whitespace(" ".join(abstract_parts))),
    }
    return metadata, abstract_heading_ref


def make_section(container_name: str, heading: str) -> dict[str, Any]:
    section_name = heading
    section_path = f"{container_name}/{section_name}"
    return {
        "section_name": section_name,
        "section_path": section_path,
        "section_slug": f"{container_name}/{slugify(section_name)}",
        "heading": heading,
        "start_page": None,
        "end_page": None,
        "blocks": [],
    }


def add_block(section: dict[str, Any], block: dict[str, Any]) -> None:
    page_number = block.get("page_number")
    if section["start_page"] is None and page_number is not None:
        section["start_page"] = page_number
    if page_number is not None:
        section["end_page"] = page_number
    section["blocks"].append(block)


def build_paper_json(
    doc: Any,
    output_dir: Path,
    pdf_path: Path,
    image_scale: float,
    formula_enrichment: bool,
) -> dict[str, Any]:
    from docling_core.types.doc.document import FormulaItem, ListItem, PictureItem, SectionHeaderItem, TableItem, TextItem

    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    picture_map, picture_assets = export_pictures(doc, assets_dir, output_dir)
    table_map, table_assets = export_tables(doc, assets_dir, output_dir)
    formula_blocks = {item["self_ref"]: item for item in collect_formulas(doc)}
    asset_inventory = picture_assets + table_assets

    items = [item for item, _level in doc.iterate_items()]
    metadata, abstract_heading_ref = extract_metadata(items)

    appendix_start_page: int | None = None
    references_start_page: int | None = None
    in_references = False
    current_container = "maintext"
    current_section: dict[str, Any] | None = None
    sections = {"maintext": {"subsections": []}, "appendix": {"subsections": []}}

    for item in items:
        prov = item.prov[0] if getattr(item, "prov", None) else None
        page_number = int(prov.page_no) if prov else None
        bbox_pdf = bbox_to_list(prov) if prov else None

        if isinstance(item, SectionHeaderItem):
            if is_visual_child(item):
                continue
            heading = normalize_whitespace(item.text)
            if item.self_ref == abstract_heading_ref:
                continue
            if heading == metadata.get("title"):
                continue
            if is_references_heading(heading):
                if current_container == "maintext" and references_start_page is None:
                    references_start_page = page_number
                in_references = True
                continue
            if is_appendix_heading(heading):
                current_container = "appendix"
                in_references = False
                if appendix_start_page is None:
                    appendix_start_page = page_number

            if in_references and current_container == "maintext":
                continue

            current_section = make_section(current_container, heading)
            sections[current_container]["subsections"].append(current_section)
            add_block(
                current_section,
                {
                    "block_type": "heading",
                    "text": heading,
                    "page_number": page_number,
                    "bbox_pdf": bbox_pdf,
                    "container_section": current_container,
                },
            )
            continue

        if current_container == "maintext" and in_references:
            continue

        if isinstance(item, (TextItem, ListItem)):
            if is_visual_child(item):
                continue
            text = normalize_whitespace(item.text)
            if looks_like_noise_text(text):
                continue
            if current_section is None:
                continue
            add_block(
                current_section,
                {
                    "block_type": "text",
                    "text": text,
                    "page_number": page_number,
                    "bbox_pdf": bbox_pdf,
                    "container_section": current_container,
                },
            )
            continue

        if isinstance(item, PictureItem):
            if current_section is None:
                continue
            block = dict(picture_map[item.self_ref])
            block["container_section"] = current_container
            add_block(current_section, block)
            continue

        if isinstance(item, TableItem):
            if current_section is None:
                continue
            block = dict(table_map[item.self_ref])
            block["container_section"] = current_container
            add_block(current_section, block)
            continue

        if isinstance(item, FormulaItem):
            if is_visual_child(item):
                continue
            if current_section is None:
                continue
            block = dict(formula_blocks[item.self_ref])
            block["container_section"] = current_container
            add_block(current_section, block)

    for container_name in ("maintext", "appendix"):
        for subsection in sections[container_name]["subsections"]:
            for block in subsection["blocks"]:
                block["subsection_name"] = subsection["section_name"]
                block["subsection_path"] = subsection["section_path"]
        sections[container_name]["subsection_count"] = len(sections[container_name]["subsections"])

    combined_markdown = build_combined_markdown(metadata, sections)

    return {
        "pdf_path": str(pdf_path),
        "output_dir": str(output_dir),
        "page_count": len(doc.pages),
        "appendix_start_page": appendix_start_page,
        "processing": {
            "parser": "docling",
            "image_scale": image_scale,
            "formula_enrichment": formula_enrichment,
            "references_start_page": references_start_page,
        },
        "metadata": metadata,
        "sections": sections,
        "assets": asset_inventory,
        "combined_interleaved_markdown": combined_markdown,
    }


def build_combined_markdown(metadata: dict[str, Any], sections: dict[str, Any]) -> str:
    parts: list[str] = []
    if metadata.get("title"):
        parts.append(f"# {metadata['title']}")
    if metadata.get("authors"):
        parts.append("\n## Authors")
        parts.extend(f"- {item}" for item in metadata["authors"])
    if metadata.get("institutions"):
        parts.append("\n## Institutions")
        parts.extend(f"- {item}" for item in metadata["institutions"])
    if metadata.get("abstract"):
        parts.append(f"\n## Abstract\n{metadata['abstract']}")

    for container_name in ("maintext", "appendix"):
        for subsection in sections[container_name]["subsections"]:
            parts.append(f"\n## {subsection['heading']}")
            for block in subsection["blocks"][1:]:
                block_type = block["block_type"]
                if block_type == "text":
                    parts.append(block["text"])
                elif block_type == "figure":
                    if block.get("asset_path"):
                        parts.append(f"![{block.get('caption') or 'figure'}]({block['asset_path']})")
                    if block.get("caption"):
                        parts.append(block["caption"])
                elif block_type == "table":
                    if block.get("caption"):
                        parts.append(block["caption"])
                    if block.get("asset_path"):
                        parts.append(f"![{block.get('caption') or 'table'}]({block['asset_path']})")
                    if block.get("text"):
                        parts.append(block["text"])
                elif block_type == "equation":
                    equation_text = block.get("orig") or block.get("text")
                    if equation_text:
                        parts.append(f"```text\n{equation_text}\n```")
    return "\n\n".join(part for part in parts if part).strip() + "\n"


def export_interleaved_docling_artifacts(
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    formula_enrichment: bool = False,
    image_scale: float = 3.0,
) -> dict[str, Any]:
    from docling_core.types.doc.base import ImageRefMode

    pdf_path = Path(pdf_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    converter = build_converter(
        enable_formula_enrichment=formula_enrichment,
        image_scale=image_scale,
    )
    result = converter.convert(str(pdf_path))
    doc = result.document

    markdown_path = output_dir / f"{pdf_path.stem}.docling.md"
    html_path = output_dir / f"{pdf_path.stem}.docling.html"
    json_path = output_dir / f"{pdf_path.stem}.docling.json"
    paper_json_path = output_dir / "interleaved_content.docling.json"
    paper_markdown_path = output_dir / "interleaved_content.docling.md"

    doc.save_as_markdown(
        markdown_path,
        artifacts_dir=artifacts_dir,
        image_mode=ImageRefMode.REFERENCED,
        page_break_placeholder="\n\n---\n\n",
    )
    html_path.write_text(
        doc.export_to_html(
            image_mode=ImageRefMode.EMBEDDED,
            formula_to_mathml=True,
        ),
        encoding="utf-8",
    )
    doc.save_as_json(json_path, artifacts_dir=artifacts_dir, image_mode=ImageRefMode.REFERENCED)

    paper_payload = build_paper_json(
        doc=doc,
        output_dir=output_dir,
        pdf_path=pdf_path,
        image_scale=image_scale,
        formula_enrichment=formula_enrichment,
    )
    paper_json_path.write_text(json.dumps(paper_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paper_markdown_path.write_text(paper_payload["combined_interleaved_markdown"], encoding="utf-8")

    return {
        "markdown_path": str(markdown_path),
        "html_path": str(html_path),
        "json_path": str(json_path),
        "paper_json_path": str(paper_json_path),
        "paper_markdown_path": str(paper_markdown_path),
        "page_count": len(doc.pages),
        "paper_payload": paper_payload,
    }


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    result = export_interleaved_docling_artifacts(
        pdf_path=pdf_path,
        output_dir=output_dir,
        formula_enrichment=args.formula_enrichment,
        image_scale=args.image_scale,
    )
    paper_payload = result["paper_payload"]

    print(f"Parsed PDF: {pdf_path}")
    print(f"Markdown: {result['markdown_path']}")
    print(f"HTML: {result['html_path']}")
    print(f"JSON: {result['json_path']}")
    print(f"Paper JSON: {result['paper_json_path']}")
    print(f"Paper Markdown: {result['paper_markdown_path']}")
    print(f"Sections: {paper_payload['sections']['maintext']['subsection_count']} maintext, {paper_payload['sections']['appendix']['subsection_count']} appendix")
    print(f"Assets: {len(paper_payload['assets'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
