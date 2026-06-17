#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm
from utils.docling_parse import export_interleaved_docling_artifacts, parse_pdf_text_only


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_target_ids(queries_dir: Path) -> set[str]:
    target_ids: set[str] = set()
    for name in ("text_grounded.jsonl", "multimodal_grounded.jsonl"):
        path = queries_dir / name
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            target_ids.add(str(row["target_paper_id"]))
    return target_ids


def _write_pdf_from_base64(raw_pdf_base64: str, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    binary = base64.b64decode(raw_pdf_base64)
    tmp_path = pdf_path.with_suffix(".tmp")
    tmp_path.write_bytes(binary)
    tmp_path.replace(pdf_path)


def _run_text_only_docling_parse(pdf_path: Path) -> dict[str, Any]:
    payload = parse_pdf_text_only(pdf_path, disable_table_structure=False)
    if payload.get("ok"):
        payload["parse_mode"] = "table"
        return payload
    error_text = str(payload.get("error") or "")
    if "is not valid" in error_text.lower():
        payload["parse_mode"] = "table"
        return payload
    fallback_payload = parse_pdf_text_only(pdf_path, disable_table_structure=True)
    if fallback_payload.get("ok"):
        fallback_payload["parse_mode"] = "no_table"
        return fallback_payload
    fallback_payload["error"] = str(fallback_payload.get("error") or "docling_parse_failed")
    fallback_payload["parse_mode"] = "failed"
    return fallback_payload


def _run_interleaved_docling_parse(pdf_path: Path, output_dir: Path) -> dict[str, Any]:
    try:
        export_payload = export_interleaved_docling_artifacts(pdf_path, output_dir)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    interleaved_md = output_dir / "interleaved_content.docling.md"
    interleaved_json = output_dir / "interleaved_content.docling.json"
    raw_md = output_dir / f"{pdf_path.stem}.docling.md"
    raw_json = output_dir / f"{pdf_path.stem}.docling.json"
    raw_html = output_dir / f"{pdf_path.stem}.docling.html"
    return {
        "ok": interleaved_md.exists() and interleaved_json.exists(),
        "interleaved_markdown_path": str(interleaved_md),
        "interleaved_json_path": str(interleaved_json),
        "raw_markdown_path": str(raw_md),
        "raw_json_path": str(raw_json),
        "raw_html_path": str(raw_html),
    }


def _load_existing_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("papers") or []
    return {str(entry.get("paper_id")): entry for entry in entries if entry.get("paper_id")}


def _iter_corpus_rows(corpus_path: Path):
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build MAPLE paper Docling artifacts from the fixed2 corpus, including "
            "Stage5-style text-only markdown and interleaved markdown/json outputs."
        )
    )
    parser.add_argument(
        "--queries-dir",
        default="/data3/weiyiyang/dataset_cache/MAPLE/queries",
        help="Directory containing MAPLE query JSONL files.",
    )
    parser.add_argument(
        "--corpus-jsonl",
        default="/data3/weiyiyang/code/SciFullMMBench/outputs/hf_paper_retrieval_dataset_fixed2/corpus/corpus.jsonl",
        help="MAPLE fixed2 corpus JSONL with embedded PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data3/weiyiyang/dataset_cache/MAPLE_docling_dual_outputs_fixed2",
        help="Output root for PDFs, text-only markdown, interleaved markdown/json, and manifests.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit for testing.")
    parser.add_argument("--force", action="store_true", help="Reprocess even when outputs already exist.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index for multi-worker processing.")
    parser.add_argument("--shard-count", type=int, default=1, help="Total shard count for multi-worker processing.")
    parser.add_argument("--worker-label", default="", help="Optional label written into manifest/log metadata.")
    parser.add_argument(
        "--mode",
        choices=["both", "text_only", "interleaved"],
        default="both",
        help="Processing mode: both (default), text_only (first pass), interleaved (second pass).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    queries_dir = Path(args.queries_dir).resolve()
    corpus_jsonl = Path(args.corpus_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()

    target_ids = _load_target_ids(queries_dir)
    shard_index = int(args.shard_index)
    shard_count = int(args.shard_count)
    if shard_count <= 0:
        raise SystemExit("--shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise SystemExit("--shard-index must be in [0, shard_count)")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = output_dir / "pdfs"
    text_md_dir = output_dir / "text_only_md"
    text_meta_dir = output_dir / "text_only_meta"
    interleaved_root = output_dir / "interleaved"
    metadata_dir = output_dir / "metadata"
    for directory in (pdf_dir, text_md_dir, text_meta_dir, interleaved_root, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mode = str(args.mode)
    manifest_suffix = {"text_only": "_textonly", "interleaved": "_interleaved", "both": ""}[mode]
    manifest_path = output_dir / f"conversion_manifest{manifest_suffix}.json"
    manifest_by_id = _load_existing_manifest(manifest_path)

    do_text = mode in ("both", "text_only")
    do_interleaved = mode in ("both", "interleaved")

    # Count total papers for this shard
    shard_total = sum(
        1 for i, _ in enumerate(_iter_corpus_rows(corpus_jsonl), start=1)
        if ((i - 1) % shard_count) == shard_index
    )
    pbar = tqdm(
        total=shard_total,
        desc=f"[{args.worker_label or 'shard'+str(shard_index)}] {mode}",
        unit="paper",
        dynamic_ncols=True,
        position=shard_index,
    )

    processed = 0
    for row_index, row in enumerate(_iter_corpus_rows(corpus_jsonl), start=1):
        if ((row_index - 1) % shard_count) != shard_index:
            continue
        paper_id = str(row.get("paper_id") or "").strip()
        if not paper_id:
            pbar.update(1)
            continue
        if args.limit and processed >= int(args.limit):
            pbar.update(1)
            continue

        role = "target_paper" if paper_id in target_ids else "hard_negative"
        pdf_path = pdf_dir / f"{paper_id}.pdf"
        text_md_path = text_md_dir / f"{paper_id}.md"
        text_meta_path = text_meta_dir / f"{paper_id}.json"
        interleaved_dir = interleaved_root / paper_id
        metadata_path = metadata_dir / f"{paper_id}.json"

        # Mode-aware skip check
        skip = False
        if not args.force:
            if mode == "text_only" and text_md_path.exists():
                skip = True
            elif mode == "interleaved" and (interleaved_dir / "interleaved_content.docling.md").exists():
                skip = True
            elif mode == "both" and text_md_path.exists() and (interleaved_dir / "interleaved_content.docling.md").exists():
                skip = True
        if skip:
            processed += 1
            pbar.update(1)
            continue

        started_at = time.time()
        entry: dict[str, Any] = {
            "paper_id": paper_id,
            "role": role,
            "title": str(row.get("title") or ""),
            "abstract": str(row.get("abstract") or ""),
            "pdf_url": str(row.get("pdf_url") or ""),
            "order": row_index,
            "pdf_path": str(pdf_path),
            "text_only_markdown_path": str(text_md_path),
            "text_only_meta_path": str(text_meta_path),
            "interleaved_dir": str(interleaved_dir),
            "metadata_path": str(metadata_path),
            "worker_label": str(args.worker_label or ""),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "mode": mode,
        }

        try:
            if args.force or not pdf_path.exists():
                _write_pdf_from_base64(str(row.get("pdf_base64") or ""), pdf_path)

            # --- Text-only pass ---
            if do_text:
                text_payload = _run_text_only_docling_parse(pdf_path=pdf_path)
                text_meta_path.write_text(json.dumps(text_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                entry["text_only_ok"] = bool(text_payload.get("ok"))
                entry["text_only_parse_mode"] = str(text_payload.get("parse_mode") or "")
                if text_payload.get("ok"):
                    markdown = str(text_payload.get("markdown") or "")
                    text_md_path.write_text(markdown, encoding="utf-8")
                    entry["text_only_markdown_chars"] = len(markdown)
                    entry["page_count"] = int(text_payload.get("page_count") or 0)
                else:
                    entry["text_only_error"] = str(text_payload.get("error") or "text_only_failed")
            else:
                # Load existing text-only results for metadata continuity
                if text_meta_path.exists():
                    prev = json.loads(text_meta_path.read_text(encoding="utf-8"))
                    entry["text_only_ok"] = bool(prev.get("ok"))
                    entry["text_only_parse_mode"] = str(prev.get("parse_mode") or "")
                    entry["text_only_markdown_chars"] = prev.get("text_only_markdown_chars")
                    entry["page_count"] = prev.get("page_count")

            # --- Interleaved pass ---
            if do_interleaved:
                interleaved_dir.mkdir(parents=True, exist_ok=True)
                interleaved_payload = _run_interleaved_docling_parse(pdf_path=pdf_path, output_dir=interleaved_dir)
                entry["interleaved_ok"] = bool(interleaved_payload.get("ok"))
                if interleaved_payload.get("ok"):
                    inter_md_path = Path(str(interleaved_payload["interleaved_markdown_path"]))
                    entry["interleaved_markdown_path"] = str(inter_md_path)
                    entry["interleaved_json_path"] = str(interleaved_payload["interleaved_json_path"])
                    entry["interleaved_markdown_chars"] = len(inter_md_path.read_text(encoding="utf-8", errors="replace"))
                    entry["raw_markdown_path"] = str(interleaved_payload.get("raw_markdown_path") or "")
                    entry["raw_json_path"] = str(interleaved_payload.get("raw_json_path") or "")
                    entry["raw_html_path"] = str(interleaved_payload.get("raw_html_path") or "")
                else:
                    entry["interleaved_error"] = str(interleaved_payload.get("error") or "interleaved_failed")

            if mode == "both":
                entry["status"] = "ok" if entry.get("text_only_ok") and entry.get("interleaved_ok") else "partial"
            elif mode == "text_only":
                entry["status"] = "ok" if entry.get("text_only_ok") else "failed"
            else:
                entry["status"] = "ok" if entry.get("interleaved_ok") else "failed"
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "failed"
            entry["error"] = str(exc)

        entry["elapsed_seconds"] = round(time.time() - started_at, 3)
        metadata_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_by_id[paper_id] = entry
        manifest_path.write_text(
            json.dumps(
                {
                    "queries_dir": str(queries_dir),
                    "corpus_jsonl": str(corpus_jsonl),
                    "output_dir": str(output_dir),
                    "mode": mode,
                    "target_paper_count": len(target_ids),
                    "processed_count": len(manifest_by_id),
                    "worker_label": str(args.worker_label or ""),
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "papers": [manifest_by_id[key] for key in sorted(manifest_by_id)],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        processed += 1
        pbar.update(1)
        pbar.set_postfix_str(f"{paper_id} {entry['status']} {entry['elapsed_seconds']:.0f}s")

    pbar.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
