#!/usr/bin/env python3
"""Continuously embed new markdown papers and store in PostgreSQL + BM25 index."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

from utils.db.paper_text_embeddings import (
    PAPER_TEXT_EMBEDDINGS_TABLE,
    ensure_table,
    existing_embeddings,
    get_engine,
    insert_embeddings,
)

# Set JAVA_HOME for pyserini (jnius needs JDK at import time)
_JDK_HOME = "/data3/weiyiyang/jdk21"
if os.path.isdir(_JDK_HOME) and "JAVA_HOME" not in os.environ:
    os.environ["JAVA_HOME"] = _JDK_HOME

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MD_DIR = Path("/data3/weiyiyang/dataset_cache/MAPLE_docling_dual_outputs_fixed2/text_only_md")
META_DIR = Path("/data3/weiyiyang/dataset_cache/MAPLE_docling_dual_outputs_fixed2/metadata")
BM25_INDEX_DIR = Path("/data3/weiyiyang/dataset_cache/MAPLE_docling_dual_outputs_fixed2/bm25_index")
BM25_JSONL = Path("/data3/weiyiyang/dataset_cache/MAPLE_docling_dual_outputs_fixed2/bm25_docs.jsonl")
BM25_TRACK = Path("/data3/weiyiyang/dataset_cache/MAPLE_docling_dual_outputs_fixed2/bm25_indexed.txt")

BGE_URL = "http://localhost:18080/embed"
MULTI_URL = "http://localhost:18081/embed/{model_name}"
GRITLM_URL = "http://localhost:18082/embed"

# Model configs: name → (port, source, max_chars)
# source: "full" = full markdown text, "ta" = title + abstract
# Full-text models: each paper fills the token limit → one at a time
FULL_TEXT_MODELS: list[dict[str, Any]] = [
    {"name": "bge-m3", "url": BGE_URL, "max_chars": 8000},
    {"name": "qwen3-embed-8b", "url": MULTI_URL, "max_chars": 32000},
    {"name": "gritlm-7b", "url": GRITLM_URL, "max_chars": 14000},
]

# TA (title+abstract) models: short text → can batch many
TA_MODELS: list[dict[str, Any]] = [
    {"name": "specter2", "url": MULTI_URL, "max_chars": 1000, "batch_size": 32},
    {"name": "scincl", "url": MULTI_URL, "max_chars": 1000, "batch_size": 32},
    {"name": "instructor-xl", "url": MULTI_URL, "max_chars": 1000, "batch_size": 32},
]
# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def read_text(paper_id: str) -> str | None:
    md_path = MD_DIR / f"{paper_id}.md"
    if md_path.exists():
        return md_path.read_text(encoding="utf-8", errors="replace")
    return None


def read_title_abstract(paper_id: str) -> str | None:
    meta_path = META_DIR / f"{paper_id}.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        title = str(meta.get("title") or "").strip()
        abstract = str(meta.get("abstract") or "").strip()
        if title and abstract:
            return f"{title}\n{abstract}"
        return title or abstract or None
    except (json.JSONDecodeError, OSError):
        return None


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Embedding API
# ---------------------------------------------------------------------------

def call_embed_api(url: str, texts: list[str], timeout: int = 120) -> list[list[float]]:
    """Call embedding API and return list of embedding vectors."""
    resp = requests.post(
        url,
        json={"texts": texts},
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("embeddings", [])


def embed_papers(
    model: dict[str, Any],
    paper_ids: list[str],
    texts: list[str],
) -> tuple[list[str], list[list[float]], dict[str, int]]:
    """Embed papers. Batches multiple if batch_size set, otherwise one at a time."""
    url = model["url"]
    if "{model_name}" in url:
        url = url.format(model_name=model["name"])
    bs = model.get("batch_size", 1)

    ok_pids: list[str] = []
    ok_embs: list[list[float]] = []
    char_map: dict[str, int] = {}
    error_count = 0
    max_errors = 5

    total = len(paper_ids)
    for i in range(0, total, bs):
        batch_pids = paper_ids[i:i + bs]
        batch_texts = texts[i:i + bs]
        try:
            embs = call_embed_api(url, batch_texts)
            if len(embs) != len(batch_pids):
                print(f"  WARNING: got {len(embs)} embeddings for {len(batch_pids)}", flush=True)
                embs = embs[:len(batch_pids)]
            ok_pids.extend(batch_pids)
            ok_embs.extend(embs)
            for pid, txt in zip(batch_pids, batch_texts):
                char_map[pid] = len(txt)
            error_count = 0
            if bs == 1 and (i + 1) % 100 == 0:
                print(f"  [{model['name']}] progress: {i + 1}/{total}", flush=True)
        except Exception as exc:
            error_count += 1
            print(f"  ERROR [{model['name']}]: {exc}", flush=True)
            if error_count >= max_errors:
                print(f"  [{model['name']}] too many errors, skipping remaining", flush=True)
                break

    return ok_pids, ok_embs, char_map


# ---------------------------------------------------------------------------
# BM25 via Pyserini
# ---------------------------------------------------------------------------

def bm25_indexed_papers() -> set[str]:
    if BM25_TRACK.exists():
        return set(BM25_TRACK.read_text(encoding="utf-8").strip().splitlines())
    return set()


def bm25_add_and_reindex(new_paper_ids: list[str]) -> int:
    """Append new papers to JSONL, rebuild Pyserini index. Returns count added."""
    if not new_paper_ids:
        return 0

    existing_ids = bm25_indexed_papers()
    added = 0
    with BM25_JSONL.open("a", encoding="utf-8") as f:
        for pid in new_paper_ids:
            if pid in existing_ids:
                continue
            text = read_text(pid)
            if text is None:
                continue
            doc = {"id": pid, "contents": text}
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            existing_ids.add(pid)
            added += 1

    if added > 0:
        _build_bm25_index()
        BM25_TRACK.write_text("\n".join(sorted(existing_ids)), encoding="utf-8")

    return added


def _build_bm25_index() -> None:
    """Build BM25 index from JSONL using pyserini LuceneIndexer."""
    try:
        import json as _json

        # pyserini imports openai at module level — needs dummy key
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "dummy-for-bm25"

        from pyserini.index.lucene import LuceneIndexer

        # Remove existing index to rebuild clean
        import shutil
        if BM25_INDEX_DIR.exists():
            shutil.rmtree(str(BM25_INDEX_DIR))
        BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        docs = []
        if BM25_JSONL.exists():
            for line in BM25_JSONL.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    docs.append(_json.loads(line))

        if docs:
            indexer = LuceneIndexer(
                index_dir=str(BM25_INDEX_DIR),
                threads=4,
            )
            indexer.add_batch_dict(docs)
            indexer.close()
            print(f"  BM25 index built with {len(docs)} docs at {BM25_INDEX_DIR}", flush=True)
        else:
            print("  BM25: no documents to index", flush=True)
    except Exception:
        print("  WARNING: pyserini failed, skipping BM25 index rebuild", flush=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def process_loop(engine, once: bool = False, interval: int = 60, skip_bm25: bool = False) -> None:
    print("Ensuring table exists...", flush=True)
    ensure_table(engine)

    while True:
        t_start = time.time()

        # 1. Discover papers
        all_md = {p.stem for p in MD_DIR.glob("*.md")}
        embedded_pairs = existing_embeddings(engine)
        bm25_done = bm25_indexed_papers()

        new_papers = all_md - {pid for pid, _ in embedded_pairs}
        bm25_new = all_md - bm25_done

        total = len(all_md)
        embedded_count = len({pid for pid, _ in embedded_pairs})
        print(f"\n[{time.strftime('%H:%M:%S')}] {total} papers on disk | "
              f"{embedded_count} have embeddings | {len(bm25_done)} in BM25", flush=True)

        # 2. BM25 index new papers
        if not skip_bm25 and bm25_new:
            print(f"  BM25: indexing {len(bm25_new)} new papers...", flush=True)
            n = bm25_add_and_reindex(list(bm25_new))
            print(f"  BM25: {n} added to index", flush=True)

        # 3. Dense embeddings — BGE-M3 first (independent, always available)
        for model in FULL_TEXT_MODELS:
            model_name = model["name"]
            todo = [pid for pid in sorted(all_md)
                    if (pid, model_name) not in embedded_pairs]
            if not todo:
                continue
            print(f"  [{model_name}] {len(todo)} papers to embed (max {model['max_chars']} chars)", flush=True)

            texts = []
            valid_pids = []
            for pid in todo:
                txt = read_text(pid)
                if txt:
                    valid_pids.append(pid)
                    texts.append(truncate(txt, model["max_chars"]))

            if valid_pids:
                try:
                    pids, embs, chars = embed_papers(model, valid_pids, texts)
                    if pids:
                        n = insert_embeddings(engine, pids, model_name, embs, chars)
                        print(f"  [{model_name}] inserted {n}", flush=True)
                except Exception as exc:
                    print(f"  [{model_name}] FATAL: {exc}", flush=True)

        # 4. TA models: title+abstract variant
        for model in TA_MODELS:
            base_name = model["name"]
            ta_name = f"{base_name}_ta"
            todo = [pid for pid in sorted(all_md)
                    if (pid, ta_name) not in embedded_pairs]
            if not todo:
                continue
            print(f"  [{ta_name}] {len(todo)} papers (title+abstract, max {model['max_chars']} chars)", flush=True)

            texts = []
            valid_pids = []
            for pid in todo:
                txt = read_title_abstract(pid)
                if txt:
                    valid_pids.append(pid)
                    texts.append(truncate(txt, model["max_chars"]))

            if valid_pids:
                pids, embs, chars = embed_papers(model, valid_pids, texts)
                if pids:
                    n = insert_embeddings(engine, pids, ta_name, embs, chars)
                    print(f"  [{ta_name}] inserted {n}", flush=True)

        # 5. TA models: full-text (truncated) variant
        for model in TA_MODELS:
            base_name = model["name"]
            full_name = f"{base_name}_full"
            todo = [pid for pid in sorted(all_md)
                    if (pid, full_name) not in embedded_pairs]
            if not todo:
                continue
            print(f"  [{full_name}] {len(todo)} papers (full text truncated, max {model['max_chars']} chars)", flush=True)

            texts = []
            valid_pids = []
            for pid in todo:
                txt = read_text(pid)
                if txt:
                    valid_pids.append(pid)
                    texts.append(truncate(txt, model["max_chars"]))

            if valid_pids:
                pids, embs, chars = embed_papers(model, valid_pids, texts)
                if pids:
                    n = insert_embeddings(engine, pids, full_name, embs, chars)
                    print(f"  [{full_name}] inserted {n}", flush=True)

        elapsed = time.time() - t_start
        print(f"  Loop took {elapsed:.1f}s", flush=True)

        if once:
            break

        sleep_time = max(0, interval - elapsed)
        if sleep_time > 0:
            print(f"  Sleeping {sleep_time:.0f}s...", flush=True)
            time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuously embed papers and build BM25 index."
    )
    parser.add_argument("--once", action="store_true", help="Run one pass and exit.")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between polling cycles (default: 60).")
    parser.add_argument("--db-url", type=str, default=None,
                        help="PostgreSQL connection URL (default: localhost:5433).")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Specific models to embed (e.g. bge-m3 specter2_ta). Default: all.")
    parser.add_argument("--skip-bm25", action="store_true", help="Skip BM25 indexing.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    engine = get_engine(db_url=args.db_url)
    process_loop(engine, once=args.once, interval=args.interval, skip_bm25=args.skip_bm25)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
