from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openreview_pipeline.utils.golden_retrieval_icl import (
    DEFAULT_OUTPUT_PATH,
    IR_CONSENSUS_PATH,
    QA_CONSENSUS_PATH,
    build_examples_from_csv_paths,
    write_examples_json,
)


def _build_answer_tldr_generator(
    config_path: str,
    base_url: str | None,
    model: str | None,
    cache_path: Path,
):
    from openreview_pipeline.runner import build_llm_backend

    llm = build_llm_backend(config_path, base_url=base_url, model=model)
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cache = {}

    def generate_tldr(answer_text: str) -> str:
        cache_key = hashlib.sha256(answer_text.encode("utf-8")).hexdigest()
        if cache_key in cache:
            return str(cache[cache_key])
        prompt = f"""Summarize this CrossValidated/StackExchange answer for retrieval-ICL.

Return one concise paragraph, 2-4 sentences.
Preserve the main claim, cited-paper use, and any important experimental/method detail.
Do not add facts not present in the answer.

Answer:
{answer_text}
"""
        tldr = llm.generate(prompt).strip()
        cache[cache_key] = tldr
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return tldr

    return generate_tldr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare normalized retrieval-ICL golden examples.")
    parser.add_argument(
        "--ir-csv",
        default=str(IR_CONSENSUS_PATH),
        help="Final human consensus IR CSV.",
    )
    parser.add_argument(
        "--qa-csv",
        default=str(QA_CONSENSUS_PATH),
        help="Final human consensus QA CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON path for expanded retrieval-ICL examples.",
    )
    parser.add_argument(
        "--resolve-web-titles",
        action="store_true",
        help="Resolve missing QA paper titles from DOI/arXiv URLs using public web APIs.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml for LLM TLDR generation.")
    parser.add_argument("--base-url", default=None, help="LLM API base URL override for answer TLDR generation.")
    parser.add_argument("--model", default=None, help="LLM model override for answer TLDR generation.")
    parser.add_argument(
        "--generate-answer-tldr",
        action="store_true",
        help="Use the configured LLM to generate query-level QA answer_tldr values.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    answer_tldr_generator = None
    if args.generate_answer_tldr:
        output_path = Path(args.output)
        cache_path = output_path.with_name(output_path.stem + "_answer_tldr_cache.json")
        answer_tldr_generator = _build_answer_tldr_generator(
            args.config,
            args.base_url,
            args.model,
            cache_path,
        )
    examples, report = build_examples_from_csv_paths(
        [Path(args.ir_csv), Path(args.qa_csv)],
        resolve_web_titles=bool(args.resolve_web_titles),
        answer_tldr_generator=answer_tldr_generator,
    )
    output_path = Path(args.output)
    write_examples_json(examples, output_path, report=report)
    print(f"Wrote {len(examples)} retrieval-ICL examples to {output_path}.")
    print(f"Wrote preparation report to {output_path.with_name(output_path.stem + '_report.json')}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
