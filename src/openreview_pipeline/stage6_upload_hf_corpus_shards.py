#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload sharded corpus JSONL files to a Hugging Face dataset repo.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--delete-monolith", action="store_true")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir).expanduser().resolve()
    shards = sorted(shard_dir.glob("corpus-*.jsonl"))
    if not shards:
        raise SystemExit(f"no shards found in {shard_dir}")

    api = HfApi()
    print(
        {
            "repo_id": args.repo_id,
            "repo_type": args.repo_type,
            "num_shards": len(shards),
            "first_shard": shards[0].name,
            "last_shard": shards[-1].name,
        },
        flush=True,
    )

    for idx, shard in enumerate(shards, 1):
        path_in_repo = f"corpus/{shard.name}"
        last_err = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                out = api.upload_file(
                    path_or_fileobj=str(shard),
                    path_in_repo=path_in_repo,
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    token=args.token,
                    commit_message=f"Add {shard.name} ({idx}/{len(shards)})",
                )
                print(
                    f"OK shard={shard.name} idx={idx}/{len(shards)} attempt={attempt} commit={out}",
                    flush=True,
                )
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                print(
                    f"RETRY shard={shard.name} idx={idx}/{len(shards)} attempt={attempt} "
                    f"err={type(exc).__name__}: {str(exc)[:300]}",
                    flush=True,
                )
                time.sleep(15 * attempt)
        if last_err is not None:
            raise last_err

    if args.delete_monolith:
        try:
            out = api.delete_file(
                path_in_repo="corpus/corpus.jsonl",
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                token=args.token,
                commit_message="Remove monolithic corpus.jsonl after sharded upload",
            )
            print(f"DELETE_OK commit={out}", flush=True)
        except Exception as exc:
            print(f"DELETE_RETRY_NEEDED err={type(exc).__name__}: {exc}", flush=True)

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
