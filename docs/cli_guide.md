# OpenReview Pipeline CLI Guide

This guide documents how to use the Click CLI defined in [src/openreview_pipeline/cli.py](/Users/marswei/Documents/SciFullMMBench/src/openreview_pipeline/cli.py:1).

## Entry Points

The project exposes a console script via [pyproject.toml](/Users/marswei/Documents/SciFullMMBench/pyproject.toml:1):

```bash
openreview-pipeline --help
```

If the package is not installed as a console script yet, run it from the repo root with:

```bash
PYTHONPATH=src python3 -m openreview_pipeline.cli --help
```

## Global Option

Enable verbose logging:

```bash
openreview-pipeline --verbose <command> ...
```

## Command Overview

The CLI provides these commands:

- `download`
- `filter`
- `summarize`
- `init-golden-query-embeddings`
- `generate-queries`
- `decontextualize-queries`
- `hard-negative-mining`
- `query-analysis`
- `run-all`
- `update-final-json`

## Typical Workflow

The normal stage order is:

1. `download`
2. `filter`
3. `summarize`
4. `init-golden-query-embeddings` once before Stage 3
5. `generate-queries`
6. `decontextualize-queries`
7. `query-analysis`
8. `hard-negative-mining`
9. `update-final-json` if you need to rebuild the combined final JSON later

## Commands

### 1. Download

Download OpenReview papers into stage-0 JSON.

```bash
openreview-pipeline download \
  --venue ICLR \
  --year 2025 \
  --max-papers 10 \
  --output data/00_downloaded.json
```

Useful options:

- `--venue`: conference venue such as `ICLR`, `NeurIPS`, `ICML`
- `--year`: target year
- `--max-papers`: max number of papers
- `--forum-id`: fetch one or more OpenReview forum ids as a comma-separated list, for example `ID_1, ID_2`
- `--username`, `--password`, `--token`: override OpenReview credentials from `configs/config.yaml`

### 2. Filter

Run stage-1 rule filtering on downloaded papers.

```bash
openreview-pipeline filter \
  --input data/00_downloaded.json \
  --output data/01_filtered.json
```

### 3. Summarize

Run stage-2 summarization on filtered papers.

```bash
openreview-pipeline summarize \
  --input data/01_filtered.json \
  --output data/02_summarized.json
```

Useful options:

- `--base-url`
- `--model`

These override `configs/config.yaml` for the LLM backend. API keys are read from `llm.api_tokens`.

### 4. Generate Queries

Run stage-3 query generation from summarized papers. Stage 3 emits both IR and QA queries
for each supported view (`motivation`, `method`, `experiment/result`) and retrieves few-shot examples
from PostgreSQL/pgvector.

Before the first run, prepare the normalized retrieval-ICL JSON and initialize the pgvector table:

```bash
openreview-pipeline prepare-golden-retrieval-icl-examples
openreview-pipeline init-golden-query-embeddings
```

The prepare step reads final human-consensus labels, keeps `accept` and `fix`, maps
`experiment` to `experiment/result`, stores title-only targets, and builds `indexing_content`
and `retrieval_content`. If `golden_query_embeddings` is empty, Stage 3 reads
`stages.generate_queries.golden_classifications_path`, embeds `indexing_content` with BGE-M3,
and upserts rows into `golden_query_embeddings` automatically.

```bash
openreview-pipeline generate-queries \
  --input data/02_summarized.json \
  --output data/03_queries.json
```

Useful options:

- `--base-url`
- `--model`

### 5.5. Decontextualize Queries

Run the new post-stage-3 rewrite step. This command uses the stage-2 title and abstract context to
paraphrase each generated query into more natural search wording with lower lexical overlap.

```bash
openreview-pipeline decontextualize-queries \
  --summarized-input data/02_summarized.json \
  --queries-input data/03_queries.json \
  --output data/03_queries_decontextualized.json
```

Useful options:

- `--base-url`
- `--model`

The output preserves the original query in `original_query_text` and writes the rewritten text into
`query_text`, so downstream stages can consume the decontextualized version directly.

### 6. Query Analysis

Run stage-4 query analysis. This command analyzes the decontextualized query set, using stage-2 summaries and optional stage-0 metadata.

```bash
openreview-pipeline query-analysis \
  --summarized-input data/02_summarized.json \
  --queries-input data/03_queries_decontextualized.json \
  --downloaded-input data/00_downloaded.json \
  --output-dir data/04_query_analysis
```

Useful options:

- `--downloaded-input`: optional stage-0 dataset
- `--base-url`
- `--model`

Stage-4 processes papers concurrently according to `stages.max_concurrent_papers`. For each paper,
all queries are sent together in one style-judge request and one retrieval-evaluation request.

Expected output directory contents typically include:

- `query_analysis.json`
- `query_analysis.md`

### 7. Hard Negative Mining

Run stage-5 hard negative mining from generated queries. If `--query-analysis-input` is provided, only stage-4 surviving queries are mined.

```bash
openreview-pipeline hard-negative-mining \
  --input data/03_queries_decontextualized.json \
  --query-analysis-input data/04_query_analysis \
  --output data/05_hard_negatives.json
```

Useful options:

- `--base-url`
- `--model`
- `--query-analysis-input`: optional stage-4 query analysis directory used to keep only surviving queries
- `--scholar-provider`: `serpapi` or `scholarly`
- `--serpapi-api-key`
- `--scholar-max-results`
- `--scholar-language`
- `--download-selected-pdfs`: download selected candidate PDFs instead of storing URL-only metadata

### 7. Run All

Run the full pipeline end to end.

```bash
openreview-pipeline run-all \
  --output-dir data \
  --venue ICLR \
  --year 2025 \
  --max-papers 10 \
  --summarize-limit 5
```

Useful options:

- `--forum-id`: run against one or more known papers as a comma-separated list
- `--base-url`
- `--model`
- `--summarize-limit`: maximum papers to summarize with the LLM
- `--skip-filter`: create an all-passed `01_filtered.json` instead of applying stage-1 rules

For a single-paper smoke test:

```bash
openreview-pipeline run-all \
  --output-dir outputs/single_paper_test \
  --venue ICLR \
  --year 2025 \
  --forum-id <OPENREVIEW_FORUM_ID> \
  --summarize-limit 1
```

For multiple known papers, pass comma-separated forum ids. In forum-id mode, rerunning against the same output directory merges new papers into the existing `00_downloaded.json`; newly fetched papers replace existing rows with the same paper id.

```bash
openreview-pipeline run-all \
  --output-dir outputs/forum_batch \
  --forum-id XZNXSM4rHG,ID_2,ID_3
```

To bypass stage-1 filtering and summarize every downloaded paper:

```bash
openreview-pipeline run-all \
  --output-dir outputs/forum_batch \
  --forum-id XZNXSM4rHG,ID_2,ID_3 \
  --skip-filter
```

Later, add more papers to the same folder:

```bash
openreview-pipeline run-all \
  --output-dir outputs/forum_batch \
  --forum-id ID_4,ID_5
```

Each stage is resume-aware. If an output file already exists, the stage loads completed
paper/query rows from that file, reports them as successful, and continues from the
first missing item. This is useful after API rate limits, interrupted LLM calls, or
aborted hard-negative mining runs.

If you do not have a forum id, use:

```bash
openreview-pipeline run-all \
  --output-dir outputs/single_paper_test \
  --venue ICLR \
  --year 2025 \
  --max-papers 1 \
  --summarize-limit 1
```

### 8. Update Final JSON

Rebuild the final combined output JSON from stage artifact files already on disk.

```bash
openreview-pipeline update-final-json \
  --base-dir data \
  --output data/final_pipeline_output.json
```

Useful overrides:

- `--downloaded-path`
- `--filtered-path`
- `--summarized-path`
- `--queries-path`
- `--hard-negatives-path`
- `--query-analysis-dir`

## Common Patterns

### Show top-level help

```bash
openreview-pipeline --help
```

### Show command-specific help

```bash
openreview-pipeline query-analysis --help
```

### Use the module path instead of the installed script

```bash
PYTHONPATH=src python3 -m openreview_pipeline.cli run-all --help
```

## Configuration Notes

The CLI reads configuration from:

- [configs/config.yaml](/Users/marswei/Documents/SciFullMMBench/configs/config.yaml:1)

CLI flags such as `--base-url`, `--model`, `--username`, and `--token` override config values for that invocation. LLM API keys are configured only through `llm.api_tokens`.

The LLM config uses a unified multi-key request manager:

```yaml
llm:
  base_url: "https://api.chatanywhere.tech/v1"
  model: "gpt-5.4-mini-ca"
  api_tokens:
    - "key_1"
    - "key_2"
  per_key_request_interval_seconds: 2.0
  per_key_max_concurrent_requests: 1
  max_retries: 3
  retry_backoff_seconds: 8.0
  max_tokens: 4096
  temperature: 0.0

stages:
  max_concurrent_papers: 5
  hard_negative_mining:
    review_max_workers: 1
```

`stages.max_concurrent_papers` controls paper-level concurrency in stage-2, stage-3, and stage-4.
The LLM manager then assigns each request to an API key using round-robin selection. Each key has
its own request interval and in-flight request cap.

## Output Layout

A typical output layout under `data/` is:

```text
data/
  00_downloaded.json
  01_filtered.json
  02_summarized.json
  03_queries.json
  04_query_analysis/
  05_hard_negatives.json
  final_pipeline_output.json
```

## Recommendation

Use `run-all` for the simplest end-to-end path.

Use the stage-specific commands when:

- you want to debug one stage in isolation
- you already have intermediate JSON artifacts
- you want to rerun query analysis without rerunning earlier stages



## Note
```bash
PYTHONPATH=src python3 -m openreview_pipeline.cli run-all \
  --venue ICLR \
  --year 2026 \
  --forum-id XZNXSM4rHG,ID_2 \
  --output-dir outputs/test_single

```

```bash
PYTHONPATH=src python3 -m openreview_pipeline.cli summarize \
  --input outputs/test_single/01_filtered.json \
  --output outputs/test_single/02_summarized.json

```

```bash
PYTHONPATH=src python3 -m openreview_pipeline.cli update-final-json \
  --base-dir ../outputs/test_run_10

```
```bash
PYTHONPATH=src python3 -m openreview_pipeline.cli run-all \
  --output-dir outputs/test_run_10 \
  --venue ICLR \
  --year 2025 \
  --max-papers 10 \
  --summarize-limit 10
```
```bash
PYTHONPATH=src python3 -m openreview_pipeline.cli query-analysis \
  --summarized-input ../outputs/test_run_10/02_summarized.json \
  --queries-input ../outputs/test_run_10/03_queries.json \
  --downloaded-input ../outputs/test_run_10/00_downloaded.json \
  --output-dir ../outputs/test_run_10/04_query_analysis
```
```commandline
PYTHONPATH=src python3 -m openreview_pipeline.cli run-all \
  --output-dir outputs/forum_batch \
  --forum-id txmqENuRcc,OBMcxeSK5U,yfcpdY4gMP,WMx2DTFltx
```



## server
```commandline
ssh weiyiyang@10.0.1.226 'cd ~/code/SciFullMMBench && \
OUT=outputs/multimodal_pipeline_JEGDp1E4OH_$(date +%Y%m%d_%H%M%S) && \
docker compose exec -T gradio python -m openreview_pipeline.cli run-pipeline \
  --stages 0-4 \
  --output-dir "$OUT" \
  --forum-id JEGDp1E4OH \
  --max-papers 1 \
  --summarize-limit 1 && \
docker compose exec -T gradio python tests/scripts/render_query_bullet_comments_md.py \
  "$OUT/final_pipeline_output.json" \
  -o "$OUT/query_bullet_original_comments.md" \
  --max-comment-chars 2000 && \
echo "$OUT"'

```
```commandline
ssh weiyiyang@10.0.1.226 'cd ~/code/SciFullMMBench && \
FINAL_JSON=outputs/multimodal_pipeline_JEGDp1E4OH_20260526_201634/final_pipeline_output.json \
docker compose up -d --force-recreate gradio'
```
