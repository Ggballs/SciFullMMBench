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
- `generate-queries`
- `hard-negative-mining`
- `query-analysis`
- `run-all`
- `update-final-json`

## Typical Workflow

The normal stage order is:

1. `download`
2. `filter`
3. `summarize`
4. `generate-queries`
5. `query-analysis`
6. `hard-negative-mining`
7. `update-final-json` if you need to rebuild the combined final JSON later

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
- `--username`, `--password`, `--token`: override OpenReview credentials from `config.yaml`

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
- `--api-token`
- `--model`

These override `config.yaml` for the LLM backend.

### 4. Generate Queries

Run stage-3 query generation from summarized papers.

```bash
openreview-pipeline generate-queries \
  --input data/02_summarized.json \
  --output data/03_queries.json
```

Useful options:

- `--base-url`
- `--api-token`
- `--model`

### 5. Query Analysis

Run stage-4 query analysis. This command analyzes the stage-3 query set, using stage-2 summaries and optional stage-0 metadata.

```bash
openreview-pipeline query-analysis \
  --summarized-input data/02_summarized.json \
  --queries-input data/03_queries.json \
  --downloaded-input data/00_downloaded.json \
  --output-dir data/04_query_analysis
```

Useful options:

- `--downloaded-input`: optional stage-0 dataset
- `--base-url`
- `--api-token`
- `--model`
- `--judge-batch-size`: default `10`; ignored when `--llm-judge-mode single_query`
- `--llm-judge-mode`: `batch` or `single_query`
- `--judge-max-concurrency`: default `1`
- `--retrieval-batch-size`: default `10`; batches retrieval-effectiveness checks per paper

Expected output directory contents typically include:

- `query_analysis.json`
- `query_analysis.md`

### 6. Hard Negative Mining

Run stage-5 hard negative mining from generated queries. If `--query-analysis-input` is provided, only stage-4 surviving queries are mined.

```bash
openreview-pipeline hard-negative-mining \
  --input data/03_queries.json \
  --query-analysis-input data/04_query_analysis \
  --output data/05_hard_negatives.json
```

Useful options:

- `--base-url`
- `--api-token`
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
- `--api-token`
- `--model`
- `--summarize-limit`: maximum papers to summarize with the LLM
- `--judge-batch-size`
- `--llm-judge-mode`
- `--judge-max-concurrency`
- `--retrieval-batch-size`

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

Later, add more papers to the same folder:

```bash
openreview-pipeline run-all \
  --output-dir outputs/forum_batch \
  --forum-id ID_4,ID_5
```

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

- [config.yaml](/Users/marswei/Documents/SciFullMMBench/config.yaml:1)

CLI flags such as `--base-url`, `--api-token`, `--model`, `--username`, and `--token` override config values for that invocation.

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
