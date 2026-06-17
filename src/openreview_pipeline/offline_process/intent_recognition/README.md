# Cross Validated Offline Data Filter

This folder is intentionally minimal. It downloads and filters Cross Validated data, then exports retained question queries.

## Usage

Run the full Cross Validated pipeline:

```bash
python3 src/openreview_pipeline/offline_process/intent_recognition/download_and_filter.py \
  --process-crossvalidated
```

This downloads the March 2026 Cross Validated archive `stats.stackexchange.com.7z`, extracts `Posts.xml` with `bsdtar`, filters paper-grounded questions from 2020-present, and exports retained queries.

Filter an already extracted Cross Validated `Posts.xml`:

```bash
python3 src/openreview_pipeline/offline_process/intent_recognition/download_and_filter.py \
  --crossvalidated-posts /path/to/stats.stackexchange.com.20260331/Posts.xml
```

Download the Cross Validated archive only:

```bash
python3 src/openreview_pipeline/offline_process/intent_recognition/download_and_filter.py \
  --download-crossvalidated-archive
```

## Outputs

- `data/raw/crossvalidated_questions.jsonl`: normalized 2020-present candidate questions with score greater than 5.
- `data/filtered/paper_grounded_posts.jsonl`: retained paper-grounded records.
- `data/filtered/queries.json`: one JSON array of cleaned query records extracted from retained records.
- `reports/filter_summary.json` and `reports/filter_summary.md`: counts and latest QA timestamps.

Each filtered record includes `source`, `id`, `title`, `body`, `score`, `created_at`, `created_date`, `created_utc`, `tags`, `flair`, `url`, and `qualifying_answers`.

## Filter

Cross Validated records are kept when:

- question creation date is 2020-01-01 or later
- question title ends with `?`
- question score is greater than 5
- at least one answer has score greater than 5
- that answer contains `arxiv`, `doi`, or `aclanthology`
