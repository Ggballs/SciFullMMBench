# Multi-View Query Analysis

Utilities for checking whether golden QA/IR queries align with the `motivation`,
`method`, and `experiment` view taxonomy.

## Golden Query View Classification

Classify all 428 golden query rows under `tests/test_data/query_analysis` with two
independent LLM calls:

```bash
PYTHONPATH=src python3 src/openreview_pipeline/offline_process/multi_view_analysis/classify_golden_query_views.py \
  --input-root tests/test_data/query_analysis \
  --output-dir outputs/query_analysis/golden_view_classification_two_call \
  --config configs/config.yaml \
  --num-calls 2 \
  --sample-size 100 \
  --high-confidence 0.8 \
  --audit-size 15 \
  --batch-size 2 \
  --max-concurrent-requests 0
```

The classifier uses these labels:

- `motivation`: targets WHY this research problem exists and WHY this paper's approach is necessary, such as the research problem, need, gap, goal, hypothesis, or reason the work matters.
- `method`: targets HOW the proposed approach works mechanistically, such as the proposed approach, model, algorithm, system, dataset construction process, or implementation design.
- `experiment`: targets WHAT was empirically tested and what the results showed and analyzed, such as evaluation setup, benchmarks, datasets used for testing, metrics, baselines, ablations, empirical findings, comparisons, analytical observations, and observed limitations.

The task is multi-label. The LLM may assign more than one label and marks
underspecified cases as ambiguous. It should use `unclear` only when none of the
three view labels fits with reasonable confidence.

The default procedure runs two independent LLM calls. It then triages each query:

- `high_confidence`: both calls assign the same label set, and both confidence scores are at least `--high-confidence`.
- `low_confidence`: both calls assign the same label set, but at least one confidence score is below `--high-confidence`.
- `conflict`: the two calls assign different label sets, including partial conflicts such as `method` vs `method|experiment`.

The blind human-review sample contains all conflict cases, a stratified sample of
low-confidence cases across query type and assigned label, and a small
high-confidence audit sample controlled by `--audit-size`.

Use `--batch-size 2` for this task so each LLM request classifies at most two
queries. `--max-concurrent-requests 0` automatically uses one active request slot
per configured API key via `llm/base.py`.

Outputs:

- `golden_query_view_call1_classifications.jsonl` / `.csv`: first LLM pass.
- `golden_query_view_call2_classifications.jsonl` / `.csv`: second LLM pass.
- `golden_query_view_consensus.jsonl` / `.csv`: merged agreement and triage buckets.
- `golden_query_view_distribution.json`: primary-label and multi-label counts.
- `golden_query_view_review_sample.csv`: sampled annotation sheet for human checking.

The review sample includes blank human columns:

- `human_motivation`
- `human_method`
- `human_experiment`
- `human_ambiguous`
- `human_primary_label`
- `human_decision`
- `human_notes`

Use `1`/`0` or `yes`/`no` in the human label columns. Suggested decisions are
`accept`, `fix`, and `unclear`.

## Human Annotation UI

Launch the Gradio annotation page:

```bash
python3 src/openreview_pipeline/offline_process/multi_view_analysis/annotate_golden_query_views.py \
  --input-csv outputs/query_analysis/golden_view_classification/golden_query_view_review_sample.csv \
  --annotations-csv outputs/query_analysis/golden_view_classification/human_annotations.csv \
  --port 7870
```

The UI pre-fills the human labels from the LLM prediction. Use `Save & Next`
for normal review, and `Next Pending` to resume unfinished annotation. It writes
human labels to `human_annotations.csv`; `Export JSONL` creates a matching
`human_annotations.jsonl`.

For blind review, add `--blind`. In blind mode the LLM labels and rationales are
hidden, and the human label fields start empty.
