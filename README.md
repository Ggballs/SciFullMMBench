# OpenReview Pipeline

A staged OpenReview processing pipeline for filtering, summarizing, generating retrieval queries, analyzing query quality, and mining hard negatives from academic papers.

## Features

- **Stage 0**: Download recent OpenReview papers (configurable venue and year range)
- **Stage 1**: Rule-based filtering of papers based on quality criteria
- **Stage 2**: LLM-based bullet-point summarization from multiple views
- **Stage 3**: LLM-based retrieval query generation
- **Stage 4**: LLM-based query analysis and query quality filtering
- **Stage 5**: Hard-negative and positive candidate mining for surviving queries

## Installation

```bash
pip install -e .
```

## Project Structure

```
.
├── pyproject.toml
├── prompts/
│   ├── summarize_by_view.txt
│   ├── generate_queries.txt
│   └── query_analysis/
│       ├── retrieval_effectiveness.txt
│       └── style_analysis.txt
├── src/openreview_pipeline/
│   ├── __init__.py
│   ├── cli.py
│   ├── schemas.py
│   ├── schemas_filter.py
│   ├── schemas_summarize.py
│   ├── schemas_queries.py
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── stage0_download.py
│   │   ├── stage1_filter.py
│   │   ├── stage2_summarize.py
│   │   ├── stage3_generate_queries.py
│   │   ├── stage4_query_analysis.py
│   │   └── stage5_hard_negative_mining.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── base.py
│   └── utils/
│       └── __init__.py
└── tests/
    └── test_pipeline.py
```

## Usage

### CLI Commands

```bash
# Download papers
openreview-pipeline download --output data/00_downloaded.json --venue ICLR --year 2025 --max-papers 10

# Filter papers
openreview-pipeline filter --input data/00_downloaded.json --output data/01_filtered.json

# Summarize papers
openreview-pipeline summarize --input data/01_filtered.json --output data/02_summarized.json

# Generate queries
openreview-pipeline generate-queries --input data/02_summarized.json --output data/03_queries.json

# Analyze generated query quality
openreview-pipeline query-analysis --summarized-input data/02_summarized.json --queries-input data/03_queries.json --output-dir data/04_query_analysis --judge-batch-size 10 --retrieval-batch-size 10

# Mine hard negatives for surviving queries
openreview-pipeline hard-negative-mining --input data/03_queries.json --query-analysis-input data/04_query_analysis --output data/05_hard_negatives.json

# Run all stages
openreview-pipeline run-all --output-dir data --venue ICLR --year 2025 --max-papers 10 --summarize-limit 5 --judge-batch-size 10 --retrieval-batch-size 10
```

### Python API

```python
from openreview_pipeline.stages import (
    DatasetDownloader,
    RuleBasedFilter,
    Summarizer,
    QueryGenerator,
    HardNegativeMiner,
)
from openreview_pipeline.llm import MockLLMBackend
from pathlib import Path

# Initialize LLM backend
llm = MockLLMBackend()

# Download
downloader = DatasetDownloader(venue="ICLR")
downloader.run(Path("data/00_downloaded.json"))

# Filter
filter_stage = RuleBasedFilter()
filter_stage.run(Path("data/00_downloaded.json"), Path("data/01_filtered.json"))

# Summarize
summarizer = Summarizer(llm=llm)
summarizer.run(Path("data/01_filtered.json"), Path("data/02_summarized.json"))

# Generate queries
generator = QueryGenerator(llm=llm)
generator.run(Path("data/02_summarized.json"), Path("data/03_queries.json"))

# Hard-negative mining
miner = HardNegativeMiner(llm=llm)
miner.run(Path("data/03_queries.json"), Path("data/05_hard_negatives.json"))
```

## LLM Backend

The pipeline uses an abstract `LLMBackend` interface. Currently available backends:

- `MockLLMBackend` - Returns mock responses (for testing)
- `OpenAIBackend` - OpenAI API integration (stub, needs implementation)
- `AnthropicBackend` - Anthropic API integration (stub, needs implementation)

To add a new backend, implement the `LLMBackend` abstract class:

```python
from openreview_pipeline.llm import LLMBackend

class MyLLMBackend(LLMBackend):
    def generate(self, prompt: str, **kwargs) -> str:
        # Your implementation
        return response

    def generate_json(self, prompt: str, **kwargs) -> dict:
        # Your implementation
        return {"key": "value"}
```

## Output Format

All outputs are JSON files with structured data:

- `00_downloaded.json` - DownloadedPapersDataset
- `01_filtered.json` - FilteredPapersDataset
- `02_summarized.json` - SummarizedPapersDataset
- `03_queries.json` - GeneratedQueriesDataset
- `04_filtered_queries.json` - FilteredQueriesDataset

## Testing

```bash
pytest tests/
```

## Development

```bash
pip install -e ".[dev]"
ruff check src/
```

---

## Query Analysis Module

The `query_analysis` module analyzes human-written queries from academic IR datasets (LitSearch, PASA) to generate instruction prompts for rewriting OpenReview bullet points into realistic human search queries.

### Purpose

Convert OpenReview paper bullet points into natural, human-like search queries that researchers would actually type into academic search systems.

### Input Format Assumptions

Datasets should have:
- **Query column**: Contains the query text (configurable via `--query_column`)
- **Human flag column**: Indicates if query is human-written (configurable via `--human_flag_column`)
- **Supported formats**: CSV, TSV, JSON, JSONL, or HuggingFace dataset ID

**LitSearch** (princeton-nlp/LitSearch):
- Uses `query_set` column to distinguish human vs synthetic
- Human queries: `manual_acl` (155) + `manual_iclr` (91) = 246 queries
- Synthetic queries: `inline_acl` (98) + `inline_nonacl` (253) = 351 GPT-4 generated queries
- Can load directly via HuggingFace dataset ID (e.g., `princeton-nlp/LitSearch`)

**PASA** (CarlanLark/pasa-dataset on HuggingFace):
- **AutoScholarQuery**: 35k synthetic queries (GPT-4o generated) - for training
- **RealScholarQuery**: 50 real human queries from AI researchers - for evaluation
- RealScholarQuery has NO human flag column - all queries are human by definition
- AutoScholarQuery has NO human flag column - all queries are synthetic (should be excluded)
- If no human flag column exists, loader assumes all queries are human (for RealScholarQuery)

### Project Structure

```
src/query_analysis/
├── __init__.py
├── loaders.py          # Dataset adapters (LitSearch, PASA)
├── features.py         # Reusable analysis functions
├── analyze_query_style.py  # Main analysis script
└── build_rewrite_prompt.py # Prompt generation script
```

### How to Run

**Step 1: Analyze query style**

```bash
# For LitSearch (from HuggingFace)
python -m query_analysis.analyze_query_style \
    --litsearch_path princeton-nlp/LitSearch \
    --output_dir outputs/query_analysis \
    --human_flag_column query_set

# For LitSearch + PASA/RealScholarQuery
python -m query_analysis.analyze_query_style \
    --litsearch_path princeton-nlp/LitSearch \
    --pasa_path CarlanLark/pasa-dataset/RealScholarQuery \
    --output_dir outputs/query_analysis \
    --human_flag_column query_set

# For local files (CSV/JSONL)
python -m query_analysis.analyze_query_style \
    --litsearch_path /path/to/litsearch.csv \
    --pasa_path /path/to/realscholarquery.jsonl \
    --output_dir outputs/query_analysis \
    --min_tokens 2 \
    --seed 42

# Also render query word-count distribution artifacts
python -m query_analysis.analyze_query_style \
    --litsearch_path /path/to/litsearch.csv \
    --output_dir outputs/query_analysis \
    --plot_length_distribution
```

**Step 2: Generate rewrite instruction prompt**

```bash
python -m query_analysis.build_rewrite_prompt \
    --analysis_path outputs/query_analysis/style_analysis.json \
    --output_path outputs/query_analysis/rewrite_instruction_prompt.txt
```

### Outputs

After running both scripts:

```
outputs/query_analysis/
├── style_analysis.json           # Detailed analysis in JSON format
├── style_analysis.md             # Markdown report
├── query_length_distribution.png # Optional word-count histogram + boxplot
├── query_length_distribution_summary.json # Optional plot summary stats
└── rewrite_instruction_prompt.txt # Operational prompt for rewriting
```

**style_analysis.json** contains:
- `dataset_overview`: Per-dataset statistics
- `filtering_rules`: How human queries were filtered
- `length_stats`: Token/char length distributions
- `lexical_patterns`: Punctuation, stopwords, surface patterns
- `template_patterns`: Common query templates (e.g., "X for Y", "X vs Y")
- `intent_patterns`: Exploratory, comparison, lookup, diagnostic
- `comparative_findings`: Cross-dataset comparison
- `human_style_principles`: Derived rewrite rules
- `representative_examples`: Example queries

**rewrite_instruction_prompt.txt** is an operational prompt containing:
- Role and task definition
- Explicit style rules from analysis
- What to preserve from bullet points
- What to avoid (hallucination, verbosity, formulaic outputs)
- Fragment bullet handling guidelines
- One vs multiple query guidance
- 8 example rewrites
- Output format specification

### CLI Arguments

**analyze_query_style.py:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--litsearch_path` | None | Path to LitSearch CSV/JSON/JSONL |
| `--pasa_path` | None | Path to PASA dataset |
| `--output_dir` | outputs/query_analysis | Output directory |
| `--query_column` | query | Query column name |
| `--human_flag_column` | category | Human flag column |
| `--human_flag_value` | human | Value for human-written |
| `--pasa_human_value` | true | PASA human flag value |
| `--min_tokens` | 2 | Minimum token count |
| `--seed` | 42 | Random seed |

**build_rewrite_prompt.py:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--analysis_path` | outputs/query_analysis/style_analysis.json | Input analysis JSON |
| `--output_path` | outputs/query_analysis/rewrite_instruction_prompt.txt | Output prompt file |
