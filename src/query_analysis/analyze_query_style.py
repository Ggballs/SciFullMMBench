#!/usr/bin/env python3
"""
Query Style Analyzer

Analyze the style of human-written queries from LitSearch and PASA datasets,
and generate insights for rewriting OpenReview bullet points into human-like queries.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from query_analysis.loaders import (
    LitSearchLoader,
    PasaLoader,
)
from query_analysis.features import (
    compute_all_metrics,
    extract_representative_examples,
)
from query_analysis.llm_judge import (
    compute_llm_judged_metrics,
    compute_llm_semantic_constraint_metrics,
    load_llm_config,
    summarize_judge_results,
)
from query_analysis.length_visualization import save_length_distribution_plot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetAnalysis:
    """Container for per-dataset analysis results."""
    name: str
    total_queries: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    representative_examples: List[Dict[str, str]] = field(default_factory=list)
    queries_list: List[str] = field(default_factory=list)


class QueryStyleAnalyzer:
    """Main analyzer for query style analysis."""

    def __init__(
        self,
        litsearch_path: Optional[str] = None,
        pasa_path: Optional[str] = None,
        litsearch_query_column: str = "query",
        pasa_query_column: str = "question",
        human_flag_column: str = "query_set",
        human_flag_value: str = "manual",
        pasa_human_flag_value: Any = True,
        min_tokens: int = 2,
        seed: int = 42,
        config_path: str = "config.yaml",
        llm_batch_size: int = 25,
        llm_judge_mode: str = "batch",
        llm_max_concurrency: int = 1,
        llm_seed: Optional[int] = None,
        use_llm_judge: bool = True,
    ):
        self.litsearch_path = litsearch_path
        self.pasa_path = pasa_path
        self.litsearch_query_column = litsearch_query_column
        self.pasa_query_column = pasa_query_column
        self.human_flag_column = human_flag_column
        self.human_flag_value = human_flag_value
        self.pasa_human_flag_value = pasa_human_flag_value
        self.min_tokens = min_tokens
        self.seed = seed
        self.config_path = Path(config_path)
        self.llm_batch_size = llm_batch_size
        self.llm_judge_mode = llm_judge_mode
        self.llm_max_concurrency = llm_max_concurrency
        self.llm_seed = llm_seed
        self.use_llm_judge = use_llm_judge
        self.llm_config = load_llm_config(self.config_path) if use_llm_judge else None
        if self.llm_config and self.llm_seed is None:
            self.llm_seed = self.llm_config.pop("seed", None)
        elif self.llm_config:
            self.llm_config.pop("seed", None)

        np.random.seed(seed)

        self.litsearch_loader: Optional[LitSearchLoader] = None
        self.pasa_loader: Optional[PasaLoader] = None
        self.litsearch_analysis: Optional[DatasetAnalysis] = None
        self.pasa_analysis: Optional[DatasetAnalysis] = None
        self.combined_analysis: Optional[DatasetAnalysis] = None

    def load_datasets(self) -> Dict[str, List[str]]:
        """Load and filter datasets, return list of queries."""
        datasets = {}

        if self.litsearch_path:
            logger.info(f"Loading LitSearch dataset from: {self.litsearch_path}")
            self.litsearch_loader = LitSearchLoader(
                query_column=self.litsearch_query_column,
                human_flag_column=self.human_flag_column,
                human_flag_value=self.human_flag_value,
            )
            df = self.litsearch_loader.load(self.litsearch_path)
            df_human = self.litsearch_loader.filter_human(df)
            df_filtered = self._filter_by_tokens(df_human)
            queries = df_filtered["query"].tolist()
            datasets["litsearch"] = queries
            logger.info(f"LitSearch: {len(queries)} queries loaded")

        if self.pasa_path:
            logger.info(f"Loading PASA dataset from: {self.pasa_path}")
            self.pasa_loader = PasaLoader(
                query_column=self.pasa_query_column,
                human_flag_column=self.human_flag_column,
                human_flag_value=self.pasa_human_flag_value,
            )
            df = self.pasa_loader.load(self.pasa_path)
            df_human = self.pasa_loader.filter_human(df)
            df_filtered = self._filter_by_tokens(df_human)
            queries = df_filtered["query"].tolist()
            datasets["pasa"] = queries
            logger.info(f"PASA: {len(queries)} queries loaded")

        return datasets

    def _filter_by_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter queries by minimum token count."""
        if "query" not in df.columns:
            return df

        df = df.copy()
        df["token_count"] = df["query"].apply(lambda x: len(str(x).split()))
        df_filtered = df[df["token_count"] >= self.min_tokens].copy()
        df_filtered = df_filtered.drop(columns=["token_count"])

        return df_filtered.reset_index(drop=True)

    def analyze_dataset(self, name: str, queries: List[str]) -> DatasetAnalysis:
        """Analyze a single dataset."""
        logger.info(f"Analyzing {len(queries)} queries from {name}...")

        metrics = compute_all_metrics(queries)
        if self.use_llm_judge and queries:
            metrics.update(
                compute_llm_judged_metrics(
                    queries=queries,
                    batch_size=self.llm_batch_size,
                    judge_mode=self.llm_judge_mode,
                    max_concurrency=self.llm_max_concurrency,
                    seed=self.llm_seed,
                    **self.llm_config,
                )
            )
            metrics.update(
                compute_llm_semantic_constraint_metrics(
                    queries=queries,
                    batch_size=self.llm_batch_size,
                    judge_mode=self.llm_judge_mode,
                    max_concurrency=self.llm_max_concurrency,
                    seed=self.llm_seed,
                    **self.llm_config,
                )
            )

        examples = extract_representative_examples(queries, n=20, diversity=True)

        analysis = DatasetAnalysis(
            name=name,
            total_queries=len(queries),
            metrics=metrics,
            representative_examples=examples,
            queries_list=queries,
        )

        return analysis

    def analyze(self) -> Dict[str, Any]:
        """Run full analysis on all loaded datasets."""
        datasets = self.load_datasets()

        if "litsearch" in datasets:
            self.litsearch_analysis = self.analyze_dataset("litsearch", datasets["litsearch"])

        if "pasa" in datasets:
            self.pasa_analysis = self.analyze_dataset("pasa", datasets["pasa"])

        if self.litsearch_analysis and self.pasa_analysis:
            logger.info("Computing combined analysis...")
            all_queries = self.litsearch_analysis.queries_list + self.pasa_analysis.queries_list
            combined_metrics = compute_all_metrics(all_queries)
            if self.use_llm_judge:
                combined_per_query = (
                    self.litsearch_analysis.metrics.get("llm_judge", {}).get("per_query", [])
                    + self.pasa_analysis.metrics.get("llm_judge", {}).get("per_query", [])
                )
                combined_metrics.update(summarize_judge_results(combined_per_query))
                combined_semantic_per_query = (
                    self.litsearch_analysis.metrics.get("semantic_constraint_analysis", {}).get("per_query", [])
                    + self.pasa_analysis.metrics.get("semantic_constraint_analysis", {}).get("per_query", [])
                )
                if combined_semantic_per_query:
                    from query_analysis.llm_judge import summarize_semantic_constraints

                    combined_metrics.update(summarize_semantic_constraints(combined_semantic_per_query))
            self.combined_analysis = DatasetAnalysis(
                name="combined",
                total_queries=len(all_queries),
                metrics=combined_metrics,
                representative_examples=extract_representative_examples(all_queries, n=20, diversity=True),
                queries_list=all_queries,
            )
        elif self.litsearch_analysis:
            self.combined_analysis = self._copy_as_combined(self.litsearch_analysis)
        elif self.pasa_analysis:
            self.combined_analysis = self._copy_as_combined(self.pasa_analysis)
        else:
            logger.warning("No datasets loaded!")
            self.combined_analysis = DatasetAnalysis(name="combined", total_queries=0)

        return self._build_output()

    def _copy_as_combined(self, analysis: DatasetAnalysis) -> DatasetAnalysis:
        """Copy analysis as combined format."""
        return DatasetAnalysis(
            name="combined",
            total_queries=analysis.total_queries,
            metrics=analysis.metrics,
            representative_examples=analysis.representative_examples,
            queries_list=analysis.queries_list,
        )

    def _build_output(self) -> Dict[str, Any]:
        """Build final output dictionary."""
        output = {
            "dataset_overview": self._build_dataset_overview(),
            "filtering_rules": self._build_filtering_rules(),
            "metrics": {},
            "comparative_findings": self._build_comparative_findings(),
            "human_style_principles": self._build_human_style_principles(),
            "representative_examples": [],
        }

        if self.combined_analysis:
            output["metrics"]["combined"] = self._flatten_metrics(self.combined_analysis.metrics)
            output["representative_examples"] = self.combined_analysis.representative_examples

        if self.litsearch_analysis:
            output["metrics"]["litsearch"] = self._flatten_metrics(self.litsearch_analysis.metrics)

        if self.pasa_analysis:
            output["metrics"]["pasa"] = self._flatten_metrics(self.pasa_analysis.metrics)

        output["question_template_distribution"] = self._build_template_distribution()

        return output

    def _build_template_distribution(self) -> Dict[str, Any]:
        """Build template distribution with examples, filtering templates with < 5 queries."""
        result = {}

        for name, analysis in [
            ("litsearch", self.litsearch_analysis),
            ("pasa", self.pasa_analysis),
            ("combined", self.combined_analysis),
        ]:
            if not analysis:
                continue

            templates = analysis.metrics.get("question_templates", {})
            dist = templates.get("template_distribution", {})
            examples = templates.get("template_examples", {})
            total = analysis.total_queries

            filtered_templates = {}
            for template, count in sorted(dist.items(), key=lambda x: -x[1]):
                if count >= 4:
                    ratio = count / total if total > 0 else 0
                    example_list = examples.get(template, [])
                    filtered_templates[template] = {
                        "count": count,
                        "ratio": round(ratio, 4),
                        "example": example_list[0] if example_list else None,
                    }

            result[name] = {
                "templates": filtered_templates,
                "total_queries": total,
                "templates_with_5plus": len(filtered_templates),
            }

        return result

    def _flatten_metrics(self, metrics: Dict) -> Dict:
        """Flatten metrics dict for JSON output."""
        return {
            "length_stats": metrics.get("length_stats", {}),
            "constraint_count": metrics.get("constraint_count", {}),
            "question_templates": metrics.get("question_templates", {}),
            "qualitative_metrics": metrics.get("qualitative_metrics", {}),
            "semantic_constraint_analysis": metrics.get("semantic_constraint_analysis", {}),
        }

    def _build_dataset_overview(self) -> Dict[str, Any]:
        """Build dataset overview section."""
        overview = {}

        if self.litsearch_analysis:
            overview["litsearch"] = {
                "total_queries": self.litsearch_analysis.total_queries,
                "filtering_applied": f"query_set starts with 'manual', min_tokens={self.min_tokens}",
            }

        if self.pasa_analysis:
            overview["pasa"] = {
                "total_queries": self.pasa_analysis.total_queries,
                "filtering_applied": f"all human queries, min_tokens={self.min_tokens}",
            }

        if self.combined_analysis:
            overview["combined"] = {
                "total_queries": self.combined_analysis.total_queries,
            }

        return overview

    def _build_filtering_rules(self) -> Dict[str, Any]:
        """Build filtering rules section."""
        return {
            "litsearch": {
                "human_flag_column": self.human_flag_column,
                "human_flag_value_prefix": self.human_flag_value,
                "min_token_count": self.min_tokens,
            },
            "pasa": {
                "human_flag_column": "none (all human)",
                "human_flag_value": "all queries are human",
                "min_token_count": self.min_tokens,
            },
        }

    def _build_comparative_findings(self) -> Dict[str, Any]:
        """Build comparative findings section."""
        findings = {"shared_traits": [], "dataset_specific": {}}

        if not self.litsearch_analysis or not self.pasa_analysis:
            return findings

        ls_metrics = self.litsearch_analysis.metrics
        ps_metrics = self.pasa_analysis.metrics

        ls_templates = ls_metrics.get("question_templates", {}).get("template_distribution", {})
        ps_templates = ps_metrics.get("question_templates", {}).get("template_distribution", {})

        ls_avg_len = ls_metrics.get("length_stats", {}).get("token_length", {}).get("mean", 0)
        ps_avg_len = ps_metrics.get("length_stats", {}).get("token_length", {}).get("mean", 0)

        ls_constraint = ls_metrics.get("constraint_count", {}).get("constraints_per_query", {}).get("mean", 0)
        ps_constraint = ps_metrics.get("constraint_count", {}).get("constraints_per_query", {}).get("mean", 0)

        ls_qual = ls_metrics.get("qualitative_metrics", {})
        ps_qual = ps_metrics.get("qualitative_metrics", {})
        ls_spec_raw = ls_qual.get("specificity_calibration", {}).get("mean")
        ps_spec_raw = ps_qual.get("specificity_calibration", {}).get("mean")
        ls_spec_fit = ls_qual.get("specificity_calibration_fit", {}).get("mean")
        ps_spec_fit = ps_qual.get("specificity_calibration_fit", {}).get("mean")
        ls_lex_raw = ls_qual.get("lexical_naturalism", {}).get("mean")
        ps_lex_raw = ps_qual.get("lexical_naturalism", {}).get("mean")
        ls_lex_fit = ls_qual.get("lexical_naturalism_fit", {}).get("mean")
        ps_lex_fit = ps_qual.get("lexical_naturalism_fit", {}).get("mean")

        findings["shared_traits"] = [
            f"Both datasets have similar avg query length (~{ls_avg_len:.0f} vs ~{ps_avg_len:.0f} tokens)",
        ]
        if ls_metrics.get("constraint_count") and ps_metrics.get("constraint_count"):
            findings["shared_traits"].append(
                f"Both show comparable semantic constraint density (~{ls_constraint:.1f} vs ~{ps_constraint:.1f} constraints/query)"
            )
        if ls_spec_raw is not None and ps_spec_raw is not None:
            findings["shared_traits"].append(
                f"Both are close to the ideal specificity calibration of 3 (~{ls_spec_raw:.2f} vs ~{ps_spec_raw:.2f})"
            )
        if ls_lex_fit is not None and ps_lex_fit is not None:
            findings["shared_traits"].append(
                f"Lexical naturalism fit is strong in both (~{ls_lex_fit:.2f} vs ~{ps_lex_fit:.2f})"
            )

        findings["dataset_specific"]["litsearch"] = {
            "total_queries": self.litsearch_analysis.total_queries,
            "avg_token_length": round(ls_avg_len, 1),
            "avg_constraints": round(ls_constraint, 2),
            "top_templates": dict(sorted(ls_templates.items(), key=lambda x: -x[1])[:5]),
        }
        if ls_spec_raw is not None:
            findings["dataset_specific"]["litsearch"]["specificity_calibration_mean"] = round(ls_spec_raw, 3)
        if ls_spec_fit is not None:
            findings["dataset_specific"]["litsearch"]["specificity_calibration_fit_mean"] = round(ls_spec_fit, 3)
        if ls_lex_raw is not None:
            findings["dataset_specific"]["litsearch"]["lexical_naturalism_mean"] = round(ls_lex_raw, 3)
        if ls_lex_fit is not None:
            findings["dataset_specific"]["litsearch"]["lexical_naturalism_fit_mean"] = round(ls_lex_fit, 3)

        findings["dataset_specific"]["pasa"] = {
            "total_queries": self.pasa_analysis.total_queries,
            "avg_token_length": round(ps_avg_len, 1),
            "avg_constraints": round(ps_constraint, 2),
            "top_templates": dict(sorted(ps_templates.items(), key=lambda x: -x[1])[:5]),
        }
        if ps_spec_raw is not None:
            findings["dataset_specific"]["pasa"]["specificity_calibration_mean"] = round(ps_spec_raw, 3)
        if ps_spec_fit is not None:
            findings["dataset_specific"]["pasa"]["specificity_calibration_fit_mean"] = round(ps_spec_fit, 3)
        if ps_lex_raw is not None:
            findings["dataset_specific"]["pasa"]["lexical_naturalism_mean"] = round(ps_lex_raw, 3)
        if ps_lex_fit is not None:
            findings["dataset_specific"]["pasa"]["lexical_naturalism_fit_mean"] = round(ps_lex_fit, 3)

        return findings

    def _build_human_style_principles(self) -> Dict[str, Any]:
        """Build human style principles section."""
        principles = {
            "template_usage": {
                "principle": "Use common question templates",
                "description": "Human queries follow recognizable patterns like 'Is there a paper that...', 'Give me papers on...', 'How to...'",
                "guidance": [
                    "Prefer templates like 'Are there any papers on X?', 'Give me papers that X', 'How to achieve Y?'",
                    "Avoid overly formal or mechanical phrasing",
                    "Match the natural patterns researchers actually use",
                ],
            },
            "constraint_expression": {
                "principle": "Express constraints naturally",
                "description": "Constraints (methods, settings, datasets) are expressed through prepositional phrases",
                "guidance": [
                    "Include retrieval-narrowing constraints such as task, method, dataset, or comparison when they matter",
                    "Multiple semantic constraints are common and acceptable",
                    "Constraints should help retrieval without turning the query into a paper reconstruction",
                ],
            },
            "length_and_specificity": {
                "principle": "Balance length with specificity",
                "description": "Queries should be long enough to be specific but not overloaded",
                "guidance": [
                    "Optimal length is around 10-20 tokens",
                    "Include enough detail to retrieve relevant papers",
                    "Avoid very short queries (under 5 tokens) unless very specific",
                ],
            },
            "lexical_naturalism": {
                "principle": "Use natural researcher register",
                "description": "Queries should sound like something a real researcher would type, not keyword dumps and not polished prose",
                "guidance": [
                    "Avoid keyword-dump fragments and SEO-style noun lists",
                    "Avoid polished abstract-like prose and polite instruction wording",
                    "Prefer direct, query-like phrasing with natural technical vocabulary",
                ],
            },
        }

        return principles


def build_markdown_report(analysis: Dict[str, Any]) -> str:
    """Build markdown report from analysis results."""
    metrics_by_dataset = analysis.get("metrics", {})
    has_qualitative_metrics = any(
        metrics_by_dataset.get(ds_name, {}).get("qualitative_metrics", {}).get("specificity_calibration")
        or metrics_by_dataset.get(ds_name, {}).get("qualitative_metrics", {}).get("lexical_naturalism")
        for ds_name in ["combined", "litsearch", "pasa"]
    )

    lines = [
        "# Query Style Analysis Report",
        "",
        "## Summary",
        "",
        "This analysis examines human-written queries from LitSearch and PASA datasets",
        "to identify characteristics of natural, human-authored academic search queries.",
        "",
        "## Methodology",
        "",
        "1. Load queries from each dataset (LitSearch + PASA)",
        "2. Compute structural metrics (length, question templates)",
        "3. Identify question templates",
        "4. Optionally run LLM judge metrics (specificity calibration, lexical naturalism, semantic constraints)",
        "5. Compare across datasets to identify shared traits",
        "6. Derive rewrite principles for converting bullet points to human-like queries",
        "",
    ]

    overview = analysis.get("dataset_overview", {})
    lines.append("## Dataset Overview")
    lines.append("")
    for dataset, info in overview.items():
        lines.append(f"### {dataset.capitalize()}")
        for key, value in info.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    lines.append("## Quantitative Metrics")
    lines.append("")

    for ds_name in ["combined", "litsearch", "pasa"]:
        metrics = analysis.get("metrics", {}).get(ds_name, {})
        if not metrics:
            continue

        lines.append(f"### {ds_name.capitalize()}")

        length = metrics.get("length_stats", {}).get("token_length", {})
        lines.append("**Length**:")
        lines.append(f"- Mean tokens: {length.get('mean', 0):.1f}")
        lines.append(f"- Median tokens: {length.get('median', 0):.1f}")
        lines.append(f"- Range: {length.get('min', 0)} - {length.get('max', 0)}")
        lines.append(f"- Std: {length.get('std', 0):.1f}")
        lines.append("")

        constraints = metrics.get("constraint_count", {})
        cpq = constraints.get("constraints_per_query", {})
        if constraints and cpq:
            lines.append("**Semantic Constraint Count (LLM Judge)**:")
            lines.append(f"- Avg constraints/query: {cpq.get('mean', 0):.2f}")
            lines.append("")

    lines.append("## Question Templates (>= 5 queries)")
    template_dist = analysis.get("question_template_distribution", {}).get("combined", {}).get("templates", {})
    for t, info in sorted(template_dist.items(), key=lambda x: -x[1].get("count", 0)):
        example = info.get("example", "")
        example_short = example[:80] + "..." if len(example) > 80 else example
        lines.append(f"- **{t}** ({info['count']} queries, {info['ratio']:.1%})")
        lines.append(f"  - Example: \"{example_short}\"")
    lines.append("")

    if has_qualitative_metrics:
        lines.append("## Qualitative Metrics")
        lines.append("")
        for ds_name in ["combined", "litsearch", "pasa"]:
            metrics = analysis.get("metrics", {}).get(ds_name, {})
            if not metrics:
                continue
            qual = metrics.get("qualitative_metrics", {})
            if not qual:
                continue
            spec_raw = qual.get("specificity_calibration", {}).get("mean")
            spec_fit = qual.get("specificity_calibration_fit", {}).get("mean")
            lex_raw = qual.get("lexical_naturalism", {}).get("mean")
            lex_fit = qual.get("lexical_naturalism_fit", {}).get("mean")
            if all(value is None for value in [spec_raw, spec_fit, lex_raw, lex_fit]):
                continue
            lines.append(f"### {ds_name.capitalize()}")
            if spec_raw is not None:
                lines.append(
                    f"- **Specificity Calibration**: {spec_raw:.3f} "
                    "(1-5, 3 is ideal)"
                )
            if spec_fit is not None:
                lines.append(
                    f"- **Specificity Calibration Fit**: {spec_fit:.3f} "
                    "(0-1, higher = closer to ideal)"
                )
            if lex_raw is not None:
                lines.append(
                    f"- **Lexical Naturalism**: {lex_raw:.3f} "
                    "(1-5, 3 is ideal)"
                )
            if lex_fit is not None:
                lines.append(
                    f"- **Lexical Naturalism Fit**: {lex_fit:.3f} "
                    "(0-1, higher = closer to ideal)"
                )
            lines.append("")

    comparative = analysis.get("comparative_findings", {})
    if comparative:
        lines.append("## Comparative Findings")
        lines.append("")
        lines.append("### Shared Traits")
        for trait in comparative.get("shared_traits", []):
            lines.append(f"- {trait}")
        lines.append("")
        lines.append("### Dataset-Specific")
        for ds, patterns in comparative.get("dataset_specific", {}).items():
            lines.append(f"**{ds}**:")
            for key, value in patterns.items():
                if isinstance(value, dict):
                    lines.append(f"  - {key}: {value}")
                else:
                    lines.append(f"  - {key}: {value}")
        lines.append("")

    examples = analysis.get("representative_examples", [])
    if examples:
        lines.append("## Representative Examples")
        lines.append("")
        for ex in examples[:15]:
            template = ex.get("template", "other")
            query = ex.get("query", "")
            lines.append(f"- **[{template}]** {query}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze query style from LitSearch and PASA datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--litsearch_path",
        type=str,
        default=None,
        help="Path to LitSearch dataset (CSV/JSON/JSONL/parquet) or HuggingFace ID",
    )
    parser.add_argument(
        "--pasa_path",
        type=str,
        default=None,
        help="Path to PASA/RealScholarQuery dataset (JSONL)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/query_analysis",
        help="Output directory for analysis files",
    )
    parser.add_argument(
        "--query_column",
        type=str,
        default="query",
        help="Column name for query text (used for LitSearch)",
    )
    parser.add_argument(
        "--pasa_query_column",
        type=str,
        default="question",
        help="Column name for query text in PASA/RealScholarQuery dataset",
    )
    parser.add_argument(
        "--human_flag_column",
        type=str,
        default="query_set",
        help="Column name for human-written flag",
    )
    parser.add_argument(
        "--human_flag_value",
        type=str,
        default="manual",
        help="Prefix for human-written query sets in LitSearch",
    )
    parser.add_argument(
        "--pasa_human_value",
        type=str,
        default="true",
        help="Value indicating human-written in PASA dataset",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=2,
        help="Minimum token count for queries",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml with LLM judge settings",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=25,
        help="Batch size for LLM-as-a-judge scoring",
    )
    parser.add_argument(
        "--llm_judge_mode",
        type=str,
        choices=["batch", "single_query"],
        default="batch",
        help="Whether to score in multi-query batches or one query per prompt",
    )
    parser.add_argument(
        "--llm_max_concurrency",
        type=int,
        default=1,
        help="Maximum concurrent LLM requests for judge scoring",
    )
    parser.add_argument(
        "--llm_seed",
        type=int,
        default=None,
        help="Optional seed forwarded to compatible OpenAI-style backends",
    )
    parser.add_argument(
        "--no_llm_judge",
        action="store_true",
        help="Disable LLM-as-a-judge metrics and keep only local structural metrics",
    )
    parser.add_argument(
        "--plot_length_distribution",
        action="store_true",
        help="Generate a PNG visualization and JSON summary of query word-count distributions",
    )

    args = parser.parse_args()

    if not args.litsearch_path and not args.pasa_path:
        parser.error("At least one of --litsearch_path or --pasa_path is required")

    pasa_human_value: Any = args.pasa_human_value
    if pasa_human_value.lower() == "true":
        pasa_human_value = True
    elif pasa_human_value.lower() == "false":
        pasa_human_value = False

    analyzer = QueryStyleAnalyzer(
        litsearch_path=args.litsearch_path,
        pasa_path=args.pasa_path,
        litsearch_query_column=args.query_column,
        pasa_query_column=args.pasa_query_column,
        human_flag_column=args.human_flag_column,
        human_flag_value=args.human_flag_value,
        pasa_human_flag_value=pasa_human_value,
        min_tokens=args.min_tokens,
        seed=args.seed,
        config_path=args.config,
        llm_batch_size=args.llm_batch_size,
        llm_judge_mode=args.llm_judge_mode,
        llm_max_concurrency=args.llm_max_concurrency,
        llm_seed=args.llm_seed,
        use_llm_judge=not args.no_llm_judge,
    )

    logger.info("Starting query style analysis...")
    analysis = analyzer.analyze()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "style_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON analysis to: {json_path}")

    md_content = build_markdown_report(analysis)
    md_path = output_dir / "style_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved MD report to: {md_path}")

    if args.plot_length_distribution:
        query_sets = {}
        if analyzer.litsearch_analysis and analyzer.litsearch_analysis.queries_list:
            query_sets["litsearch"] = analyzer.litsearch_analysis.queries_list
        if analyzer.pasa_analysis and analyzer.pasa_analysis.queries_list:
            query_sets["pasa"] = analyzer.pasa_analysis.queries_list
        if analyzer.combined_analysis and analyzer.combined_analysis.queries_list:
            query_sets["combined"] = analyzer.combined_analysis.queries_list

        png_path = output_dir / "query_length_distribution.png"
        summary_path = output_dir / "query_length_distribution_summary.json"
        summary = save_length_distribution_plot(query_sets, png_path)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved query length plot to: {png_path}")
        logger.info(f"Saved query length summary to: {summary_path}")

    logger.info("Analysis complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
