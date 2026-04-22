#!/usr/bin/env python3
"""
Rewrite Prompt Builder

Read style_analysis.json and generate an operational rewrite instruction prompt
for converting OpenReview bullet points into human-like queries.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


class RewritePromptBuilder:
    """Build rewrite prompts from style analysis."""

    def __init__(self, analysis_path: str):
        self.analysis_path = Path(analysis_path)
        self.analysis: Dict[str, Any] = {}

    def load_analysis(self) -> Dict[str, Any]:
        """Load analysis JSON file."""
        with open(self.analysis_path, "r", encoding="utf-8") as f:
            self.analysis = json.load(f)
        return self.analysis

    def _get_combined_metrics(self) -> Dict[str, Any]:
        """Get combined metrics."""
        return self.analysis.get("metrics", {}).get("combined", {})

    def _generate_style_rules(self) -> List[str]:
        """Generate explicit style rules from analysis."""
        metrics = self._get_combined_metrics()
        rules = []

        length_stats = metrics.get("length_stats", {})
        token_stats = length_stats.get("token_length", {})
        mean_tokens = token_stats.get("mean", 15)
        median_tokens = token_stats.get("median", 14)

        rules.append(f"1. LENGTH: Keep queries concise (optimal ~{median_tokens:.0f}-{mean_tokens:.0f} tokens). Human queries are direct and to-the-point.")

        template_dist = self.analysis.get("question_template_distribution", {}).get("combined", {}).get("templates", {})

        if template_dist:
            rules.append("2. TEMPLATES: Use common question patterns (with examples from human queries):")
            for t, info in sorted(template_dist.items(), key=lambda x: -x[1].get("count", 0)):
                example = info.get("example", "")
                if example:
                    example_short = example[:60] + "..." if len(example) > 60 else example
                    rules.append(f"   - '{t}' ({info['count']} queries)")
                    rules.append(f"     Example: \"{example_short}\"")

        constraint_stats = metrics.get("constraint_count", {})
        constraint_mean = constraint_stats.get("constraints_per_query", {}).get("mean", 1.5)
        rules.append(f"3. CONSTRAINTS: Include ~{constraint_mean:.1f} semantic retrieval constraints per query when needed.")
        rules.append("   Favor meaningful narrowing conditions such as task, method, dataset, comparison, or scope.")

        qual = metrics.get("qualitative_metrics", {})
        spec = qual.get("specificity_calibration", {}).get("mean", 3.0)
        nat = qual.get("lexical_naturalism", {}).get("mean", 3.0)

        rules.append(f"4. QUALITY TARGETS: Specificity Calibration ~{spec:.2f} (ideal near 3), Lexical Naturalism ~{nat:.2f} (ideal near 3)")
        rules.append("   - Include enough detail to guide retrieval without reconstructing a known paper")
        rules.append("   - Sound like a direct researcher query, not a keyword dump and not polished prose")

        return rules

    def _generate_preservation_rules(self) -> List[str]:
        """Generate rules for what to preserve from bullet points."""
        return [
            "Preserve all technical terms, model names, dataset names, and acronyms (e.g., BERT, CLIP, RLHF)",
            "Preserve specific numbers, metrics, and quantitative claims",
            "Preserve the core research focus or contribution",
            "Maintain the scope/boundary described in the bullet",
            "Preserve method names, task names, and domain-specific terminology",
        ]

    def _generate_avoidance_rules(self) -> List[str]:
        """Generate rules for what to avoid."""
        return [
            "NO HALLUCINATION: Do not add information not present in the bullet point",
            "NO VERBOSITY: Don't expand short bullets into long explanations",
            "NO FORMULAIC OUTPUTS: Avoid 'In this paper, we propose...' or 'This paper investigates...'",
            "NO POLISHED PROSE: Don't create grammatically complete sentences if fragments work better",
            "NO UNNECESSARY ARTICLES: Drop 'a', 'an', 'the' when they don't affect meaning",
            "NO FILLER WORDS: Remove 'research on', 'study of', 'investigation of' unless specific",
            "NO MECHANICAL PHRASING: Don't sound like you're filling a template",
        ]

    def _generate_fragment_handling(self) -> List[str]:
        """Generate rules for handling fragment bullets."""
        return [
            "If bullet is a noun phrase fragment (e.g., 'novel attention mechanism'), convert to question form",
            "Common conversions: 'X for Y' → 'Are there any papers on X for Y?'",
            "If bullet starts with 'We propose/develop/present X', convert to 'Are there papers on X?'",
            "If bullet is a full sentence, extract the key technical claim and reshape as a question",
            "Keep question fragments ('How to achieve X?') as-is or expand naturally",
        ]

    def _generate_query_count_guidance(self) -> List[str]:
        """Generate guidance on when to produce one vs multiple queries."""
        return [
            "ONE query when: Bullet is focused on a single concept/method/dataset",
            "MULTIPLE queries when: Bullet contains multiple distinct claims separated by 'and'",
            "SPLIT when: A bullet covers both 'what' and 'for what purpose' - separate queries",
            "MERGE when: Multiple short bullets cover the same concept - combine concisely",
            "DEFAULT: When in doubt, prefer 1-2 focused queries over many fragmented ones",
        ]

    def _get_synthetic_examples(self) -> List[Dict[str, str]]:
        """Get synthetic examples of bullet-to-query rewrites."""
        return [
            {
                "input": "We propose a novel attention mechanism called Multi-Head Attention",
                "output": "Are there any papers on Multi-Head Attention mechanism?",
                "reasoning": "Convert proposal to question form with 'Are there any papers on'",
            },
            {
                "input": "The method achieves state-of-the-art performance on ImageNet",
                "output": "What is the state-of-the-art method on ImageNet?",
                "reasoning": "Convert claim to 'What is X on Y?' question form",
            },
            {
                "input": "How does contrastive learning compare with supervised learning?",
                "output": "Are there any papers comparing contrastive learning with supervised learning?",
                "reasoning": "Expand question fragment to full 'Are there papers on X vs Y?'",
            },
            {
                "input": "We present a new benchmark for evaluating multimodal models",
                "output": "Is there a benchmark for evaluating multimodal models?",
                "reasoning": "Convert proposal to 'Is there X?' question form",
            },
            {
                "input": "Training data efficiency through data augmentation techniques",
                "output": "Are there any papers on data augmentation for training efficiency?",
                "reasoning": "Convert fragment to 'Are there papers on X for Y?'",
            },
            {
                "input": "Robustness to adversarial attacks in vision transformers",
                "output": "Are there any papers on adversarial attacks in vision transformers?",
                "reasoning": "Convert noun phrase to question form",
            },
            {
                "input": "This paper investigates the effect of batch size on neural network training",
                "output": "How does batch size affect neural network training?",
                "reasoning": "Convert 'investigates X' to 'How does X affect Y?'",
            },
            {
                "input": "A unified framework for both image and text understanding",
                "output": "Is there a unified framework for image and text understanding?",
                "reasoning": "Convert noun phrase to 'Is there X?' question form",
            },
        ]

    def _format_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format examples for prompt."""
        lines = []
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"  Input: {ex['input']}")
            lines.append(f"  Output: {ex['output']}")
            lines.append(f"  Reasoning: {ex['reasoning']}")
            lines.append("")
        return "\n".join(lines)

    def build_prompt(self) -> str:
        """Build the complete rewrite instruction prompt."""
        prompt_parts = []

        prompt_parts.append("# Rewrite Instruction Prompt: OpenReview Bullet Points to Human-like Queries")
        prompt_parts.append("")
        prompt_parts.append("## Role and Task Definition")
        prompt_parts.append("")
        prompt_parts.append("You are an expert at rewriting academic paper bullet points into realistic, natural human search queries for information retrieval systems.")
        prompt_parts.append("")
        prompt_parts.append("**Task**: Convert given bullet points from OpenReview paper summaries into queries that a researcher would naturally type into a search engine or academic paper search system.")
        prompt_parts.append("")
        prompt_parts.append("## Core Objective")
        prompt_parts.append("")
        prompt_parts.append("Transform bullet points from this format:")
        prompt_parts.append("  'We propose a novel method for X that achieves Y'")
        prompt_parts.append("")
        prompt_parts.append("Into human search queries like:")
        prompt_parts.append("  'Are there any papers on X for Y?' or 'What is the method for X?'")
        prompt_parts.append("")
        prompt_parts.append("## Explicit Style Rules (Derived from Human Query Analysis)")
        prompt_parts.append("")
        for rule in self._generate_style_rules():
            prompt_parts.append(f"- {rule}")
        prompt_parts.append("")
        prompt_parts.append("## What to Preserve from Bullet Points")
        prompt_parts.append("")
        for rule in self._generate_preservation_rules():
            prompt_parts.append(f"- {rule}")
        prompt_parts.append("")
        prompt_parts.append("## What to Avoid")
        prompt_parts.append("")
        for rule in self._generate_avoidance_rules():
            prompt_parts.append(f"- {rule}")
        prompt_parts.append("")
        prompt_parts.append("## Handling Bullet Point Conversion")
        prompt_parts.append("")
        for rule in self._generate_fragment_handling():
            prompt_parts.append(f"- {rule}")
        prompt_parts.append("")
        prompt_parts.append("## When to Produce One vs Multiple Queries")
        prompt_parts.append("")
        prompt_parts.append("General guidance:")
        prompt_parts.append("")
        for guidance in self._generate_query_count_guidance():
            prompt_parts.append(f"- {guidance}")
        prompt_parts.append("")
        prompt_parts.append("## Output Format")
        prompt_parts.append("")
        prompt_parts.append("Return a JSON object with the following structure:")
        prompt_parts.append("```json")
        prompt_parts.append("{")
        prompt_parts.append('  "queries": [')
        prompt_parts.append('    {')
        prompt_parts.append('      "query": "<rewritten query text>",')
        prompt_parts.append('      "reasoning": "<brief explanation of rewrite choices>"')
        prompt_parts.append('    }')
        prompt_parts.append('  ]')
        prompt_parts.append("}")
        prompt_parts.append("```")
        prompt_parts.append("")
        prompt_parts.append("## Example Rewrites")
        prompt_parts.append("")
        examples = self._get_synthetic_examples()
        prompt_parts.append(self._format_examples(examples))
        prompt_parts.append("")
        prompt_parts.append("## Final Notes")
        prompt_parts.append("")
        prompt_parts.append("- The goal is retrieval recall: queries should help find relevant papers")
        prompt_parts.append("- Natural queries = queries that sound like what a researcher would actually type")
        prompt_parts.append("- When uncertain, prefer brevity and key terms over grammatical completeness")
        prompt_parts.append("- Technical accuracy and preserving meaning are more important than style")

        return "\n".join(prompt_parts)

    def save_prompt(self, output_path: str) -> None:
        """Save prompt to file."""
        prompt = self.build_prompt()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"Rewrite instruction prompt saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build rewrite instruction prompt from style analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--analysis_path",
        type=str,
        default="outputs/query_analysis/style_analysis.json",
        help="Path to style_analysis.json from analyze_query_style.py",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/query_analysis/rewrite_instruction_prompt.txt",
        help="Output path for rewrite instruction prompt",
    )

    args = parser.parse_args()

    if not Path(args.analysis_path).exists():
        print(f"Error: Analysis file not found: {args.analysis_path}")
        print("Please run analyze_query_style.py first to generate style_analysis.json")
        return 1

    builder = RewritePromptBuilder(args.analysis_path)
    builder.load_analysis()
    builder.save_prompt(args.output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
