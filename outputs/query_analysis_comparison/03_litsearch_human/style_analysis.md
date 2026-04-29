# Query Style Analysis Report

## Summary

This analysis examines human-written queries from LitSearch and PASA datasets
to identify characteristics of natural, human-authored academic search queries.

## Methodology

1. Load queries from each dataset (LitSearch + PASA)
2. Compute quantitative metrics (length, LLM semantic constraint counts)
3. Identify question templates
4. Evaluate judge-based style metrics (specificity calibration, lexical naturalism)
5. Compare across datasets to identify shared traits
6. Derive rewrite principles for converting bullet points to human-like queries

## Dataset Overview

### Litsearch
- **total_queries**: 111
- **filtering_applied**: query_set starts with 'manual', min_tokens=2

### Combined
- **total_queries**: 111

## Quantitative Metrics

### Combined
**Length**:
- Mean tokens: 17.5
- Median tokens: 17.0
- Range: 6 - 33
- Std: 5.1

**Semantic Constraint Count (LLM Judge)**:
- Avg constraints/query: 2.19

### Litsearch
**Length**:
- Mean tokens: 17.5
- Median tokens: 17.0
- Range: 6 - 33
- Std: 5.1

**Semantic Constraint Count (LLM Judge)**:
- Avg constraints/query: 2.19

## Question Templates (>= 5 queries)
- **What [open-source/latest/first] method...** (63 queries, 56.8%)
  - Example: "What are some data-efficient ways to learn text embeddings thru contrastive lear..."
- **Is there a paper/method that/which...** (42 queries, 37.8%)
  - Example: "Are there any papers that build dense retrievers with mixture-of-experts archite..."

## Qualitative Metrics

### Combined
- **Specificity Calibration**: 3.153 (1-5, 3 is ideal)
- **Specificity Calibration Fit**: 0.707 (0-1, higher = closer to ideal)
- **Lexical Naturalism**: 2.703 (1-5, 3 is ideal)
- **Lexical Naturalism Fit**: 0.707 (0-1, higher = closer to ideal)

### Litsearch
- **Specificity Calibration**: 3.153 (1-5, 3 is ideal)
- **Specificity Calibration Fit**: 0.707 (0-1, higher = closer to ideal)
- **Lexical Naturalism**: 2.703 (1-5, 3 is ideal)
- **Lexical Naturalism Fit**: 0.707 (0-1, higher = closer to ideal)

## Comparative Findings

### Shared Traits

### Dataset-Specific

## Representative Examples

- **[Is there a paper/method that/which...]** Are there any papers that build dense retrievers with mixture-of-experts architecture where each expert is responsible for different types of queries?
- **[Is there a paper/method that/which...]** Is there a decoder-only language model that does not use a tokenizer and operates on raw text bytes?
- **[Is there a paper/method that/which...]** Is there a method for measuring the critical errors that a dialogue system makes in its responses?
- **[Is there a paper/method that/which...]** Is there a method that measures the information provided in a (model generated) rationale beyond what the original context provided?
- **[How to [achieve/do]...]** How to achieve zero-shot lip reading?
- **[How to [achieve/do]...]** How to better attract readers to news articles by generating personalized headlines?
- **[If one would like/wants...]** If one would like to train (or evaluate) a helpful assistant agent that can converse with humans while the humans traverse an environment, which work has the most suitable resource?
- **[What [open-source/latest/first] method...]** What are some data-efficient ways to learn text embeddings thru contrastive learning?
- **[What [open-source/latest/first] method...]** What are some methods for solving the class-incremetal continual learning problems?
- **[What [open-source/latest/first] method...]** What limitations do large language models have in evaluating information-seeking question answering?
- **[What [open-source/latest/first] method...]** What paper compares humans' and language models' non-literal interpretations of utterances featuring phenomena like deceit, irony, and humor?
- **[Can [we/X]...]** Can we find the solution of the Bilevel optimization when the lower-level problem is nonconvex?
- **[Can [we/X]...]** Can you find a dataset that shows LLM-based evaluation may not be reliable enough?
- **[Can [we/X]...]** Can you find a research paper that discusses using structured pruning techniques to scale down language models, where the original model being pruned has billions of parameters?
