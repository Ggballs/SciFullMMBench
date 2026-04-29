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

### Pasa
- **total_queries**: 50
- **filtering_applied**: all human queries, min_tokens=2

### Combined
- **total_queries**: 50

## Quantitative Metrics

### Combined
**Length**:
- Mean tokens: 15.6
- Median tokens: 14.0
- Range: 6 - 38
- Std: 7.1

**Semantic Constraint Count (LLM Judge)**:
- Avg constraints/query: 2.00

### Pasa
**Length**:
- Mean tokens: 15.6
- Median tokens: 14.0
- Range: 6 - 38
- Std: 7.1

**Semantic Constraint Count (LLM Judge)**:
- Avg constraints/query: 2.00

## Question Templates (>= 5 queries)
- **Give me papers [that/on/about]...** (34 queries, 68.0%)
  - Example: "Give me papers which show that using a smaller dataset in large language model p..."

## Qualitative Metrics

### Combined
- **Specificity Calibration**: 2.660 (1-5, 3 is ideal)
- **Specificity Calibration Fit**: 0.710 (0-1, higher = closer to ideal)
- **Lexical Naturalism**: 2.740 (1-5, 3 is ideal)
- **Lexical Naturalism Fit**: 0.630 (0-1, higher = closer to ideal)

### Pasa
- **Specificity Calibration**: 2.660 (1-5, 3 is ideal)
- **Specificity Calibration Fit**: 0.710 (0-1, higher = closer to ideal)
- **Lexical Naturalism**: 2.740 (1-5, 3 is ideal)
- **Lexical Naturalism Fit**: 0.630 (0-1, higher = closer to ideal)

## Comparative Findings

### Shared Traits

### Dataset-Specific

## Representative Examples

- **[Give me papers [that/on/about]...]** Give me papers which show that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets.
- **[I am looking for...]** I am looking for research papers on the construction of multimodal foundation models that support both visual and audio inputs. These models should be pre-trained on large-scale datasets, including visual, audio, and audio-visual data. Please exclude survey papers.
- **[Do [X/they]...]** Do you know some papers about using reward shaping methods to train large language model agent.
- **[Is there a paper/method that/which...]** Is there any work that analyzes the scaling law of the multi-module models, such as video-text, image-text models?
- **[What [open-source/latest/first] method...]** What papers discuss the use of transformer architecture in 3d video generation
- **[Can [we/X]...]** Can LLMs detect LLM-generated text in a zero-shot manner? Do they perform better than supervised fine-tuned small classification models? Provide related papers.
- **[other]** Video aesthetics score, using multimodal large models.
- **[I would like to...]** I would like to find some research papers about test time training topic, in LLM research area.
- **[Help me search/find...]** Help me search for the work related to the synthetic data of large language models. I want to know how to automatically generate large-scale, high-quality, diverse, difficult, and valuable long thought data for learning.
- **[Could [we/X]...]** Could you list research that demonstrates the advantages of Quantization-Aware Training (QAT), which can enable the model to learn better representations for low-bit weights?.
- **[Using [X for Y]...]** Using synthesis data for scaling up sft data.
- **[How can [we/X]...]** How can LLM agents be evaluated and benchmarked for financial tasks? Note that I am referring to agents.
