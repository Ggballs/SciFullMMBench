# Human IR Query Intent Summary

Source files:

- `tests/test_data/query_analysis/litsearch-human-queries_quality2.jsonl`
- `tests/test_data/query_analysis/pasa-realscholarquery.jsonl`

This summary focuses on intent: the retrieval job behind the query, independent
from surface wording such as "Which paper..." or "Give me papers...".

## Dataset Snapshot

| Dataset | Queries | Dominant intent style |
| --- | ---: | --- |
| LitSearch human | 111 | Needle-like paper/work recovery from a remembered contribution |
| PASA real-scholar | 50 | Literature gathering and evidence seeking across a topic |
| Combined | 161 | Mixed: exact-work lookup plus topic-level literature search |

Approximate primary intent distribution, based on rule-assisted reading of the
query text. Some queries naturally express more than one intent, so these labels
are best used as generation guidance rather than gold annotations.

| Primary intent | LitSearch | PASA | Combined |
| --- | ---: | ---: | ---: |
| Known-paper or first-work lookup | 51, 45.9% | 1, 2.0% | 52, 32.3% |
| Existence check for a specific work or method | 33, 29.7% | 1, 2.0% | 34, 21.1% |
| Literature set or topic survey | 1, 0.9% | 28, 56.0% | 29, 18.0% |
| Claim or evidence seeking | 10, 9.0% | 8, 16.0% | 18, 11.2% |
| Method or solution discovery | 4, 3.6% | 8, 16.0% | 12, 7.5% |
| Resource, dataset, benchmark, or tool discovery | 9, 8.1% | 2, 4.0% | 11, 6.8% |
| Mechanism, theory, or evaluation question | 3, 2.7% | 2, 4.0% | 5, 3.1% |

## Intent Taxonomy

### Known-paper or first-work lookup

The user seems to remember one distinctive contribution and wants to recover the
paper, work, method, or model.

- Retrieval target: usually one paper, sometimes a tiny set.
- Common query shape: "Which paper first...", "What paper...", "Which work...".
- Use when a summary bullet contains a distinctive novelty, first use, named
  technical move, or unusual empirical finding.
- Generation guidance: keep one memorable contribution as the hook; do not
  reconstruct the whole abstract.
- Examples:
  - "Which paper first applied the chain-of-thought technique in the text summarization field?"
  - "What paper first extends rotary positional encoding (RoPE) for camera-geometry encoding in multi-view transformers?"

### Existence check for a specific work or method

The user does not necessarily know whether such a paper exists and asks for a
match to a constrained idea.

- Retrieval target: one or a few matching works.
- Common query shape: "Is there a paper/method/tool that...", "Are there any papers that...".
- Use when the source bullet describes a particular mechanism, capability, or
  setting, but not necessarily a first contribution.
- Generation guidance: phrase the query as a natural feasibility check; preserve
  the central method/task pair and at most one qualifier.
- Examples:
  - "Are there any papers that build dense retrievers with mixture-of-experts architecture where each expert is responsible for different types of queries?"
  - "Is there any paper that uses prompt tuning in multi-answer QA?"

### Literature set or topic survey

The user wants a set of papers around a topic, task, domain, or application.

- Retrieval target: many papers, often a short reading list.
- Common query shape: "Give me papers...", "Show me research on...", "List all papers...".
- Most common in PASA.
- Use when the source material points to a research area rather than one
  distinctive contribution.
- Generation guidance: let the query be broader, but include one technical
  anchor such as method, task, modality, or domain.
- Examples:
  - "List all papers that use autoregressive transformer to generate videos."
  - "Show me research on rejection sampling finetuning."

### Claim or evidence seeking

The user wants papers that support, refute, explain, or demonstrate a proposition.

- Retrieval target: papers usable as evidence for a claim.
- Common query shape: "papers which show...", "papers demonstrating...",
  "papers claiming...", "papers explaining why...".
- Use when the source bullet is a result, negative result, comparison, limitation,
  or surprising empirical observation.
- Generation guidance: encode the claim directly, but avoid turning the query into
  a fully specified theorem or benchmark requirement.
- Examples:
  - "Give me papers which show that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets."
  - "Provide papers demonstrating that the self-correction of LLMs does not enhance their performance."

### Method or solution discovery

The user asks how to solve a task, build a model, apply a training technique, or
address a practical problem.

- Retrieval target: methods, approaches, or applied systems.
- Common query shape: "How to...", "papers that apply...", "research papers on the construction of...".
- Use when the source bullet is mainly about a method or intervention.
- Generation guidance: make the task and mechanism visible; do not require the
  generated query to know the exact paper identity.
- Examples:
  - "How to achieve zero-shot lip reading?"
  - "Papers that apply RLHF to address the hallucination problem in image and video description."

### Resource, dataset, benchmark, or tool discovery

The user wants a reusable artifact: dataset, benchmark, tool, corpus, resource,
or evaluation setting.

- Retrieval target: artifact papers or resource descriptions.
- Common query shape: "Is there a dataset/tool...", "What open-source dataset...",
  "Show me code evaluation datasets...".
- Use when the source bullet describes a benchmark, evaluation resource, data
  construction process, or software/tool contribution.
- Generation guidance: name the resource function and target setting rather than
  all construction details.
- Examples:
  - "Is there a tool that can automatically segment speech and the corresponding text transcriptions, to obtain a finer grained alignment?"
  - "Show me code evaluation datasets with a mid-level hardness. It show be harder than HumanEval and MBPP, but easier than code_contests."

### Mechanism, theory, or evaluation question

The user wants an explanation, analysis, guarantee, limitation, or evaluation
framework rather than a plain paper list.

- Retrieval target: analysis papers, theory papers, evaluation studies, or benchmarks.
- Common query shape: "What limitations...", "Are there guarantees...",
  "How can X be evaluated...".
- Use when the source bullet is about why something happens, when a method fails,
  how to evaluate a capability, or what theory supports a behavior.
- Generation guidance: preserve the analytical question, but keep the query
  compact and search-like.
- Examples:
  - "What limitations do large language models have in evaluating information-seeking question answering?"
  - "How can LLM agents be evaluated and benchmarked for financial tasks? Note that I am referring to agents."

## Dataset-Specific Takeaways

LitSearch human queries are mostly tip-of-the-tongue recovery queries. The user
often knows a distinctive contribution, a "first" claim, or a rare method-task
combination and wants the matching paper. To imitate this dataset, generate many
single-target queries with one memorable hook:

- "Which paper first used X for Y?"
- "Is there a paper that does X under Y?"
- "What paper proposed X to handle Y?"

PASA real-scholar queries are more often active literature-search requests. The
user asks for sets of papers, related work around an application, evidence for a
claim, or papers using a broad method family. To imitate this dataset, generate
more multi-paper requests:

- "Give me papers about X."
- "Show me research on X for Y."
- "Provide papers demonstrating that X happens."

The combined human distribution should not be controlled by wording templates
alone. A better generation process is:

1. Choose the retrieval intent.
2. Choose target cardinality: one paper, a small set, or a broader set.
3. Select one core retrieval hook from the source bullet.
4. Add at most one natural qualifier.
5. Then choose a surface template that fits the intent.

## Suggested Intent Mix for Query Generation

For a combined human-like generator:

- 30-35% known-paper or first-work lookup
- 20-25% existence check for a specific work or method
- 15-20% literature set or topic survey
- 10-15% claim or evidence seeking
- 5-10% method or solution discovery
- 5-8% resource, dataset, benchmark, or tool discovery
- 3-5% mechanism, theory, or evaluation question

For LitSearch-like generation, increase known-paper lookup and existence checks.
For PASA-like generation, increase literature-set and claim/evidence queries.

## Prompt Block

Use this block when updating query-generation prompts:

```text
Before writing each query, silently choose one search intent:
1. known-paper or first-work lookup;
2. existence check for a specific work or method;
3. literature set or topic survey;
4. claim or evidence seeking;
5. method or solution discovery;
6. resource, dataset, benchmark, or tool discovery;
7. mechanism, theory, or evaluation question.

Generate the wording after choosing the intent. Do not imitate only the surface
template. The query should reveal what the researcher is trying to retrieve:
one remembered paper, a small set of matching works, evidence for a claim, a
dataset/benchmark/tool, a method, or an explanation/evaluation paper.

Use one core retrieval hook from the source bullet and at most one supporting
qualifier. Prefer compact, natural researcher phrasing over full abstract-style
requirements.
```
