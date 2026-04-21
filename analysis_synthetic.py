import json
from pathlib import Path
from query_analysis.features import compute_all_metrics, detect_question_template
from collections import Counter

# Load pipeline output
with open("data/iclr_2026/pipeline_output_single.json") as f:
    data = json.load(f)

# Extract all synthetic queries
queries = []
for paper in data.get("papers", []):
    for q in paper.get("queries", []):
        qt = q.get("query_text", "")
        if qt:
            queries.append(qt)

print(f"Total synthetic queries: {len(queries)}")

# Compute metrics
metrics = compute_all_metrics(queries)

# Build template distribution summary
template_dist = Counter()
template_examples = {}
for q in queries:
    t, _ = detect_question_template(q)
    t = t or "other"
    template_dist[t] += 1
    if t not in template_examples:
        template_examples[t] = []
    if len(template_examples[t]) < 3:
        template_examples[t].append(q)

# Get top templates
total = len(queries)
result = {
    "dataset": "pipeline_output_single",
    "total_queries": total,
    "metrics": {
        "combined": {
            **metrics,
            "question_templates": {
                "template_distribution": dict(template_dist),
                "template_ratios": {t: c/total for t, c in template_dist.items()},
                "template_examples": template_examples,
                "total_templates": len(template_dist)
            }
        }
    }
}

# Save
output_path = Path("outputs/query_analysis/synthetic_single_analysis.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"Saved to {output_path}")