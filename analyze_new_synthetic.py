import sys
sys.path.insert(0, 'src')

import json
from pathlib import Path
from query_analysis.features import compute_all_metrics, detect_question_template
from collections import Counter

# Load the new generated queries
with open("data/iclr_2026_single/03_queries_single_new.json") as f:
    data = json.load(f)

# Extract all queries
queries = []
for paper_queries in data.get("papers_queries", []):
    for q in paper_queries.get("queries_by_view", []):
        qt = q.get("query_text", "")
        if qt:
            queries.append(qt)

print(f"Total new synthetic queries: {len(queries)}")

# Compute metrics
metrics = compute_all_metrics(queries)

# Build template distribution
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

total = len(queries)
result = {
    "dataset": "new_synthetic_single",
    "total_queries": total,
    "query_examples": queries,
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
output_path = Path("outputs/query_analysis/new_synthetic_single_analysis.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"Saved to {output_path}")

# Print summary
print("\n=== Template Distribution ===")
for t, c in template_dist.most_common():
    print(f"  {t}: {c} ({c/total*100:.1f}%)")