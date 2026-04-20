import json
from pathlib import Path

FILES = {
    "GPT-3.5"  : Path("results/gpt35_results.json"),
    "LLaMA-2"  : Path("results/llama2_results.json"),
}

def load(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def fmt(val):
    return f"{val:5.1f}%" if val is not None else "  N/A "

print("\n── Model Comparison ────────────────────────────────────────────────────")
print(f"{'Dataset':<12} {'Mode':<6} {'Model':<10} {'EM':>7} {'F1':>7} "
      f"{'Hall%':>7} {'Overconf%':>10}")
print("─" * 70)

for model_name, fpath in FILES.items():
    data = load(fpath)
    if data is None:
        print(f"[{model_name}] No results file found at {fpath}")
        continue
    for row in data["results"]:
        tag = "RAG " if row["rag"] else "BASE"
        ds  = row["dataset"]
        print(f"{ds:<12} {tag:<6} {model_name:<10} "
              f"{fmt(row['exact_match'])} "
              f"{fmt(row['avg_f1'])} "
              f"{fmt(row['hallucination_pct'])} "
              f"{fmt(row['overconfident_pct']):>10}")
    print()

print("─" * 70)
print("Files read from: results/gpt35_results.json, results/llama2_results.json")
print("Open dashboard.html to visualize (serves results from results/summary.json)")
