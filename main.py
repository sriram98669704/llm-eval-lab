"""
main.py — Orchestrator / entry point for the LLM Eval Lab.

Runs the full benchmark pipeline end to end:
  1. Benchmark each model across all prompt categories (models.run_model)
  2. Report GSM8K accuracy (ground-truth math)
  3. Report HumanEval pass@1 (sandboxed code execution)
  4. Route each category to its best model
  5. Print the final leaderboard
  6. Print the strength profile (model vs category)
  7. Save summary.json for the Streamlit dashboard

Run with:  python main.py
"""

import json
from datetime import datetime
from models import run_model, RUNS_PER_PROMPT
from prompts import PROMPTS
from router import route_model
from metrics import gsm8k_accuracy, humaneval_pass_at_k

# Wipe last run's detail log so results.jsonl always matches this single session
print("Resetting results.jsonl...")
open("results.jsonl", "w").close()

# Models to benchmark. Second line overrides the first — use ["mistral"] for a
# fast single-model test; switch to the full list for a real comparison run.
MODELS = ["mistral", "llama3"]

results = {}

# -----------------------------
# STEP 1: BENCHMARK
# -----------------------------
for model in MODELS:

    print("\n========================")
    print("MODEL:", model)
    print("========================")

    results[model] = run_model(model, PROMPTS)

# -----------------------------
# STEP 2: GSM8K ACCURACY
# -----------------------------
print("\n========================")
print("GSM8K ACCURACY")
print("========================")

for model, stats in results.items():
    acc = gsm8k_accuracy(stats.get("gsm8k_results", []))
    print(f"{model}: {acc['correct']}/{acc['total']} ({acc['accuracy']:.2%})")

# -----------------------------
# STEP 3: HUMANEVAL RESULTS
# -----------------------------
print("\n========================")
print("HUMANEVAL RESULTS")
print("========================")

for model, stats in results.items():
    he = humaneval_pass_at_k(stats.get("humaneval_results", []), k=RUNS_PER_PROMPT)
    print(f"{model}: pass@1={he['pass@1']:.2%} ({he['passed']}/{he['total']})")

# -----------------------------
# STEP 4: ROUTING DEMO
# -----------------------------
print("\n========================")
print("ROUTING DEMO")
print("========================")

# Auto-route every category that was actually benchmarked (read keys off any
# model's category_scores — all models share the same set of categories)
test_categories = list(next(iter(results.values()))["category_scores"].keys())

for category in test_categories:
    best = route_model(category, results)
    score = results[best]["category_scores"].get(category, 0) if best else None
    print(f"  {category:<18} -> {best}  (score: {score:.4f})" if best else f"  {category:<18} -> no data")

# -----------------------------
# STEP 5: LEADERBOARD
# -----------------------------
print("\n========================")
print("FINAL LEADERBOARD")
print("========================")

sorted_models = sorted(
    results.items(),
    key=lambda x: x[1]["final_score"],
    reverse=True
)

for model, stats in sorted_models:

    # Compute once and reuse throughout
    acc = gsm8k_accuracy(stats.get("gsm8k_results", []))
    he  = humaneval_pass_at_k(stats.get("humaneval_results", []), k=RUNS_PER_PROMPT)
    judge = stats["avg_judge"]

    print(f"\n  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  Model: {model:<41}║")
    print(f"  ╚══════════════════════════════════════════════════╝")

    print(f"\n  Final Score    : {stats['final_score']:.4f}")
    print(f"  Avg Latency    : {round(stats['avg_latency'], 2)}s")
    print(f"  Speed Score    : {stats['normalized_speed']:.3f}")
    print(f"  Stability      : {stats['model_stability']:.3f}")

    print(f"\n  ─── Score Breakdown ─────────────────────────────")

    # Judge signal — all 4 dimensions
    if stats["category_judge_scores"]:
        print(f"  Judge          : correctness={judge['correctness']:.3f}  reasoning={judge['reasoning']:.3f}  clarity={judge['clarity']:.3f}  hallucination_free={judge['hallucination_free']:.3f}")
    else:
        print(f"  Judge          : n/a (no judged categories run)")

    # GSM8K
    if acc["total"] > 0:
        print(f"  GSM8K          : {acc['correct']}/{acc['total']} ({acc['accuracy']:.2%})")
    else:
        print(f"  GSM8K          : n/a")

    # HumanEval
    if he["total"] > 0:
        he_str = f"pass@1={he['pass@1']:.2%}"
        if RUNS_PER_PROMPT > 1:
            he_str += f"  pass@{RUNS_PER_PROMPT}={he[f'pass@{RUNS_PER_PROMPT}']:.2%}"
        print(f"  HumanEval      : {he_str} ({he['passed']}/{he['total']})")
    else:
        print(f"  HumanEval      : n/a")

    print(f"\n  ─── Category Breakdown ──────────────────────────")

    for cat in stats["category_stabilities"]:

        cat_score = stats.get("category_scores", {}).get(cat, None)
        score_str = f"{cat_score:.4f}" if cat_score is not None else "n/a"

        print(f"\n    {cat.upper()}")
        print(f"      Score     : {score_str}")
        print(f"      Stability : {stats['category_stabilities'][cat]:.3f}")
        print(f"      Latency   : {stats['category_latencies'][cat]:.2f}s")

        if cat in stats["category_judge_scores"]:
            j = stats["category_judge_scores"][cat]
            print(f"      Judge     : correctness={j.get('correctness',0):.3f}  reasoning={j.get('reasoning',0):.3f}  clarity={j.get('clarity',0):.3f}  hallucination_free={j.get('hallucination_free',0):.3f}")
        elif cat == "gsm8k":
            print(f"      Accuracy  : {acc['correct']}/{acc['total']} ({acc['accuracy']:.2%})")
        elif cat == "humaneval":
            he_str = f"pass@1={he['pass@1']:.2%}"
            if RUNS_PER_PROMPT > 1:
                he_str += f"  pass@{RUNS_PER_PROMPT}={he[f'pass@{RUNS_PER_PROMPT}']:.2%}"
            print(f"      {he_str} ({he['passed']}/{he['total']})")

    print()

# -----------------------------
# STEP 6: STRENGTH PROFILE
# -----------------------------
print("\n========================")
print("STRENGTH PROFILE")
print("========================")

# Collect all categories across all models
all_cats = []
for stats in results.values():
    for cat in stats.get("category_scores", {}).keys():
        if cat not in all_cats:
            all_cats.append(cat)

if all_cats and results:
    model_names = list(results.keys())

    # Header
    col_w = 12
    header = f"  {'Category':<18}" + "".join(f"{m:<{col_w}}" for m in model_names)
    print(header)
    print("  " + "─" * (18 + col_w * len(model_names)))

    for cat in all_cats:
        row = f"  {cat:<18}"
        for m in model_names:
            score = results[m].get("category_scores", {}).get(cat)
            row += f"{score:.4f}      " if score is not None else f"{'n/a':<{col_w}}"
        print(row)

    print()

    # Winner per category
    print("  Winner per category:")
    for cat in all_cats:
        scores = {m: results[m].get("category_scores", {}).get(cat) for m in model_names}
        scores = {m: s for m, s in scores.items() if s is not None}
        if scores:
            winner = max(scores, key=lambda m: scores[m])
            print(f"    {cat:<18}: {winner} ({scores[winner]:.4f})")

# -----------------------------
# STEP 7: SAVE SUMMARY
# -----------------------------
# summary.json holds the aggregated, in-memory results that aren't in
# results.jsonl (final scores, category scores, etc.) so the dashboard can
# render the leaderboard without re-running any computation.
from judge import JUDGE_MODEL
summary = {
    "run_at": datetime.now().isoformat(),
    "runs_per_prompt": RUNS_PER_PROMPT,
    "judge_model": JUDGE_MODEL,
    "models": {}
}

for model, stats in results.items():
    acc = gsm8k_accuracy(stats.get("gsm8k_results", []))
    he  = humaneval_pass_at_k(stats.get("humaneval_results", []), k=RUNS_PER_PROMPT)
    summary["models"][model] = {
        "final_score":          stats["final_score"],
        "avg_latency":          stats["avg_latency"],
        "normalized_speed":     stats["normalized_speed"],
        "model_stability":      stats["model_stability"],
        "avg_judge":            stats["avg_judge"],
        "category_scores":      stats.get("category_scores", {}),
        "category_stabilities": stats["category_stabilities"],
        "category_latencies":   stats["category_latencies"],
        "category_judge_scores":stats["category_judge_scores"],
        "gsm8k_accuracy":       acc,
        "humaneval_results":    he
    }

with open("summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSummary saved to summary.json  ->  run: streamlit run dashboard.py")
