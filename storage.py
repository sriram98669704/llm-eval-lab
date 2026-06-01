"""
storage.py — Append-only logger for per-run results.

Every individual model run (one prompt, one attempt) is written as a single
JSON line to results.jsonl. This is the raw, detailed record that the dashboard's
GSM8K / HumanEval / Quality tabs read from.

results.jsonl is reset at the start of each main.py run, so it always reflects
exactly one benchmark session (kept in sync with summary.json).
"""

import json
from datetime import datetime

# -----------------------------
# STORE RESULTS
# -----------------------------
def log_result(
    model,
    category,
    prompt,
    response,
    latency,
    embedding=None,
    stability=None,
    judge_score=None,
    gsm8k_predicted=None,
    gsm8k_expected=None,
    gsm8k_correct=None,
    humaneval_passed=None,
    humaneval_error=None,
    humaneval_code=None
):
    record = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "category": category,
        "prompt": prompt,
        "response": response,
        "latency": latency,
        "embedding": embedding,
        "stability": stability,
        "judge_score": judge_score,
        "gsm8k_predicted": gsm8k_predicted,
        "gsm8k_expected": gsm8k_expected,
        "gsm8k_correct": gsm8k_correct,
        "humaneval_passed": humaneval_passed,
        "humaneval_error": humaneval_error,
        "humaneval_code": humaneval_code
    }

    with open("results.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


# -----------------------------
# FUTURE EXTENSIONS
# -----------------------------
# 1. Hallucination detection
#    - Compare embeddings across models to detect hallucinations
# 2. Response clustering
#    - Group similar responses to analyze patterns
# 3. Reranking system
#    - Use stored embeddings instead of recomputing
# 4. Dashboard (Streamlit or similar)
#    - Visualize response similarity, latency, and judge scores
