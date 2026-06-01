"""
models.py — The evaluation engine.

For each model and each prompt category, this:
  - Calls the model (with a hard timeout to prevent hangs)
  - Computes an embedding for stability measurement
  - Scores the output via either ground truth (gsm8k/humaneval) or the LLM judge
  - Runs generated code in a sandbox (humaneval)
  - Logs every run to results.jsonl
  - Aggregates per-category and per-model scores

Returns a rich stats dict consumed by main.py for the leaderboard and summary.
Tunable constants (RUNS_PER_PROMPT, EMBEDDING_MODEL, etc.) live at the top.
"""

import time
import ollama
import concurrent.futures
from metrics import (
    stability_score,
    latency_score,
    extract_model_answer,
    extract_ground_truth,
    extract_function_code,
    category_score,
    compute_final_score
)
from storage import log_result
from judge import judge_response
from sandbox import run_code_in_sandbox

RUNS_PER_PROMPT = 3
JUDGE_KEYS = ["correctness", "reasoning", "clarity", "hallucination_free"]

# Categories that use ground truth instead of LLM judge
GROUND_TRUTH_CATEGORIES = {"gsm8k", "humaneval"}

# Model used for generating embeddings — separate from benchmarked models
EMBEDDING_MODEL = "nomic-embed-text"  # dedicated embedding model — neutral, not a benchmarked model

# Whether to store full embedding vectors in results.jsonl
# Set False to keep file size small (embeddings are large)
STORE_EMBEDDINGS = False

# Timeout for individual model calls in seconds (prevents infinite hangs)
MODEL_CALL_TIMEOUT = 300  # 5 minutes

# -----------------------------
# EMBEDDING
# -----------------------------
def get_embedding(text):
    """Get vector embedding using EMBEDDING_MODEL."""
    return ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )["embedding"]


def _call_model(model, prompt_text):
    """Inner model call — runs in a thread so we can apply a timeout."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt_text}]
    )
    return response["message"]["content"]


# -----------------------------
# SINGLE RUN
# -----------------------------
def run_once(model, prompt_text):
    """
    Run the model once on a given prompt.
    Enforces MODEL_CALL_TIMEOUT to prevent infinite hangs.
    """
    start = time.perf_counter()

    # ollama.chat has no native timeout, so we run it in a worker thread and
    # bound it with future.result(timeout=...). Without this, a stuck model
    # would hang the entire benchmark indefinitely.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call_model, model, prompt_text)
        try:
            text = future.result(timeout=MODEL_CALL_TIMEOUT)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Model call timed out after {MODEL_CALL_TIMEOUT}s")

    latency = time.perf_counter() - start
    embedding = get_embedding(text)  # used downstream to measure run-to-run stability

    return latency, text, embedding

# -----------------------------
# MODEL EVALUATION
# -----------------------------
def run_model(model, PROMPTS):
    """
    Run a model on all categories in PROMPTS.
    Returns detailed metrics.
    """

    category_stabilities = {}
    category_latencies = {}
    category_judge_scores = {}
    category_scores = {}
    all_latencies = []
    gsm8k_results = []
    humaneval_results = []

    for category, prompts in PROMPTS.items():
        print(f"\n")
        print(f"  ╔══════════════════════════════════════════════════╗")
        print(f"  ║  Category: {category.upper():<38}║")
        print(f"  ╚══════════════════════════════════════════════════╝")
        prompt_stabilities = []
        prompt_latencies = []
        prompt_judge_scores = []
        # gsm8k/humaneval have objective answers, so they skip the LLM judge entirely
        use_judge = category not in GROUND_TRUTH_CATEGORIES

        for item in prompts:
            prompt_text = item["input"]
            ground_truth = item.get("reference", None)

            embeddings = []
            latencies = []
            judge_scores = []

            display_prompt = prompt_text[:90] + "..." if len(prompt_text) > 90 else prompt_text
            print(f"\n  ┌─ Prompt ─────────────────────────────────────────")
            print(f"  │  {display_prompt}")
            print(f"  │  Runs: {RUNS_PER_PROMPT}")
            print(f"  └───────────────────────────────────────────────────")
            print()

            for run_idx in range(RUNS_PER_PROMPT):
                print(f"    ▶ Run {run_idx + 1}/{RUNS_PER_PROMPT} ...", end=" ", flush=True)
                latency, text, emb = run_once(model, prompt_text)
                print(f"done  ({round(latency, 2)}s)")
                embeddings.append(emb)
                latencies.append(latency)

                # Stability so far for THIS prompt's runs. With RUNS_PER_PROMPT=1
                # there's only one embedding, so it's trivially 1.0; it becomes
                # meaningful once multiple runs are compared pairwise.
                current_stability = stability_score(embeddings)

                # Judge scoring — skip for ground truth categories
                judge = None
                if use_judge:
                    judge = judge_response(prompt_text, text)
                    judge_scores.append(judge)

                # GSM8K ground truth comparison
                gsm8k_predicted = None
                gsm8k_expected = None
                gsm8k_correct = None

                if category == "gsm8k" and ground_truth:
                    gsm8k_predicted = extract_model_answer(text)
                    gsm8k_expected = extract_ground_truth(ground_truth)
                    # round() both sides: GSM8K answers are integers, guards against
                    # float artifacts like 72.0 vs 72 or a stray ".0" in the response
                    gsm8k_correct = (
                        gsm8k_predicted is not None and
                        gsm8k_expected is not None and
                        round(gsm8k_predicted) == round(gsm8k_expected)
                    )
                    gsm8k_results.append({"response": text, "reference": ground_truth})
                    print(f"      Predicted : {gsm8k_predicted}")
                    print(f"      Expected  : {gsm8k_expected}")
                    print(f"      Result    : {'Correct' if gsm8k_correct else 'Wrong'}")

                # HumanEval sandbox execution
                humaneval_passed = None
                humaneval_error = None
                humaneval_code = None

                if category == "humaneval" and ground_truth:
                    print(f"      Sandbox   : running...", end=" ", flush=True)
                    # Clean/repair the model output into runnable code
                    full_code = extract_function_code(text, ground_truth["original_prompt"])
                    humaneval_code = full_code  # store exactly what runs in sandbox
                    # Assemble: candidate function + official tests + the check() call
                    test_code = (
                        full_code +
                        "\n\n" +
                        ground_truth["test"] +
                        f"\ncheck({ground_truth['entry_point']})"
                    )
                    sandbox_result = run_code_in_sandbox(test_code, timeout=5)
                    humaneval_passed = sandbox_result["passed"]
                    humaneval_error = sandbox_result["error"]
                    humaneval_results.append({
                        "passed": humaneval_passed,
                        "error": humaneval_error
                    })
                    status = 'Passed' if humaneval_passed else 'Failed'
                    print(f"{status}  ({sandbox_result['time']}s)")
                    if not humaneval_passed and humaneval_error:
                        print(f"      Error     : {humaneval_error[:120]}")

                # Log to results.jsonl — stability now correctly logged per run
                log_result(
                    model=model,
                    category=category,
                    prompt=prompt_text,
                    response=text,
                    latency=latency,
                    embedding=emb if STORE_EMBEDDINGS else None,
                    stability=current_stability,
                    judge_score=judge,
                    gsm8k_predicted=gsm8k_predicted,
                    gsm8k_expected=gsm8k_expected,
                    gsm8k_correct=gsm8k_correct,
                    humaneval_passed=humaneval_passed,
                    humaneval_error=humaneval_error,
                    humaneval_code=humaneval_code
                )

            # Prompt-level summary
            avg_latency = sum(latencies) / len(latencies)
            stability = stability_score(embeddings)

            prompt_latencies.append(avg_latency)
            prompt_stabilities.append(stability)

            print()
            print(f"    ─── Summary ───────────────────────────────────")
            print(f"    Avg Latency : {round(avg_latency, 2)}s")
            print(f"    Stability   : {round(stability, 3)}")

            if use_judge and judge_scores:
                avg_judge_score = {k: sum(d.get(k, 0) for d in judge_scores) / len(judge_scores) for k in JUDGE_KEYS}
                prompt_judge_scores.append(avg_judge_score)
                print(f"    Judge Scores:")
                print(f"      Correctness      : {avg_judge_score['correctness']:.2f}")
                print(f"      Reasoning        : {avg_judge_score['reasoning']:.2f}")
                print(f"      Clarity          : {avg_judge_score['clarity']:.2f}")
                print(f"      Hallucination    : {avg_judge_score['hallucination_free']:.2f}")
            else:
                print(f"    Judge       : skipped (ground truth used)")
            print()

        # Category-level summary
        category_stability = sum(prompt_stabilities) / len(prompt_stabilities)
        category_latency = sum(prompt_latencies) / len(prompt_latencies)

        category_stabilities[category] = category_stability
        category_latencies[category] = category_latency
        all_latencies.append(category_latency)

        print(f"  ┌─ Category Summary ───────────────────────────────")
        print(f"  │  Avg Latency : {round(category_latency, 2)}s")
        print(f"  │  Stability   : {round(category_stability, 3)}")

        if use_judge and prompt_judge_scores:
            category_judge = {k: sum(d.get(k, 0) for d in prompt_judge_scores) / len(prompt_judge_scores) for k in JUDGE_KEYS}
            category_judge_scores[category] = category_judge
            print(f"  │  Judge:")
            print(f"  │    Correctness      : {category_judge['correctness']:.2f}")
            print(f"  │    Reasoning        : {category_judge['reasoning']:.2f}")
            print(f"  │    Clarity          : {category_judge['clarity']:.2f}")
            print(f"  │    Hallucination    : {category_judge['hallucination_free']:.2f}")

        # Compute category score using task-appropriate weights
        cat_score = category_score(
            category,
            judge_scores=category_judge_scores.get(category),
            gsm8k_results=gsm8k_results if category == "gsm8k" else None,
            humaneval_results=humaneval_results if category == "humaneval" else None,
            stability=category_stabilities.get(category, 1.0)
        )
        category_scores[category] = cat_score
        print(f"  │  Category Score    : {cat_score:.4f}")
        print(f"  └───────────────────────────────────────────────────")

    # -----------------------------
    # MODEL-LEVEL METRICS
    # -----------------------------
    # Roll category-level numbers up to single model-level figures
    avg_latency = sum(all_latencies) / len(all_latencies)
    model_stability = sum(category_stabilities.values()) / len(category_stabilities)
    normalized_speed = latency_score(avg_latency)

    # Average one judge dimension across all judged categories (0 if none judged)
    def _avg_judge_dim(dim):
        if not category_judge_scores:
            return 0.0
        return sum(c.get(dim, 0) for c in category_judge_scores.values()) / len(category_judge_scores)

    avg_judge_correctness     = _avg_judge_dim("correctness")
    avg_judge_reasoning       = _avg_judge_dim("reasoning")
    avg_judge_clarity         = _avg_judge_dim("clarity")
    avg_judge_hallucination   = _avg_judge_dim("hallucination_free")

    # Task-aware final score — built from per-category scores
    final_score = compute_final_score(
        category_scores=category_scores,
        model_stability=model_stability,
        avg_latency=avg_latency
    )

    return {
        "avg_latency": avg_latency,
        "normalized_speed": normalized_speed,
        "model_stability": model_stability,
        "final_score": final_score,
        "avg_judge": {
            "correctness":       avg_judge_correctness,
            "reasoning":         avg_judge_reasoning,
            "clarity":           avg_judge_clarity,
            "hallucination_free": avg_judge_hallucination
        },
        "category_stabilities": category_stabilities,
        "category_latencies": category_latencies,
        "category_judge_scores": category_judge_scores,
        "category_scores": category_scores,
        "gsm8k_results": gsm8k_results,
        "humaneval_results": humaneval_results
    }
