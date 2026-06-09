"""
models.py — The evaluation engine.

For each model and each prompt category, this:
  - Calls the model once via dispatcher (routes to the correct SDK by name prefix)
  - Scores the output via the LLM jury judge (leave-one-out)
  - Logs every call to results.jsonl
  - Aggregates per-category and per-model QUALITY scores (+ latency + cost)

Returns a stats dict consumed by main.py for the leaderboard, routing, and summary.

ONE call per prompt. At temperature 0 repeated runs of the same prompt are
~identical, so they add no signal — the reliability measure that matters is the
spread of quality ACROSS the different prompts in a category (score_std / error
bars), which needs only one run each. More robustness = more *different* prompts.

All model/judge choices come from config.py. All SDK calls go through
dispatcher.py — no SDK logic here.
"""

from dispatcher import call_model
from metrics import category_score, compute_final_score, score_std
from storage import log_result
from judge import judge_response, jury_pool, JUDGE_KEYS  # JUDGE_KEYS: single source of truth in judge.py
from config import CATEGORIES


# -----------------------------
# SINGLE CALL
# -----------------------------
def run_once(model, prompt_text):
    """
    Run the model once on a given prompt via the dispatcher.

    Returns a dict: {text, latency, cost_usd}
    Returns None on failure — caller skips the prompt gracefully.
    """
    result = call_model(model, prompt_text)
    if result is None:
        print(f"    [run_once] call_model returned None for {model} — skipping prompt")
        return None

    return {
        "text":     result["text"],
        "latency":  result["latency"],
        "cost_usd": result.get("cost_usd"),
    }


# -----------------------------
# MODEL EVALUATION
# -----------------------------
def run_model(model, PROMPTS):
    """
    Run a model on all categories in PROMPTS.
    Failed calls are skipped gracefully — the benchmark never crashes.
    Returns a detailed metrics dict (quality + latency + cost per category).
    """
    category_latencies    = {}
    category_judge_scores = {}
    category_scores       = {}
    category_score_stds   = {}   # std dev of per-prompt quality — error bars
    category_costs        = {}   # avg cost_usd per prompt — used by router for cheapest-model logic
    all_latencies         = []

    # Judge health: per category — how many responses were fully scored,
    # single-judged (only 1 of 2 judges succeeded), or completely unscored.
    # Used by main.py to populate meta.json.
    judge_health = {}   # { category: { "scored": int, "single_judge": int, "unscored": int } }

    # Per-judge reliability — ground truth from judge_raw keys (which judges
    # actually returned a score). This is what fixes the old "blame both judges"
    # bug: on a single-judged response we now know EXACTLY which judge dropped
    # out, so the miss is charged to the real culprit, never its innocent partner.
    pool = jury_pool(model)
    judge_reliability = {j: {"attempted": 0, "succeeded": 0} for j in pool}

    for category, prompts in PROMPTS.items():
        print(f"\n")
        print(f"  ╔══════════════════════════════════════════════════╗")
        print(f"  ║  Category: {category.upper():<38}║")
        print(f"  ╚══════════════════════════════════════════════════╝")

        prompt_latencies    = []
        prompt_judge_scores = []   # one {correctness,reasoning,clarity,hallucination_free} per prompt
        prompt_costs        = []   # one cost_usd per prompt

        # Health counters for this category
        cat_health = {"scored": 0, "single_judge": 0, "unscored": 0}
        expected_judges = len(pool)  # how many judges should score each response

        for item in prompts:
            prompt_text = item["prompt"]

            display_prompt = prompt_text[:90] + "..." if len(prompt_text) > 90 else prompt_text
            print(f"\n  ┌─ Prompt ─────────────────────────────────────────")
            print(f"  │  {display_prompt}")
            print(f"  └───────────────────────────────────────────────────")
            print(f"    ▶ calling {model} ...", end=" ", flush=True)

            run = run_once(model, prompt_text)
            if run is None:
                print(f"failed  (prompt skipped)")
                continue

            text     = run["text"]
            latency  = run["latency"]
            cost_usd = run["cost_usd"]
            print(f"done  ({round(latency, 2)}s)")

            # Judge scoring — leave-one-out jury
            judge = judge_response(prompt_text, text, model)
            if judge is not None:
                # judge_raw keys = which judges actually responded
                judges_responded = len(judge.get("judge_raw", {}))
                if judges_responded >= expected_judges:
                    cat_health["scored"] += 1
                else:
                    cat_health["single_judge"] += 1
                prompt_judge_scores.append({k: judge.get(k, 0.0) for k in JUDGE_KEYS})
            else:
                cat_health["unscored"] += 1
                print(f"    [models] Judge returned None — excluded from scoring")

            # Per-judge reliability: every judge in the pool was asked to score
            # this response. Record which ones actually did (judge_raw keys =
            # ground truth). When judge is None, both failed → none succeeded.
            responded = set((judge or {}).get("judge_raw", {}).keys())
            for j in pool:
                judge_reliability[j]["attempted"] += 1
                if j in responded:
                    judge_reliability[j]["succeeded"] += 1

            log_result(
                model=model,
                category=category,
                prompt=prompt_text,
                response=text,
                latency=latency,
                cost_usd=cost_usd,
                judge_score=judge,
            )

            prompt_latencies.append(latency)
            if cost_usd is not None:
                prompt_costs.append(cost_usd)

            if judge is not None:
                print(f"    Judge: "
                      f"correctness={judge['correctness']:.2f}  "
                      f"reasoning={judge['reasoning']:.2f}  "
                      f"clarity={judge['clarity']:.2f}  "
                      f"hallucination_free={judge['hallucination_free']:.2f}")

        # Skip category if every prompt failed the model call
        if not prompt_latencies:
            print(f"  ⚠ No successful calls in category {category} — skipping")
            continue

        category_latency = sum(prompt_latencies) / len(prompt_latencies)
        category_latencies[category] = category_latency
        all_latencies.append(category_latency)

        if prompt_costs:
            category_costs[category] = round(sum(prompt_costs) / len(prompt_costs), 6)

        # Category-level judge averages (across prompts)
        if prompt_judge_scores:
            category_judge = {
                k: sum(d.get(k, 0) for d in prompt_judge_scores) / len(prompt_judge_scores)
                for k in JUDGE_KEYS
            }
            category_judge_scores[category] = category_judge

        judge_health[category] = cat_health

        # Quality score for the category (from the averaged judge dims)
        cat_score = category_score(
            category,
            judge_scores=category_judge_scores.get(category),
        )
        category_scores[category] = cat_score

        # Error bars: score each prompt individually, measure the spread
        per_prompt_scores = [
            category_score(category, judge_scores=ps)
            for ps in prompt_judge_scores
        ]
        cat_std = score_std(per_prompt_scores)
        category_score_stds[category] = cat_std

        print(f"\n  ┌─ Category Summary ───────────────────────────────")
        print(f"  │  Quality     : {cat_score:.4f} ± {cat_std:.4f}")
        print(f"  │  Avg Latency : {round(category_latency, 2)}s")
        if category in category_costs:
            print(f"  │  Avg Cost    : ${category_costs[category]:.6f}")
        if category in category_judge_scores:
            cj = category_judge_scores[category]
            print(f"  │  Judge       : "
                  f"correctness={cj['correctness']:.2f}  "
                  f"reasoning={cj['reasoning']:.2f}  "
                  f"clarity={cj['clarity']:.2f}  "
                  f"hallucination_free={cj['hallucination_free']:.2f}")
        print(f"  └───────────────────────────────────────────────────")

    # -----------------------------
    # MODEL-LEVEL METRICS
    # -----------------------------
    if not all_latencies:
        print(f"  ⚠ No successful runs at all for {model}")
        return {}

    avg_latency = sum(all_latencies) / len(all_latencies)

    def _avg_judge_dim(dim):
        if not category_judge_scores:
            return 0.0
        return sum(c.get(dim, 0) for c in category_judge_scores.values()) / len(category_judge_scores)

    # Overall quality = average of per-category quality (the leaderboard rank).
    # Fixed denominator over all 6 CEO categories: a model that fails a whole
    # category gets 0.0 for it rather than a smaller denominator (which would
    # artificially inflate its score).
    final_score = compute_final_score(category_scores, all_categories=CATEGORIES)

    return {
        "final_score":          final_score,   # overall quality (rank metric)
        "avg_latency":          avg_latency,   # seconds (shown, never blended into quality)
        "avg_judge": {
            "correctness":        _avg_judge_dim("correctness"),
            "reasoning":          _avg_judge_dim("reasoning"),
            "clarity":            _avg_judge_dim("clarity"),
            "hallucination_free": _avg_judge_dim("hallucination_free"),
        },
        "category_scores":       category_scores,
        "category_score_stds":   category_score_stds,
        "category_latencies":    category_latencies,
        "category_costs":        category_costs,
        "category_judge_scores": category_judge_scores,
        "judge_health":          judge_health,
        "judge_reliability":     judge_reliability,   # per-judge attempted/succeeded (ground truth)
    }
