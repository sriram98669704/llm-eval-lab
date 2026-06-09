"""
main.py — Orchestrator / entry point for the LLM Eval Lab.

Runs the full benchmark pipeline end to end:
  1. Benchmark each model across all 6 prompt categories (models.run_model)
  2. Route each category to its best model (routing demo)
  3. Print the final leaderboard
  4. Print the strength profile (model vs category)
  5. Save summary.json for the Streamlit dashboard

Run with:
  python main.py           # full benchmark — 5 prompts per category
  python main.py --mini    # mini benchmark — 3 prompts per category
                           # (verifies the full pipeline end-to-end, ~13-15 min,
                           #  3 prompts gives real error bars unlike 1-2)
"""

import json
import hashlib
import subprocess
import sys
from datetime import datetime
from config import (
    EVALUATED_MODELS, JUDGE_MODELS, TEMPERATURE,
    FAMILY_SKIP, DEFAULT_TOLERANCE, TIGHT_TOLERANCE, TIGHT_CATEGORIES,
    SPEED_SENSITIVE, CATEGORIES,
)
from models import run_model
from prompts import PROMPTS
from router import derive_routing_table
from storage import init_run, save_meta, save_summary, update_index, save_routing


# -----------------------------
# META HELPERS
# -----------------------------
def _git_info():
    """
    Capture git provenance so the exact code state is recorded.

    git_dirty matters: a SHA alone is a false promise if you have uncommitted
    edits — two runs with the same SHA but different dirty state can produce
    different results. Recording dirty=True flags that the SHA isn't enough
    to fully reproduce the run.
    """
    def _run(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    sha    = _run(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    dirty  = bool(_run(["git", "status", "--porcelain"]))  # any uncommitted changes?

    return {"git_sha": sha, "git_branch": branch, "git_dirty": dirty}


def _env_info():
    """
    Record Python version and key SDK versions.
    Model behaviour can shift between SDK releases — this ties results to exact deps.
    Never crashes: unknown packages just record 'unknown'.
    """
    def _pkg(name):
        try:
            import importlib.metadata
            return importlib.metadata.version(name)
        except Exception:
            return "unknown"

    return {
        "python_version": sys.version.split()[0],
        "packages": {
            "openai":     _pkg("openai"),
            "anthropic":  _pkg("anthropic"),
            "together":   _pkg("together"),
        },
    }




def _run_fingerprint(prompts):
    """
    Hash of everything that directly affects results — inputs, config, and
    scoring weights. If this matches between two runs, any score difference
    is due to code logic changes (caught by git SHA) or model API drift.

    Includes:
      - evaluated_models      — which models ran
      - judge_models          — who scored them
      - family_skip           — which judges were skipped per contestant
      - temperature           — affects model outputs
      - tolerances            — affect the routing quality_bar
      - speed_sensitive       — affects which eligible model becomes the default
      - scoring_weights       — category profiles + unknown-category fallback
      - pricing               — per-token rates; they decide the cheapest eligible
                                default, so a price change can change the routing product
      - prompts               — the exact text every model saw

    Does NOT include: code logic (covered by git SHA), SDK versions
    (covered by environment), or run timing. One model call per prompt at
    temperature 0, so there is no runs-per-prompt knob to capture.
    """
    from metrics import CATEGORY_WEIGHTS, DEFAULT_WEIGHTS
    from pricing import PRICING

    # Full, honest capture of every scoring weight: the per-category profiles and
    # the unknown-category fallback. These live as plain data in metrics.py, so
    # the fingerprint changes whenever ANY weight changes — not just correctness.
    scoring_weights = {
        "category_profiles": CATEGORY_WEIGHTS,
        "default_profile":   DEFAULT_WEIGHTS,
    }

    payload = json.dumps({
        "evaluated_models":  EVALUATED_MODELS,
        "judge_models":      JUDGE_MODELS,
        "family_skip":       FAMILY_SKIP,
        "temperature":       TEMPERATURE,
        "tolerances": {
            "default":           DEFAULT_TOLERANCE,
            "tight":             TIGHT_TOLERANCE,
            "tight_categories":  TIGHT_CATEGORIES,
        },
        "speed_sensitive":   SPEED_SENSITIVE,
        "scoring_weights":   scoring_weights,
        "pricing":           PRICING,
        "prompts":           prompts,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _check_category_prompt_sync():
    """
    Warn (non-fatally) if config.CATEGORIES and prompts.json drift apart.

    They must agree: the leaderboard denominator and the dashboard both iterate
    config.CATEGORIES, while the benchmark loops the prompts.json keys. A mismatch
    is silent and corrupting — a prompts-only category never gets scored, and a
    config-only category counts as 0.0 for every model. Surface it loudly at the
    start of a run instead of letting it skew the numbers.
    """
    config_cats  = set(CATEGORIES)
    prompt_cats  = set(PROMPTS.keys())
    if config_cats != prompt_cats:
        missing_prompts = config_cats - prompt_cats   # in config, no prompts
        missing_config  = prompt_cats - config_cats   # have prompts, not in config
        print("\n  ⚠ CATEGORY MISMATCH between config.CATEGORIES and prompts.json:")
        if missing_prompts:
            print(f"      in config but NO prompts (will score 0.0): {sorted(missing_prompts)}")
        if missing_config:
            print(f"      have prompts but NOT in config (will be ignored): {sorted(missing_config)}")
        print("      Fix config.py or prompts.json so the two agree before trusting results.\n")


def run_benchmark(mini=False):

    run_started_at = datetime.now()

    # Pre-flight: config and prompts must describe the same categories.
    _check_category_prompt_sync()

    # Create timestamped run folder — all writes go here + root copy
    run_dir = init_run(run_started_at)
    print(f"Run folder: {run_dir}")

    # --mini: slice each category to the first 3 prompts.
    # 3 prompts is the minimum for meaningful error bars (std dev needs ≥ 2 points;
    # 3 gives a real spread rather than a single number vs zero).
    # Full run uses all 5 prompts per category.
    if mini:
        PROMPTS_TO_USE = {cat: prompts[:3] for cat, prompts in PROMPTS.items()}
        print("Mode: MINI (3 prompts per category — pipeline verification run)")
    else:
        PROMPTS_TO_USE = PROMPTS
        print("Mode: FULL (5 prompts per category)")

    # No wipe needed: each run writes into its own fresh data/runs/<timestamp>/
    # folder, so results.jsonl always reflects exactly this run.

    MODELS = EVALUATED_MODELS  # gpt-4o, claude-sonnet-4-5, deepseek-ai/DeepSeek-V4-Pro — set in config.py

    results = {}

    # -----------------------------
    # STEP 1: BENCHMARK
    # -----------------------------
    for model in MODELS:
        print("\n========================")
        print("MODEL:", model)
        print("========================")

        results[model] = run_model(model, PROMPTS_TO_USE)

    # -----------------------------
    # STEP 2: ROUTING TABLE
    # -----------------------------
    # Derive the full routing policy (default / escalate_to / bar per category)
    # from the benchmark results. Saved to routing.json in the save section below.
    print("\n========================")
    print("ROUTING TABLE")
    print("========================")

    # Categories that had at least one successful run across any model.
    # (Reused later by the strength profile.)
    all_run_cats = []
    for stats in results.values():
        for cat in stats.get("category_scores", {}).keys():
            if cat not in all_run_cats:
                all_run_cats.append(cat)

    prompt_counts = {cat: len(prompts) for cat, prompts in PROMPTS_TO_USE.items()}
    routing = derive_routing_table(results, prompt_counts=prompt_counts)

    if routing["policy"]:
        for category in all_run_cats:
            pol = routing["policy"].get(category)
            if not pol:
                print(f"  {category:<18} -> no data")
                continue
            print(
                f"  {category:<18} "
                f"default={pol['default']:<18} "
                f"escalate_to={pol['escalate_to']:<18} "
                f"quality_bar={pol['quality_bar']:.4f} "
                f"({pol['selection']})"
            )
    else:
        print("  No categories benchmarked successfully.")

    # -----------------------------
    # STEP 3: LEADERBOARD
    # -----------------------------
    print("\n========================")
    print("FINAL LEADERBOARD")
    print("========================")

    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get("final_score", 0),
        reverse=True
    )

    for model, stats in sorted_models:

        if not stats:
            print(f"\n  {model}: no successful runs")
            continue

        judge = stats.get("avg_judge", {})

        print(f"\n  ╔══════════════════════════════════════════════════╗")
        print(f"  ║  Model: {model:<41}║")
        print(f"  ╚══════════════════════════════════════════════════╝")

        print(f"\n  Quality Score  : {stats['final_score']:.4f}   (avg category quality — the rank metric)")
        print(f"  Avg Latency    : {round(stats['avg_latency'], 2)}s")

        print(f"\n  ─── Score Breakdown ─────────────────────────────")

        if stats.get("category_judge_scores"):
            print(
                f"  Judge          : "
                f"correctness={judge.get('correctness', 0):.3f}  "
                f"reasoning={judge.get('reasoning', 0):.3f}  "
                f"clarity={judge.get('clarity', 0):.3f}  "
                f"hallucination_free={judge.get('hallucination_free', 0):.3f}"
            )
        else:
            print(f"  Judge          : n/a (no judged categories ran successfully)")

        print(f"\n  ─── Category Breakdown ──────────────────────────")

        for cat in stats.get("category_scores", {}):

            cat_score = stats.get("category_scores", {}).get(cat)
            cat_std   = stats.get("category_score_stds", {}).get(cat)
            cat_lat   = stats.get("category_latencies", {}).get(cat)
            cat_cost  = stats.get("category_costs", {}).get(cat)
            score_str = f"{cat_score:.4f} ± {cat_std:.4f}" if cat_score is not None and cat_std is not None else "n/a"

            print(f"\n    {cat.upper()}")
            print(f"      Quality   : {score_str}")
            if cat_lat is not None:
                print(f"      Latency   : {cat_lat:.2f}s")
            if cat_cost is not None:
                print(f"      Cost      : ${cat_cost:.6f}")

            if cat in stats.get("category_judge_scores", {}):
                j = stats["category_judge_scores"][cat]
                print(
                    f"      Judge     : "
                    f"correctness={j.get('correctness', 0):.3f}  "
                    f"reasoning={j.get('reasoning', 0):.3f}  "
                    f"clarity={j.get('clarity', 0):.3f}  "
                    f"hallucination_free={j.get('hallucination_free', 0):.3f}"
                )

        print()

    # -----------------------------
    # STEP 4: STRENGTH PROFILE
    # -----------------------------
    print("\n========================")
    print("STRENGTH PROFILE")
    print("========================")

    if all_run_cats and results:
        model_names = list(results.keys())

        col_w = 12
        header = f"  {'Category':<18}" + "".join(f"{m:<{col_w}}" for m in model_names)
        print(header)
        print("  " + "─" * (18 + col_w * len(model_names)))

        for cat in all_run_cats:
            row = f"  {cat:<18}"
            for m in model_names:
                score = results[m].get("category_scores", {}).get(cat)
                row += f"{score:.4f}      " if score is not None else f"{'n/a':<{col_w}}"
            print(row)

        print()

        print("  Winner per category:")
        for cat in all_run_cats:
            scores = {
                m: results[m].get("category_scores", {}).get(cat)
                for m in model_names
            }
            scores = {m: s for m, s in scores.items() if s is not None}
            if scores:
                winner = max(scores, key=lambda m: scores[m])
                print(f"    {cat:<18}: {winner} ({scores[winner]:.4f})")

    # -----------------------------
    # STEP 5: SAVE SUMMARY
    # -----------------------------
    # summary.json holds aggregated results the dashboard reads.
    # results.jsonl holds the per-run detail.

    summary = {
        "run_at":          datetime.now().isoformat(),
        "judge_models":    JUDGE_MODELS,
        "models":          {}
    }

    for model, stats in results.items():
        if not stats:
            continue
        summary["models"][model] = {
            "final_score":           stats["final_score"],   # overall quality (rank metric)
            "avg_latency":           stats["avg_latency"],
            "avg_judge":             stats["avg_judge"],
            "category_scores":       stats.get("category_scores", {}),
            "category_score_stds":   stats.get("category_score_stds", {}),
            "category_latencies":    stats.get("category_latencies", {}),
            "category_costs":        stats.get("category_costs", {}),
            "category_judge_scores": stats.get("category_judge_scores", {}),
        }

    save_summary(summary)

    # -----------------------------
    # STEP 6: SAVE ROUTING TABLE
    # -----------------------------
    # routing.json is THE artifact — the machine-readable answer to
    # "which model for which task". Computed in STEP 2 above.
    save_routing(routing)
    print("Routing saved to routing.json")

    # -----------------------------
    # STEP 7: SAVE META.JSON
    # -----------------------------
    run_finished_at = datetime.now()
    duration_seconds = round((run_finished_at - run_started_at).total_seconds(), 1)

    # Judge health: merge per-model per-category counts into one structure
    # Shape: { contestant_model: { category: { scored, single_judge, unscored } } }
    judge_health_by_contestant = {
        model: results[model].get("judge_health", {})
        for model in MODELS
        if results.get(model)
    }

    # Roll up per-judge reliability. models.py records, per judge, how many
    # responses it was ASKED to score (attempted) and how many it ACTUALLY
    # scored (succeeded) — read straight from the judge_raw keys of each response.
    # That ground truth means a single-judged response is charged to the judge
    # that genuinely dropped out, never to its innocent partner. (The old code
    # here couldn't tell them apart and blamed both — fixed.)
    judge_reliability = {}
    for model, stats in results.items():
        if not stats:
            continue
        for judge_model, counts in stats.get("judge_reliability", {}).items():
            agg = judge_reliability.setdefault(judge_model, {"attempted": 0, "succeeded": 0})
            agg["attempted"] += counts["attempted"]
            agg["succeeded"] += counts["succeeded"]

    meta = {
        "schema_version": 1,

        "run": {
            "started_at":       run_started_at.isoformat(),
            "finished_at":      run_finished_at.isoformat(),
            "duration_seconds": duration_seconds,
        },

        "code": _git_info(),

        "environment": _env_info(),

        "config": {
            "temperature":        TEMPERATURE,
            "evaluated_models":   EVALUATED_MODELS,
            "judge_models":       JUDGE_MODELS,
            "family_skip":        FAMILY_SKIP,
            "tolerances": {
                "default":          DEFAULT_TOLERANCE,
                "tight":            TIGHT_TOLERANCE,
                "tight_categories": TIGHT_CATEGORIES,
            },
            "speed_sensitive":    SPEED_SENSITIVE,
        },

        # What ACTUALLY ran this run — PROMPTS_TO_USE, not the full suite. On a
        # --mini run this honestly records 3 per category, so the receipt and the
        # dashboard reflect the real call count instead of the full-suite total.
        "prompts": {
            "total":  sum(len(v) for v in PROMPTS_TO_USE.values()),
            "counts": {cat: len(prompts) for cat, prompts in PROMPTS_TO_USE.items()},
        },

        "run_fingerprint": _run_fingerprint(PROMPTS_TO_USE),

        "judge_health": {
            "by_contestant_category": judge_health_by_contestant,
            "by_judge_model":         judge_reliability,
        },
    }

    save_meta(meta)

    # Register this run in data/index.json — lightweight entry, not full meta
    update_index({
        "id":               run_dir.name,
        "started_at":       run_started_at.isoformat(),
        "finished_at":      run_finished_at.isoformat(),
        "duration_seconds": duration_seconds,
        "git_sha":          meta["code"]["git_sha"],
        "git_dirty":        meta["code"]["git_dirty"],
        "run_fingerprint":  meta["run_fingerprint"],
        "models":           EVALUATED_MODELS,
        "prompt_counts":    meta["prompts"]["counts"],
    })

    print(f"\nRun saved to {run_dir}/")
    print("  ├─ summary.json · routing.json · meta.json · results.jsonl")
    print("  └─ registered in data/index.json")
    print("\nLaunch the dashboard:  streamlit run dashboard.py")


if __name__ == "__main__":
    run_benchmark(mini="--mini" in sys.argv)
