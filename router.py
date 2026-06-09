"""
router.py — Picks the model for each task category from benchmark results.

ONE currency: quality (category_score). Quality alone decides:
  - the leaderboard rank,
  - which models are ELIGIBLE to be a category's default (the quality bar),
  - the escalate target (the highest-quality model).

Cost and latency NEVER enter the quality score. They are used only to choose
AMONG the models that already clear the quality bar:
  - speed-sensitive categories (config.SPEED_SENSITIVE) -> the FASTEST eligible
  - everything else                                     -> the CHEAPEST eligible
                                                           (latency breaks ties)

The whole design in one line: you escalate to GAIN quality, so only a quality
shortfall can decide who is best or whether to escalate — cost and latency only
decide which eligible model you start with, never whether you escalate.

This module has two halves:
  - derive_routing_table() : OFFLINE — turns benchmark results into the policy.
  - run_live()             : LIVE — applies that policy to a single real prompt,
                             judging it live and upgrading only on a measured
                             quality shortfall (never preemptively).
"""

from datetime import datetime
from config import (
    TIGHT_CATEGORIES, TIGHT_TOLERANCE, DEFAULT_TOLERANCE, SPEED_SENSITIVE,
)
from dispatcher import call_model
from judge import judge_response, JUDGE_KEYS
from metrics import category_score
from storage import load_routing


# -----------------------------
# MODEL ROUTER (best by quality)
# -----------------------------
def route_model(category, results):
    """
    Returns the highest-QUALITY model for a category (the escalate target),
    or None if no model has data for it. Quality is the single ranking signal,
    so this can never disagree with the leaderboard.
    """
    best_model   = None
    best_quality = float("-inf")

    for model, stats in results.items():
        if not stats:
            continue
        quality = stats.get("category_scores", {}).get(category)
        if quality is None:
            continue  # no data for this category — skip
        if quality > best_quality:
            best_quality = quality
            best_model   = model

    return best_model


# -----------------------------
# TOLERANCE PER CATEGORY
# -----------------------------
def _tolerance_for(category):
    """
    Tight bar for precision-critical categories (coding, architecture),
    looser bar for everything else. Pulled from config so it's tunable in one place.
    """
    return TIGHT_TOLERANCE if category in TIGHT_CATEGORIES else DEFAULT_TOLERANCE


# -----------------------------
# QUALITY / COST / LATENCY PER MODEL FOR ONE CATEGORY
# -----------------------------
def _score_models_for_category(category, results):
    """
    Returns {model: {quality, cost, latency}} for every model that has a quality
    score for this category. `cost` and `latency` are the real measured per-prompt
    averages for the category (either may be None if unmeasured).
    """
    scored = {}
    for model, stats in results.items():
        if not stats:
            continue
        quality = stats.get("category_scores", {}).get(category)
        if quality is None:
            continue  # this model has no data for this category
        scored[model] = {
            "quality": quality,
            "cost":    stats.get("category_costs", {}).get(category),      # may be None
            "latency": stats.get("category_latencies", {}).get(category),  # may be None
        }
    return scored


# -----------------------------
# DERIVE ROUTING TABLE (the product)
# -----------------------------
def derive_routing_table(results, prompt_counts=None):
    """
    Build the routing policy from benchmark results — the machine-readable
    answer to "which model for which task".

    Per category:
      - best_quality : the highest category quality score.
      - quality_bar  : best_quality - tolerance (tight 0.03 for coding/
                       architecture, else 0.05). The best model always clears
                       its own bar, so at least one model is always eligible.
      - eligible     : models whose quality >= quality_bar.
      - default      : chosen AMONG the eligible models —
                         speed-sensitive category -> lowest latency
                         otherwise                -> lowest cost (latency tiebreak)
                       Unknown cost/latency is treated as +inf so it never wins by accident.
      - escalate_to  : the highest-quality model (route_model) — where a live
                       answer goes when its measured quality falls below the bar.
      - models       : per-model transparency {quality, cost_usd, latency, clears_bar}
                       so anyone can audit exactly why each decision was made.

    Args:
        results:       {model: stats_dict} from models.run_model
        prompt_counts: optional {category: int} for "decision based on N prompts"

    Returns:
      {
        "generated_at": iso timestamp,
        "policy": { category: { default, escalate_to, quality_bar, tolerance,
                                best_quality, selection, prompt_count, models } }
      }
    """
    prompt_counts = prompt_counts or {}
    policy = {}

    # Every category that at least one model scored
    categories = []
    for stats in results.values():
        if not stats:
            continue
        for cat in stats.get("category_scores", {}):
            if cat not in categories:
                categories.append(cat)

    for category in categories:
        scored = _score_models_for_category(category, results)
        if not scored:
            continue

        # Highest quality = escalate target. Reuse route_model for one source of truth.
        escalate_to = route_model(category, results)
        if escalate_to is None or escalate_to not in scored:
            continue
        best_quality = scored[escalate_to]["quality"]

        tolerance   = _tolerance_for(category)
        quality_bar = round(best_quality - tolerance, 4)

        # Eligible = clears the quality bar (the best model always qualifies → never empty)
        eligible = {m: d for m, d in scored.items() if d["quality"] >= quality_bar}

        speed_sensitive = category in SPEED_SENSITIVE

        def _latency_then_cost_key(m):
            lat = eligible[m]["latency"]
            c   = eligible[m]["cost"]
            latency = lat if lat is not None else float("inf")
            cost    = c   if c   is not None else float("inf")
            return (latency, cost)  # fastest first; cost breaks latency ties

        def _cost_then_latency_key(m):
            c   = eligible[m]["cost"]
            lat = eligible[m]["latency"]
            cost    = c   if c   is not None else float("inf")
            latency = lat if lat is not None else float("inf")
            return (cost, latency)  # cheapest first; latency breaks cost ties

        # Among eligible models, cost/latency only chooses WHICH one — never quality.
        # Symmetric: each path uses a secondary axis to break ties on the primary.
        if speed_sensitive:
            default = min(eligible, key=_latency_then_cost_key)
        else:
            default = min(eligible, key=_cost_then_latency_key)

        # Per-model transparency block
        models_block = {
            m: {
                "quality":    d["quality"],
                "cost_usd":   d["cost"],
                "latency":    round(d["latency"], 4) if d["latency"] is not None else None,
                "clears_bar": d["quality"] >= quality_bar,
            }
            for m, d in scored.items()
        }

        policy[category] = {
            "default":      default,
            "escalate_to":  escalate_to,
            "quality_bar":  quality_bar,
            "tolerance":    tolerance,
            "best_quality": best_quality,
            "selection":    "fastest" if speed_sensitive else "cheapest",
            "prompt_count": prompt_counts.get(category),
            "models":       models_block,
        }

    return {
        "generated_at": datetime.now().isoformat(),
        "policy": policy,
    }


# -----------------------------
# LIVE RUN (apply the policy to one real prompt, escalating only if needed)
# -----------------------------
def _run_and_score(model, prompt, category):
    """
    Call ONE model on a live prompt, judge it with the live leave-one-out jury,
    and score that single response's quality using the category's weights.

    Returns:
        dict {model, response, quality, latency, cost_usd, judge_cost_usd,
              judge_costs, judge}
          - cost_usd       = the answer call's cost
          - judge_cost_usd = the jury's cost to score THIS answer (0.0 if unjudged)
          - judge_costs    = per-judge-model cost breakdown (stored, rarely shown)
          - quality is None if the jury failed to score it (the call still
            succeeded, so we keep the response but have no live score).
        None if the model call itself failed entirely (no response at all).
    """
    call = call_model(model, prompt)
    if call is None:
        return None  # the model never answered — caller decides what to do

    judge = judge_response(prompt, call["text"], model)
    if judge is None:
        quality        = None  # we have a response but couldn't score it
        judge_cost_usd = 0.0   # no successful judge calls to attribute
        judge_costs    = {}
    else:
        dims = {k: judge.get(k, 0.0) for k in JUDGE_KEYS}
        quality        = category_score(category, dims)
        judge_cost_usd = judge.get("judge_cost_usd", 0.0)  # judging cost for THIS answer
        judge_costs    = judge.get("judge_costs", {})      # per-judge detail (stored)

    return {
        "model":          model,
        "response":       call["text"],
        "quality":        quality,
        "latency":        round(call["latency"], 4),
        "cost_usd":       call.get("cost_usd"),   # answer cost
        "judge_cost_usd": judge_cost_usd,          # judging cost for this answer
        "judge_costs":    judge_costs,             # per-judge breakdown (stored, not always shown)
        "judge":          judge,
    }


def _print_trace(trace):
    """
    Per-answer summary of one escalation decision: one line per answer
    (model · role · quality · answer cost · judging cost), then the three
    subtotals (answers / judging / all-in). Store rich, display lean — the
    per-judge breakdown stays in the trace; here we show only the rollups.
    """
    print(f"\n  [run_live] category={trace['category']}   bar={trace['quality_bar']:.4f}\n")

    def _leg_line(leg, role):
        if not leg:
            return
        q   = leg.get("quality")
        ans = leg.get("cost_usd")
        jdg = leg.get("judge_cost_usd")
        q_str   = f"{q:.4f}"    if q   is not None else "n/a"
        ans_str = f"${ans:.6f}" if ans is not None else "n/a"
        jdg_str = f"${jdg:.6f}" if jdg is not None else "n/a"
        print(f"    {leg['model']:<18} ({role:<9}) q={q_str:<8} answer={ans_str:<11} judging={jdg_str}")

    _leg_line(trace.get("default"), "default")
    if trace.get("escalated"):
        _leg_line(trace.get("escalated_call"), "escalated")

    print(f"    --- totals ---")
    print(f"    answers:  ${trace.get('answer_cost_usd', 0.0):.6f}")
    print(f"    judging:  ${trace.get('judge_cost_usd', 0.0):.6f}")
    print(f"    all-in:   ${trace.get('total_cost_usd', 0.0):.6f}")
    print(f"    -> returning {trace['final_model']}  ({trace.get('reason', '')})")


def _emit(callback, message):
    """
    Best-effort live status update for the UI. `callback` is an optional
    one-arg function (e.g. the dashboard's st.write). Wrapped in try/except so
    that reporting a step can never break the actual run — if the UI hook
    raises, the run continues unaffected.
    """
    if callback is None:
        return
    try:
        callback(message)
    except Exception:
        pass


def run_live(prompt, category, policy=None, status_callback=None):
    """
    Apply the routing policy to ONE live prompt — the live counterpart to
    derive_routing_table(). The offline table decides the policy; this applies it.

    Named run_live (not "escalate") because it ALWAYS runs the default model and
    judges it; escalation is just one possible step inside, taken only when the
    measured quality falls below the bar.

    Flow:
      1. Look up the category's policy -> default, escalate_to, quality_bar.
      2. Run the DEFAULT model, judge it live, score its quality.
      3. Decide:
           - couldn't score it          -> return default, flagged (never guess)
           - quality >= bar             -> keep the default (cheap/fast path)
           - quality <  bar, but default
             IS already the best model  -> keep it (nothing better to go to)
           - quality <  bar, better model
             exists                     -> escalate: run + judge escalate_to,
                                           return ITS response
      4. Return a full trace of every measured value, and print a summary.

    Pure of file I/O: it makes API calls but writes nothing to disk. The returned
    trace IS the record — persistent live-logging is a later step.

    Args:
        prompt:   the live prompt text
        category: which category it belongs to (must exist in `policy`)
        policy:   the routing policy mapping {category: {default, escalate_to,
                  quality_bar, ...}} — i.e. the inner "policy" dict from
                  routing.json, or the full routing dict (auto-unwrapped).
                  If None, auto-loaded from routing.json — so the simplest
                  valid call is just: run_live(prompt, category)
        status_callback: optional one-arg function called with a short
                  human-readable string at each internal milestone (default
                  model call, live score, escalation decision, escalated call).
                  Lets a UI show a truly live step-by-step trace. Best-effort —
                  errors in the callback never affect the run. Default None.

    Returns:
        trace dict (see code), or None if the category has no policy.
    """
    # --- resolve policy: accept None, the full routing dict, or the inner policy ---
    if policy is None:
        loaded = load_routing()
        if not loaded or not loaded.get("policy"):
            print("  [run_live] No routing.json policy found — run a benchmark first.")
            return None
        policy = loaded["policy"]
    elif "policy" in policy and "generated_at" in policy:
        # caller passed the full routing.json dict — unwrap silently
        policy = policy["policy"]

    cat_policy = policy.get(category)
    if cat_policy is None:
        print(f"  [run_live] No policy for category '{category}' — cannot route.")
        return None

    default_model = cat_policy["default"]
    escalate_to   = cat_policy["escalate_to"]
    quality_bar   = cat_policy["quality_bar"]

    # --- 1. Run + judge + score the DEFAULT model ---
    _emit(status_callback, f"⚡ Calling **{default_model}** — the benchmark's recommended model for **{category}** tasks. Two independent graders will score the response…")
    default_result = _run_and_score(default_model, prompt, category)
    if default_result is None:
        # The default model never answered — nothing to return, fail honestly.
        _emit(status_callback, f"❌ **{default_model}** failed to respond after retries — nothing to return.")
        print(f"  [run_live] Default model {default_model} failed — no response.")
        return {
            "category":        category,
            "prompt":          prompt,
            "quality_bar":     quality_bar,
            "default_model":   default_model,
            "escalate_to":     escalate_to,
            "escalated":       False,
            "error":           "default_call_failed",
            "final_model":     None,
            "final_response":  None,
            "default":         None,
            "answer_cost_usd": 0.0,
            "judge_cost_usd":  0.0,
            "total_cost_usd":  0.0,
        }

    # Base trace — assumes we keep the default; escalation overrides below.
    # Costs split three ways: answers (responses kept) + judging (the live jury)
    # = all-in. Stored separately so the true cost is never hidden.
    base_answer = default_result["cost_usd"] or 0.0
    base_judge  = default_result["judge_cost_usd"] or 0.0
    trace = {
        "category":        category,
        "prompt":          prompt,
        "quality_bar":     quality_bar,
        "default_model":   default_model,
        "escalate_to":     escalate_to,
        "escalated":       False,
        "final_model":     default_model,
        "final_response":  default_result["response"],
        "default":         default_result,
        "answer_cost_usd": round(base_answer, 6),
        "judge_cost_usd":  round(base_judge, 6),
        "total_cost_usd":  round(base_answer + base_judge, 6),
    }

    default_quality = default_result["quality"]

    # --- 2. Decide ---
    # (a) Couldn't score the default -> don't guess, return it, flag it.
    if default_quality is None:
        trace["judged"] = False
        trace["reason"] = "default unjudged (jury failed) — not escalating without evidence"
        _emit(status_callback, "⚠️ Both graders failed to produce a score — returning the answer ungraded. We never escalate without a measured quality shortfall.")
        _print_trace(trace)
        return trace

    trace["judged"] = True
    _emit(status_callback, f"📊 **{default_model}** responded — two graders scored it **{default_quality:.2f}** (quality bar for **{category}**: **{quality_bar:.2f}**).")

    # (b) Default clears the bar -> keep the cheap/fast choice.
    if default_quality >= quality_bar:
        trace["reason"] = "default cleared the quality bar"
        _emit(status_callback, f"✅ **{default_quality:.2f}** cleared the bar of **{quality_bar:.2f}** — **{default_model}** handled it well. No escalation needed.")
        _print_trace(trace)
        return trace

    # (c) Below bar, but the default IS already the best model -> nothing better.
    if escalate_to == default_model:
        trace["reason"] = "below bar but already on the best model — no higher option"
        _emit(status_callback, f"⚠️ **{default_quality:.2f}** fell below the bar of **{quality_bar:.2f}** — but **{default_model}** is already the highest-quality model for **{category}**, nowhere to escalate. Returning its response.")
        _print_trace(trace)
        return trace

    # (d) Below bar and a higher-quality model exists -> escalate.
    _emit(status_callback, f"⚠️ **{default_quality:.2f}** fell below the bar of **{quality_bar:.2f}** — escalating to **{escalate_to}** for a stronger answer…")
    escalated_result = _run_and_score(escalate_to, prompt, category)
    if escalated_result is None:
        # Escalation target failed — fall back to the default response, flagged.
        trace["reason"] = "below bar; escalation target failed — returning default"
        trace["escalation_failed"] = True
        _emit(status_callback, f"❌ **{escalate_to}** failed to respond after retries — falling back to **{default_model}**'s answer.")
        _print_trace(trace)
        return trace

    trace["escalated"]      = True
    trace["final_model"]    = escalate_to
    trace["final_response"] = escalated_result["response"]
    trace["escalated_call"] = escalated_result
    trace["reason"]         = "default below bar — escalated to highest-quality model"

    _esc_q = escalated_result["quality"]
    _esc_q_str = f"{_esc_q:.2f}" if _esc_q is not None else "n/a"
    _emit(status_callback, f"📊 **{escalate_to}** responded — escalated score **{_esc_q_str}** vs default **{default_quality:.2f}** (bar: **{quality_bar:.2f}**).")

    # Roll both legs into the three cost fields (answers + judging = all-in).
    answer_cost = (default_result["cost_usd"] or 0.0) + (escalated_result["cost_usd"] or 0.0)
    judge_cost  = (default_result["judge_cost_usd"] or 0.0) + (escalated_result["judge_cost_usd"] or 0.0)
    trace["answer_cost_usd"] = round(answer_cost, 6)
    trace["judge_cost_usd"]  = round(judge_cost, 6)
    trace["total_cost_usd"]  = round(answer_cost + judge_cost, 6)

    _print_trace(trace)
    return trace


# -----------------------------
# ROUTE LOOKUP (which model — without calling it)
# -----------------------------
def _why(task, pol):
    """One-line, human-readable explanation of a routing decision."""
    default     = pol["default"]
    selection   = pol.get("selection", "chosen")
    quality_bar = pol.get("quality_bar")
    escalate_to = pol.get("escalate_to")
    quality     = pol.get("models", {}).get(default, {}).get("quality")

    q_str   = f"{quality:.2f}"     if quality     is not None else "n/a"
    bar_str = f"{quality_bar:.2f}" if quality_bar is not None else "n/a"

    return (
        f"{task} -> {default}: {selection} model clearing the quality bar "
        f"({bar_str}); measured quality {q_str}. "
        f"Escalates to {escalate_to} if a live answer falls below the bar."
    )


def route(task, policy=None):
    """
    Look up which model to use for a task — WITHOUT calling any model.

    The lightweight, importable counterpart to run_live(): run_live() runs the
    prompt live and may upgrade; route() just reads the precomputed policy and
    explains the choice. Any other system can import and call this.

    Args:
        task:   a category name, e.g. "coding". (Free-text prompts are mapped to
                a category by categorizer.py / the Live Test tab; route() takes
                the already-resolved category.)
        policy: the routing policy mapping {category: {...}}. If None, it is
                auto-loaded from routing.json so a caller can simply do
                route("coding") with nothing else. Pass a dict to stay pure
                (e.g. in tests) — no file is read then.

    Returns:
        {model, why, cost_estimate, escalate_to} for a known task, or
        None if there is no policy for the task (or no routing.json yet).
    """
    # Auto-load the policy from routing.json when the caller didn't pass one.
    if policy is None:
        loaded = load_routing()
        if not loaded or not loaded.get("policy"):
            print("  [route] No routing.json policy found — run a benchmark first.")
            return None
        policy = loaded["policy"]

    pol = policy.get(task)
    if pol is None:
        print(f"  [route] No policy for task '{task}'.")
        return None

    default       = pol["default"]
    cost_estimate = pol.get("models", {}).get(default, {}).get("cost_usd")

    return {
        "model":         default,
        "why":           _why(task, pol),
        "cost_estimate": cost_estimate,
        "escalate_to":   pol.get("escalate_to"),
    }
