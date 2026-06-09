"""
findings.py — Derives key human-readable findings from benchmark data.

Pure functions — no API calls, no I/O.
Takes the summary and routing dicts (already loaded from disk) and returns
a structured findings dict the dashboard renders as headline cards.
"""

from __future__ import annotations


def _short_name(model: str) -> str:
    """Return a display-friendly short name for a model string."""
    if model.startswith("gpt"):
        return "GPT-4o"
    if model.startswith("claude"):
        return "Claude Sonnet"
    if "deepseek" in model.lower():
        return "DeepSeek V4 Pro"
    return model


def _avg_cost(model_data: dict) -> float:
    """Average per-prompt cost across categories (from category_costs)."""
    costs = list(model_data.get("category_costs", {}).values())
    return sum(costs) / len(costs) if costs else 1.0


def compute_findings(summary: dict, routing: dict | None) -> dict:
    """
    Compute key findings from a benchmark run.

    Parameters
    ----------
    summary : dict   — contents of summary.json
    routing : dict   — contents of routing.json (may be None)

    Returns
    -------
    dict with keys:
        winner                  — {model, short_name, score}
        runner_up               — {model, short_name, score}
        trailing                — {model, short_name, score}
        biggest_gap             — {category, gap, best_model, best_short, worst_model, worst_short, best_score, worst_score}
        best_value              — {model, short_name, quality, cost, ratio}
        escalation_categories   — list of category names where default != escalate_to
        no_escalation_categories — list of category names already routed to the best model
        category_winners        — {category: {model, short_name, score}}
    """
    models_data: dict = summary.get("models", {})
    policy: dict = (routing or {}).get("policy", {})

    if not models_data:
        return {}

    # ── 1. Overall ranking ─────────────────────────────────────────────────────
    ranked = sorted(
        models_data.items(),
        key=lambda x: x[1].get("final_score", 0.0),
        reverse=True,
    )
    winner_name, winner_data = ranked[0]
    runner_up_name, runner_up_data = ranked[1] if len(ranked) > 1 else ranked[0]
    last_name, last_data = ranked[-1]

    # ── 2. Per-category winners ────────────────────────────────────────────────
    all_cats: set[str] = set()
    for _, d in models_data.items():
        all_cats.update(d.get("category_scores", {}).keys())

    category_winners: dict = {}
    for cat in all_cats:
        scores = {
            m: d["category_scores"].get(cat, 0.0)
            for m, d in models_data.items()
        }
        best_m = max(scores, key=scores.get)  # type: ignore[arg-type]
        category_winners[cat] = {
            "model": best_m,
            "short_name": _short_name(best_m),
            "score": scores[best_m],
        }

    # ── 3. Biggest per-category quality gap ───────────────────────────────────
    biggest_gap: dict = {
        "category": None,
        "gap": 0.0,
        "best_model": None,
        "best_short": None,
        "worst_model": None,
        "worst_short": None,
        "best_score": 0.0,
        "worst_score": 0.0,
    }
    for cat in all_cats:
        scores = {
            m: d["category_scores"].get(cat, 0.0)
            for m, d in models_data.items()
        }
        best_m = max(scores, key=scores.get)  # type: ignore[arg-type]
        worst_m = min(scores, key=scores.get)  # type: ignore[arg-type]
        gap = scores[best_m] - scores[worst_m]
        if gap > biggest_gap["gap"]:
            biggest_gap = {
                "category": cat,
                "gap": gap,
                "best_model": best_m,
                "best_short": _short_name(best_m),
                "worst_model": worst_m,
                "worst_short": _short_name(worst_m),
                "best_score": scores[best_m],
                "worst_score": scores[worst_m],
            }

    # ── 4. Best value (quality ÷ avg cost per prompt) ─────────────────────────
    value_scores: dict = {}
    for m, d in models_data.items():
        c = _avg_cost(d)
        if c > 0:
            value_scores[m] = d.get("final_score", 0.0) / c

    best_value_model = max(value_scores, key=value_scores.get)  # type: ignore[arg-type]
    bv_data = models_data[best_value_model]
    best_value = {
        "model": best_value_model,
        "short_name": _short_name(best_value_model),
        "quality": bv_data.get("final_score", 0.0),
        "avg_cost": _avg_cost(bv_data),
        "ratio": value_scores[best_value_model],
    }

    # ── 5. Escalation coverage ────────────────────────────────────────────────
    escalation_cats: list[str] = []
    no_escalation_cats: list[str] = []
    for cat, rule in policy.items():
        if rule.get("default") != rule.get("escalate_to"):
            escalation_cats.append(cat)
        else:
            no_escalation_cats.append(cat)

    return {
        "winner": {
            "model": winner_name,
            "short_name": _short_name(winner_name),
            "score": winner_data.get("final_score", 0.0),
        },
        "runner_up": {
            "model": runner_up_name,
            "short_name": _short_name(runner_up_name),
            "score": runner_up_data.get("final_score", 0.0),
        },
        "trailing": {
            "model": last_name,
            "short_name": _short_name(last_name),
            "score": last_data.get("final_score", 0.0),
        },
        "biggest_gap": biggest_gap,
        "best_value": best_value,
        "escalation_categories": escalation_cats,
        "no_escalation_categories": no_escalation_cats,
        "category_winners": category_winners,
    }
