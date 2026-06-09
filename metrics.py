"""
metrics.py — Scoring math helpers (no I/O, no side effects).

ONE currency: quality.
  - category_score()      : task-aware 0–1 quality for a single category
                            (dot-product of the judge dimensions with that
                             category's weight profile)
  - compute_final_score() : a model's overall quality = the average of its
                            per-category quality scores (the leaderboard rank)
  - score_std()           : error bars — spread of quality ACROSS the
                            different prompts in a category

Cost and latency are deliberately NOT in here. They are separate axes, measured
elsewhere, and used only to choose AMONG models that already clear the quality
bar — they never enter the quality score. (You escalate to GAIN quality, so only
a quality shortfall can decide who is best or whether to escalate.)

All 6 categories (architecture/coding/debugging/summarization/
business_writing/agent_planning) are judge-scored — no ground-truth metrics.

Pure functions — easy to unit test in isolation.
"""

import math


# -----------------------------
# SCORE STD (error bars)
# -----------------------------
def score_std(scores):
    """
    Population standard deviation of a list of 0–1 category scores.

    The scores are one-per-prompt within a category, so this measures how
    consistent a model is ACROSS the different prompts of the same task type:
        small std = reliable    (similar score on every prompt)
        large std = inconsistent (great on some prompts, poor on others)

    This is the reliability signal. It replaces the old run-to-run "stability"
    metric, which was meaningless at temperature 0 (repeated runs of one prompt
    are ~identical, so every model scored ~1.0). Spread across *different*
    prompts is the honest measure — and it needs only one run per prompt.

    Returns 0.0 if fewer than 2 scores — can't measure spread from one point.
    """
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return round(math.sqrt(variance), 4)


# -----------------------------
# CATEGORY SCORING PROFILES (task-aware weights)
# -----------------------------
# Each category weights the judge dimensions that matter most for that task.
# Weights are kept as DATA (not buried inside a formula) so they can be:
#   - hashed directly into the run_fingerprint — a full, honest capture; the
#     fingerprint changes whenever ANY weight changes, not just correctness
#   - changed in one place without touching scoring logic
# Every profile sums to 1.0, so a perfect response scores exactly 1.0.
# Dimensions: correctness, reasoning, clarity, hallucination_free (each 0–1).
CATEGORY_WEIGHTS = {
    "architecture":     {"correctness": 0.4, "reasoning": 0.4, "hallucination_free": 0.2},  # design: correct + sound reasoning + no fabricated tech
    "coding":           {"correctness": 0.6, "reasoning": 0.3, "clarity": 0.1},             # code: correctness dominates
    "debugging":        {"correctness": 0.5, "reasoning": 0.4, "clarity": 0.1},             # debug: right root cause + reasoning
    "summarization":    {"clarity": 0.4, "correctness": 0.3, "hallucination_free": 0.3},    # summary: clear + accurate + no hallucination
    "business_writing": {"clarity": 0.4, "correctness": 0.3, "reasoning": 0.2, "hallucination_free": 0.1},  # writing: clear + correct + reasoned
    "agent_planning":   {"reasoning": 0.4, "correctness": 0.4, "clarity": 0.2},             # planning: sound reasoning + correct steps
}

# Fallback for any unknown category — balanced correctness/reasoning/clarity.
DEFAULT_WEIGHTS = {"correctness": 0.4, "reasoning": 0.4, "clarity": 0.2}


# -----------------------------
# CATEGORY SCORE (task-aware quality)
# -----------------------------
def category_score(category, judge_scores=None):
    """
    Returns a single 0–1 quality score for a category using task-appropriate
    weights. All categories are judge-scored — no ground-truth metrics used.

    Args:
        category:     one of the 6 CEO categories
        judge_scores: dict with keys correctness/reasoning/clarity/hallucination_free

    Returns:
        float 0–1, clamped and rounded to 4 decimal places.
        Returns 0.0 if judge_scores is missing.
    """
    if not judge_scores:
        return 0.0

    # Dot-product the judge dimensions with this category's weights.
    weights = CATEGORY_WEIGHTS.get(category, DEFAULT_WEIGHTS)
    score = sum(judge_scores.get(dim, 0.0) * w for dim, w in weights.items())

    return round(min(max(score, 0.0), 1.0), 4)  # clamp to [0,1] for safety


# -----------------------------
# FINAL SCORE = overall quality (the rank metric)
# -----------------------------
def compute_final_score(category_scores, all_categories=None):
    """
    A model's overall quality = the average of its per-category quality scores.

    Each category_score is already task-appropriately weighted, so a simple
    average gives one comparable 0–1 number used to rank the leaderboard.
    Quality is the only thing in here — cost and latency are shown as separate
    columns, never blended in.

    Args:
        category_scores: {category: float} from the model's scored categories.
        all_categories:  optional fixed list (e.g. config.CATEGORIES). When
                         provided the denominator is always len(all_categories)
                         and any missing category counts as 0.0 — so a model
                         that fails a whole category is penalised rather than
                         rewarded with a smaller denominator. Pass None to use
                         the old variable-denominator behaviour (backward compat).

    Returns 0.0 if the model produced no category scores at all.
    """
    if not category_scores:
        return 0.0
    if all_categories:
        scores = [category_scores.get(c, 0.0) for c in all_categories]
        return round(sum(scores) / len(all_categories), 4)
    # fallback: average over whatever scored (variable denominator)
    return round(sum(category_scores.values()) / len(category_scores), 4)
