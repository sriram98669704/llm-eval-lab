"""
router.py — Picks the best model for a given task category.

Given the benchmark results for all models, returns the model that scores
highest on a category using its task-aware category_score (the same 0–1 signal
shown in the strength profile). Works for every category type — including
gsm8k and humaneval, which earlier judge-only routing couldn't handle.

Routing score = 0.70 * category_score + 0.20 * stability + 0.10 * speed
"""


# -----------------------------
# MODEL ROUTER
# -----------------------------
def route_model(category, results):
    """
    Routes a category task to the best model using:

    1. Category score (0.70) — task-appropriate performance signal
       Uses category_scores from Phase 4, works for ALL categories
       including gsm8k (accuracy) and humaneval (pass@1)
    2. Stability (0.20) — consistency across runs
    3. Speed (0.10) — latency

    Returns the best model name, or None if no data available.
    """

    best_model = None
    best_score = float("-inf")

    for model, stats in results.items():

        # -----------------------------
        # 1. CATEGORY SCORE (PRIMARY)
        # -----------------------------
        cat_score = stats.get("category_scores", {}).get(category)

        if cat_score is None:
            continue  # no data for this category — skip

        # -----------------------------
        # 2. STABILITY SIGNAL
        # -----------------------------
        stability = stats.get("category_stabilities", {}).get(category, 0.0)

        # -----------------------------
        # 3. SPEED SIGNAL
        # -----------------------------
        speed = stats.get("normalized_speed", 0.0)

        # -----------------------------
        # 4. ROUTING SCORE
        # -----------------------------
        score = (
            0.70 * cat_score +
            0.20 * stability +
            0.10 * speed
        )

        if score > best_score:
            best_score = score
            best_model = model

    return best_model
