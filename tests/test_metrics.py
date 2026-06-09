"""
tests/test_metrics.py — Unit tests for metrics.py (pure functions, no API calls).
"""

import pytest
from metrics import category_score, compute_final_score, score_std, CATEGORY_WEIGHTS


PERFECT = {"correctness": 1.0, "reasoning": 1.0, "clarity": 1.0, "hallucination_free": 1.0}
ZERO    = {"correctness": 0.0, "reasoning": 0.0, "clarity": 0.0, "hallucination_free": 0.0}


# --- category_score ---

class TestCategoryScore:
    def test_perfect_scores_all_categories(self):
        for cat in CATEGORY_WEIGHTS:
            assert category_score(cat, PERFECT) == 1.0, f"failed for {cat}"

    def test_zero_scores_all_categories(self):
        for cat in CATEGORY_WEIGHTS:
            assert category_score(cat, ZERO) == 0.0, f"failed for {cat}"

    def test_missing_judge_scores_returns_zero(self):
        assert category_score("coding", None) == 0.0
        assert category_score("coding", {}) == 0.0

    def test_unknown_category_uses_default_weights(self):
        # Default = correctness 0.4, reasoning 0.4, clarity 0.2 — all 1.0 → 1.0
        assert category_score("unknown_cat", PERFECT) == 1.0

    def test_coding_correctness_dominates(self):
        # coding weights: correctness 0.6, reasoning 0.3, clarity 0.1
        scores = {"correctness": 1.0, "reasoning": 0.0, "clarity": 0.0, "hallucination_free": 0.0}
        assert category_score("coding", scores) == 0.6

    def test_summarization_clarity_and_hallucination(self):
        # summarization: clarity 0.4, correctness 0.3, hallucination_free 0.3
        scores = {"clarity": 1.0, "correctness": 0.0, "hallucination_free": 1.0, "reasoning": 0.0}
        assert category_score("summarization", scores) == pytest.approx(0.7, abs=1e-4)

    def test_clamped_above_one(self):
        # Shouldn't happen with valid 0–1 inputs, but safety clamp should hold
        over = {"correctness": 2.0, "reasoning": 2.0, "clarity": 2.0, "hallucination_free": 2.0}
        assert category_score("coding", over) <= 1.0

    def test_partial_dims_only_weights_present_keys(self):
        # Only provide correctness — other dims default to 0
        scores = {"correctness": 1.0}
        result = category_score("coding", scores)
        # coding: correctness 0.6 + reasoning 0 + clarity 0 = 0.6
        assert result == pytest.approx(0.6, abs=1e-4)


# --- compute_final_score ---

class TestComputeFinalScore:
    def test_empty_returns_zero(self):
        assert compute_final_score({}) == 0.0
        assert compute_final_score(None) == 0.0

    def test_single_category(self):
        assert compute_final_score({"coding": 0.8}) == 0.8

    def test_average_of_multiple(self):
        scores = {"coding": 0.8, "debugging": 0.6, "architecture": 0.7}
        assert compute_final_score(scores) == pytest.approx(0.7, abs=1e-4)

    def test_all_categories_fixed_denominator(self):
        from config import CATEGORIES
        # Model that only scored 3 of 6 — missing ones count as 0
        scores = {CATEGORIES[0]: 0.9, CATEGORIES[1]: 0.9, CATEGORIES[2]: 0.9}
        result = compute_final_score(scores, all_categories=CATEGORIES)
        expected = (0.9 * 3) / 6
        assert result == pytest.approx(expected, abs=1e-4)

    def test_fixed_denominator_penalises_missing_categories(self):
        from config import CATEGORIES
        partial = {CATEGORIES[0]: 1.0}
        full    = {c: 1.0 for c in CATEGORIES}
        assert compute_final_score(partial, all_categories=CATEGORIES) < compute_final_score(full, all_categories=CATEGORIES)

    def test_variable_denominator_fallback(self):
        # Without all_categories, denominator = len of provided dict
        scores = {"coding": 1.0, "debugging": 0.0}
        assert compute_final_score(scores) == pytest.approx(0.5, abs=1e-4)


# --- score_std ---

class TestScoreStd:
    def test_single_score_returns_zero(self):
        assert score_std([0.8]) == 0.0

    def test_empty_returns_zero(self):
        assert score_std([]) == 0.0

    def test_identical_scores_zero_std(self):
        assert score_std([0.7, 0.7, 0.7]) == 0.0

    def test_known_std(self):
        # [0.0, 1.0] → mean 0.5, variance 0.25, std 0.5
        assert score_std([0.0, 1.0]) == pytest.approx(0.5, abs=1e-4)

    def test_larger_spread_higher_std(self):
        tight = [0.7, 0.72, 0.71]
        wide  = [0.2, 0.9, 0.5]
        assert score_std(wide) > score_std(tight)
