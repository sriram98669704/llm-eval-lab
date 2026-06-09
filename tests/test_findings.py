"""
tests/test_findings.py — Unit tests for findings.compute_findings.

No API calls — pure function over synthetic dicts.
"""

import pytest
from findings import compute_findings, _short_name, _avg_cost

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_model(final_score, cat_scores, cat_costs, latency=10.0):
    """Build a minimal model entry matching the summary.json shape."""
    return {
        "final_score": final_score,
        "avg_latency": latency,
        "category_scores": cat_scores,
        "category_costs": cat_costs,
        "category_score_stds": {k: 0.0 for k in cat_scores},
        "category_latencies": {k: latency for k in cat_scores},
    }


CATS = ["architecture", "coding", "debugging", "summarization", "business_writing", "agent_planning"]


def _make_summary(gpt_scores, claude_scores, ds_scores,
                  gpt_costs=None, claude_costs=None, ds_costs=None):
    gpt_costs    = gpt_costs    or {c: 0.008 for c in CATS}
    claude_costs = claude_costs or {c: 0.050 for c in CATS}
    ds_costs     = ds_costs     or {c: 0.003 for c in CATS}
    gpt_final    = sum(gpt_scores.values())    / len(gpt_scores)
    claude_final = sum(claude_scores.values()) / len(claude_scores)
    ds_final     = sum(ds_scores.values())     / len(ds_scores)
    return {
        "models": {
            "gpt-4o":                      _make_model(gpt_final,    gpt_scores,    gpt_costs),
            "claude-sonnet-4-5":           _make_model(claude_final, claude_scores, claude_costs),
            "deepseek-ai/DeepSeek-V4-Pro": _make_model(ds_final,     ds_scores,     ds_costs),
        }
    }


def _make_routing(defaults):
    """
    defaults: dict of {category: (default_model, escalate_to_model)}
    """
    policy = {}
    for cat, (default, escalate) in defaults.items():
        policy[cat] = {
            "default": default,
            "escalate_to": escalate,
            "quality_bar": 0.85,
        }
    return {"policy": policy}


# ── _short_name ───────────────────────────────────────────────────────────────

class TestShortName:
    def test_gpt(self):
        assert _short_name("gpt-4o") == "GPT-4o"

    def test_claude(self):
        assert _short_name("claude-sonnet-4-5") == "Claude Sonnet"

    def test_deepseek(self):
        assert _short_name("deepseek-ai/DeepSeek-V4-Pro") == "DeepSeek V4 Pro"

    def test_unknown_returns_as_is(self):
        assert _short_name("llama-3/something") == "llama-3/something"


# ── _avg_cost ─────────────────────────────────────────────────────────────────

class TestAvgCost:
    def test_basic_average(self):
        data = {"category_costs": {"a": 0.010, "b": 0.020}}
        assert _avg_cost(data) == pytest.approx(0.015)

    def test_empty_returns_one(self):
        assert _avg_cost({}) == 1.0

    def test_single_category(self):
        data = {"category_costs": {"coding": 0.007}}
        assert _avg_cost(data) == pytest.approx(0.007)


# ── compute_findings — winner ─────────────────────────────────────────────────

class TestWinner:
    def _run(self):
        gpt    = {c: 0.79 for c in CATS}
        claude = {c: 0.91 for c in CATS}
        ds     = {c: 0.88 for c in CATS}
        return compute_findings(_make_summary(gpt, claude, ds), None)

    def test_winner_is_highest_final_score(self):
        f = self._run()
        assert f["winner"]["short_name"] == "Claude Sonnet"

    def test_runner_up_is_second(self):
        f = self._run()
        assert f["runner_up"]["short_name"] == "DeepSeek V4 Pro"

    def test_trailing_is_lowest(self):
        f = self._run()
        assert f["trailing"]["short_name"] == "GPT-4o"

    def test_winner_score_correct(self):
        f = self._run()
        assert f["winner"]["score"] == pytest.approx(0.91)


# ── compute_findings — biggest gap ───────────────────────────────────────────

class TestBiggestGap:
    def test_identifies_category_with_largest_spread(self):
        # coding: gpt=0.5, others=0.9 → gap 0.4 — should win
        # all other cats: uniform 0.85 → gap 0
        gpt    = {c: 0.85 for c in CATS}
        claude = {c: 0.90 for c in CATS}
        ds     = {c: 0.90 for c in CATS}
        gpt["coding"] = 0.50
        f = compute_findings(_make_summary(gpt, claude, ds), None)
        assert f["biggest_gap"]["category"] == "coding"

    def test_gap_value_correct(self):
        gpt    = {c: 0.85 for c in CATS}
        claude = {c: 0.85 for c in CATS}
        ds     = {c: 0.85 for c in CATS}
        gpt["architecture"] = 0.60   # gap = 0.25
        f = compute_findings(_make_summary(gpt, claude, ds), None)
        assert f["biggest_gap"]["gap"] == pytest.approx(0.25)

    def test_best_and_worst_models_identified(self):
        gpt    = {c: 0.85 for c in CATS}
        claude = {c: 0.95 for c in CATS}
        ds     = {c: 0.85 for c in CATS}
        gpt["summarization"] = 0.55
        f = compute_findings(_make_summary(gpt, claude, ds), None)
        assert f["biggest_gap"]["best_short"] == "Claude Sonnet"
        assert f["biggest_gap"]["worst_short"] == "GPT-4o"


# ── compute_findings — best value ────────────────────────────────────────────

class TestBestValue:
    def test_high_quality_low_cost_wins(self):
        # DS: high quality (0.92) at very low cost (0.001) → highest ratio
        gpt    = {c: 0.80 for c in CATS}
        claude = {c: 0.91 for c in CATS}
        ds     = {c: 0.92 for c in CATS}
        gpt_costs    = {c: 0.008 for c in CATS}
        claude_costs = {c: 0.050 for c in CATS}
        ds_costs     = {c: 0.001 for c in CATS}
        f = compute_findings(
            _make_summary(gpt, claude, ds, gpt_costs, claude_costs, ds_costs),
            None
        )
        assert f["best_value"]["short_name"] == "DeepSeek V4 Pro"

    def test_best_value_contains_quality_and_cost(self):
        gpt    = {c: 0.80 for c in CATS}
        claude = {c: 0.91 for c in CATS}
        ds     = {c: 0.92 for c in CATS}
        f = compute_findings(_make_summary(gpt, claude, ds), None)
        bv = f["best_value"]
        assert "quality" in bv
        assert "avg_cost" in bv
        assert bv["avg_cost"] > 0


# ── compute_findings — escalation ────────────────────────────────────────────

class TestEscalation:
    def _routing_mixed(self):
        # coding + summarization have different default vs escalate_to
        defaults = {
            "architecture":    ("deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro"),
            "coding":          ("deepseek-ai/DeepSeek-V4-Pro", "claude-sonnet-4-5"),
            "debugging":       ("claude-sonnet-4-5",           "claude-sonnet-4-5"),
            "summarization":   ("gpt-4o",                      "claude-sonnet-4-5"),
            "business_writing":("deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro"),
            "agent_planning":  ("deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro"),
        }
        return _make_routing(defaults)

    def test_escalation_categories_count(self):
        gpt = claude = ds = {c: 0.85 for c in CATS}
        f = compute_findings(_make_summary(gpt, claude, ds), self._routing_mixed())
        assert len(f["escalation_categories"]) == 2

    def test_escalation_category_names(self):
        gpt = claude = ds = {c: 0.85 for c in CATS}
        f = compute_findings(_make_summary(gpt, claude, ds), self._routing_mixed())
        assert set(f["escalation_categories"]) == {"coding", "summarization"}

    def test_no_escalation_categories_count(self):
        gpt = claude = ds = {c: 0.85 for c in CATS}
        f = compute_findings(_make_summary(gpt, claude, ds), self._routing_mixed())
        assert len(f["no_escalation_categories"]) == 4

    def test_no_routing_data_returns_empty_lists(self):
        gpt = claude = ds = {c: 0.85 for c in CATS}
        f = compute_findings(_make_summary(gpt, claude, ds), None)
        assert f["escalation_categories"] == []
        assert f["no_escalation_categories"] == []


# ── compute_findings — category winners ──────────────────────────────────────

class TestCategoryWinners:
    def test_each_category_has_a_winner(self):
        gpt    = {c: 0.80 for c in CATS}
        claude = {c: 0.85 for c in CATS}
        ds     = {c: 0.90 for c in CATS}
        f = compute_findings(_make_summary(gpt, claude, ds), None)
        assert set(f["category_winners"].keys()) == set(CATS)

    def test_winner_per_category_correct(self):
        gpt    = {c: 0.80 for c in CATS}
        claude = {c: 0.85 for c in CATS}
        ds     = {c: 0.90 for c in CATS}
        # Give Claude the win on coding
        claude["coding"] = 0.95
        f = compute_findings(_make_summary(gpt, claude, ds), None)
        assert f["category_winners"]["coding"]["short_name"] == "Claude Sonnet"
        # DeepSeek wins everything else
        for cat in CATS:
            if cat != "coding":
                assert f["category_winners"][cat]["short_name"] == "DeepSeek V4 Pro"


# ── edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_summary_returns_empty_dict(self):
        assert compute_findings({}, None) == {}

    def test_no_models_key_returns_empty_dict(self):
        assert compute_findings({"run_at": "2026-01-01"}, None) == {}

    def test_none_routing_handled_gracefully(self):
        gpt = claude = ds = {c: 0.85 for c in CATS}
        f = compute_findings(_make_summary(gpt, claude, ds), None)
        assert "winner" in f
        assert f["escalation_categories"] == []

    def test_all_models_equal_score(self):
        gpt = claude = ds = {c: 0.85 for c in CATS}
        f = compute_findings(_make_summary(gpt, claude, ds), None)
        # Should still return a winner (arbitrary but deterministic)
        assert f["winner"]["score"] == pytest.approx(0.85)
        assert f["biggest_gap"]["gap"] == pytest.approx(0.0)
