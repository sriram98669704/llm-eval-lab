"""
tests/test_router.py — Unit tests for router.py logic (no API calls).

Tests cover:
  - _tolerance_for: tight vs default tolerance by category
  - route_model: picks the highest-quality model
  - derive_routing_table: quality_bar calc, eligible set, default selection,
    escalate_to assignment, speed-sensitive vs cost-sensitive categories
"""

import pytest
from router import route_model, derive_routing_table, _tolerance_for
from config import TIGHT_TOLERANCE, DEFAULT_TOLERANCE, TIGHT_CATEGORIES, SPEED_SENSITIVE


# --- Helpers ---

def _results(models_quality, cost=0.01, latency=1.0):
    """Build a minimal results dict from {model: quality_for_category}."""
    return {
        m: {
            "category_scores":   {"coding": q},
            "category_costs":    {"coding": cost},
            "category_latencies":{"coding": latency},
        }
        for m, q in models_quality.items()
    }


def _results_cat(category, models_quality, cost=0.01, latency=1.0):
    return {
        m: {
            "category_scores":   {category: q},
            "category_costs":    {category: cost},
            "category_latencies":{category: latency},
        }
        for m, q in models_quality.items()
    }


# --- _tolerance_for ---

class TestToleranceFor:
    def test_tight_for_coding(self):
        assert _tolerance_for("coding") == TIGHT_TOLERANCE

    def test_tight_for_architecture(self):
        assert _tolerance_for("architecture") == TIGHT_TOLERANCE

    def test_default_for_summarization(self):
        assert _tolerance_for("summarization") == DEFAULT_TOLERANCE

    def test_default_for_business_writing(self):
        assert _tolerance_for("business_writing") == DEFAULT_TOLERANCE

    def test_default_for_unknown(self):
        assert _tolerance_for("some_unknown_category") == DEFAULT_TOLERANCE


# --- route_model ---

class TestRouteModel:
    def test_returns_highest_quality_model(self):
        results = _results({"gpt-4o": 0.9, "claude": 0.8, "deepseek": 0.7})
        assert route_model("coding", results) == "gpt-4o"

    def test_returns_none_for_missing_category(self):
        results = _results({"gpt-4o": 0.9})
        assert route_model("debugging", results) is None

    def test_returns_none_for_empty_results(self):
        assert route_model("coding", {}) is None

    def test_handles_none_stats(self):
        results = {"gpt-4o": None, "claude": {"category_scores": {"coding": 0.8}, "category_costs": {}, "category_latencies": {}}}
        assert route_model("coding", results) == "claude"

    def test_single_model(self):
        results = _results({"gpt-4o": 0.75})
        assert route_model("coding", results) == "gpt-4o"


# --- derive_routing_table ---

class TestDeriveRoutingTable:
    def test_quality_bar_calculation(self):
        results = _results({"gpt-4o": 0.9, "claude": 0.8})
        table = derive_routing_table(results)
        cat = table["policy"]["coding"]
        # bar = best_quality - tight_tolerance = 0.9 - 0.03 = 0.87
        assert cat["quality_bar"] == pytest.approx(0.87, abs=1e-4)

    def test_best_model_always_clears_bar(self):
        results = _results({"gpt-4o": 0.9, "claude": 0.85, "deepseek": 0.7})
        table = derive_routing_table(results)
        cat = table["policy"]["coding"]
        assert cat["models"]["gpt-4o"]["clears_bar"] is True

    def test_escalate_to_is_highest_quality(self):
        results = _results({"gpt-4o": 0.9, "claude": 0.8})
        table = derive_routing_table(results)
        assert table["policy"]["coding"]["escalate_to"] == "gpt-4o"

    def test_default_is_cheapest_for_non_speed_sensitive(self):
        results = {
            "cheap_model": {
                "category_scores":    {"debugging": 0.88},
                "category_costs":     {"debugging": 0.005},
                "category_latencies": {"debugging": 2.0},
            },
            "expensive_model": {
                "category_scores":    {"debugging": 0.90},
                "category_costs":     {"debugging": 0.05},
                "category_latencies": {"debugging": 1.0},
            },
        }
        table = derive_routing_table(results)
        cat = table["policy"]["debugging"]
        # expensive is best, cheap clears bar (0.90 - 0.05 = 0.85, cheap = 0.88 >= 0.85)
        assert cat["default"] == "cheap_model"
        assert cat["selection"] == "cheapest"

    def test_default_is_fastest_for_speed_sensitive(self):
        results = {
            "fast_model": {
                "category_scores":    {"summarization": 0.88},
                "category_costs":     {"summarization": 0.05},
                "category_latencies": {"summarization": 0.5},
            },
            "slow_model": {
                "category_scores":    {"summarization": 0.90},
                "category_costs":     {"summarization": 0.005},
                "category_latencies": {"summarization": 3.0},
            },
        }
        table = derive_routing_table(results)
        cat = table["policy"]["summarization"]
        assert cat["default"] == "fast_model"
        assert cat["selection"] == "fastest"

    def test_model_below_bar_not_eligible(self):
        results = _results({"gpt-4o": 0.9, "claude": 0.8, "deepseek": 0.5})
        table = derive_routing_table(results)
        cat = table["policy"]["coding"]
        # bar = 0.9 - 0.03 = 0.87 → deepseek (0.5) and claude (0.8) don't clear it
        assert cat["models"]["deepseek"]["clears_bar"] is False
        assert cat["models"]["claude"]["clears_bar"] is False

    def test_table_has_generated_at(self):
        results = _results({"gpt-4o": 0.9})
        table = derive_routing_table(results)
        assert "generated_at" in table

    def test_empty_results_produces_empty_policy(self):
        table = derive_routing_table({})
        assert table["policy"] == {}
