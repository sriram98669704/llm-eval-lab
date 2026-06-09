"""
tests/test_report.py — Unit tests for report.generate_report.

No API calls — pure function over synthetic dicts.
"""

import pytest
from report import generate_report, _short, _md_table
from findings import compute_findings

CATS = ["architecture", "coding", "debugging", "summarization", "business_writing", "agent_planning"]


def _make_model(final_score, cat_scores, cat_costs):
    return {
        "final_score": final_score,
        "avg_latency": 15.0,
        "category_scores": cat_scores,
        "category_costs": cat_costs,
        "category_score_stds": {k: 0.01 for k in cat_scores},
        "category_latencies": {k: 15.0 for k in cat_scores},
    }


def _make_summary():
    scores = {c: 0.85 for c in CATS}
    return {
        "run_at": "2026-06-09T12:00:00",
        "models": {
            "gpt-4o":                      _make_model(0.79, {c: 0.79 for c in CATS}, {c: 0.008 for c in CATS}),
            "claude-sonnet-4-5":           _make_model(0.91, {c: 0.91 for c in CATS}, {c: 0.050 for c in CATS}),
            "deepseek-ai/DeepSeek-V4-Pro": _make_model(0.92, {c: 0.92 for c in CATS}, {c: 0.003 for c in CATS}),
        },
    }


def _make_routing():
    return {
        "policy": {
            cat: {
                "default": "deepseek-ai/DeepSeek-V4-Pro",
                "escalate_to": "claude-sonnet-4-5" if cat == "coding" else "deepseek-ai/DeepSeek-V4-Pro",
                "quality_bar": 0.87,
                "selection": "fastest" if cat == "summarization" else "cheapest",
            }
            for cat in CATS
        }
    }


def _make_meta():
    return {
        "run_fingerprint": "abc123def456abc123def456abc123def456abc123def456abc123def456abc1",
        "prompt_count": 18,
        "code": {"git_sha": "a1b2c3d", "git_branch": "main", "git_dirty": False},
    }


# ── _short ────────────────────────────────────────────────────────────────────

class TestShort:
    def test_gpt(self):     assert _short("gpt-4o") == "GPT-4o"
    def test_claude(self):  assert _short("claude-sonnet-4-5") == "Claude Sonnet"
    def test_deepseek(self): assert _short("deepseek-ai/DeepSeek-V4-Pro") == "DeepSeek V4 Pro"
    def test_unknown(self): assert _short("llama-3") == "llama-3"


# ── _md_table ─────────────────────────────────────────────────────────────────

class TestMdTable:
    def test_contains_headers(self):
        out = _md_table(["A", "B"], [["1", "2"]])
        assert "| A | B |" in out

    def test_contains_separator(self):
        out = _md_table(["Col"], [["val"]])
        assert "---" in out

    def test_contains_row_values(self):
        out = _md_table(["X"], [["hello"]])
        assert "hello" in out

    def test_multiple_rows(self):
        out = _md_table(["H"], [["r1"], ["r2"], ["r3"]])
        assert out.count("r") == 3


# ── generate_report — structure ───────────────────────────────────────────────

class TestReportStructure:
    def _report(self):
        s = _make_summary()
        r = _make_routing()
        f = compute_findings(s, r)
        m = _make_meta()
        return generate_report(s, r, f, m)

    def test_returns_string(self):
        assert isinstance(self._report(), str)

    def test_non_empty(self):
        assert len(self._report()) > 100

    def test_has_title(self):
        assert "# LLM Eval Lab" in self._report()

    def test_has_key_findings_section(self):
        assert "## Key Findings" in self._report()

    def test_has_leaderboard_section(self):
        assert "## Model Leaderboard" in self._report()

    def test_has_per_category_section(self):
        assert "## Per-Category Scores" in self._report()

    def test_has_routing_section(self):
        assert "## Routing Policy" in self._report()

    def test_has_category_winners_section(self):
        assert "## Category Winners" in self._report()

    def test_ends_with_newline(self):
        assert self._report().endswith("\n")


# ── generate_report — content ─────────────────────────────────────────────────

class TestReportContent:
    def _report(self):
        s = _make_summary()
        r = _make_routing()
        f = compute_findings(s, r)
        return generate_report(s, r, f, None)

    def test_contains_model_short_names(self):
        out = self._report()
        assert "GPT-4o" in out
        assert "Claude Sonnet" in out
        assert "DeepSeek V4 Pro" in out

    def test_contains_run_date(self):
        assert "2026-06-09" in self._report()

    def test_contains_winner(self):
        assert "Overall winner" in self._report()

    def test_contains_biggest_gap(self):
        assert "Biggest category gap" in self._report() or "biggest" in self._report().lower()

    def test_contains_best_quality_per_dollar(self):
        assert "Best quality per dollar" in self._report()

    def test_contains_all_categories(self):
        out = self._report()
        for cat in CATS:
            assert cat.replace("_", " ").title() in out

    def test_contains_quality_score_explanation(self):
        assert "Quality score" in self._report()

    def test_contains_footer_disclaimer(self):
        assert "Report generated" in self._report()


# ── generate_report — meta handling ──────────────────────────────────────────

class TestReportMetaHandling:
    def test_none_meta_does_not_crash(self):
        s = _make_summary()
        r = _make_routing()
        f = compute_findings(s, r)
        out = generate_report(s, r, f, None)
        assert "# LLM Eval Lab" in out

    def test_meta_fingerprint_included(self):
        s = _make_summary()
        r = _make_routing()
        f = compute_findings(s, r)
        m = _make_meta()
        out = generate_report(s, r, f, m)
        assert "abc123def456" in out

    def test_meta_git_sha_included(self):
        s = _make_summary()
        r = _make_routing()
        f = compute_findings(s, r)
        m = _make_meta()
        out = generate_report(s, r, f, m)
        assert "a1b2c3d" in out

    def test_meta_prompt_count_included(self):
        s = _make_summary()
        r = _make_routing()
        f = compute_findings(s, r)
        m = _make_meta()
        out = generate_report(s, r, f, m)
        assert "18" in out


# ── generate_report — routing section ────────────────────────────────────────

class TestReportRouting:
    def test_routing_table_present_when_policy_exists(self):
        s = _make_summary()
        r = _make_routing()
        f = compute_findings(s, r)
        out = generate_report(s, r, f, None)
        assert "Quality Bar" in out

    def test_no_routing_shows_fallback_message(self):
        s = _make_summary()
        f = compute_findings(s, None)
        out = generate_report(s, None, f, None)
        assert "No routing policy found" in out
