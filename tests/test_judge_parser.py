"""
tests/test_judge_parser.py — Unit tests for judge.parse_judge_json (no API calls).
"""

import pytest
from judge import parse_judge_json

KEYS = ["correctness", "reasoning", "clarity", "hallucination_free"]


def _valid(d):
    return d is not None and all(k in d for k in KEYS)


# --- Clean JSON ---

class TestCleanJson:
    def test_plain_json(self):
        raw = '{"correctness": 0.9, "reasoning": 0.8, "clarity": 0.7, "hallucination_free": 1.0}'
        result = parse_judge_json(raw)
        assert _valid(result)
        assert result["correctness"] == pytest.approx(0.9)

    def test_integer_values_cast_to_float(self):
        raw = '{"correctness": 1, "reasoning": 1, "clarity": 1, "hallucination_free": 1}'
        result = parse_judge_json(raw)
        assert _valid(result)
        assert isinstance(result["correctness"], float)


# --- Markdown fences ---

class TestMarkdownFences:
    def test_json_fenced_block(self):
        raw = '```json\n{"correctness": 0.8, "reasoning": 0.7, "clarity": 0.9, "hallucination_free": 0.6}\n```'
        assert _valid(parse_judge_json(raw))

    def test_plain_fenced_block(self):
        raw = '```\n{"correctness": 0.5, "reasoning": 0.5, "clarity": 0.5, "hallucination_free": 0.5}\n```'
        assert _valid(parse_judge_json(raw))

    def test_prose_before_json(self):
        raw = 'Here is my evaluation:\n{"correctness": 0.9, "reasoning": 0.85, "clarity": 0.8, "hallucination_free": 1.0}'
        assert _valid(parse_judge_json(raw))


# --- Trailing commas ---

class TestTrailingCommas:
    def test_trailing_comma_in_object(self):
        raw = '{"correctness": 0.8, "reasoning": 0.7, "clarity": 0.6, "hallucination_free": 0.9,}'
        assert _valid(parse_judge_json(raw))

    def test_trailing_comma_fenced(self):
        raw = '```json\n{"correctness": 0.8, "reasoning": 0.7, "clarity": 0.6, "hallucination_free": 0.9,}\n```'
        assert _valid(parse_judge_json(raw))


# --- Regex last resort ---

class TestRegexFallback:
    def test_extra_keys_ignored(self):
        raw = '{"correctness": 0.9, "reasoning": 0.8, "clarity": 0.7, "hallucination_free": 0.6, "extra": "ignored"}'
        result = parse_judge_json(raw)
        assert _valid(result)

    def test_missing_key_defaults_to_zero(self):
        # 3 of 4 keys present — hallucination_free missing, should default to 0.0
        raw = '{"correctness": 0.9, "reasoning": 0.8, "clarity": 0.7}'
        result = parse_judge_json(raw)
        assert _valid(result)
        assert result["hallucination_free"] == 0.0
        assert result["correctness"] == pytest.approx(0.9)


# --- Failure cases ---

class TestFailureCases:
    def test_none_input(self):
        assert parse_judge_json(None) is None

    def test_empty_string(self):
        assert parse_judge_json("") is None

    def test_plain_text_no_json(self):
        assert parse_judge_json("The response was great and clear.") is None

    def test_empty_object(self):
        assert parse_judge_json("{}") is None

    def test_totally_malformed(self):
        assert parse_judge_json("{{{not valid json at all}}}") is None
