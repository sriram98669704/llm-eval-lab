"""
tests/test_dispatcher_routing.py — Tests for dispatcher.call_model SDK routing.

No real API calls — patches the three private _call_* functions to return a
fixed dict so we can verify the correct SDK is selected by model-name prefix.
"""

import pytest
from unittest.mock import patch, MagicMock

FAKE_RESULT = {
    "text": "hello",
    "input_tokens": 10,
    "output_tokens": 5,
    "latency": 0.5,
}


def _fake_cost(model, inp, out):
    return 0.001


# --- SDK routing by model-name prefix ---

class TestCallModelRouting:
    def test_gpt_prefix_routes_to_openai(self):
        with patch("dispatcher._call_openai", return_value=FAKE_RESULT) as mock, \
             patch("dispatcher.calculate_cost", side_effect=_fake_cost):
            from dispatcher import call_model
            result = call_model("gpt-4o", "test prompt")
            mock.assert_called_once_with("gpt-4o", "test prompt")
            assert result is not None

    def test_claude_prefix_routes_to_anthropic(self):
        with patch("dispatcher._call_anthropic", return_value=FAKE_RESULT) as mock, \
             patch("dispatcher.calculate_cost", side_effect=_fake_cost):
            from dispatcher import call_model
            result = call_model("claude-sonnet-4-5", "test prompt")
            mock.assert_called_once_with("claude-sonnet-4-5", "test prompt")
            assert result is not None

    def test_deepseek_prefix_routes_to_together(self):
        with patch("dispatcher._call_together", return_value=FAKE_RESULT) as mock, \
             patch("dispatcher.calculate_cost", side_effect=_fake_cost):
            from dispatcher import call_model
            result = call_model("deepseek-ai/DeepSeek-V4-Pro", "test prompt")
            mock.assert_called_once_with("deepseek-ai/DeepSeek-V4-Pro", "test prompt")
            assert result is not None

    def test_unknown_prefix_returns_none(self):
        with patch("dispatcher.calculate_cost", side_effect=_fake_cost):
            from dispatcher import call_model
            result = call_model("llama-3/something", "test prompt")
            assert result is None

    def test_result_contains_cost_field(self):
        with patch("dispatcher._call_openai", return_value=FAKE_RESULT), \
             patch("dispatcher.calculate_cost", return_value=0.002):
            from dispatcher import call_model
            result = call_model("gpt-4o", "test")
            assert "cost_usd" in result
            assert result["cost_usd"] == 0.002

    def test_sdk_failure_returns_none(self):
        with patch("dispatcher._call_openai", side_effect=Exception("network error")), \
             patch("dispatcher.calculate_cost", side_effect=_fake_cost):
            from dispatcher import call_model
            result = call_model("gpt-4o", "test")
            assert result is None
