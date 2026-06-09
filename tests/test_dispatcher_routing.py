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
            mock.assert_called_once_with("gpt-4o", "test prompt", api_key=None)
            assert result is not None

    def test_claude_prefix_routes_to_anthropic(self):
        with patch("dispatcher._call_anthropic", return_value=FAKE_RESULT) as mock, \
             patch("dispatcher.calculate_cost", side_effect=_fake_cost):
            from dispatcher import call_model
            result = call_model("claude-sonnet-4-5", "test prompt")
            mock.assert_called_once_with("claude-sonnet-4-5", "test prompt", api_key=None)
            assert result is not None

    def test_deepseek_prefix_routes_to_together(self):
        with patch("dispatcher._call_together", return_value=FAKE_RESULT) as mock, \
             patch("dispatcher.calculate_cost", side_effect=_fake_cost):
            from dispatcher import call_model
            result = call_model("deepseek-ai/DeepSeek-V4-Pro", "test prompt")
            mock.assert_called_once_with("deepseek-ai/DeepSeek-V4-Pro", "test prompt", api_key=None)
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


# --- BYOK: per-session keys are forwarded explicitly, never via os.environ ---

class TestCallModelBYOKKeys:
    def test_openai_key_forwarded(self):
        with patch("dispatcher._call_openai", return_value=FAKE_RESULT) as mock, \
             patch("dispatcher.calculate_cost", side_effect=_fake_cost):
            from dispatcher import call_model
            call_model("gpt-4o", "p", keys={"openai": "sk-session-123"})
            mock.assert_called_once_with("gpt-4o", "p", api_key="sk-session-123")

    def test_anthropic_key_forwarded(self):
        with patch("dispatcher._call_anthropic", return_value=FAKE_RESULT) as mock, \
             patch("dispatcher.calculate_cost", side_effect=_fake_cost):
            from dispatcher import call_model
            call_model("claude-sonnet-4-5", "p", keys={"anthropic": "sk-ant-session"})
            mock.assert_called_once_with("claude-sonnet-4-5", "p", api_key="sk-ant-session")

    def test_together_key_forwarded(self):
        with patch("dispatcher._call_together", return_value=FAKE_RESULT) as mock, \
             patch("dispatcher.calculate_cost", side_effect=_fake_cost):
            from dispatcher import call_model
            call_model("deepseek-ai/DeepSeek-V4-Pro", "p", keys={"together": "tog-session"})
            mock.assert_called_once_with("deepseek-ai/DeepSeek-V4-Pro", "p", api_key="tog-session")

    def test_only_relevant_provider_key_used(self):
        # Passing all three keys must forward ONLY the matching provider's key.
        with patch("dispatcher._call_openai", return_value=FAKE_RESULT) as mock, \
             patch("dispatcher.calculate_cost", side_effect=_fake_cost):
            from dispatcher import call_model
            call_model("gpt-4o", "p", keys={
                "openai": "sk-oai", "anthropic": "sk-ant", "together": "tog",
            })
            mock.assert_called_once_with("gpt-4o", "p", api_key="sk-oai")

    def test_missing_provider_key_falls_back_to_none(self):
        # keys present but not for this provider → api_key=None → env fallback downstream.
        with patch("dispatcher._call_openai", return_value=FAKE_RESULT) as mock, \
             patch("dispatcher.calculate_cost", side_effect=_fake_cost):
            from dispatcher import call_model
            call_model("gpt-4o", "p", keys={"anthropic": "sk-ant"})
            mock.assert_called_once_with("gpt-4o", "p", api_key=None)
