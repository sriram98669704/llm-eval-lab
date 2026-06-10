"""
tests/test_byok.py — Unit tests for byok.py (BYOK key resolution + redaction).

No API calls, no Streamlit, no real environment. Pure functions over plain
dicts. The security-critical assertions:
  - os.environ is NEVER written to (the cross-session leak we refuse to create)
  - missing keys resolve to None (never silently fall through to someone else's)
  - redact() removes anything key-shaped before it could reach the UI or a log
"""

import os
import pytest
from byok import (
    PROVIDERS, ENV_VARS,
    keys_from_env, resolve_keys, validate_key_format, redact,
)


# ── keys_from_env ─────────────────────────────────────────────────────────────

class TestKeysFromEnv:
    def test_all_present(self):
        env = {"OPENAI_API_KEY": "sk-o", "ANTHROPIC_API_KEY": "sk-ant-a", "TOGETHER_API_KEY": "tgp_t"}
        keys = keys_from_env(env)
        assert keys == {"openai": "sk-o", "anthropic": "sk-ant-a", "together": "tgp_t"}

    def test_missing_become_none(self):
        assert keys_from_env({}) == {"openai": None, "anthropic": None, "together": None}

    def test_empty_string_becomes_none(self):
        env = {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "sk-ant-a", "TOGETHER_API_KEY": "tgp_t"}
        assert keys_from_env(env)["openai"] is None

    def test_does_not_mutate_input(self):
        env = {"OPENAI_API_KEY": "sk-o"}
        before = dict(env)
        keys_from_env(env)
        assert env == before

    def test_covers_all_providers(self):
        assert set(keys_from_env({}).keys()) == set(PROVIDERS)


# ── resolve_keys ──────────────────────────────────────────────────────────────

class TestResolveKeys:
    def test_deployed_no_keys_all_missing(self):
        env = keys_from_env({})
        keys, source, missing = resolve_keys(env, {})
        assert source == "byok"
        assert missing == sorted(PROVIDERS)
        assert all(keys[p] is None for p in PROVIDERS)

    def test_deployed_all_pasted(self):
        env = keys_from_env({})
        session = {"openai": "sk-o", "anthropic": "sk-ant-a", "together": "tgp_t"}
        keys, source, missing = resolve_keys(env, session)
        assert source == "byok"
        assert missing == []
        assert keys["openai"] == "sk-o"

    def test_local_env_complete_is_env_source(self):
        env = keys_from_env({
            "OPENAI_API_KEY": "sk-o", "ANTHROPIC_API_KEY": "sk-ant-a", "TOGETHER_API_KEY": "tgp_t",
        })
        keys, source, missing = resolve_keys(env, {})
        assert source == "env"
        assert missing == []

    def test_env_wins_over_session_when_both_present(self):
        env = keys_from_env({"OPENAI_API_KEY": "sk-env"})
        keys, _, _ = resolve_keys(env, {"openai": "sk-session"})
        assert keys["openai"] == "sk-env"

    def test_session_fills_gaps_left_by_env(self):
        env = keys_from_env({"OPENAI_API_KEY": "sk-o"})  # only openai in env
        session = {"anthropic": "sk-ant-a", "together": "tgp_t"}
        keys, source, missing = resolve_keys(env, session)
        assert missing == []
        assert source == "byok"  # not all three came from env
        assert keys["anthropic"] == "sk-ant-a"

    def test_partial_keys_reports_only_truly_missing(self):
        env = keys_from_env({})
        keys, source, missing = resolve_keys(env, {"openai": "sk-o"})
        assert missing == ["anthropic", "together"]

    def test_none_session_handled(self):
        env = keys_from_env({})
        keys, source, missing = resolve_keys(env, None)
        assert missing == sorted(PROVIDERS)

    def test_missing_provider_resolves_to_none_never_blank(self):
        # Critical: a missing key must be None so dispatcher has nothing to call,
        # rather than an empty string that might be mistaken for a value.
        env = keys_from_env({})
        keys, _, _ = resolve_keys(env, {"openai": ""})
        assert keys["openai"] is None


# ── the security invariant: os.environ is NEVER mutated ──────────────────────

class TestNeverMutatesEnviron:
    def test_keys_from_env_does_not_write_os_environ(self):
        before = dict(os.environ)
        keys_from_env(os.environ)
        assert dict(os.environ) == before

    def test_resolve_keys_does_not_write_os_environ(self):
        before = dict(os.environ)
        resolve_keys(keys_from_env(os.environ), {"openai": "sk-should-not-leak-to-env"})
        assert dict(os.environ) == before
        # And the pasted key never became an env var.
        assert os.environ.get("OPENAI_API_KEY") != "sk-should-not-leak-to-env"


# ── validate_key_format ───────────────────────────────────────────────────────

class TestValidateKeyFormat:
    def test_empty_rejected(self):
        ok, _ = validate_key_format("openai", "")
        assert ok is False

    def test_none_rejected(self):
        ok, _ = validate_key_format("openai", None)
        assert ok is False

    def test_too_short_rejected(self):
        ok, msg = validate_key_format("openai", "sk-1")
        assert ok is False
        assert "short" in msg

    def test_openai_without_prefix_rejected(self):
        ok, msg = validate_key_format("openai", "abcdefghijklmnop")
        assert ok is False
        assert "sk-" in msg

    def test_anthropic_without_prefix_rejected(self):
        ok, msg = validate_key_format("anthropic", "sk-abcdefghijklmnop")
        assert ok is False
        assert "sk-ant" in msg

    def test_good_openai_key(self):
        ok, msg = validate_key_format("openai", "sk-abcdefghijklmnop")
        assert ok is True

    def test_good_anthropic_key(self):
        ok, msg = validate_key_format("anthropic", "sk-ant-abcdefghijklmnop")
        assert ok is True

    def test_together_any_long_token(self):
        ok, _ = validate_key_format("together", "tgp_v1_abcdefghijklmnop")
        assert ok is True


# ── redact ────────────────────────────────────────────────────────────────────

class TestRedact:
    def test_redacts_openai_key(self):
        out = redact("Incorrect API key provided: sk-abc123def456ghi789")
        assert "sk-abc123def456ghi789" not in out
        assert "redacted" in out

    def test_redacts_anthropic_key(self):
        out = redact("auth failed for sk-ant-api03-abcdef123456")
        assert "sk-ant-api03-abcdef123456" not in out

    def test_redacts_together_key(self):
        out = redact("bad token tgp_v1_abcdef123456ghijkl")
        assert "tgp_v1_abcdef123456ghijkl" not in out

    def test_leaves_plain_text_unchanged(self):
        msg = "The default model failed to respond after 3 retries."
        assert redact(msg) == msg

    def test_handles_none(self):
        assert redact(None) is None

    def test_handles_empty(self):
        assert redact("") == ""

    def test_redacts_multiple_keys_in_one_string(self):
        out = redact("tried sk-aaaaaaaa and sk-ant-bbbbbbbb")
        assert "sk-aaaaaaaa" not in out
        assert "sk-ant-bbbbbbbb" not in out

    def test_redacts_key_in_realistic_error(self):
        # Mimics how an SDK auth error can echo a partial key.
        err = "AuthenticationError: Incorrect API key provided: sk-proj-AbCd1234EfGh5678. You can find your API key at ..."
        out = redact(err)
        assert "sk-proj-AbCd1234EfGh5678" not in out
        assert "Incorrect API key provided" in out  # the human part survives

    def test_redacts_bare_hex_together_key(self):
        # Legacy Together AI keys are bare 64-char hex — no tgp_ prefix.
        key = "a1b2c3d4" * 8  # 64 hex chars
        out = redact(f"invalid api key: {key}")
        assert key not in out
        assert "redacted" in out

    def test_redacts_uppercase_hex_key(self):
        key = "A1B2C3D4" * 8
        out = redact(f"invalid api key: {key}")
        assert key not in out

    def test_leaves_short_hex_alone(self):
        # A git SHA (40 hex chars) is not key-shaped — must survive.
        sha = "f" * 40
        msg = f"deployed commit {sha}"
        assert redact(msg) == msg

    def test_leaves_hex_inside_longer_token_alone(self):
        # 64 hex chars embedded in a longer word ≠ a key boundary.
        msg = "id_" + "a" * 64 + "z"
        assert redact(msg) == msg
