"""
byok.py — Bring-Your-Own-Keys resolution for the Live Test tab.

Pure functions — no Streamlit, no file I/O, and (critically) NO writes to
os.environ. The dashboard passes in the current environment and the keys a
visitor pasted this session; these functions decide which key to use per
provider and whether a live run can proceed.

Security model
--------------
  - Keys are only ever READ — from the environment (a local `.env`) or from the
    pasted session dict. They are NEVER written back to os.environ, a file, a
    cache, or a log. os.environ is process-global and shared across every
    visitor on a server instance, so writing a pasted key there could leak one
    session's key into another's. We never do that — keys flow as explicit
    function arguments down to the SDK client (see dispatcher.call_model).
  - On the deployed app there is no `.env`, so resolution falls through to the
    pasted session keys, scoped to that one browser session's RAM.

These functions are unit-tested in tests/test_byok.py with zero API calls.
"""

from __future__ import annotations

import re

PROVIDERS = ("openai", "anthropic", "together")

ENV_VARS = {
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "together":  "TOGETHER_API_KEY",
}


def keys_from_env(environ) -> dict:
    """
    Pull the three provider keys from an environment mapping (e.g. os.environ).

    Returns {provider: key_or_None}. Reads only — never mutates `environ`.
    Empty-string values are normalised to None so "present but blank" counts
    as missing.
    """
    return {p: (environ.get(ENV_VARS[p]) or None) for p in PROVIDERS}


def resolve_keys(env_keys: dict, session_keys: dict | None) -> tuple[dict, str, list]:
    """
    Decide which keys Live Test should use, env-first then pasted (BYOK).

    Parameters
    ----------
    env_keys     : {provider: key|None} from the environment (a local `.env`).
    session_keys : {provider: key|None} pasted this session (BYOK), or None.

    Per provider the environment wins if present, otherwise the pasted key.

    Returns
    -------
    (keys, source, missing)
      keys    : {provider: key|None} to forward to run_live (None where absent;
                dispatcher then has nothing to call → fails cleanly, never silently
                using someone else's key).
      source  : "env"  if all three keys came from the environment (no paste needed)
                "byok" otherwise (at least one key is pasted, or some are missing).
      missing : sorted list of providers with no key from either source.
    """
    session_keys = session_keys or {}
    keys = {p: (env_keys.get(p) or session_keys.get(p) or None) for p in PROVIDERS}

    missing = sorted(p for p in PROVIDERS if not keys[p])
    all_from_env = all(env_keys.get(p) for p in PROVIDERS)
    source = "env" if all_from_env else "byok"
    return keys, source, missing


def validate_key_format(provider: str, key: str | None) -> tuple[bool, str]:
    """
    Light, non-cryptographic sanity check so obvious junk fails fast with a clean
    message instead of bouncing off the provider. NEVER logs or echoes the key.

    Returns (ok, message). `ok=False` means the string is clearly not a key;
    it does NOT prove a key is valid — only the provider can do that.
    """
    if not key or not key.strip():
        return False, "empty"
    key = key.strip()
    if len(key) < 8:
        return False, "too short to be a real key"
    if provider == "openai" and not key.startswith("sk-"):
        return False, "OpenAI keys usually start with 'sk-'"
    if provider == "anthropic" and not key.startswith("sk-ant"):
        return False, "Anthropic keys usually start with 'sk-ant-'"
    return True, "ok"


# Order matters: match the more specific 'sk-ant-' before the generic 'sk-'.
_KEY_RE = re.compile(
    r"(sk-ant-[A-Za-z0-9_\-]{6,}|sk-[A-Za-z0-9_\-]{6,}|tgp_[A-Za-z0-9_\-]{6,})"
)


def redact(text):
    """
    Mask anything that looks like an API key in a string before it is shown in
    the UI or written to a log. Defence-in-depth: provider auth errors sometimes
    echo a partial key, and we never want even that to surface. Returns the input
    unchanged when there is nothing key-shaped in it.
    """
    if not text:
        return text
    return _KEY_RE.sub("«redacted-key»", str(text))
