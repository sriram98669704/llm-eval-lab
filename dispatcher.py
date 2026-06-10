"""
dispatcher.py — Single entry point for all model calls.

Every file in the project calls call_model() here.
No SDK logic anywhere else.

Routing by model name prefix:
  gpt-*          -> OpenAI
  claude-*       -> Anthropic
  deepseek-ai/*  -> Together AI (DeepSeek V4 Pro)

Each SDK call retried up to 3 times with exponential backoff (tenacity).
Returns None if all retries fail — callers handle None gracefully.

Returned dict from call_model():
  {
      "text":          str,    # model's response
      "input_tokens":  int,    # tokens sent
      "output_tokens": int,    # tokens received
      "latency":       float,  # seconds
      "cost_usd":      float,  # dollars (None if model not in PRICING)
  }
"""

import os
import time
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from byok import redact
from pricing import calculate_cost
from config import MAX_OUTPUT_TOKENS

load_dotenv()

logger = logging.getLogger(__name__)


def _log_retry(retry_state):
    """
    tenacity before_sleep hook. Logs each retry with the exception text passed
    through redact() — provider auth errors can echo key fragments, and retry
    logs land in server stdout (e.g. Streamlit Cloud logs), so they get the
    same redaction as the UI path.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    fn = retry_state.fn.__name__ if retry_state.fn else "call"
    logger.warning(
        "Retrying %s (attempt %d) after error: %s",
        fn, retry_state.attempt_number, redact(str(exc)),
    )


# -----------------------------
# OPENAI
# -----------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    before_sleep=_log_retry,
    reraise=True
)
def _call_openai(model, prompt, api_key=None):
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_OUTPUT_TOKENS,   # parity with the other two providers — same output budget
        temperature=0
    )
    latency = time.perf_counter() - start

    return {
        "text":          response.choices[0].message.content,
        "input_tokens":  response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "latency":       latency
    }


# -----------------------------
# ANTHROPIC
# -----------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    before_sleep=_log_retry,
    reraise=True
)
def _call_anthropic(model, prompt, api_key=None):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    latency = time.perf_counter() - start

    return {
        "text":          response.content[0].text,
        "input_tokens":  response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency":       latency
    }


# -----------------------------
# TOGETHER AI (DeepSeek V4 Pro)
# -----------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    before_sleep=_log_retry,
    reraise=True
)
def _call_together(model, prompt, api_key=None):
    from together import Together
    client = Together(api_key=api_key or os.getenv("TOGETHER_API_KEY"))

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_OUTPUT_TOKENS,   # explicit so Together's low default can't truncate long answers
        temperature=0
    )
    latency = time.perf_counter() - start

    return {
        "text":          response.choices[0].message.content,
        "input_tokens":  response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "latency":       latency
    }


# -----------------------------
# EMBEDDING — used by categorizer.py (Step 22, kNN live-prompt categorization).
# Not used in the benchmark or routing pipeline.
# -----------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True
)
def embed(text, api_key=None):
    """
    Get vector embedding(s) using text-embedding-3-small.

    `text` can be a single string or a list of strings.
    - Single string → returns one vector (list of floats).
    - List of strings → returns a list of vectors in the same order.
      The entire list is sent in one API call, so callers should batch
      where possible (e.g. build_fingerprints sends all 30 prompts at once).

    Used by categorizer.py (Step 22, kNN live-prompt categorization).
    NOT used in the benchmark or routing pipeline — keep it out of the eval loop.
    Retried up to 3 times on failure.
    """
    from openai import OpenAI
    from config import EMBEDDING_MODEL
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    if isinstance(text, list):
        return [item.embedding for item in response.data]
    return response.data[0].embedding


# -----------------------------
# MAIN ENTRY POINT
# -----------------------------
def call_model(model, prompt, keys=None):
    """
    Route a prompt to the correct SDK based on model name prefix.
    Each SDK call is retried up to 3 times with exponential backoff.
    Returns standard dict on success, None on total failure.

    keys: optional {"openai": ..., "anthropic": ..., "together": ...} of
          per-session API keys (BYOK). When a provider's key is supplied it is
          passed EXPLICITLY to that SDK client; when absent the call falls back
          to the process environment (.env / local run). Keys are never written
          to os.environ, so one user session's key can never leak into another's
          — see dashboard.py Live Test (BYOK).
    """
    keys = keys or {}
    try:
        if model.startswith("gpt-"):
            result = _call_openai(model, prompt, api_key=keys.get("openai"))
        elif model.startswith("claude-"):
            result = _call_anthropic(model, prompt, api_key=keys.get("anthropic"))
        elif model.startswith("deepseek-ai/"):
            result = _call_together(model, prompt, api_key=keys.get("together"))
        else:
            raise ValueError(f"Unknown model prefix: {model}")

        result["cost_usd"] = calculate_cost(
            model,
            result["input_tokens"],
            result["output_tokens"]
        )
        return result

    except Exception as e:
        # redact() the exception text — this print reaches server stdout
        # (Streamlit Cloud logs), and provider errors can echo key fragments.
        print(f"[dispatcher] {model} failed after all retries: {redact(str(e))}")
        return None
