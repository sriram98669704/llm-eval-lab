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
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging
from pricing import calculate_cost
from config import MAX_OUTPUT_TOKENS

load_dotenv()

logger = logging.getLogger(__name__)


# -----------------------------
# OPENAI
# -----------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def _call_openai(model, prompt):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def _call_anthropic(model, prompt):
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def _call_together(model, prompt):
    from together import Together
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

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
# EMBEDDING — reserved for Step 25 kNN live-prompt categorization.
# Not used in the benchmark or routing pipeline.
# -----------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True
)
def embed(text):
    """
    Get a vector embedding for `text` using text-embedding-3-small.

    Reserved for Step 25 kNN live-prompt categorization (mapping a free-text
    prompt to one of the 6 CEO categories without the user having to label it).
    Not used in the benchmark or routing pipeline — do not call this from
    the eval loop.

    Retried up to 3 times on failure.
    """
    from openai import OpenAI
    from config import EMBEDDING_MODEL
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


# -----------------------------
# MAIN ENTRY POINT
# -----------------------------
def call_model(model, prompt):
    """
    Route a prompt to the correct SDK based on model name prefix.
    Each SDK call is retried up to 3 times with exponential backoff.
    Returns standard dict on success, None on total failure.
    """
    try:
        if model.startswith("gpt-"):
            result = _call_openai(model, prompt)
        elif model.startswith("claude-"):
            result = _call_anthropic(model, prompt)
        elif model.startswith("deepseek-ai/"):
            result = _call_together(model, prompt)
        else:
            raise ValueError(f"Unknown model prefix: {model}")

        result["cost_usd"] = calculate_cost(
            model,
            result["input_tokens"],
            result["output_tokens"]
        )
        return result

    except Exception as e:
        print(f"[dispatcher] {model} failed after all retries: {e}")
        return None
