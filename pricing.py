"""
pricing.py — Cost tracking for all model API calls.

PRICING dict holds input/output cost per 1K tokens for every model.
calculate_cost() converts token counts → dollars.

Rates as of June 2026 — verify at:
  OpenAI:    https://platform.openai.com/pricing
  Anthropic: https://www.anthropic.com/pricing
  Together:  https://www.together.ai/pricing

Cost is tracked per response and stored in results.jsonl.
Cost NEVER enters the quality score — it's a separate axis shown in the dashboard.
"""

# Cost per 1K tokens (input and output billed separately)
PRICING = {
    "gpt-4o": {
        "input":  0.0025,   # $2.50 per 1M input tokens
        "output": 0.0100,   # $10.00 per 1M output tokens
    },
    "claude-sonnet-4-5": {
        "input":  0.0030,   # $3.00 per 1M input tokens
        "output": 0.0150,   # $15.00 per 1M output tokens
    },
    "deepseek-ai/DeepSeek-V4-Pro": {
        "input":  0.0021,   # $2.10 per 1M input tokens
        "output": 0.0044,   # $4.40 per 1M output tokens
    },
}


def calculate_cost(model, input_tokens, output_tokens):
    """
    Calculate cost in USD for a single API call.

    Args:
        model:         model name (must match a key in PRICING)
        input_tokens:  number of tokens sent to the model
        output_tokens: number of tokens received from the model

    Returns:
        float: cost in USD, rounded to 6 decimal places
        None:  if model not in PRICING (unknown model — don't crash)
    """
    rates = PRICING.get(model)
    if rates is None:
        return None

    cost = (
        (input_tokens  / 1000) * rates["input"] +
        (output_tokens / 1000) * rates["output"]
    )
    return round(cost, 6)
