"""
judge.py — Leave-one-out LLM jury for scoring open-ended responses.

Each contestant is judged by the OTHER 2 models — never itself.
No self-judging. No family bias. Average of 2 judges per response.

Judges (same 3 models as contestants, set in config.py as JUDGE_MODELS):
  - gpt-4o                      (OpenAI)
  - claude-sonnet-4-5           (Anthropic)
  - deepseek-ai/DeepSeek-V4-Pro (DeepSeek via Together AI)

Leave-one-out table:
  gpt-4o response              → judged by claude-sonnet-4-5 + deepseek-ai/DeepSeek-V4-Pro
  claude-sonnet-4-5            → judged by gpt-4o + deepseek-ai/DeepSeek-V4-Pro
  deepseek-ai/DeepSeek-V4-Pro  → judged by gpt-4o + claude-sonnet-4-5

Public API:
  judge_response(prompt, response, contestant_model)
    → {"correctness": float, "reasoning": float,
       "clarity": float, "hallucination_free": float,
       "judge_raw": dict}   on success
    → None                                           on total failure

Robustness:
  - Each of the 2 active judges retried once on failure
  - Failed judges excluded from average (not zeroed)
  - parse_judge_json() handles markdown fences, trailing commas, malformed JSON
  - Returns None only if both remaining judges fail all attempts
"""

import re
import json
from config import JUDGE_MODELS, FAMILY_SKIP
from dispatcher import call_model


def jury_pool(contestant_model):
    """
    Returns the list of judge models that will score this contestant.
    Excludes the same-family judge (leave-one-out).
    Single source of truth — used by _call_jury and models.py health tracking.
    """
    skip = FAMILY_SKIP.get(contestant_model)
    return [m for m in JUDGE_MODELS if m != skip]

JUDGE_KEYS = ["correctness", "reasoning", "clarity", "hallucination_free"]

JUDGE_SYSTEM_PROMPT = (
    "You are a calibrated evaluation engine scoring an AI response.\n"
    "Score each dimension on a 0.0–1.0 scale:\n"
    "  0.0–0.4 = poor    (wrong, incoherent, or heavily hallucinated)\n"
    "  0.5     = average (partially correct, some gaps)\n"
    "  0.8     = good    (mostly correct, minor issues)\n"
    "  0.9+    = exceptional (accurate, well-reasoned, clear)\n"
    "  1.0     = perfect (flawless on every dimension)\n\n"
    "Dimensions:\n"
    "  correctness:       Is the answer factually/logically correct?\n"
    "  reasoning:         Are the steps and logic sound?\n"
    "  clarity:           Is the response clear and well-structured?\n"
    "  hallucination_free: Are all claims accurate and verifiable?\n\n"
    "Return ONLY valid JSON — no explanations, no extra text:\n"
    "{\n"
    '  "correctness": 0.0,\n'
    '  "reasoning": 0.0,\n'
    '  "clarity": 0.0,\n'
    '  "hallucination_free": 0.0\n'
    "}"
)


# -----------------------------
# STANDALONE JSON PARSER
# -----------------------------
def parse_judge_json(content):
    """
    Robustly extract judge scores from raw judge output.
    Works for any judge model — built with gpt-4o behaviour in mind.

    Pipeline (in order):
      1. Strip markdown fences  — gpt-4o often wraps in ```json ... ```
      2. Find { ... }           — isolate the JSON object
      3. Fix trailing commas    — ,} and ,] are invalid JSON
      4. raw_decode             — cleanest parse path
      5. Regex last resort      — extract key:value pairs manually
      6. Return None            — if everything fails, never crash

    Returns dict with JUDGE_KEYS as float values, or None on total failure.
    Independently testable — no side effects, no judge calls.
    """
    if not content:
        return None

    text = content.strip()

    # Step 1: strip markdown fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Step 2: find the JSON object — first { to last }
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    json_str = text[start:end + 1]

    # Step 3: fix trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)

    # Step 4: try raw_decode
    try:
        result, _ = json.JSONDecoder().raw_decode(json_str)
        if isinstance(result, dict) and any(k in result for k in JUDGE_KEYS):
            return {k: float(result.get(k, 0.0)) for k in JUDGE_KEYS}
    except (json.JSONDecodeError, ValueError):
        pass

    # Step 5: regex last resort — pull out key: value pairs directly
    scores = {}
    for key in JUDGE_KEYS:
        match = re.search(rf'"{key}"\s*:\s*([0-9]*\.?[0-9]+)', json_str)
        if match:
            scores[key] = float(match.group(1))
    if len(scores) == len(JUDGE_KEYS):
        return scores

    # Step 6: total failure
    return None


# -----------------------------
# SINGLE JUDGE, SINGLE ATTEMPT
# -----------------------------
def _judge_once(judge_model, prompt, response, keys=None):
    """
    One attempt with one judge model via dispatcher.
    Returns (scores, raw, cost_usd) on success, raises on any failure.
    cost_usd is the dispatcher's cost for this judge call (None if unpriced).

    keys: optional per-session BYOK keys, forwarded to call_model (see dispatcher).
    """
    result = call_model(
        judge_model,
        f"System: {JUDGE_SYSTEM_PROMPT}\n\nPrompt:\n{prompt}\n\nResponse:\n{response}",
        keys=keys,
    )

    if result is None:
        raise ValueError(f"dispatcher returned None for judge {judge_model}")

    raw = result["text"].strip()
    scores = parse_judge_json(raw)

    if scores is None:
        raise ValueError(f"parse_judge_json failed for {judge_model}: {raw[:200]}")

    return scores, raw, result.get("cost_usd")


# -----------------------------
# JURY: LEAVE-ONE-OUT
# -----------------------------
def _call_jury(prompt, response, contestant_model, keys=None):
    """
    Call all judges except the one from the same family as the contestant.
    Each judge retried once on failure.
    Returns list of (judge_model, scores_dict, raw_str, cost_usd) for judges that succeeded.
    Failed judges excluded — not zeroed.

    Leave-one-out means:
      gpt-4o response              → judged by claude-sonnet-4-5 + deepseek-ai/DeepSeek-V4-Pro
      claude-sonnet-4-5            → judged by gpt-4o + deepseek-ai/DeepSeek-V4-Pro
      deepseek-ai/DeepSeek-V4-Pro  → judged by gpt-4o + claude-sonnet-4-5
    """
    results = []

    for judge_model in jury_pool(contestant_model):
        # jury_pool() already excludes the same-family judge

        for attempt in range(2):
            try:
                scores, raw, cost = _judge_once(judge_model, prompt, response, keys=keys)
                results.append((judge_model, scores, raw, cost))
                break  # success — move to next judge
            except Exception as e:
                print(f"[judge] {judge_model} attempt {attempt + 1}/2 failed: {e}")

    return results


# -----------------------------
# PUBLIC API
# -----------------------------
def judge_response(prompt, response, contestant_model, keys=None):
    """
    Score a response using leave-one-out jury.
    Skips the judge from the same family as the contestant.
    Averages scores across the 2 remaining judges.

    Args:
        prompt:           the original prompt
        response:         the contestant's response
        contestant_model: name of the model being judged (used to skip sibling)
        keys:             optional per-session BYOK keys (forwarded to call_model)

    Returns:
        On success: {
            "correctness": float, "reasoning": float,
            "clarity": float, "hallucination_free": float,
            "judge_raw":      {model: raw_str, ...},
            "judge_costs":    {model: cost_usd, ...},   # per-judge call cost
            "judge_cost_usd": float                     # rolled-up jury cost
        }
        On total failure (both judges fail): None
    """
    jury_results = _call_jury(prompt, response, contestant_model, keys=keys)

    if not jury_results:
        print(f"[judge] All jury members failed for {contestant_model} — returning None")
        return None

    # Average each dimension across successful judges
    averaged = {}
    for key in JUDGE_KEYS:
        averaged[key] = sum(r[1][key] for r in jury_results) / len(jury_results)

    averaged["judge_raw"] = {r[0]: r[2] for r in jury_results}

    # Per-judge call cost (None if that judge model was unpriced) + rolled-up
    # total for THIS answer. Stored rich; the router rolls it into the live trace.
    judge_costs = {r[0]: r[3] for r in jury_results}
    averaged["judge_costs"]    = judge_costs
    averaged["judge_cost_usd"] = round(sum(c for c in judge_costs.values() if c is not None), 6)

    return averaged
