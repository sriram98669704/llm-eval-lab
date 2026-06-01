"""
judge.py — LLM-as-a-judge scorer for open-ended categories.

Used for categories that have no ground truth (factual, creative, opinion, etc.).
Sends the prompt + response to a judge model and asks for a strict JSON score on
four dimensions: correctness, reasoning, clarity, hallucination_free (each 0–1).

Robustness:
  - System prompt forces JSON-only output and harsh, calibrated scoring
  - raw_decode safely extracts the JSON object even if the model adds stray text
  - Any failure falls back to all-zeros so the pipeline never crashes

NOTE: JUDGE_MODEL should differ from the models being benchmarked to avoid
self-preference bias. "mistral" is set for quick local testing only.
"""

import re
import ollama
import json

JUDGE_MODEL = "phi3"  # neutral 3rd-party judge — not in the benchmarked model set
# JUDGE_MODEL = "gemma2"  # alternative neutral judge

# -----------------------------
# LLM JUDGE FUNCTION
# -----------------------------
def judge_response(prompt, response):
    """
    Uses an LLM to judge a response based on correctness, reasoning,
    clarity, and hallucination-free metrics.
    Returns a dictionary with float scores between 0.0 and 1.0.
    """

    try:
        feedback = ollama.chat(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a harsh, critical evaluation engine. Be skeptical and penalize heavily.\n"
                        "Score each dimension honestly — do NOT give 1.0 unless the response is truly perfect.\n"
                        "Most responses should score between 0.5 and 0.85. Reserve 0.9+ for exceptional answers.\n"
                        "Penalize: vague answers, missing steps, verbosity without substance, any factual errors.\n\n"
                        "For hallucination_free specifically:\n"
                        "- Score LOW (0.0–0.4) if the response contains made-up facts, wrong numbers, or invented names/sources.\n"
                        "- Score MEDIUM (0.5–0.7) if the response is mostly correct but includes unverifiable or suspicious claims.\n"
                        "- Score HIGH (0.8–1.0) only if every factual claim in the response is accurate and verifiable.\n"
                        "- Be especially strict on math, code, and factual categories.\n\n"
                        "You MUST return ONLY valid JSON. No explanations. No extra text.\n\n"
                        "Schema:\n"
                        "{\n"
                        '  "correctness": 0.0,\n'
                        '  "reasoning": 0.0,\n'
                        '  "clarity": 0.0,\n'
                        '  "hallucination_free": 0.0\n'
                        "}\n\n"
                        "All values must be floats between 0 and 1."
                    )
                },
                {
                    "role": "user",
                    "content": f"Prompt:\n{prompt}\n\nResponse:\n{response}"
                }
            ],
            options={
                "temperature": 0
            }
        )

        content = feedback["message"]["content"]
        if not content:
            raise ValueError("Empty response from judge model")

        content = content.strip()

        # -----------------------------
        # Extract JSON safely
        # -----------------------------
        start = content.find("{")
        if start == -1:
            raise ValueError(f"No JSON found in: {content}")

        # Strip trailing commas before the closing brace — phi3 sometimes
        # produces malformed JSON like {"key": 0.8,} which JSONDecoder rejects
        json_str = content[start:]
        json_str = re.sub(r',\s*}', '}', json_str)   # trailing comma in object
        json_str = re.sub(r',\s*]', ']', json_str)   # trailing comma in array

        result, _ = json.JSONDecoder().raw_decode(json_str)

        # -----------------------------
        # Ensure all keys exist
        # -----------------------------
        return {
            "correctness": float(result.get("correctness", 0.0)),
            "reasoning": float(result.get("reasoning", 0.0)),
            "clarity": float(result.get("clarity", 0.0)),
            "hallucination_free": float(result.get("hallucination_free", 0.0))
        }

    except Exception as e:
        print("Judge LLM failed:", e)

        # Safe fallback to prevent pipeline break
        return {
            "correctness": 0.0,
            "reasoning": 0.0,
            "clarity": 0.0,
            "hallucination_free": 0.0
        }