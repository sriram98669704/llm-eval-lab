"""
datasets_loader.py — Loads real benchmark datasets from HuggingFace.

  load_gsm8k     : grade-school math word problems. Each item carries strict
                   formatting instructions (forcing a 'FINAL ANSWER: <n>' line)
                   plus the ground-truth answer string for verification.

  load_humaneval : Python coding problems. Each item carries instructions plus
                   a reference dict with the unit tests, entry_point function
                   name, canonical_solution, and the clean original_prompt
                   (used to reconstruct/repair the model's code before testing).
"""

from datasets import load_dataset

# -----------------------------
# GSM8K (Math reasoning)
# -----------------------------
def load_gsm8k(limit=5):
    dataset = load_dataset("gsm8k", "main", split="train")

    data = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break

        data.append({
            "input": (
                item["question"] +
                "\n\nInstructions:"
                "\n- Solve step by step, showing your reasoning clearly."
                "\n- Your LAST line MUST be exactly in this format (no exceptions):"
                "\n  FINAL ANSWER: 42"
                "\n- Write ONLY a plain number after FINAL ANSWER — no units, no $, no words, no approximations."
                "\n- Example of correct format: FINAL ANSWER: 72"
                "\n- Example of WRONG format: FINAL ANSWER: $72 or FINAL ANSWER: 72 pages"
            ),
            "reference": item["answer"]
        })

    return data


# -----------------------------
# HumanEval (Coding)
# -----------------------------
def load_humaneval(limit=5):
    dataset = load_dataset("openai_humaneval", split="test")

    data = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break

        data.append({
            "input": (
                item["prompt"] +
                "\n# Instructions:"
                "\n# - Complete the function body only."
                "\n# - Do NOT repeat the function signature or docstring."
                "\n# - Do NOT add a main block or example usage."
                "\n# - Return only valid indented Python code."
                "\n# - Do NOT wrap in markdown code blocks."
            ),
            "reference": {
                "canonical_solution": item["canonical_solution"],
                "test": item["test"],
                "entry_point": item["entry_point"],
                "original_prompt": item["prompt"]  # clean prompt for code reconstruction
            }
        })

    return data
