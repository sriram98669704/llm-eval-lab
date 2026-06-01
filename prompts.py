"""
prompts.py — Defines the benchmark suite (PROMPTS).

PROMPTS maps each category name to a list of {"input", "reference"} items:

  Judge-scored categories (phi3 evaluates correctness, reasoning, clarity, hallucination-free):
  - factual      : open-ended knowledge questions
  - creative     : stories, poetry, expressive writing
  - architecture : system design and engineering tradeoffs
  - edge_case    : paradoxes, ambiguous logic, unusual reasoning

  Ground-truth categories (no judge — objective verification):
  - gsm8k        : real math word problems, answer verified against ground truth
  - humaneval    : real coding problems, solution executed in a sandbox

  Dataset sizes controlled via the limit= args in datasets_loader.py.
"""

from datasets_loader import load_gsm8k, load_humaneval

# -----------------------------
# LOAD RAW DATASETS
# -----------------------------
gsm8k_data = load_gsm8k(limit=5)
humaneval_data = load_humaneval(limit=5)

# -----------------------------
# NORMALIZED PROMPTS
# -----------------------------
PROMPTS = {
    "factual": [
        {"input": "What is vector search?", "reference": ""},
        {"input": "Explain TCP vs UDP simply.", "reference": ""}
    ],

    "creative": [
        {"input": "Write a short story about AI learning emotions.", "reference": ""},
        {"input": "Summarize AI in a poetic way.", "reference": ""}
    ],

    "architecture": [
        {"input": "Design a scalable chat system.", "reference": ""},
        {"input": "Explain microservices vs monolith architecture.", "reference": ""}
    ],

    "edge_case": [
        {"input": "If time could move backwards, how would causality work?", "reference": ""},
        {"input": "A statement says: 'This sentence is false'. Analyze it.", "reference": ""}
    ],

    # -----------------------------
    # REAL BENCHMARKS (from datasets_loader)
    # -----------------------------
    "gsm8k": gsm8k_data,
    "humaneval": humaneval_data
}