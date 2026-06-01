"""
prompts.py — Defines the benchmark suite (PROMPTS).

PROMPTS maps each category name to a list of {"input", "reference"} items:
  - factual   : open-ended prompt, scored by the LLM judge (no reference)
  - gsm8k     : real math problems loaded from HuggingFace (ground-truth)
  - humaneval : real coding problems loaded from HuggingFace (sandbox-tested)

The commented-out block below holds extra judge-based categories (math, coding,
creative, opinion, architecture, edge_case) — uncomment to expand the suite.
Dataset sizes are controlled via the limit= args.
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