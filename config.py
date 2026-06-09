"""
config.py — Single source of truth for all model and run configuration.

Change a model here and it changes everywhere. Nothing is hardcoded elsewhere.
"""

# -----------------------------
# MODELS UNDER EVALUATION
# Three frontier families: OpenAI, Anthropic, DeepSeek (via Together AI).
# One per family — fair like-for-like comparison.
# -----------------------------
EVALUATED_MODELS = [
    "gpt-4o",                       # OpenAI
    "claude-sonnet-4-5",            # Anthropic
    "deepseek-ai/DeepSeek-V4-Pro",  # DeepSeek via Together AI
]

# -----------------------------
# JURY JUDGES — leave-one-out
# Same 3 models as contestants. Each response judged by the OTHER 2.
# No self-judging. No family bias. No separate judge tier needed.
#
# | Contestant                  | Judged by                                          |
# |-----------------------------|----------------------------------------------------|
# | gpt-4o                      | claude-sonnet-4-5 + deepseek-ai/DeepSeek-V4-Pro   |
# | claude-sonnet-4-5           | gpt-4o + deepseek-ai/DeepSeek-V4-Pro              |
# | deepseek-ai/DeepSeek-V4-Pro | gpt-4o + claude-sonnet-4-5                        |
# -----------------------------
JUDGE_MODELS = [
    "gpt-4o",                       # OpenAI judge
    "claude-sonnet-4-5",            # Anthropic judge
    "deepseek-ai/DeepSeek-V4-Pro",  # DeepSeek judge
]

# Maps each contestant to the judge to SKIP (same model = skip)
FAMILY_SKIP = {
    "gpt-4o":                       "gpt-4o",
    "claude-sonnet-4-5":            "claude-sonnet-4-5",
    "deepseek-ai/DeepSeek-V4-Pro":  "deepseek-ai/DeepSeek-V4-Pro",
}

# -----------------------------
# EMBEDDER — reserved for Step 25 (kNN live-prompt categorization).
# NOT used in the benchmark/eval pipeline. Kept here so the model id lives in
# one place when kNN routing is built. (Stability scoring was removed: at
# temperature 0 it was meaningless; reliability now = error bars across prompts.)
# -----------------------------
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI

# -----------------------------
# CATEGORIES (CEO's Tilicho-specific tasks — all judge-scored)
# -----------------------------
CATEGORIES = [
    "architecture",      # multi-agent design, enterprise AI workflows
    "coding",            # FastAPI, backend services, API integrations
    "debugging",         # async failures, Redis queues, bottlenecks
    "summarization",     # meeting transcripts, client docs, technical docs
    "business_writing",  # proposals, recommendations, client comms
    "agent_planning",    # multi-agent workflows, AI orchestration
]

# -----------------------------
# RUN SETTINGS
# -----------------------------
TEMPERATURE = 0          # Deterministic outputs for reproducibility

# Max output tokens per model call. Applied IDENTICALLY to all three providers so
# the comparison is like-for-like — no model gets a longer answer budget than
# another. Anthropic requires it; for OpenAI and Together it prevents a low
# provider default from silently truncating long answers. Tune here, nowhere else.
MAX_OUTPUT_TOKENS = 4096
# One model call per prompt. At temperature 0 repeated runs of the same prompt
# are ~identical, so they add no signal — the reliability measure that matters is
# the spread of quality ACROSS the different prompts in a category (error bars),
# which needs only one run each. More robustness = more *different* prompts.

# -----------------------------
# ROUTING SETTINGS
# -----------------------------
# quality_bar = best_quality - tolerance. A model is ELIGIBLE to be a category's
# default only if its quality clears this bar. Tight bar where precision matters.
DEFAULT_TOLERANCE = 0.05   # loose — most categories
TIGHT_TOLERANCE   = 0.03   # tight — coding / architecture (precision matters)
TIGHT_CATEGORIES  = ["coding", "architecture"]

# Among the models that clear the quality bar, how to pick the default:
#   - category in SPEED_SENSITIVE  -> pick the FASTEST  (lowest latency)
#   - otherwise                    -> pick the CHEAPEST (lowest cost), latency breaks ties
# Quality decides WHO is eligible; cost/latency only choose AMONG the eligible.
SPEED_SENSITIVE = ["summarization"]

# -----------------------------
# ESCALATION TRIGGER (for live data / promotion mechanic)
# -----------------------------
ESCALATION_RATE_THRESHOLD = 0.40   # >40% escalation rate triggers re-eval alert
MIN_PROMPTS_FOR_TRIGGER = 20       # need at least 20 live prompts before triggering
MIN_PROMPTS_NEW_CATEGORY = 10      # new category needs 10+ prompts before mini eval

# -----------------------------
# JUDGE HEALTH TRIGGER
# -----------------------------
# If a judge model fails more than this fraction of its attempts in a run,
# the dashboard shows a warning: scores from that run may be unreliable.
# Action: re-run the benchmark (almost always a transient API issue).
JUDGE_FAILURE_THRESHOLD = 0.20     # >20% failure rate triggers re-run alert
