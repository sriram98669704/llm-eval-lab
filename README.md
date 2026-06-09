# LLM Eval Lab

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Tests](https://img.shields.io/badge/tests-150%20passing-brightgreen) ![No API calls in tests](https://img.shields.io/badge/test%20suite-offline-lightgrey) ![Built with Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b)

**Which model should we use for which task — and when should we automatically upgrade to a better one?**

Picking one LLM for everything leaves quality or money on the table. The best model for summarization is rarely the best for debugging, and the most capable model is often the slowest and most expensive. LLM Eval Lab answers the question empirically: it benchmarks three frontier models across six real task categories, derives a quality-driven routing policy from the measured scores, and then applies that policy live — sending each prompt to the cheapest model that meets the quality bar, and escalating to the best model only when a response actually falls short.

> 🎥 **[Watch the walkthrough](https://drive.google.com/file/d/1L5qdMgL_nA6xgG5BPA5tWBXI-Sv_NXLh/view)**  ·  🚀 **[Live app](https://llm-eval-tilicho-labs.streamlit.app/)**

---

## At a glance

```
python main.py                  # benchmark → score → derive routing policy
streamlit run dashboard.py      # explore results + run live prompts
```

- **3 models** benchmarked head-to-head: GPT-4o, Claude Sonnet 4.5, DeepSeek-V4-Pro
- **6 task categories**, 5 prompts each — 30 prompts, 90 model responses per run
- **Leave-one-out jury** — every response graded by the other two models, never by itself
- **Quality-driven routing** — one default + one escalation target derived per category
- **Live Test** — route any prompt in real time, with live grading and automatic escalation
- **150 unit tests** — the *test suite* runs fully offline (the benchmark and Live Test make real API calls; the tests don't), covering scoring math, parsing, routing, and key-handling

---

## Headline results

From the latest full run (30 prompts, 3 models, leave-one-out grading):

| Rank | Model | Provider | Overall quality | Avg cost/prompt | Avg latency |
|---:|---|---|---:|---:|---:|
| 🥇 | `deepseek-ai/DeepSeek-V4-Pro` | DeepSeek via Together AI | **0.915** | ~$0.006 | 34.1s |
| 🥈 | `claude-sonnet-4-5` | Anthropic | 0.909 | ~$0.038 | 36.3s |
| 🥉 | `gpt-4o` | OpenAI | 0.807 | ~$0.006 | 6.7s |

> **The finding that makes routing worth it:** the highest-quality model here (DeepSeek, 0.915) is also one of the two *cheapest* — roughly **6× cheaper per prompt than Claude**, whose quality is statistically comparable. GPT-4o is the fastest and just as cheap, but trails on quality. No model wins on all three axes of quality, cost, and speed — so committing to a single vendor always overpays on one of them. The routing policy picks the right model per task instead.

**Quality is close at the top, but the cost and speed profiles are wildly different** — which is exactly why per-category routing pays off. GPT-4o is ~5× faster but trails on quality; DeepSeek matches or beats Claude on quality at roughly a sixth of the cost in several categories. No single model dominates all six tasks:

| Category | Quality winner | Score |
|---|---|---:|
| `architecture` | DeepSeek-V4-Pro | 0.898 |
| `coding` | Claude Sonnet 4.5 | 0.901 |
| `debugging` | Claude Sonnet 4.5 | 0.903 |
| `summarization` | Claude Sonnet 4.5 | 0.965 |
| `business_writing` | DeepSeek-V4-Pro | 0.948 |
| `agent_planning` | DeepSeek-V4-Pro | 0.921 |

*(Numbers are a point-in-time snapshot of these 30 prompts — see [Limitations](#limitations). They will change when you run it yourself.)*

---

## The derived routing policy

After scoring, the system derives one policy per category: a **default** model (cheapest, or fastest for speed-sensitive tasks, among those clearing the quality bar) and an **escalation target** (the highest-quality model). This is the headline artifact — `routing.json`:

| Category | Default model | Why default | Escalates to |
|---|---|---|---|
| `architecture` | DeepSeek-V4-Pro | cheapest clearing the bar | DeepSeek-V4-Pro |
| `coding` | Claude Sonnet 4.5 | only model clearing the tight bar | Claude Sonnet 4.5 |
| `debugging` | DeepSeek-V4-Pro | clears bar at ~⅙ the cost of best | Claude Sonnet 4.5 |
| `summarization` | GPT-4o | fastest clearing the bar (2.1s) | Claude Sonnet 4.5 |
| `business_writing` | DeepSeek-V4-Pro | cheapest *and* highest quality | DeepSeek-V4-Pro |
| `agent_planning` | DeepSeek-V4-Pro | cheapest clearing the bar | DeepSeek-V4-Pro |

When `default == escalate_to`, the default is already the best model in that category — there's nowhere to escalate, so the response is returned as-is.

---

## How it works

### One currency: quality

Quality is the **only** thing that decides leaderboard rank, routing eligibility, and escalation. Cost and latency are tracked and displayed, but they only break ties *among* models that have already cleared the quality bar — they never decide who ranks higher or when escalation fires.

```
quality_bar  = best_quality_in_category − tolerance
               (tolerance: 0.03 for coding/architecture, 0.05 elsewhere)

default      = cheapest model that clears the bar
               (fastest for summarization)

escalate_to  = the highest-quality model in that category
```

### Quality is a task-aware weighted score

Each response is graded on four dimensions (0–1): **correctness**, **reasoning**, **clarity**, **hallucination-free**. These are combined into a single quality score using **per-category weights** — because a code snippet lives or dies on correctness, while a meeting summary lives or dies on clarity and not inventing things that weren't said.

| Category | correctness | reasoning | clarity | hallucination-free |
|---|---:|---:|---:|---:|
| `architecture` | 0.40 | 0.40 | — | 0.20 |
| `coding` | 0.60 | 0.30 | 0.10 | — |
| `debugging` | 0.50 | 0.40 | 0.10 | — |
| `summarization` | 0.30 | — | 0.40 | 0.30 |
| `business_writing` | 0.30 | 0.20 | 0.40 | 0.10 |
| `agent_planning` | 0.40 | 0.40 | 0.20 | — |

Weights sum to 1.0 per category, so a perfect response scores exactly 1.0. Full detail in [METHODOLOGY.md](METHODOLOGY.md).

### Leave-one-out jury

Each response is graded by the **other two models** — never by itself. This removes self-serving grades without needing a separate, more expensive grader tier.

```
gpt-4o response              →  graded by claude-sonnet-4-5 + deepseek-ai/DeepSeek-V4-Pro
claude-sonnet-4-5 response   →  graded by gpt-4o + deepseek-ai/DeepSeek-V4-Pro
deepseek-ai/DeepSeek-V4-Pro  →  graded by gpt-4o + claude-sonnet-4-5
```

A judge that fails to return parseable scores is **excluded** from the average, not zeroed — zeroing would unfairly punish the contestant for the judge's failure. If more than 20% of a model×category's responses were graded by fewer than two judges, the dashboard raises a reliability warning.

### Live routing

```
prompt arrives
    ↓
auto-detect category (kNN over prompt embeddings) — or pick manually
    ↓
default model responds
    ↓
graded live by the other two models
    ↓
quality ≥ bar?  ──yes──▶  return answer  (cheap/fast path)
    │
    no
    ↓
escalate to best model → re-grade → return answer
```

Every cost component is tracked — default answer, default grading, escalated answer (if any), escalated grading (if any) — so the dashboard always shows the true total spend per prompt.

---

## The dashboard

```bash
streamlit run dashboard.py
```

| Tab | What it shows |
|---|---|
| **🏆 Leaderboard** | Overall quality, cost, and latency per model, with headline findings |
| **📊 Eval** | Per-model × per-category quality heatmap, with score spread (std dev) on hover |
| **🔬 Benchmark** | The raw data — every prompt, every answer, every grader's score |
| **🗺️ Routing** | Quality-bar table, routing-flow diagram, and grading-reliability checks |
| **⚡ Live Test** | Run any prompt through the live policy with your own keys — streams each step as it happens |

The **Live Test** tab auto-detects the category from your prompt using k-nearest-neighbour over prompt embeddings, showing its confidence (cosine similarity) so you can override with **Pick manually** when a prompt sits on a category boundary.

---

## Bring Your Own Keys (Live Test)

The Live Test tab makes real, paid API calls, so it needs keys. The security model is strict and deliberate:

- Keys are entered once per browser session in the BYOK panel — **never** committed, never in the repo.
- They live in **browser session memory only** — never written to disk, logs, or environment variables, and never shared between visitors.
- They are passed as explicit function arguments straight to the provider SDKs and **never written to `os.environ`** (which is process-global and shared across sessions — writing a key there could leak it between visitors).
- Anything key-shaped is **redacted** before it can ever reach the UI or a log, as defence-in-depth against provider errors that echo partial keys.
- Keys vanish when you close the tab, and a **Clear keys** button wipes them on demand.

### Where each key path is used

There are two separate places keys come from, for two separate jobs:

- **Benchmark + local dev → `.env`.** The benchmark pipeline (`python main.py`, which runs the 30 prompts in `prompts.json` across all models) reads keys from a local `.env`. That's the offline batch job that produces the leaderboard and routing policy — it needs your keys on the machine running it.
- **Deployed Live Test → BYOK.** The deployed app has no `.env`, so the Live Test tab uses bring-your-own-keys — each visitor's pasted keys, scoped to their own browser session.

**Precedence is env-first.** If a local `.env` supplies all three keys, Live Test uses them directly and the BYOK panel stays hidden — convenient when you're developing locally and already have keys on hand. It only appears when the environment isn't *fully* populated: the app checks whether all three keys are present, so to see the BYOK panel on your own machine you can either delete `.env` entirely, or simply remove one provider's line from it (e.g. drop `TOGETHER_API_KEY`) — a single missing key flips it to BYOK mode. Either way, keys are only ever *read* — never written to disk, logs, or `os.environ`.

The four read-only tabs (Leaderboard, Eval, Benchmark, Routing) need no keys at all — they run entirely off the pre-computed benchmark data. Key resolution is pure, side-effect-free, and covered by [`tests/test_byok.py`](tests/test_byok.py), including an explicit assertion that `os.environ` is never mutated.

---

## Quickstart

### Prerequisites

- Python 3.9+
- API keys: OpenAI, Anthropic, and Together AI (for DeepSeek)

### Install

```bash
git clone https://github.com/sriram98669704/llm-eval-lab.git
cd llm-eval-lab

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# fill in OPENAI_API_KEY, ANTHROPIC_API_KEY, TOGETHER_API_KEY
```

`.env` is gitignored and never committed. For the deployed app, no `.env` exists — Live Test uses the BYOK panel instead.

### Run the benchmark

```bash
python main.py
```

Benchmarks all models, scores responses, derives the routing policy. Saves results under `data/runs/<timestamp>/` and registers them in `data/index.json`.

### Launch the dashboard

```bash
streamlit run dashboard.py
```

### Run the tests

```bash
pytest tests/
```

150 tests. The test suite runs fully offline — no API calls, no keys needed — so it's fast and free to run. Covers scoring math, judge output parsing, SDK routing, routing-table derivation, BYOK key resolution, and security invariants. (The benchmark pipeline and Live Test do make real API calls — these tests verify the logic around them.)

---

## Using the routing policy in code

```python
from router import route

# Free — reads routing.json, no API call
result = route("coding")
# → {"model": "claude-sonnet-4-5", "why": "...", "escalate_to": "claude-sonnet-4-5"}
```

```python
from router import run_live

trace = run_live("Write a FastAPI endpoint that validates a webhook payload.", "coding")
print(trace["final_response"])
print(f"Cost: ${trace['total_cost_usd']:.4f}")
```

---

## Project structure

```
llm-eval-lab/
├── main.py           # Orchestrator — runs the full benchmark pipeline
├── models.py         # Evaluation engine — calls models, grades, aggregates
├── judge.py          # Leave-one-out grading logic + JSON parser
├── router.py         # derive_routing_table() + run_live() + route()
├── dispatcher.py     # Single entry point for all provider API calls
├── categorizer.py    # kNN live-prompt categorization over embeddings
├── byok.py           # Key resolution, format validation, redaction (no os.environ writes)
├── metrics.py        # Scoring math — category_score, compute_final_score
├── pricing.py        # Token cost calculation
├── findings.py       # Auto-generated headline findings for the dashboard
├── report.py         # Exportable run report
├── prompts.py        # Prompt loader
├── prompts.json      # 30 prompts (6 categories × 5 prompts)
├── storage.py        # All file I/O — writes runs/, reads the latest
├── config.py         # Single source of truth for all constants
├── dashboard.py      # Streamlit dashboard
├── .streamlit/
│   └── config.toml   # Deploy + security hardening (telemetry off, XSRF/CORS on)
├── tests/            # 150 unit tests — no API calls
└── data/
    ├── index.json    # Registry of every run
    └── runs/
        └── <timestamp>/
            ├── summary.json   # Aggregated scores per model
            ├── routing.json   # Routing policy (the headline artifact)
            ├── results.jsonl  # Per-prompt detail records
            └── meta.json      # Reproducibility receipt
```

---

## Task categories

| Category | What it covers |
|---|---|
| `architecture` | Multi-agent design, enterprise AI workflows |
| `coding` | FastAPI, backend services, API integrations |
| `debugging` | Async failures, Redis queues, performance bottlenecks |
| `summarization` | Meeting transcripts, client and technical docs |
| `business_writing` | Proposals, recommendations, client comms |
| `agent_planning` | Multi-agent workflows, AI orchestration |

These are deliberately practical, open-ended tasks — not multiple-choice academic benchmarks — so the scores reflect the kind of work the models would actually be routed for.

**The suite is configurable, not fixed.** The 30 prompts live in `prompts.json` (5 per category). Add, swap, or expand them and re-run `python main.py` — every downstream artifact (scores, the routing policy, the dashboard) recomputes from whatever the suite contains. The `run_fingerprint` in `meta.json` changes whenever the prompt set does, so two runs are always distinguishable. 30 is the starting suite, not a ceiling.

---

## Configuration

All tunable constants live in `config.py`.

| Constant | Default | What it controls |
|---|---|---|
| `EVALUATED_MODELS` | 3 models | Which models are benchmarked |
| `CATEGORIES` | 6 categories | Task types |
| `TEMPERATURE` | `0` | Deterministic outputs |
| `DEFAULT_TOLERANCE` | `0.05` | Quality-bar width (most categories) |
| `TIGHT_TOLERANCE` | `0.03` | Quality-bar width (coding, architecture) |
| `SPEED_SENSITIVE` | `["summarization"]` | Categories where fastest wins |
| `JUDGE_FAILURE_THRESHOLD` | `0.20` | Failure rate that triggers a reliability warning |

---

## Reproducibility

Every run produces `meta.json` with the git SHA, Python and SDK versions, a full config snapshot, and a `run_fingerprint` — a SHA-256 hash of the prompts, config, and scoring weights. If the fingerprint matches between two runs, any score difference is genuine model drift, not a setup change.

---

## Limitations

This is an honest, narrow benchmark — read [METHODOLOGY.md](METHODOLOGY.md) for the full account. In short, it **can** claim that on these 30 prompts, at this point in time, model X beats model Y on task Z by the measured margin, and that the derived routing policy is the best quality-cost-latency tradeoff among these three models. It **cannot** claim generalization to all prompts of a task type (5 per category is a small sample), stability over time (models change), or freedom from LLM-as-judge bias (known tendencies toward length and formatting). The `run_fingerprint` exists precisely so you can tell when any of those factors have changed.

---

## What's next

Phase 8 (not built for this demo — needs live traffic data):

- **Confidence-gated categorization** — when kNN similarity is low, fall back to an LLM to classify the prompt rather than forcing a nearest-neighbour match
- **Live re-benchmark triggers** — automatically flag when accumulated live traffic suggests a model's quality has drifted from the benchmark
- **Per-user routing profiles** — let teams tune tolerances and speed/cost preferences per deployment context
- **Cluster discovery** — surface new task categories from live prompt embeddings using HDBSCAN
