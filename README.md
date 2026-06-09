# LLM Eval Lab

**Which model should we use for which task — and when should we automatically upgrade to a better one?**

A structured benchmark and live routing framework for [Tilicho Labs](https://tilicho.ai). Evaluates three frontier LLMs across six real task categories, derives a quality-driven routing policy, and applies it live — escalating automatically when a response falls below the measured quality bar.

---

## Demo

> 🎥 **[Watch the walkthrough](YOUR_DEMO_URL_HERE)**  ·  🚀 **[Live app](YOUR_STREAMLIT_URL_HERE)**

---

## What It Does

```
python main.py                  # benchmark → score → derive routing policy
streamlit run dashboard.py      # explore results + run live prompts
```

1. **Benchmark** — runs 30 prompts (5 per category × 6 categories) across 3 models. Each response graded by the other two models — leave-one-out jury, no self-judging.
2. **Route** — derives a routing policy: one default model per category (cheapest or fastest among those that clear the quality bar), one escalation target (the best).
3. **Live** — apply the policy to any prompt in real time. Grades the default model's response on the fly; escalates to the best model only if quality falls short.

---

## Models

| Model | Provider | Role |
|---|---|---|
| `gpt-4o` | OpenAI | Contestant + jury member |
| `claude-sonnet-4-5` | Anthropic | Contestant + jury member |
| `deepseek-ai/DeepSeek-V4-Pro` | DeepSeek via Together AI | Contestant + jury member |

---

## Task Categories

All Tilicho-specific — no academic benchmarks.

| Category | What it covers |
|---|---|
| `architecture` | Multi-agent design, enterprise AI workflows |
| `coding` | FastAPI, backend services, API integrations |
| `debugging` | Async failures, Redis queues, bottlenecks |
| `summarization` | Meeting transcripts, client and technical docs |
| `business_writing` | Proposals, recommendations, client comms |
| `agent_planning` | Multi-agent workflows, AI orchestration |

---

## How It Works

### One currency: quality

Quality is the only thing that decides leaderboard rank, routing eligibility, and escalation. Cost and latency are shown separately and only choose *among* models that already clear the quality bar — they never affect who ranks higher or when escalation triggers.

```
quality_bar  = best_quality_in_category − tolerance
               (tolerance: 0.03 for coding/architecture, 0.05 elsewhere)

default      = cheapest model that clears the bar
               (fastest for summarization)

escalate_to  = the highest-quality model in that category
```

### Leave-one-out jury

Each response is graded by the **other two models** — never by itself. No separate grader tier needed.

```
gpt-4o response              →  graded by claude-sonnet-4-5 + deepseek-ai/DeepSeek-V4-Pro
claude-sonnet-4-5 response   →  graded by gpt-4o + deepseek-ai/DeepSeek-V4-Pro
deepseek-ai/DeepSeek-V4-Pro  →  graded by gpt-4o + claude-sonnet-4-5
```

### Live routing

```
prompt arrives
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

All costs tracked: answer cost + grading cost = total per prompt.

---

## Dashboard

```bash
streamlit run dashboard.py
```

| Tab | What it shows |
|---|---|
| **Leaderboard** | Quality score, cost, and latency per model |
| **Heatmap** | Per-model × per-category quality with std dev on hover |
| **Cost & Latency** | Side-by-side bar charts |
| **Routing** | Quality bar table, routing flow diagram, grading reliability |
| **Live Test** | Run any prompt through the live policy; streams each step as it happens |

The Live Test tab auto-detects the category from your prompt using kNN embeddings (text-embedding-3-small), or you can set it manually.

---

## Quickstart

### Prerequisites

- Python 3.9+
- API keys: OpenAI, Anthropic, Together AI

### Install

```bash
git clone https://github.com/your-org/llm-eval-lab.git
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

57 tests. No API calls. Covers scoring math, judge output parsing, SDK routing, and routing table derivation.

---

## Use the routing policy

```python
from router import route

# Free — reads routing.json, no API call
result = route("coding")
# → {"model": "gpt-4o", "why": "...", "escalate_to": "gpt-4o"}
```

```python
from router import run_live

trace = run_live("Write a FastAPI endpoint that validates a webhook payload.", "coding")
print(trace["final_response"])
print(f"Cost: ${trace['total_cost_usd']:.4f}")
```

---

## Project Structure

```
llm-eval-lab/
├── main.py           # Orchestrator — runs the full benchmark pipeline
├── models.py         # Evaluation engine — calls models, grades, aggregates
├── judge.py          # Leave-one-out grading logic + JSON parser
├── router.py         # derive_routing_table() + run_live() + route()
├── dispatcher.py     # Single entry point for all API calls
├── categorizer.py    # kNN live-prompt categorization (text-embedding-3-small)
├── metrics.py        # Scoring math — category_score, compute_final_score
├── pricing.py        # Token cost calculation
├── prompts.py        # Prompt loader
├── prompts.json      # 30 prompts (6 categories × 5 prompts)
├── storage.py        # All file I/O — writes runs/, reads the latest
├── config.py         # Single source of truth for all constants
├── dashboard.py      # Streamlit dashboard
├── tests/            # 57 unit tests — no API calls
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

## Configuration

All tunable constants live in `config.py`.

| Constant | Default | What It Controls |
|---|---|---|
| `EVALUATED_MODELS` | 3 models | Which models are benchmarked |
| `CATEGORIES` | 6 categories | Task types |
| `TEMPERATURE` | `0` | Deterministic outputs |
| `DEFAULT_TOLERANCE` | `0.05` | Quality bar width (most categories) |
| `TIGHT_TOLERANCE` | `0.03` | Quality bar width (coding, architecture) |
| `SPEED_SENSITIVE` | `["summarization"]` | Categories where fastest wins |
| `JUDGE_FAILURE_THRESHOLD` | `0.20` | Failure rate that triggers a reliability warning |

---

## Reproducibility

Every run produces `meta.json` with git SHA, Python and SDK versions, full config snapshot, and a `run_fingerprint` — a SHA-256 hash of the prompts, config, and scoring weights. If the fingerprint matches between two runs, any score difference is model drift, not a setup change.

---

## Limitations

See [METHODOLOGY.md](METHODOLOGY.md) for the full methodology and an honest account of what this benchmark can and cannot claim.

---

## What's Next

Phase 8 (not building for this demo — needs live traffic data):

- **Confidence-gated categorization** — when kNN similarity is low, fall back to an LLM to classify the prompt rather than forcing a nearest-neighbour match
- **Live re-benchmark triggers** — automatically flag when accumulated live traffic suggests a model's quality has drifted from the benchmark
- **Per-user routing profiles** — let teams tune tolerances and speed/cost preferences per deployment context
- **Cluster discovery** — surface new task categories from live prompt embeddings using HDBSCAN
