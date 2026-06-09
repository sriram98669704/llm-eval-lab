# LLM Eval Lab

A structured benchmarking and routing framework built to answer one question for
Tilicho Labs: **"Which model should we use for which type of task — and when
should we automatically upgrade to a better one?"**

> 🎥 **Demo:** [Watch the walkthrough](YOUR_DEMO_URL_HERE)

---

## What It Does

Runs a full benchmark across three frontier LLMs on six task categories specific
to Tilicho Labs. Produces a routing policy that picks the right model per task —
cheapest or fastest among those that meet the quality bar — and a live handler
that grades any prompt in real-time and escalates automatically when quality falls short.

A Streamlit dashboard surfaces everything: leaderboard, per-category heatmap,
cost and latency breakdowns, routing policy, grading reliability, and a live prompt tester.

### Three models under evaluation

| Model | Provider | Strength |
|---|---|---|
| `gpt-4o` | OpenAI | Coding, debugging, reasoning |
| `claude-sonnet-4-5` | Anthropic | Architecture, writing |
| `deepseek-ai/DeepSeek-V4-Pro` | DeepSeek (via Together AI) | Speed, summarization, low cost |

### Six task categories

`architecture` · `coding` · `debugging` · `summarization` · `business_writing` · `agent_planning`

Five prompts per category (30 total). All grader-scored — no ground-truth datasets needed.

---

## How It Works

### One currency: quality

Quality (`category_score`) is a task-aware weighted dot-product of four dimensions —
correctness, reasoning, clarity, hallucination-free — with weights tuned per category
(e.g. coding weights correctness 60%, reasoning 30%, clarity 10%).

Quality is the only thing that decides:
- **Leaderboard rank** — overall quality = average of per-category scores
- **Routing eligibility** — a model must clear the `quality_bar` to be the default
- **Escalation trigger** — only a quality shortfall causes escalation

Cost and latency are shown as separate axes. They only choose *among* models that
already clear the quality bar — they never affect who ranks higher or who escalates.

### Leave-one-out grading

Each response is graded by the **other two models** — never itself. No separate
grader tier needed. The final score is the average of two independent grades per response.

```
gpt-4o response              → graded by claude-sonnet-4-5 + deepseek-ai/DeepSeek-V4-Pro
claude-sonnet-4-5 response   → graded by gpt-4o + deepseek-ai/DeepSeek-V4-Pro
deepseek-ai/DeepSeek-V4-Pro  → graded by gpt-4o + claude-sonnet-4-5
```

### Routing policy (offline)

After the benchmark, `derive_routing_table()` builds a policy per category:

- `quality_bar` = best quality − tolerance (tight 0.03 for coding/architecture, 0.05 elsewhere)
- `default` = cheapest eligible model (or fastest for `summarization`)
- `escalate_to` = highest-quality model in that category

Saved as `routing.json` inside the run folder. The `route(task)` function loads
the latest policy instantly at $0 — no model call needed.

### Live routing (online)

`run_live(prompt, category)` applies the policy to a real prompt:

1. Runs the default (cheapest/fastest eligible) model
2. Grades it live using the other two models
3. If quality ≥ bar → returns the answer (cheap/fast path, done)
4. If quality < bar → escalates to the best model, re-grades, returns that

Costs are tracked in full: answer cost + grading cost = all-in total per prompt.

---

## Dashboard

```bash
streamlit run dashboard.py
```

Five tabs:

| Tab | What it shows |
|---|---|
| **Leaderboard** | Quality Score (avg) per model, with cost and latency columns |
| **Heatmap** | Per-model × per-category quality scores with std dev on hover |
| **Cost & Latency** | Side-by-side bar charts; cost per prompt and avg latency |
| **Routing** | Policy table (quality bar formula), routing flow diagram, grading reliability |
| **Live Test** | Run any prompt through the live policy right now; see cost breakdown |

---

## File Structure

```
llm-eval-lab/
├── main.py          # Orchestrator — runs the full benchmark pipeline
├── models.py        # Evaluation engine — calls models, grades, aggregates scores
├── judge.py         # Leave-one-out grading logic
├── router.py        # derive_routing_table() + run_live() + route()
├── dispatcher.py    # Single entry point for all API calls (OpenAI / Anthropic / Together)
├── pricing.py       # Token cost calculation (per-call USD)
├── metrics.py       # Scoring math — category_score, compute_final_score, score_std
├── prompts.py       # Prompt loader
├── prompts.json     # 30 prompts (6 categories × 5 prompts)
├── storage.py       # All file I/O — writes each run into data/runs/, reads the latest
├── config.py        # Single source of truth — models, categories, tolerances, thresholds
├── dashboard.py     # Streamlit dashboard — leaderboard, heatmap, routing, live test
└── data/
    ├── index.json   # Registry of every run — the dashboard reads the newest entry
    └── runs/
        └── 2026-06-08T23-27-26/   # One folder per run — the single source of truth
            ├── summary.json       # Aggregated scores per model
            ├── routing.json       # Routing policy (the headline artifact)
            ├── results.jsonl      # Per-prompt detail records
            └── meta.json          # Reproducibility receipt — git SHA, env, config
```

Each run writes its four artifacts into its own timestamped folder and never
overwrites a previous run. The dashboard always reads the newest run, resolved
from `data/index.json`.

---

## How to Run

### Prerequisites

- Python 3.9+
- API keys for OpenAI, Anthropic, and Together AI

### Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Set API keys

Copy `.env.example` to `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TOGETHER_API_KEY=...
```

### Run the benchmark

```bash
python main.py
```

Benchmarks all three models across all six categories. Prints the leaderboard and
routing table. Saves the run under `data/runs/<timestamp>/` (summary, routing,
meta, results) and registers it in `data/index.json`.

### Launch the dashboard

```bash
source venv/bin/activate
streamlit run dashboard.py
```

### Look up the routing policy (free, no API call)

```python
from router import route
result = route("coding")
# {"model": "gpt-4o", "why": "...", "escalate_to": "gpt-4o"}
```

### Route a live prompt

```python
from router import run_live

# Policy auto-loads from the latest run — no path needed
trace = run_live("Write a FastAPI endpoint that...", "coding")
print(trace["final_response"])
```

---

## Configuration

All tunable constants live in `config.py` — nowhere else.

| Constant | Default | What It Controls |
|---|---|---|
| `EVALUATED_MODELS` | 3 cloud models | Which models are benchmarked |
| `CATEGORIES` | 6 task categories | Task types |
| `TEMPERATURE` | `0` | Deterministic outputs |
| `DEFAULT_TOLERANCE` | `0.05` | Quality bar width (most categories) |
| `TIGHT_TOLERANCE` | `0.03` | Quality bar width (coding, architecture) |
| `SPEED_SENSITIVE` | `["summarization"]` | Categories where fastest wins (not cheapest) |
| `JUDGE_FAILURE_THRESHOLD` | `0.20` | Grading failure rate that triggers a reliability warning |

---

## Reproducibility

Every run produces `meta.json` with:
- Git SHA + dirty flag
- Python and SDK versions
- Full config snapshot
- `run_fingerprint` — SHA-256 of prompts + config + scoring weights

If the fingerprint matches between two runs, any score difference is model drift —
not a config or prompt change.

---

## What's Next

**kNN live-prompt categorization** — automatically map a free-text prompt to the
right category using embeddings, so callers don't need to pass `category` explicitly.
The embedding model (`text-embedding-3-small`) is already wired in `config.py`.
