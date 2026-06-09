# Methodology

How LLM Eval Lab benchmarks, scores, and routes — and where the numbers should and should not be trusted.

---

## Scoring

### Judge dimensions

Every response is scored on four dimensions, each 0–1:

| Dimension | What it measures |
|---|---|
| `correctness` | Is the answer factually and technically right? |
| `reasoning` | Is the logic sound and the approach well-justified? |
| `clarity` | Is it clear, structured, and easy to act on? |
| `hallucination_free` | Does it avoid fabricated facts, APIs, or citations? |

### Category weights

Quality is a **task-aware weighted dot-product** of those four dimensions. Different tasks care about different things — a code snippet lives or dies on correctness; a meeting summary lives or dies on clarity and not adding things that weren't said.

| Category | correctness | reasoning | clarity | hallucination_free |
|---|---|---|---|---|
| `architecture` | 0.40 | 0.40 | — | 0.20 |
| `coding` | 0.60 | 0.30 | 0.10 | — |
| `debugging` | 0.50 | 0.40 | 0.10 | — |
| `summarization` | 0.30 | — | 0.40 | 0.30 |
| `business_writing` | 0.30 | 0.20 | 0.40 | 0.10 |
| `agent_planning` | 0.40 | 0.40 | 0.20 | — |

Weights sum to 1.0 per category, so a perfect response always scores 1.0. Missing dimensions default to 0.0 — a judge that omits a dimension is penalised, not forgiven.

### category_score

```
category_score = Σ (judge_dimension × category_weight)
               = weighted quality for one response on one task type
```

### Overall quality (the leaderboard number)

```
overall_quality = average of category_scores across all 6 categories
```

If a model fails an entire category (no valid responses), that category counts as 0.0 — a model is not rewarded with a smaller denominator for refusing or breaking.

### Score spread (error bars)

`score_std` is the population standard deviation of a model's per-prompt scores *within* a category. Small std = consistently good. Large std = unreliable — great on some prompts, poor on others. This is shown as error bars in the heatmap.

---

## Grading

### Leave-one-out jury

Each response is graded by the **other two models**, never by itself:

```
gpt-4o               →  graded by claude-sonnet-4-5 + deepseek-ai/DeepSeek-V4-Pro
claude-sonnet-4-5    →  graded by gpt-4o + deepseek-ai/DeepSeek-V4-Pro
deepseek-ai/...      →  graded by gpt-4o + claude-sonnet-4-5
```

This eliminates self-serving grades without needing a separate grader pool. It does not eliminate cross-model bias (a model may still be lenient toward a "sibling" or harsh toward a competitor), but with three very different model families the bias is reasonably distributed.

### Failed judges

If a judge fails to return parseable scores (network error, malformed output), it is **excluded** from the average — not zeroed. Zeroing a failed judge would unfairly drag down the contestant's score. Exclusion means the score is based on however many judges responded (minimum 1). If both judges fail, the response is returned ungraded and escalation is never triggered — we never escalate without a measured quality shortfall.

### Grading reliability alert

If more than 20% of responses in a model × category combination were graded by fewer than two judges, the dashboard surfaces a reliability warning and recommends re-running the benchmark.

---

## Routing

### Quality bar

After the benchmark, one routing policy is derived per category:

```
best_quality  = highest category_score among all models
tolerance     = 0.03 for coding / architecture (precision-critical)
              = 0.05 for all other categories
quality_bar   = best_quality − tolerance
```

The best model always clears its own bar (by definition), so there is always at least one eligible model.

### Default model selection

Among models that clear the quality bar:
- **Speed-sensitive categories** (`summarization`): pick the **fastest** (lowest latency). Latency breaks ties on cost.
- **All other categories**: pick the **cheapest** (lowest cost per prompt). Cost breaks ties on latency.

Cost and latency are real measured per-prompt averages from the benchmark run — not list prices.

### Escalation

`escalate_to` is always the highest-quality model in that category. When a live response falls below the quality bar, the prompt is re-run against the escalation target.

A model that is already the highest-quality model in a category cannot escalate further — its response is returned as-is even if it falls below its own bar (which can happen due to judge variance on a single live prompt).

### Live routing flow

```
1.  Look up default model for the category
2.  Call default model
3.  Grade response live (other two models)
4.  quality ≥ quality_bar?
        yes → return answer
        no  → escalate_to == default?
                    yes → return answer (nowhere to go)
                    no  → call escalation model → grade → return answer
```

All four cost components tracked: default answer, default grading, escalated answer (if any), escalated grading (if any).

---

## kNN Categorization

When a live prompt arrives without a category label, it is mapped to the nearest of the six categories using cosine similarity over text-embedding-3-small vectors.

### How it works

1. At dashboard startup, all 30 known prompts (5 per category) are embedded in one batched API call.
2. The 5 vectors per category are averaged into one fingerprint vector per category.
3. The incoming prompt is embedded and compared to all 6 fingerprints by cosine similarity.
4. The category with the highest similarity wins — it always lands on one of the six.

### Limitations

- **5 prompts per category is a small fingerprint.** The average vector captures the centre of those 5 examples. Prompts near the boundary between two categories (e.g. a technical migration proposal that reads as both `business_writing` and `architecture`) may land on the wrong side.
- **There is no "unknown" bucket.** Every prompt is forced into one of the six categories, even if it belongs to none of them.
- **No confidence threshold.** A similarity score of 0.45 and 0.90 are treated the same way — both route to the nearest category. The score is shown in the dashboard so users can judge confidence themselves.

The **Pick manually** mode in the Live Test tab is the escape hatch for any case where auto-detection gets it wrong.

---

## Prompt Selection

The 30 prompts were written to reflect real Tilicho Labs work:
- Architecture: multi-agent system design, enterprise AI integration
- Coding: FastAPI, backend services, API integrations
- Debugging: async failures, Redis queues, performance bottlenecks
- Summarization: meeting transcripts, client documents, technical specs
- Business writing: proposals, recommendations, client communications
- Agent planning: multi-agent orchestration, AI workflow design

**Limitation:** prompts were written by a human with knowledge of the benchmark's purpose. They are not drawn from a blind sample of real production traffic. A model that performs well here may still underperform on prompt distributions it has not seen.

---

## What This Benchmark Can and Cannot Claim

### It can claim

- On these 30 prompts, at this point in time, model X scores higher than model Y on task type Z by this measured margin.
- Given those scores, the derived routing policy represents the best quality-cost-latency tradeoff available within these three models.
- The escalation logic is triggered and calibrated by actually measured quality, not assumed.

### It cannot claim

- Generalization to all prompts of that task type (5 prompts is a small sample).
- Stability over time (models are updated; this is a snapshot).
- Freedom from judge bias (LLM-as-judge has known biases toward length, formatting, and self-similar style).
- That the category labels perfectly partition the real prompt space (boundaries are fuzzy).

The `run_fingerprint` in `meta.json` (SHA-256 of prompts + config + scoring weights) makes it easy to detect when any of these factors change between runs.
