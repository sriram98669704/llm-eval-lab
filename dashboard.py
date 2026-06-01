"""
dashboard.py — Interactive Streamlit dashboard.

Visualizes the latest benchmark run by reading two files produced by main.py:
  - summary.json   : aggregated per-model scores (leaderboard, strength, latency)
  - results.jsonl  : per-run detail (GSM8K reasoning, HumanEval code, judge scores)

Six tabs: Leaderboard, Strength Profile, Quality (Judge), GSM8K, HumanEval, Latency.
Data is cached for speed — click "Refresh Data" after re-running main.py to reload.

Run with:  streamlit run dashboard.py
"""

import re
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Eval Lab",
    layout="wide"
)

st.title("LLM Eval Lab")
st.caption("Benchmarking local LLMs on factual quality, math reasoning, and code generation — "
           "with ground-truth verification, sandboxed execution, and task-aware scoring.")

# ─────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_summary():
    p = Path("summary.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())

@st.cache_data
def load_records():
    p = Path("results.jsonl")
    if not p.exists():
        return []
    lines = [l for l in p.read_text().strip().split('\n') if l.strip()]
    return [json.loads(l) for l in lines]

summary = load_summary()
records = load_records()

if not summary:
    st.warning("No data found. Run `python main.py` first, then click **Refresh Data**.")
    st.stop()

models = list(summary["models"].keys())
runs_per_prompt = summary.get("runs_per_prompt", 1)  # how many attempts per problem
judge_model = summary.get("judge_model", "an independent LLM")
he_col = f"HumanEval pass@{runs_per_prompt}"
he_help = (
    f"Coding problems where at least 1 of {runs_per_prompt} generated solution(s) "
    f"passed all unit tests in a sandboxed subprocess. "
    f"pass@{runs_per_prompt} uses the unbiased HumanEval paper estimator."
    if runs_per_prompt > 1 else
    "Coding problems where the single generated solution passed all unit tests "
    "in a sandboxed subprocess on the first attempt."
)

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def clean_question(prompt):
    """Strip injected instructions to show the raw question."""
    return prompt.split("\n\nInstructions:")[0].strip()

def humaneval_problem(prompt):
    """Extract the docstring (problem description) from a HumanEval prompt."""
    m = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"'''(.*?)'''", prompt, re.DOTALL)
    return m.group(1).strip() if m else "(no description)"

def fn_name(prompt):
    m = re.search(r'def (\w+)\(', prompt)
    return m.group(1) if m else "unknown"

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
st.sidebar.title("Controls")
selected_models = st.sidebar.multiselect("Models", models, default=models)

st.sidebar.divider()
run_at = summary.get("run_at", "unknown")[:19].replace("T", " ")
st.sidebar.markdown(f"**Last run:** `{run_at}`")
st.sidebar.markdown(f"**Runs per prompt:** `{runs_per_prompt}`")
st.sidebar.markdown(f"**Detail records:** `{len(records)}`")
st.sidebar.caption("Data reflects the most recent `python main.py` run.")

if st.sidebar.button("Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("Click after re-running main.py to load new results.")

if not selected_models:
    st.info("Select at least one model from the sidebar.")
    st.stop()

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab_lb, tab_strength, tab_quality, tab_gsm, tab_he, tab_lat = st.tabs([
    "Leaderboard",
    "Strength Profile",
    "Quality (Judge)",
    "GSM8K (Math)",
    "HumanEval (Code)",
    "Latency"
])

# ═════════════════════════════════════════════════════════
# TAB: LEADERBOARD
# ═════════════════════════════════════════════════════════
with tab_lb:
    st.header("Model Leaderboard")
    st.caption("Models ranked by Final Score — a task-aware blend of all benchmarks. Hover any column header for details.")

    rows = []
    for m in selected_models:
        s = summary["models"][m]
        acc = s.get("gsm8k_accuracy", {})
        he  = s.get("humaneval_results", {})
        rows.append({
            "Model":              m,
            "Final Score":        round(s["final_score"], 4),
            "Avg Latency (s)":    round(s["avg_latency"], 2),
            "Speed Score":        round(s["normalized_speed"], 4),
            "Stability":          round(s["model_stability"], 4),
            "Judge Correctness":  round(s["avg_judge"].get("correctness", 0), 3),
            "Judge Reasoning":    round(s["avg_judge"].get("reasoning", 0), 3),
            "GSM8K Accuracy":     f"{acc.get('correct',0)}/{acc.get('total',0)} ({acc.get('accuracy',0):.0%})" if acc.get("total", 0) > 0 else "n/a",
            he_col:               f"{he.get('passed',0)}/{he.get('total',0)} ({he.get(f'pass@{runs_per_prompt}', he.get('pass@1',0)):.0%})" if he.get("total", 0) > 0 else "n/a",
        })

    df_lb = pd.DataFrame(rows).sort_values("Final Score", ascending=False)

    st.dataframe(
        df_lb,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Final Score": st.column_config.NumberColumn(
                "Final Score",
                help=(
                    f"Overall rank metric (0-1, higher is better). "
                    f"Weighted blend: 85% average task performance across all categories, "
                    f"10% output consistency across runs, 5% response speed. "
                    f"Task performance combines: "
                    f"GSM8K math accuracy (ground-truth), "
                    f"HumanEval code pass rate (sandbox-tested), and "
                    f"open-ended quality scored by {judge_model} as an independent judge on "
                    f"correctness, reasoning, clarity, and hallucination-free — "
                    f"each category weighted for what matters most in that task type."
                )
            ),
            "Avg Latency (s)": st.column_config.NumberColumn(
                "Avg Latency (s)",
                help=(
                    "Average wall-clock time (in seconds) from sending the prompt to receiving the full response. "
                    "Measured across all categories and all runs. "
                    "Includes only model inference time — embedding and judge calls are NOT included. "
                    "Lower is better. Typical local values: fast model ~5–15s, slow model ~40–80s."
                )
            ),
            "Speed Score": st.column_config.NumberColumn(
                "Speed Score",
                help=(
                    "Latency converted to a 0–1 score using the formula: 1 / (1 + avg_latency). "
                    "A model that responds in 1s scores ~0.50. At 10s it scores ~0.09. At 40s it scores ~0.02. "
                    "This intentionally penalises slow models — speed matters in real-world use. "
                    "It contributes 5% of the Final Score. "
                    "Higher is better."
                )
            ),
            "Stability": st.column_config.NumberColumn(
                "Stability",
                help="Consistency of outputs across repeated runs (pairwise embedding similarity). 1.0 = identical every time. With 1 run it is trivially 1.0."
            ),
            "Judge Correctness": st.column_config.NumberColumn(
                "Judge Correctness",
                help=(
                    f"Factual correctness score from {judge_model} acting as an independent third-party judge (0–1). "
                    "The judge evaluates responses it did not produce, scoring whether the answer is accurate and complete. "
                    "Averaged across all open-ended categories (factual, creative, etc.)."
                )
            ),
            "Judge Reasoning": st.column_config.NumberColumn(
                "Judge Reasoning",
                help=(
                    f"Reasoning quality score from {judge_model} acting as an independent third-party judge (0–1). "
                    "Assesses whether the model's logic, steps, and justification are sound. "
                    "Averaged across all open-ended categories."
                )
            ),
            "GSM8K Accuracy": st.column_config.TextColumn(
                "GSM8K Accuracy",
                help="Math word problems solved correctly, verified against ground-truth answers."
            ),
            he_col: st.column_config.TextColumn(
                he_col,
                help=he_help
            ),
        }
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            df_lb, x="Model", y="Final Score", color="Model",
            text="Final Score", title="Final Score by Model"
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_range=[0, 1.1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            df_lb, x="Model", y="Avg Latency (s)", color="Model",
            text="Avg Latency (s)", title="Avg Latency by Model (lower = better)"
        )
        fig.update_traces(texttemplate="%{text:.1f}s", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════
# TAB: STRENGTH PROFILE
# ═════════════════════════════════════════════════════════
with tab_strength:
    st.header("Strength Profile")
    st.caption(
        f"Per-task scores (0=worst, 1=best). "
        f"GSM8K and HumanEval use ground-truth verification (no subjectivity). "
        f"All other categories are scored by {judge_model} as an independent third-party judge "
        f"across four dimensions: correctness, reasoning, clarity, and hallucination-free — "
        f"weighted differently per task type (e.g. factual emphasises correctness and hallucination-free, "
        f"creative emphasises clarity, architecture emphasises reasoning and consistency)."
    )

    all_cats = []
    for m in selected_models:
        for cat in summary["models"][m].get("category_scores", {}):
            if cat not in all_cats:
                all_cats.append(cat)

    if not all_cats:
        st.info("No category scores found.")
    else:
        heat_data = {}
        for m in selected_models:
            heat_data[m] = {
                cat: summary["models"][m].get("category_scores", {}).get(cat, None)
                for cat in all_cats
            }
        df_heat = pd.DataFrame(heat_data, index=all_cats).T

        fig = px.imshow(
            df_heat,
            text_auto=".3f",
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            title="Category Score Heatmap",
            aspect="auto",
            labels={"x": "Category", "y": "Model", "color": "Score"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Score Table")
        st.dataframe(df_heat.style.format("{:.4f}"), use_container_width=True)

        st.subheader("Winner per Category")
        winner_rows = []
        for cat in all_cats:
            cat_scores = {
                m: summary["models"][m].get("category_scores", {}).get(cat)
                for m in selected_models
            }
            cat_scores = {m: s for m, s in cat_scores.items() if s is not None}
            if cat_scores:
                winner = max(cat_scores, key=lambda m: cat_scores[m])
                winner_rows.append({
                    "Category": cat,
                    "Best Model": winner,
                    "Score": round(cat_scores[winner], 4)
                })
        if winner_rows:
            st.dataframe(pd.DataFrame(winner_rows), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════
# TAB: QUALITY (JUDGE DETAIL)
# ═════════════════════════════════════════════════════════
with tab_quality:
    st.header("Quality — LLM-Judged Categories")
    st.caption(
        f"Open-ended categories (factual, creative, opinion, etc.) scored by {judge_model} "
        f"acting as an independent third-party judge — a separate model that did not produce the responses. "
        f"Four dimensions scored 0–1: "
        f"Correctness (is the answer accurate?), "
        f"Reasoning (is the logic sound?), "
        f"Clarity (is it well-expressed?), "
        f"Hallucination-free (are all claims verifiable?)."
    )

    GROUND_TRUTH = {"gsm8k", "humaneval"}
    judge_records = [
        r for r in records
        if r["category"] not in GROUND_TRUTH
        and r["model"] in selected_models
        and r.get("judge_score")
    ]

    if not judge_records:
        st.info("No judge-scored categories found. Add categories like 'factual' or 'creative' in prompts.py.")
    else:
        # Group by model + category + prompt so all runs appear under one card
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in judge_records:
            key = (r["model"], r["category"], r["prompt"])
            grouped[key].append(r)

        for (model, category, prompt), runs in grouped.items():
            # Average judge scores across all runs
            keys = ["correctness", "reasoning", "clarity", "hallucination_free"]
            avg = {k: sum(r["judge_score"].get(k, 0) for r in runs) / len(runs) for k in keys}
            avg_latency = sum(r["latency"] for r in runs) / len(runs)

            with st.container(border=True):
                top = st.columns([3, 1])
                with top[0]:
                    st.markdown(f"**{category.upper()}** · `{model}` · {len(runs)} run{'s' if len(runs) > 1 else ''}")
                    st.markdown(f"**Q:** {clean_question(prompt)}")
                with top[1]:
                    st.metric("Avg Latency", f"{avg_latency:.1f}s")

                # Averaged judge scores
                jc = st.columns(4)
                jc[0].metric("Correctness",        f"{avg['correctness']:.2f}")
                jc[1].metric("Reasoning",          f"{avg['reasoning']:.2f}")
                jc[2].metric("Clarity",            f"{avg['clarity']:.2f}")
                jc[3].metric("Hallucination-free", f"{avg['hallucination_free']:.2f}")

                # Each run collapsed
                for i, r in enumerate(runs):
                    j = r["judge_score"]
                    with st.expander(f"Run {i+1}  —  {r['latency']:.1f}s  |  C={j.get('correctness',0):.2f}  R={j.get('reasoning',0):.2f}  Cl={j.get('clarity',0):.2f}  H={j.get('hallucination_free',0):.2f}"):
                        st.markdown(r["response"])

# ═════════════════════════════════════════════════════════
# TAB: GSM8K
# ═════════════════════════════════════════════════════════
with tab_gsm:
    st.header("GSM8K — Math Reasoning")
    st.caption("Grade-school math word problems. Answers extracted from the response and "
               "verified against ground-truth — no LLM judge involved.")

    gsm8k_records = [r for r in records if r["category"] == "gsm8k"
                     and r["model"] in selected_models]

    if not gsm8k_records:
        st.info("No GSM8K results found.")
    else:
        cols = st.columns(len(selected_models))
        for i, m in enumerate(selected_models):
            mr = [r for r in gsm8k_records if r["model"] == m]
            correct = sum(1 for r in mr if r.get("gsm8k_correct"))
            total = len(mr)
            cols[i].metric(f"{m} Accuracy", f"{correct}/{total}",
                           f"{correct/total:.0%}" if total else "n/a")

        st.divider()

        for r in gsm8k_records:
            ok = r.get("gsm8k_correct")
            with st.container(border=True):
                head = st.columns([4, 1, 1])
                with head[0]:
                    st.markdown(f"`{r['model']}` · {'Correct' if ok else 'Wrong'}")
                    st.markdown(f"**Q:** {clean_question(r['prompt'])}")
                head[1].metric("Predicted", r.get("gsm8k_predicted"))
                head[2].metric("Expected",  r.get("gsm8k_expected"))

                with st.expander("View model reasoning"):
                    st.markdown(r["response"])

# ═════════════════════════════════════════════════════════
# TAB: HUMANEVAL
# ═════════════════════════════════════════════════════════
with tab_he:
    st.header("HumanEval — Code Generation")
    st.caption("Python coding problems. Generated code is executed against hidden unit tests "
               "in a sandboxed subprocess — pass means every test passed.")

    he_records = [r for r in records if r["category"] == "humaneval"
                  and r["model"] in selected_models]

    if not he_records:
        st.info("No HumanEval results found.")
    else:
        cols = st.columns(len(selected_models))
        for i, m in enumerate(selected_models):
            mr = [r for r in he_records if r["model"] == m]
            passed = sum(1 for r in mr if r.get("humaneval_passed"))
            total = len(mr)
            cols[i].metric(f"{m} pass@{runs_per_prompt}", f"{passed}/{total}",
                           f"{passed/total:.0%}" if total else "n/a")

        st.divider()

        # Group runs by model + function so each problem shows all runs together
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in he_records:
            key = (r["model"], fn_name(r["prompt"]))
            grouped[key].append(r)

        for (model, name), runs in grouped.items():
            passed_count = sum(1 for r in runs if r.get("humaneval_passed"))
            total_runs = len(runs)
            overall_ok = passed_count == total_runs
            overall_label = f"✅ All {total_runs} passed" if overall_ok else f"❌ {passed_count}/{total_runs} passed"

            with st.container(border=True):
                head = st.columns([4, 1])
                with head[0]:
                    st.markdown(f"`{model}` · **{name}()** · {overall_label}")
                    st.markdown(f"**Task:** {humaneval_problem(runs[0]['prompt'])}")
                head[1].metric("Avg Latency", f"{sum(r['latency'] for r in runs)/len(runs):.1f}s")

                # Show every run as a row
                st.markdown("**Run-by-run results:**")
                for i, r in enumerate(runs):
                    ok = r.get("humaneval_passed")
                    icon = "✅ Passed" if ok else "❌ Failed"
                    with st.expander(f"Run {i+1} — {icon}  ({r['latency']:.1f}s)", expanded=False):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Code that ran in sandbox:**")
                            code = r.get("humaneval_code") or "(not recorded)"
                            st.code(code, language="python")
                        with c2:
                            st.markdown("**Raw model response:**")
                            st.markdown(r["response"])
                        if not ok and r.get("humaneval_error"):
                            st.markdown("**Error:**")
                            st.code(r["humaneval_error"], language="text")

# ═════════════════════════════════════════════════════════
# TAB: LATENCY
# ═════════════════════════════════════════════════════════
with tab_lat:
    st.header("Latency Analysis")
    st.caption(
        "Wall-clock response time per model and category, measured from request sent to full response received. "
        "Includes model inference time only — embedding and judge calls are excluded. "
        "Lower is better. Speed Score normalises latency to 0-1 as 1/(1+latency)."
    )

    rows = []
    for m in selected_models:
        for cat, lat in summary["models"][m].get("category_latencies", {}).items():
            rows.append({"Model": m, "Category": cat, "Avg Latency (s)": round(lat, 2)})

    if rows:
        df = pd.DataFrame(rows)
        fig = px.bar(
            df, x="Category", y="Avg Latency (s)", color="Model",
            barmode="group", title="Avg Latency per Category", text="Avg Latency (s)"
        )
        fig.update_traces(texttemplate="%{text:.1f}s", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    overall = [
        {
            "Model": m,
            "Avg Latency (s)": round(summary["models"][m]["avg_latency"], 2),
            "Speed Score": round(summary["models"][m]["normalized_speed"], 4)
        }
        for m in selected_models
    ]
    df_overall = pd.DataFrame(overall)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            df_overall, x="Model", y="Avg Latency (s)", color="Model",
            title="Overall Avg Latency", text="Avg Latency (s)"
        )
        fig.update_traces(texttemplate="%{text:.1f}s", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            df_overall, x="Model", y="Speed Score", color="Model",
            title="Speed Score (higher = faster)", text="Speed Score"
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
