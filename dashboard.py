"""
dashboard.py — Streamlit dashboard for LLM Eval Lab.

Five tabs:
  1. Leaderboard  — overall model ranking by quality
  2. Eval         — per-category quality heatmap, error bars, quality vs cost scatter
  3. Benchmark    — raw evidence: prompts, judge scores, judge health, run receipt
  4. Routing      — routing policy, escalation logic, per-category decisions
  5. Live Test    — type a prompt, route it live via run_live()

Smart Refresh: checks summary.json mtime vs session_state before reloading —
no unnecessary cache clears.

Data sources — all read from the most recent run under data/runs/<timestamp>/
(located via storage.latest_run_dir(), which reads data/index.json):
  summary.json   — aggregated scores per model
  routing.json   — routing policy per category
  results.jsonl  — per-prompt detail records
  meta.json      — run receipt, fingerprint, judge health

Run with:  streamlit run dashboard.py
"""

import json
import os
import pathlib
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import CATEGORIES, JUDGE_FAILURE_THRESHOLD
from storage import latest_run_dir
from findings import compute_findings
from report import generate_report
from byok import keys_from_env, resolve_keys, redact, PROVIDERS

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Eval Lab — Tilicho",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 LLM Eval Lab")
st.markdown(
    "Benchmarking **GPT-4o** · **Claude Sonnet** · **DeepSeek V4 Pro** across 6 Tilicho task categories. "
    "<span style='opacity:0.6;'>Quality drives everything — cost and latency only choose among "
    "models that already meet the bar.</span>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────
# FILE PATHS — resolved from the most recent run under data/runs/
# If no run exists yet, fall back to placeholder paths that simply won't
# exist, so the loaders return None and the "no data yet" guard shows.
# ─────────────────────────────────────────────────────────
_RUN_DIR = latest_run_dir()
_base = _RUN_DIR if _RUN_DIR is not None else pathlib.Path(".")
SUMMARY_PATH = _base / "summary.json"
ROUTING_PATH = _base / "routing.json"
RESULTS_PATH = _base / "results.jsonl"
META_PATH    = _base / "meta.json"


# ─────────────────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_summary():
    if not SUMMARY_PATH.exists():
        return None
    try:
        return json.loads(SUMMARY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data
def load_routing():
    if not ROUTING_PATH.exists():
        return None
    try:
        return json.loads(ROUTING_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data
def load_records():
    if not RESULTS_PATH.exists():
        return []
    try:
        lines = [ln for ln in RESULTS_PATH.read_text().strip().split("\n") if ln.strip()]
        return [json.loads(ln) for ln in lines]
    except (json.JSONDecodeError, OSError):
        return []


@st.cache_data
def load_meta():
    if not META_PATH.exists():
        return None
    try:
        return json.loads(META_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


summary = load_summary()
routing = load_routing()
records = load_records()
meta    = load_meta()


# ─────────────────────────────────────────────────────────
# GUARD — no data yet
# ─────────────────────────────────────────────────────────
if not summary or not summary.get("models"):
    st.warning(
        "**No benchmark data found.**  \n"
        "Run `python main.py` to generate results, then click **🔄 Refresh** in the sidebar."
    )
    st.stop()

ALL_MODELS = list(summary["models"].keys())


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
st.sidebar.title("Controls")

# Let long model names (e.g. deepseek-ai/DeepSeek-V4-Pro) wrap and show in full
# inside the multiselect, instead of being clipped to "deepseek…".
st.sidebar.markdown(
    """
    <style>
    section[data-testid="stSidebar"] [data-baseweb="tag"] {
        max-width: 100% !important;
        height: auto !important;
        white-space: normal !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="tag"] span {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

selected_models = st.sidebar.multiselect(
    "Models to show",
    options=ALL_MODELS,
    default=ALL_MODELS,
    help="Filter which models appear in all tabs. Deselecting a model removes it from every chart and table.",
)

_report_filename = f"llm-eval-report-{summary.get('run_at', 'unknown')[:19].replace(':', '-')}.md"
st.sidebar.download_button(
    label="📥 Download Report",
    data=generate_report(summary, routing, compute_findings(summary, routing), meta),
    file_name=_report_filename,
    mime="text/markdown",
    width="stretch",
    help="One-page Markdown report: key findings, leaderboard, per-category scores, and routing policy.",
)

st.sidebar.divider()

# Last run timestamp
run_at = summary.get("run_at", "")
run_at_display = run_at[:19].replace("T", " ") if run_at else "unknown"
st.sidebar.markdown(f"**Last run:** `{run_at_display}`")

# Run fingerprint and git info from meta.json
if meta:
    fp = meta.get("run_fingerprint", "")
    if fp:
        # Native backticks → same green code-block style as Last run / Git.
        # Streamlit's native help= tooltip (ℹ️) holds the FULL hash + explainer.
        fp_help = (
            f"**Full hash**\n\n`{fp}`\n\n"
            "SHA of prompts + config + scoring weights. "
            "Same fingerprint across two runs means any score difference is model drift — "
            "not a config or prompt change."
        )
        st.sidebar.markdown(
            f"**Fingerprint:** `{fp[:12]}…`",
            help=fp_help,
        )
    git = meta.get("code", {})
    if git.get("git_sha"):
        dirty = " ⚠️ dirty" if git.get("git_dirty") else ""
        st.sidebar.markdown(
            f"**Git:** `{git.get('git_sha', '?')}` on `{git.get('git_branch', '?')}`{dirty}"
        )

st.sidebar.divider()

# Smart Refresh — checks mtime before clearing cache
if st.sidebar.button("🔄 Refresh", width="stretch"):
    if SUMMARY_PATH.exists():
        file_mtime = SUMMARY_PATH.stat().st_mtime
        if file_mtime == st.session_state.get("loaded_at"):
            st.toast("Already up to date — no reload needed.", icon="✅")
        else:
            st.cache_data.clear()
            st.session_state["loaded_at"] = file_mtime
            st.rerun()
    else:
        st.cache_data.clear()
        st.rerun()

st.sidebar.caption("Click after running `python main.py` to load new results.")

# Record mtime on first load so Refresh can compare
if SUMMARY_PATH.exists() and "loaded_at" not in st.session_state:
    st.session_state["loaded_at"] = SUMMARY_PATH.stat().st_mtime

if not selected_models:
    st.info("Select at least one model in the sidebar.")
    st.stop()


# ─────────────────────────────────────────────────────────
# KEY FINDINGS — headline cards shown above all tabs
# ─────────────────────────────────────────────────────────
_findings = compute_findings(summary, routing)

if _findings:
    _w = _findings["winner"]
    _ru = _findings["runner_up"]
    _tr = _findings["trailing"]
    _bg = _findings["biggest_gap"]
    _bv = _findings["best_value"]
    _esc = _findings["escalation_categories"]

    _gap_to_runner = _w["score"] - _ru["score"]
    _cat_label = _bg["category"].replace("_", " ").title() if _bg["category"] else "—"
    _esc_count = len(_esc)
    _esc_names = ", ".join(c.replace("_", " ") for c in _esc) if _esc else "none"

    # Custom HTML cards — avoids st.metric's value truncation in narrow columns.
    # Each card: icon + label row, large value, small sub-line, tooltip via <title>.
    _card_css = """
    <style>
    .finding-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 10px;
        padding: 14px 16px 12px;
        height: 100%;
        box-sizing: border-box;
    }
    .finding-card .fc-label {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.55);
        margin-bottom: 6px;
        white-space: nowrap;
    }
    .finding-card .fc-value {
        font-size: 1.35rem;
        font-weight: 700;
        line-height: 1.2;
        word-break: break-word;
        margin-bottom: 4px;
    }
    .finding-card .fc-sub {
        font-size: 0.72rem;
        color: #4caf87;
    }
    .finding-card .fc-desc {
        font-size: 0.70rem;
        color: rgba(255,255,255,0.45);
        margin-top: 5px;
        line-height: 1.4;
    }
    </style>
    """
    st.markdown(_card_css, unsafe_allow_html=True)

    _c1, _c2, _c3, _c4 = st.columns(4)

    with _c1:
        st.markdown(
            f"""<div class="finding-card" title="{_w['short_name']} scores {_w['score']:.3f} — the highest average quality score across all 6 task categories. Ahead of {_ru['short_name']} ({_ru['score']:.3f}) by {_gap_to_runner:.3f}. {_tr['short_name']} trails at {_tr['score']:.3f}. Quality score = average of per-category scores, where each score is graded by the other two models on correctness, reasoning, clarity, and hallucination-free. 1.0 is perfect.">
  <div class="fc-label">🏆 Overall winner</div>
  <div class="fc-value">{_w['short_name']}</div>
  <div class="fc-sub">↑ {_w['score']:.3f} average quality score across 6 categories</div>
</div>""",
            unsafe_allow_html=True,
        )

    with _c2:
        st.markdown(
            f"""<div class="finding-card" title="{_cat_label} has the widest quality spread across models. {_bg['best_short']} scores {_bg['best_score']:.3f}; {_bg['worst_short']} scores {_bg['worst_score']:.3f} — a {_bg['gap']:.3f} gap. This means model choice matters most for {_cat_label} tasks.">
  <div class="fc-label">📊 Biggest category gap</div>
  <div class="fc-value">{_cat_label}</div>
  <div class="fc-sub">↑ ±{_bg['gap']:.2f} spread · best vs worst: {_bg['best_short']} ({_bg['best_score']:.2f}) → {_bg['worst_short']} ({_bg['worst_score']:.2f})</div>
</div>""",
            unsafe_allow_html=True,
        )

    with _c3:
        st.markdown(
            f"""<div class="finding-card" title="{_bv['short_name']} has the best quality-per-dollar ratio. Average quality score across 6 categories: {_bv['quality']:.3f}. Average measured cost per prompt: ${_bv['avg_cost']:.4f} — real API costs from the benchmark run, not list prices.">
  <div class="fc-label">💰 Best quality per dollar</div>
  <div class="fc-value">{_bv['short_name']}</div>
  <div class="fc-sub">↑ {_bv['quality']:.3f} quality · ${_bv['avg_cost']:.4f}/prompt</div>
</div>""",
            unsafe_allow_html=True,
        )

    with _c4:
        st.markdown(
            f"""<div class="finding-card" title="{_esc_count} categories have a real escalation path — the default (cheapest/fastest) model differs from the best model, so a low-quality live response can be retried against a stronger model. Categories with escalation: {_esc_names}. The other {6 - _esc_count} are already routed to their top model — a failed response is returned as-is.">
  <div class="fc-label">⚡ Escalation paths</div>
  <div class="fc-value">{_esc_count} of 6 categories</div>
  <div class="fc-sub">↑ {_esc_names}</div>
</div>""",
            unsafe_allow_html=True,
        )

    st.divider()


# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
NAV = ["🏆 Leaderboard", "📊 Eval", "🔬 Benchmark", "🗺️ Routing", "⚡ Live Test"]
active = st.segmented_control(
    "Section",
    NAV,
    default=NAV[0],
    key="active_tab",
    label_visibility="collapsed",
    width="stretch",
)
# segmented_control returns None if you click the active pill again (deselect) —
# fall back to the first section so something is always showing.
if active is None:
    active = NAV[0]


# ═════════════════════════════════════════════════════════
# TAB 1 — LEADERBOARD
# ═════════════════════════════════════════════════════════
if active == "🏆 Leaderboard":
    st.header("Model Leaderboard")
    st.caption(
        "Models ranked by Quality Score — average quality across all 6 categories. "
        "A model that fails a whole category gets 0.0 for it; "
        "the denominator is always 6 so no model can hide a bad category behind a smaller average."
    )

    rows = []
    for m in selected_models:
        s  = summary["models"][m]
        aj = s.get("avg_judge", {})
        costs = list(s.get("category_costs", {}).values())
        avg_cost = round(sum(costs) / len(costs), 6) if costs else None
        rows.append({
            "Model":              m,
            "Quality Score (avg)": round(s["final_score"], 4),
            "Avg Latency (s)":    round(s["avg_latency"], 2),
            "Avg Cost/prompt":    avg_cost,
            "Correctness":        round(aj.get("correctness", 0), 3),
            "Reasoning":          round(aj.get("reasoning", 0), 3),
            "Clarity":            round(aj.get("clarity", 0), 3),
            "Hallucination-free": round(aj.get("hallucination_free", 0), 3),
        })

    df_lb = pd.DataFrame(rows).sort_values("Quality Score (avg)", ascending=False)

    st.dataframe(
        df_lb,
        width="stretch",
        hide_index=True,
        column_config={
            "Quality Score (avg)": st.column_config.NumberColumn(
                "Quality Score (avg)",
                format="%.4f",
                help=(
                    "Average quality across all 6 categories (0–1, higher is better). "
                    "Fixed denominator of 6 — a model that scores 0 in a whole category "
                    "cannot inflate its score by having fewer categories averaged."
                ),
            ),
            "Avg Latency (s)": st.column_config.NumberColumn(
                "Avg Latency (s)",
                format="%.2f",
                help=(
                    "Wall-clock seconds from request sent to full response received. "
                    "Averaged across all 6 categories. "
                    "Model inference time only — grader calls are excluded. Lower is better."
                ),
            ),
            "Avg Cost/prompt": st.column_config.NumberColumn(
                "Avg Cost/prompt ($)",
                format="$%.6f",
                help=(
                    "Average USD cost per prompt, averaged across all categories. "
                    "Based on published token pricing for each API. Lower is better."
                ),
            ),
            "Correctness": st.column_config.NumberColumn(
                "Correctness",
                format="%.3f",
                help=(
                    "Is the answer factually accurate and complete? "
                    "Scored 0–1 by the leave-one-out graders (each response graded by the OTHER two models). "
                    "Averaged across all categories."
                ),
            ),
            "Reasoning": st.column_config.NumberColumn(
                "Reasoning",
                format="%.3f",
                help=(
                    "Is the logic sound and the steps well-structured? "
                    "Scored 0–1 by the graders. Averaged across all categories."
                ),
            ),
            "Clarity": st.column_config.NumberColumn(
                "Clarity",
                format="%.3f",
                help=(
                    "Is the response clear, well-expressed, and easy to follow? "
                    "Scored 0–1 by the graders. Averaged across all categories."
                ),
            ),
            "Hallucination-free": st.column_config.NumberColumn(
                "Hallucination-free",
                format="%.3f",
                help=(
                    "Are all claims in the response verifiable and accurate? "
                    "1.0 = no hallucinations detected by the graders. "
                    "Averaged across all categories."
                ),
            ),
        },
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            df_lb, x="Model", y="Quality Score (avg)", color="Model",
            text="Quality Score (avg)",
            title="Quality Score by Model (avg across 6 categories)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(
            showlegend=False,
            yaxis_range=[0, 1.1],
            yaxis_title="Quality score (0–1)",
        )
        st.plotly_chart(fig)

    with col2:
        fig = px.bar(
            df_lb, x="Model", y="Avg Latency (s)", color="Model",
            text="Avg Latency (s)",
            title="Avg Latency by Model (lower = faster)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(texttemplate="%{text:.1f}s", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_title="Seconds")
        st.plotly_chart(fig)


# ═════════════════════════════════════════════════════════
# TAB 2 — EVAL  (Step 18)
# ═════════════════════════════════════════════════════════
if active == "📊 Eval":
    st.header("Eval — Quality by Category")
    st.caption(
        "Where is each model strong, and where does it fall short? "
        "Green = high quality, red = low quality. "
        "Hover any cell for the score and ± std dev — a wide spread means "
        "the model was inconsistent across the prompts in that category."
    )

    # Build score and std dev matrices: rows = models, cols = categories
    score_matrix = {}
    std_matrix   = {}
    for m in selected_models:
        s = summary["models"][m]
        score_matrix[m] = {c: s.get("category_scores", {}).get(c) for c in CATEGORIES}
        std_matrix[m]   = {c: s.get("category_score_stds", {}).get(c) for c in CATEGORIES}

    df_scores = pd.DataFrame(score_matrix, index=CATEGORIES).T  # models × categories
    df_stds   = pd.DataFrame(std_matrix,   index=CATEGORIES).T

    # Rich hover text per cell
    hover_text = []
    for m in df_scores.index:
        row_hover = []
        for c in df_scores.columns:
            score = df_scores.loc[m, c]
            std   = df_stds.loc[m, c]
            if score is None:
                row_hover.append(f"<b>{m} — {c}</b><br>No data for this category")
            elif std is not None:
                row_hover.append(
                    f"<b>{m} — {c}</b><br>"
                    f"<b>Quality: {score:.4f} ± {std:.4f}</b><br>"
                    f"<i>± = std dev across the category's prompts<br>Wide spread = inconsistent model</i>"
                )
            else:
                row_hover.append(f"<b>{m} — {c}</b><br><b>Quality: {score:.4f}</b>")
        hover_text.append(row_hover)

    # Replace None with 0 for the colour scale; hover text still shows "No data"
    z_values    = [[v if v is not None else 0.0 for v in row] for row in df_scores.values.tolist()]
    text_values = [[f"{v:.3f}" if v is not None else "—"      for v in row] for row in df_scores.values.tolist()]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=list(df_scores.columns),
        y=list(df_scores.index),
        text=text_values,
        texttemplate="%{text}",
        hovertext=hover_text,
        hovertemplate="%{hovertext}<extra></extra>",
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        colorbar=dict(title="Quality<br>(0–1)"),
    ))
    fig.update_layout(
        title="Quality Score Heatmap — models × categories",
        xaxis_title="Category",
        yaxis_title="Model",
        height=max(300, 120 + 80 * len(selected_models)),
    )
    st.plotly_chart(fig)

    # Winner per category table
    st.subheader("Winner per Category")
    st.caption("Which model scored highest in each category, and how far ahead of second place.")

    winner_rows = []
    for c in CATEGORIES:
        cat_scores_map = {
            m: summary["models"][m].get("category_scores", {}).get(c)
            for m in selected_models
        }
        cat_scores_map = {m: v for m, v in cat_scores_map.items() if v is not None}
        if not cat_scores_map:
            continue
        ranked = sorted(cat_scores_map, key=lambda m: cat_scores_map[m], reverse=True)
        winner = ranked[0]
        margin = (
            round(cat_scores_map[winner] - cat_scores_map[ranked[1]], 4)
            if len(ranked) > 1 else None
        )
        winner_rows.append({
            "Category":        c,
            "Winner":          winner,
            "Score":           round(cat_scores_map[winner], 4),
            "Margin over 2nd": f"+{margin:.4f}" if margin is not None else "—",
        })

    if winner_rows:
        st.dataframe(
            pd.DataFrame(winner_rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Score": st.column_config.NumberColumn("Score", format="%.4f"),
                "Margin over 2nd": st.column_config.TextColumn(
                    "Margin over 2nd",
                    help=(
                        "How far the winner is ahead of the second-best model. "
                        "Larger margin = more dominant in this category."
                    ),
                ),
            },
        )

    # Quality vs Cost scatter
    st.subheader("Quality vs Cost")
    st.caption(
        "One dot = one model in one category. "
        "**Top-left** = best value (high quality, low cost per prompt). "
        "**Bottom-right** = expensive and underperforming."
    )

    scatter_rows = []
    for m in selected_models:
        s = summary["models"][m]
        for c in CATEGORIES:
            q    = s.get("category_scores", {}).get(c)
            cost = s.get("category_costs", {}).get(c)
            if q is not None and cost is not None:
                scatter_rows.append({
                    "Model":        m,
                    "Category":     c,
                    "Quality":      round(q, 4),
                    "Cost/prompt":  cost,
                })

    if scatter_rows:
        df_scatter = pd.DataFrame(scatter_rows)
        fig = px.scatter(
            df_scatter,
            x="Cost/prompt", y="Quality",
            color="Model", symbol="Category",
            hover_data={
                "Model":       True,
                "Category":    True,
                "Quality":     ":.4f",
                "Cost/prompt": ":.6f",
            },
            title="Quality vs Cost per prompt — one dot per model per category",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(marker=dict(size=12))
        fig.update_layout(
            yaxis_range=[0, 1.05],
            yaxis_title="Quality score (0–1)",
            xaxis_title="Avg cost per prompt (USD)",
        )
        st.plotly_chart(fig)
    else:
        st.info(
            "Cost data not available — run a benchmark with live API keys to populate this chart."
        )


# ═════════════════════════════════════════════════════════
# TAB 3 — BENCHMARK  (Step 19)
# ═════════════════════════════════════════════════════════
if active == "🔬 Benchmark":
    st.header("Benchmark — Raw Evidence")
    st.caption(
        "The raw data behind the Eval tab — every prompt, every answer, and every grader's score. "
        "The **Scoring Reliability** section below shows how many answers got both graders, "
        "only one, or none."
    )

    # ── Run receipt ─────────────────────────────────────────
    if meta:
        with st.expander("📋 Run Receipt", expanded=True):
            run_info  = meta.get("run", {})
            code_info = meta.get("code", {})
            pmt_info  = meta.get("prompts", {})
            env_info  = meta.get("environment", {})

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Started",
                run_info.get("started_at", "")[:19].replace("T", " "),
                help="When this benchmark run began.",
            )
            c2.metric(
                "Duration",
                f"{run_info.get('duration_seconds', 0):.0f}s",
                help="Total wall-clock time for the full benchmark (all models, all categories).",
            )
            n_models      = len(meta.get("config", {}).get("evaluated_models", [])) or len(ALL_MODELS)
            total_prompts = pmt_info.get("total", 0)
            c3.metric(
                "Total prompts",
                str(pmt_info.get("total", "—")),
                help=(
                    f"Prompts run per model this run. Contestant calls = "
                    f"{n_models} models × {total_prompts} prompts = {n_models * total_prompts} "
                    "(live grader calls are additional)."
                ),
            )
            fp = meta.get("run_fingerprint", "")
            c4.metric(
                "Fingerprint",
                (fp[:10] + "…") if fp else "—",
                help=(
                    "SHA-256 of prompts + config + scoring weights. "
                    "Same fingerprint = any score difference between runs is pure model drift."
                ),
            )

            git_sha    = code_info.get("git_sha", "unknown")
            git_branch = code_info.get("git_branch", "unknown")
            git_dirty  = code_info.get("git_dirty", False)
            st.caption(
                f"Git: `{git_sha}` on `{git_branch}`"
                + ("  ⚠️ dirty (uncommitted changes at run time)" if git_dirty else "")
            )

            pkgs = env_info.get("packages", {})
            if pkgs:
                st.caption("SDKs: " + "  ·  ".join(f"{k} {v}" for k, v in pkgs.items()))
    else:
        st.info("No meta.json found — run `python main.py` first to generate the run receipt.")

    # ── Judge health ─────────────────────────────────────────
    st.subheader("Scoring Reliability")
    st.caption(
        "Every answer is graded by 2 of the 3 models — a model never grades itself. "
        "**2/2 graders** = fully scored (most reliable). "
        "**1/2 graders** = only one grader responded (score is softer). "
        "**0/2 graders** = both failed, answer excluded from quality scoring entirely."
    )

    # judge_health lives in meta.json["judge_health"]["by_contestant_category"]
    # Shape: { contestant_model: { category: { scored, single_judge, unscored } } }
    judge_health_by_model = (
        meta.get("judge_health", {}).get("by_contestant_category", {})
        if meta else {}
    )

    if judge_health_by_model:
        health_rows = []
        alert_count = 0

        for m in selected_models:
            model_health = judge_health_by_model.get(m, {})
            for c in CATEGORIES:
                ch       = model_health.get(c, {})
                scored   = ch.get("scored", 0)
                single   = ch.get("single_judge", 0)
                unscored = ch.get("unscored", 0)
                total    = scored + single + unscored
                fail_pct = round((single + unscored) / total * 100, 1) if total > 0 else 0.0
                if total > 0 and fail_pct / 100 > JUDGE_FAILURE_THRESHOLD:
                    alert_count += 1
                health_rows.append({
                    "Model":         m,
                    "Category":      c,
                    "2/2 graders":   scored,
                    "1/2 graders":   single,
                    "0/2 graders":   unscored,
                    "Fail %":        fail_pct,
                })

        if alert_count:
            st.error(
                f"⚠️ {alert_count} model/category combination{'s' if alert_count > 1 else ''} "
                f"had more than {int(JUDGE_FAILURE_THRESHOLD * 100)}% of answers graded by fewer "
                "than 2 graders — those quality scores are unreliable (highlighted below). "
                "Re-run the benchmark (`python main.py`) to fix this."
            )

        health_df = pd.DataFrame(health_rows)

        def _highlight_alert(row):
            if row["Fail %"] / 100 > JUDGE_FAILURE_THRESHOLD:
                return ["background-color: rgba(220, 38, 38, 0.22)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            health_df.style.apply(_highlight_alert, axis=1),
            width="stretch",
            hide_index=True,
            column_config={
                "2/2 graders": st.column_config.NumberColumn(
                    "2/2 graders",
                    help="Both graders returned a score — full confidence in the quality number.",
                ),
                "1/2 graders": st.column_config.NumberColumn(
                    "1/2 graders",
                    help=(
                        "Only 1 of the 2 graders succeeded. "
                        "The score rests on a single grader — less reliable than a full pair."
                    ),
                ),
                "0/2 graders": st.column_config.NumberColumn(
                    "0/2 graders",
                    help=(
                        "Both graders failed — this answer is excluded from quality scoring entirely. "
                        "A high count here = re-run the benchmark (`python main.py`)."
                    ),
                ),
                "Fail %": st.column_config.NumberColumn(
                    "Fail %",
                    format="%.1f%%",
                    help=(
                        f"(1/2 + 0/2 graders) ÷ total answers × 100. "
                        f"Red banner triggers when this passes {int(JUDGE_FAILURE_THRESHOLD * 100)}%."
                    ),
                ),
            },
        )

        # Per-judge-model reliability
        judge_reliability = meta.get("judge_health", {}).get("by_judge_model", {}) if meta else {}
        if judge_reliability:
            st.markdown("#### Per-grader success rate")
            st.caption("How often each model returned a usable score when it acted as a grader.")
            rel_rows = []
            for jm, jd in judge_reliability.items():
                attempted = jd.get("attempted", 0)
                succeeded = jd.get("succeeded", 0)
                fail_pct  = round((1.0 - succeeded / attempted) * 100, 1) if attempted > 0 else 0.0
                rel_rows.append({
                    "Grader":    jm,
                    "Asked":     attempted,
                    "Scored":    succeeded,
                    "Fail %":    fail_pct,
                })
            st.dataframe(
                pd.DataFrame(rel_rows),
                width="stretch",
                hide_index=True,
                column_config={
                    "Asked": st.column_config.NumberColumn(
                        "Asked",
                        help="How many answers this model was asked to grade across the whole run.",
                    ),
                    "Scored": st.column_config.NumberColumn(
                        "Scored",
                        help="How many of those it actually returned a usable score for.",
                    ),
                    "Fail %": st.column_config.NumberColumn(
                        "Fail %",
                        format="%.1f%%",
                        help="How often this model failed to return a usable score when it was a grader.",
                    ),
                },
            )
    else:
        st.info("Scoring reliability data not available — run `python main.py` first.")

    # ── Per-prompt detail ─────────────────────────────────────
    st.subheader("Per-Prompt Detail")
    st.caption(
        "Expand a category to see each prompt with every model's answer side by side — "
        "the responses and the graders' scores, head-to-head on the same prompt."
    )

    if not records:
        st.info("No results.jsonl found — run the benchmark first.")
    else:
        for c in CATEGORIES:
            cat_records = [
                r for r in records
                if r.get("category") == c and r.get("model") in selected_models
            ]
            if not cat_records:
                continue

            # Group by prompt so all models that answered the SAME prompt sit
            # together (head-to-head), instead of a flat list of separate answers.
            prompts_in_order = []
            by_prompt = {}
            for r in cat_records:
                p = r.get("prompt", "")
                if p not in by_prompt:
                    by_prompt[p] = []
                    prompts_in_order.append(p)
                by_prompt[p].append(r)

            with st.expander(
                f"📂 {c.upper()} — {len(prompts_in_order)} prompt(s), {len(cat_records)} answer(s)"
            ):
                for pi, p in enumerate(prompts_in_order, start=1):
                    group = by_prompt[p]
                    st.markdown(f"**Prompt {pi}:** {p}")

                    for r in group:
                        js = r.get("judge_score") or {}
                        with st.container(border=True):
                            top = st.columns([3, 1, 1])
                            with top[0]:
                                st.markdown(f"**`{r.get('model', '—')}`**")

                            lat  = r.get("latency", 0) or 0
                            cost = r.get("cost_usd")
                            top[1].metric(
                                "Latency", f"{lat:.1f}s",
                                help="Wall-clock seconds for this single model call.",
                            )
                            top[2].metric(
                                "Cost", f"${cost:.6f}" if cost is not None else "n/a",
                                help="USD cost for this single model call.",
                            )

                            if js:
                                jc = st.columns(4)
                                jc[0].metric(
                                    "Correctness", f"{js.get('correctness', 0):.2f}",
                                    help="Factual accuracy of this answer, scored 0–1 by the graders.",
                                )
                                jc[1].metric(
                                    "Reasoning", f"{js.get('reasoning', 0):.2f}",
                                    help="Logic quality of this answer, scored 0–1 by the graders.",
                                )
                                jc[2].metric(
                                    "Clarity", f"{js.get('clarity', 0):.2f}",
                                    help="Clarity of expression for this answer, scored 0–1 by the graders.",
                                )
                                jc[3].metric(
                                    "Hallucination-free", f"{js.get('hallucination_free', 0):.2f}",
                                    help=(
                                        "1.0 = no hallucinations detected by the graders. "
                                        "Scored 0–1 per answer."
                                    ),
                                )
                            else:
                                st.markdown(
                                    "⚠️ **No grade for this answer** — both graders failed or returned "
                                    "nothing. Excluded from quality scoring."
                                )

                            with st.expander("View full response"):
                                st.markdown(r.get("response") or "(no response recorded)")

                    st.divider()


# ═════════════════════════════════════════════════════════
# TAB 4 — ROUTING  (Step 20)
# ═════════════════════════════════════════════════════════
if active == "🗺️ Routing":
    st.header("Routing Policy")
    st.caption(
        "Which model handles each task category by default — and exactly why. "
        "These rules are built automatically from the benchmark scores, and they're what the "
        "Live Test tab uses to route a real prompt."
    )

    # Flow diagram
    st.subheader("How live routing works")
    st.code(
        "Every live prompt\n"
        "↓\n"
        "Run DEFAULT model  ←  cheapest/fastest model that already clears the quality bar\n"
        "↓\n"
        "The other two models grade that answer  (a model never grades itself)\n"
        "↓\n"
        "Good enough? ──YES──▶  Return it  (cheap / fast path, done)\n"
        "│\n"
        "NO\n"
        "↓\n"
        "Escalate to the BEST model for this category\n"
        "↓\n"
        "The other two grade the new answer  →  Return it",
        language=None,
    )
    st.markdown(
        "**What is the \"quality bar\"?**  It's the score an answer has to beat to be accepted "
        "without escalating. Take the **best** model's average score for this category, then "
        "subtract a small cushion — **0.03** for coding and architecture (where precision really "
        "matters), **0.05** everywhere else. That average is across *all* the prompts in the "
        "category, not a single answer.  \n\n"
        "Escalation is about **quality only** — cost and speed never trigger it. If a cheaper model "
        "already clears the bar, it stays the default and we keep the savings."
    )

    if not routing or not routing.get("policy"):
        st.warning(
            "No routing.json found. Run `python main.py` to generate the routing policy, "
            "then click Refresh."
        )
    else:
        policy = routing["policy"]
        gen_at = routing.get("generated_at", "")[:19].replace("T", " ")
        st.caption(
            f"Routing rules last built on `{gen_at}` from the most recent benchmark run. "
            "Re-run `python main.py` to update."
        )

        # ── Routing summary table ────────────────────────────────
        st.subheader("Routing Table")

        table_rows = []
        for c in CATEGORIES:
            pol = policy.get(c)
            if not pol:
                continue
            table_rows.append({
                "Category":          c,
                "Default model":     pol["default"],
                "Default picked by": pol.get("selection", "cheapest"),
                "Best quality":      round(pol["best_quality"], 4),
                "Tolerance":         pol.get("tolerance", "—"),
                "Quality bar":       round(pol["quality_bar"], 4),
                "Escalates to (best model)": pol["escalate_to"],
                "Prompts used":      pol.get("prompt_count") or "—",
            })

        if table_rows:
            st.dataframe(
                pd.DataFrame(table_rows),
                width="stretch",
                hide_index=True,
                column_config={
                    "Default model": st.column_config.TextColumn(
                        "Default model",
                        help=(
                            "The model that handles live prompts for this category by default. "
                            "Selected as cheapest or fastest among all models that cleared the quality bar."
                        ),
                    ),
                    "Default picked by": st.column_config.TextColumn(
                        "Default picked by",
                        help=(
                            "'cheapest' — lowest cost per prompt among eligible models (latency breaks ties).  \n"
                            "'fastest' — lowest latency among eligible models (cost breaks ties).  \n"
                            "Only summarization is speed-sensitive; all others use cheapest."
                        ),
                    ),
                    "Quality bar": st.column_config.NumberColumn(
                        "Quality bar",
                        format="%.4f",
                        help=(
                            "A model must score ≥ this to be eligible as the category default. "
                            "= best model's average quality − tolerance. The best model always clears its own bar."
                        ),
                    ),
                    "Best quality": st.column_config.NumberColumn(
                        "Best quality",
                        format="%.4f",
                        help=(
                            "The highest average quality any model reached in this category "
                            "(averaged across its prompts) — achieved by the model in "
                            "'Escalates to (best model)'."
                        ),
                    ),
                    "Tolerance": st.column_config.NumberColumn(
                        "Tolerance",
                        format="%.2f",
                        help=(
                            "How far a model can trail the best quality score and still be eligible. "
                            "Tighter (0.03) for coding/architecture. Looser (0.05) for everything else."
                        ),
                    ),
                    "Escalates to (best model)": st.column_config.TextColumn(
                        "Escalates to (best model)",
                        help=(
                            "The highest-quality model for this category — its score is the "
                            "'Best quality' column. If a live response scores below the quality bar, "
                            "the prompt is re-run with this model."
                        ),
                    ),
                    "Prompts used": st.column_config.TextColumn(
                        "Prompts used",
                        help="Number of benchmark prompts this routing decision is based on.",
                    ),
                },
            )

        # ── Per-category cards ───────────────────────────────────
        st.subheader("Per-Category Breakdown")
        st.caption(
            "Full transparency on every model's quality, cost, and latency — "
            "and the plain-English reason for each routing decision."
        )

        for c in CATEGORIES:
            pol = policy.get(c)
            if not pol:
                continue

            default      = pol["default"]
            escalate_to  = pol["escalate_to"]
            bar          = pol["quality_bar"]
            selection    = pol.get("selection", "cheapest")
            models_block = pol.get("models", {})
            best_q       = models_block.get(escalate_to, {}).get("quality")
            default_q    = models_block.get(default, {}).get("quality")
            eligible     = [m for m, d in models_block.items() if d.get("clears_bar")]

            with st.expander(
                f"**{c.upper()}**  —  default: `{default}`  |  escalates to: `{escalate_to}`  |  bar: {bar:.4f}"
            ):
                # Plain-English explanation of this decision
                if best_q is not None and default_q is not None:
                    if default == escalate_to:
                        st.info(
                            f"**`{default}`** is both the highest-quality model "
                            f"(quality {best_q:.4f}) and the {selection} eligible model — "
                            f"it handles everything in this category with no escalation needed."
                        )
                    else:
                        eligible_str = ", ".join(f"`{m}`" for m in eligible)
                        st.info(
                            f"**`{escalate_to}`** is the best model for {c} (quality {best_q:.4f}).  \n"
                            f"**`{default}`** also clears the bar (quality {default_q:.4f}) and is the "
                            f"{selection} eligible model → it's the default.  \n"
                            f"Eligible models (all score ≥ {bar:.4f}): {eligible_str}.  \n"
                            f"If a live response falls below {bar:.4f}, it escalates to `{escalate_to}`."
                        )

                # Per-model comparison table for this category
                model_rows = []
                for mm, d in models_block.items():
                    if mm not in (selected_models if selected_models else ALL_MODELS):
                        continue
                    role = (
                        "default" if mm == default
                        else "escalate_to" if mm == escalate_to
                        else "—"
                    )
                    model_rows.append({
                        "Model":        mm,
                        "Quality":      round(d["quality"], 4),
                        "Cost/prompt":  d.get("cost_usd"),
                        "Latency (s)":  round(d["latency"], 3) if d.get("latency") is not None else None,
                        "Clears bar":   "✅" if d.get("clears_bar") else "❌",
                        "Role":         role,
                    })

                if model_rows:
                    st.dataframe(
                        pd.DataFrame(model_rows).sort_values("Quality", ascending=False),
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Quality": st.column_config.NumberColumn(
                                "Quality",
                                format="%.4f",
                                help=f"Category quality score (0–1). Quality bar for this category = {bar:.4f}.",
                            ),
                            "Cost/prompt": st.column_config.NumberColumn(
                                "Cost/prompt ($)",
                                format="$%.6f",
                                help="Average USD cost per prompt for this category.",
                            ),
                            "Latency (s)": st.column_config.NumberColumn(
                                "Latency (s)",
                                format="%.2f",
                                help="Average latency per prompt for this category (seconds).",
                            ),
                            "Clears bar": st.column_config.TextColumn(
                                "Clears bar",
                                help=f"Quality ≥ {bar:.4f} — eligible to be the default model.",
                            ),
                            "Role": st.column_config.TextColumn(
                                "Role",
                                help=(
                                    "'default' = handles live prompts. "
                                    "'escalate_to' = used when the default response falls below the bar."
                                ),
                            ),
                        },
                    )


# ═════════════════════════════════════════════════════════
# TAB 5 — LIVE TEST  (Step 21)
# ═════════════════════════════════════════════════════════
if active == "⚡ Live Test":
    st.header("Live Test")
    st.caption(
        "Route a real prompt through the policy right now. "
        "The default model runs, the two graders score it, "
        "and it escalates automatically if quality falls below the **bar** — "
        "the best model's average score for this category minus a small tolerance "
        "(see the Routing tab for the full table)."
    )

    st.info(
        "**This tab makes live API calls.**  \n"
        "Each run costs roughly **$0.01–$0.05** depending on prompt length "
        "and whether escalation triggers (escalation doubles the model call count).  \n"
        "Run `python main.py` first so there is a routing policy to use."
    )

    if not routing or not routing.get("policy"):
        st.warning(
            "No routing policy found. Run `python main.py` first to generate routing.json, "
            "then click Refresh."
        )
    else:
        # A run is in progress only during the execute pass (PASS 2 below). While
        # it runs, every control is disabled so the inputs can't change mid-flight
        # — the canonical Streamlit way to "freeze the form while busy", not a hack.
        running = st.session_state.get("live_running", False)

        # ── BYOK key panel — env-first, then per-session pasted keys ──────────
        # Pasted keys live ONLY in st.session_state (per-session server RAM) —
        # never written to os.environ, disk, or logs. Resolved fresh each run so
        # the Run-button gating below reflects what's currently entered. The
        # resolved dict is forwarded explicitly to every model + jury call; see
        # byok.py for the full security model.
        _env_keys = keys_from_env(os.environ)

        if all(_env_keys.values()):
            st.caption(
                "🔑 Using API keys from your local environment — sent only to the "
                "model providers, never stored or logged."
            )
        else:
            _already = all(
                (st.session_state.get(f"byok_saved_{_p}") or "").strip()
                for _p in PROVIDERS
            )
            with st.expander(
                "🔑 Your API keys — "
                + ("✅ set for this session" if _already else "⚠️ paste to run a live test"),
                expanded=not _already and not running,
            ):
                st.caption(
                    "Live Test calls the model providers directly with **your** keys. "
                    "They're held **in memory for this browser session only** — never written "
                    "to disk, never logged, never shared — and gone when you close the tab. "
                    "All three are needed: the answer model plus the two graders that score it."
                )
                _labels = {
                    "openai":    "OpenAI key  (sk-…)",
                    "anthropic": "Anthropic key  (sk-ant-…)",
                    "together":  "Together.ai key  (used for DeepSeek)",
                }

                _gen = st.session_state.get("byok_gen", 0)

                def _save_key(p):
                    val = st.session_state.get(f"byok_input_{p}_{_gen}", "")
                    if val:
                        st.session_state[f"byok_saved_{p}"] = val

                for _p in PROVIDERS:
                    _wkey = f"byok_input_{_p}_{_gen}"
                    if _wkey not in st.session_state and f"byok_saved_{_p}" in st.session_state:
                        st.session_state[_wkey] = st.session_state[f"byok_saved_{_p}"]
                    st.text_input(
                        _labels[_p],
                        type="password",
                        key=_wkey,
                        disabled=running,
                        help="Held in memory for this session only — never stored or logged.",
                        on_change=_save_key,
                        args=(_p,),
                    )
                if st.button("🧹 Clear keys", disabled=running, key="byok_clear"):
                    new_gen = _gen + 1
                    for _p in PROVIDERS:
                        # wipe all generations of the widget key just in case
                        for _g in range(new_gen + 1):
                            st.session_state.pop(f"byok_input_{_p}_{_g}", None)
                        st.session_state.pop(f"byok_saved_{_p}", None)
                    st.session_state["byok_gen"] = new_gen
                    st.rerun()

        # Resolve which keys to use (env-first). resolve_keys never writes os.environ.
        _cur_gen = st.session_state.get("byok_gen", 0)
        _session_keys = {
            _p: (
                st.session_state.get(f"byok_input_{_p}_{_cur_gen}", "")
                or st.session_state.get(f"byok_saved_{_p}", "")
                or ""
            ).strip() or None
            for _p in PROVIDERS
        }
        live_keys, key_source, missing_keys = resolve_keys(_env_keys, _session_keys)

        col_left, col_right = st.columns([1, 2])

        with col_left:
            detect_mode = st.radio(
                "Category detection",
                options=["Auto-detect", "Pick manually"],
                key="live_detect_mode",
                disabled=running,
                help=(
                    "**Auto-detect** — embeds your prompt using text-embedding-3-small, "
                    "compares it against fingerprint vectors for all 6 categories, "
                    "and picks the nearest match automatically.  \n"
                    "**Pick manually** — you choose the category yourself; "
                    "useful when you know the task type and want to skip the detection step."
                ),
            )
            if detect_mode == "Pick manually":
                live_category_manual = st.selectbox(
                    "Task category",
                    options=CATEGORIES,
                    key="live_category_manual",
                    disabled=running,
                    help=(
                        "Choose the Tilicho task category that best describes your prompt.  \n"
                        "The routing policy uses this to pick the default model — "
                        "each category has a different cost/quality tradeoff baked in from the benchmark."
                    ),
                )
            else:
                live_category_manual = None
            run_btn = st.button(
                "▶  Run", width="stretch", type="primary", key="live_run_btn",
                disabled=running or bool(missing_keys),
            )
            if missing_keys and not running:
                st.caption("⚠️ Paste your API keys above to enable Run.")

        # Save prompt text per mode so it disappears on switch and reappears on
        # return — consistent with how the results and live trace behave. We
        # track the previous mode in our OWN key (live_last_mode), because the
        # radio's own key is already updated to the new value by the time this
        # runs, so comparing against it would never detect the change.
        last_mode = st.session_state.get("live_last_mode", detect_mode)
        if detect_mode != last_mode and not running:
            st.session_state.setdefault("live_prompts", {})[last_mode] = st.session_state.get("live_prompt", "")
            st.session_state["live_prompt"] = st.session_state.get("live_prompts", {}).get(detect_mode, "")
        st.session_state["live_last_mode"] = detect_mode

        with col_right:
            live_prompt = st.text_area(
                "Your prompt",
                height=160,
                key="live_prompt",
                disabled=running,
                placeholder="Type your prompt here…",
                help=(
                    "This text will be sent to the default model for the selected category. "
                    "The two graders will score the response, and escalation will trigger "
                    "automatically if the quality score falls below the routing bar."
                ),
            )

        # ── PASS 1 — idle. A click snapshots the inputs, flips the running flag,
        #    and reruns so the controls re-render DISABLED before the long job
        #    starts. (You can't disable a widget already drawn this run, so we
        #    rerun to draw them disabled — then PASS 2 does the work.)
        if run_btn:
            if not live_prompt.strip():
                st.warning("Please enter a prompt before running.")
            else:
                st.session_state["live_pending"] = {
                    "prompt":          live_prompt.strip(),
                    "mode":            detect_mode,
                    "manual_category": live_category_manual,
                }
                st.session_state["live_running"] = True
                st.rerun()

        # ── PASS 2 — execute. Controls are disabled above. Do the work inside
        #    st.status, streaming the router's OWN step messages live via the
        #    status_callback (no more guessing the outcome from the trace). Store
        #    the result keyed by the mode that produced it, then rerun to re-enable
        #    the controls and hand off to PASS 3.
        if running:
            pending  = st.session_state.get("live_pending") or {}
            p_prompt = pending.get("prompt", "")
            p_mode   = pending.get("mode", "Auto-detect")
            p_manual = pending.get("manual_category")

            steps = []
            def _step(msg):
                steps.append(msg)
                st.write(msg)

            record = {"trace": None, "knn_result": None, "steps": steps}
            try:
                from router import run_live as _run_live

                with st.status("Running…", expanded=True) as status:
                    if p_mode == "Auto-detect":
                        from categorizer import build_fingerprints, categorize
                        if "fingerprints" not in st.session_state:
                            _step("📐 Embedding 30 known prompts to build category fingerprints — one-time setup, cached for the rest of this session…")
                            st.session_state["fingerprints"] = build_fingerprints(api_key=live_keys["openai"])
                        _step("🔍 Embedding your prompt into vector space and comparing it against all 6 category fingerprints to find the closest match…")
                        live_category, knn_score = categorize(
                            p_prompt, st.session_state["fingerprints"], api_key=live_keys["openai"]
                        )
                        record["knn_result"] = (live_category, knn_score)
                        _step(f"✅ Closest match: **{live_category}** — cosine similarity {knn_score:.2f} out of 1.0. Routing this as a {live_category} task.")
                    else:
                        live_category = p_manual
                        _step(f"📋 Category manually set to **{live_category}** — skipping auto-detection and routing directly.")

                    trace = _run_live(p_prompt, live_category, status_callback=_step, keys=live_keys)
                    record["trace"] = trace

                    if trace is None:
                        status.update(label="❌ No routing policy found", state="error")
                    elif trace.get("error"):
                        status.update(label="❌ Run failed", state="error")
                    elif trace.get("judged") is False:
                        status.update(label="⚠️ Done — returned ungraded", state="complete")
                    elif trace.get("escalated"):
                        status.update(label=f"✅ Done — escalated to {trace.get('final_model')}", state="complete")
                    else:
                        status.update(label=f"✅ Done — routed to {trace.get('final_model')}", state="complete")

            except Exception as exc:
                record["trace"] = {"error": "exception", "message": redact(str(exc))}

            # Persist under the mode that produced it, clear the running flag, and
            # rerun so the controls come back enabled and PASS 3 shows the result.
            st.session_state.setdefault("live_results", {})[p_mode] = record
            st.session_state["live_running"] = False
            st.session_state.pop("live_pending", None)
            st.rerun()

        # ── PASS 3 — render the result for the CURRENT mode only. Results are
        #    keyed by detection mode, so switching auto↔manual hides this mode's
        #    result and shows the other mode's (or nothing): the live trace and
        #    answer disappear on switch and reappear on return, with no manual
        #    clear/restore logic. Skipped while a run is in progress — PASS 2 owns
        #    the screen then.
        record = st.session_state.get("live_results", {}).get(detect_mode) if not running else None
        if record:
            trace      = record.get("trace")
            steps      = record.get("steps") or []
            knn_result = record.get("knn_result")

            if detect_mode == "Auto-detect" and knn_result:
                knn_cat, knn_score = knn_result
                st.caption(f"Detected category: **{knn_cat}** — similarity {knn_score:.2f}")

            if steps:
                with st.expander("🧾 Run log", expanded=False):
                    for _s in steps:
                        st.markdown(f"- {_s}")

            st.divider()
            with st.container():
                if trace is None:
                    st.error(
                        "run_live() returned None — check that routing.json exists "
                        "and the selected category has a policy entry."
                    )
                elif trace.get("error") == "exception":
                    st.error(f"Error calling run_live(): `{trace.get('message')}`")
                elif trace.get("error") == "default_call_failed":
                    st.error(
                        f"The default model (`{trace.get('default_model')}`) failed to respond.  \n"
                        "Check that your API keys are valid and try again."
                    )
                else:
                    escalated   = trace.get("escalated", False)
                    final_model = trace.get("final_model", "—")

                    if escalated:
                        st.success(
                            f"✅ **Escalated to best model** → final answer from **`{final_model}`**"
                        )
                    else:
                        st.success(
                            f"✅ Answered by **`{final_model}`** — default model, quality cleared the bar"
                        )

                    reason = trace.get("reason", "")
                    if reason:
                        st.markdown(f"**Decision:** _{reason}_")

                    # Response
                    st.subheader("Response")
                    with st.container(border=True):
                        st.markdown(trace.get("final_response") or "(no response)")

                    st.divider()

                    # Metrics row
                    m1, m2, m3, m4 = st.columns(4)

                    default_result = trace.get("default") or {}
                    live_quality   = default_result.get("quality")
                    quality_bar    = trace.get("quality_bar")

                    if live_quality is not None and quality_bar is not None:
                        delta_val = round(live_quality - quality_bar, 4)
                        m1.metric(
                            "Response quality",
                            f"{live_quality:.4f}",
                            delta=f"{delta_val:+.4f} vs bar",
                            delta_color="normal",
                            help=(
                                "Quality score the graders gave to the default model's response. "
                                "Positive delta = cleared the bar (no escalation). "
                                "Negative delta = fell below the bar (escalation triggered)."
                            ),
                        )
                    elif live_quality is not None:
                        m1.metric(
                            "Response quality", f"{live_quality:.4f}",
                            help="Quality score from the graders for the default model's response.",
                        )
                    else:
                        m1.metric(
                            "Response quality", "n/a",
                            help=(
                                "The graders couldn't score this response — both failed. "
                                "Escalation was skipped (we never escalate without measured evidence)."
                            ),
                        )

                    if quality_bar is not None:
                        m2.metric(
                            "Quality bar",
                            f"{quality_bar:.4f}",
                            help=(
                                "The threshold from routing.json for this category. "
                                "The default response must score ≥ this to avoid escalation."
                            ),
                        )

                    m3.metric(
                        "Answer cost",
                        f"${trace.get('answer_cost_usd', 0):.6f}",
                        help=(
                            "USD cost of the model call(s) that produced the final answer. "
                            "Includes both default and escalated call if escalation occurred."
                        ),
                    )
                    m4.metric(
                        "Judging cost",
                        f"${trace.get('judge_cost_usd', 0):.6f}",
                        help=(
                            "USD cost of the live grader calls. "
                            "Stored separately so the true all-in cost is always visible."
                        ),
                    )

                    total = trace.get("total_cost_usd", 0)
                    ans   = trace.get("answer_cost_usd", 0)
                    jdg   = trace.get("judge_cost_usd", 0)
                    st.markdown(
                        f"**💰 Total cost: \\${total:.6f}**  \n"
                        f"&nbsp;&nbsp;&nbsp;├─ Answer generation: &nbsp;\\${ans:.6f}  \n"
                        f"&nbsp;&nbsp;&nbsp;└─ Grading (2 models): \\${jdg:.6f}",
                        unsafe_allow_html=True,
                    )

                    # Escalation detail card
                    if escalated:
                        with st.expander("📋 Escalation detail"):
                            esc_result  = trace.get("escalated_call") or {}
                            esc_quality = esc_result.get("quality")

                            ec1, ec2, ec3 = st.columns(3)
                            ec1.metric(
                                f"Response quality (`{trace.get('default_model')}`)",
                                f"{live_quality:.4f}" if live_quality is not None else "n/a",
                                help="Quality of the default model's response — this fell below the bar.",
                            )
                            ec2.metric(
                                f"Escalated quality (`{final_model}`)",
                                f"{esc_quality:.4f}" if esc_quality is not None else "n/a",
                                help="Quality of the escalated model's response.",
                            )
                            ec3.metric(
                                "Quality bar",
                                f"{quality_bar:.4f}" if quality_bar is not None else "—",
                                help="The bar the default response failed to clear.",
                            )

                            default_lat  = default_result.get("latency") or 0
                            esc_lat      = esc_result.get("latency") or 0
                            default_cost = default_result.get("cost_usd") or 0
                            esc_cost     = esc_result.get("cost_usd") or 0
                            st.markdown(
                                f"Default call:   {default_lat:.1f}s · ${default_cost:.6f}  |  "
                                f"Escalated call: {esc_lat:.1f}s · ${esc_cost:.6f}"
                            )

                    # Unjudged response warning
                    if trace.get("judged") is False:
                        st.warning(
                            "⚠️ The graders failed to score the default response — "
                            "escalation was skipped (we never escalate without measured evidence of a shortfall). "
                            "The default model's response was returned as-is. "
                            "If this happens repeatedly, check your API keys."
                        )
