"""
report.py — Generates a one-page Markdown benchmark report.

Pure function — no API calls, no file I/O.
Takes the dicts already loaded by the dashboard and returns a Markdown string
ready for st.download_button or writing to disk.
"""

from __future__ import annotations
from datetime import datetime


def _short(model: str) -> str:
    if model.startswith("gpt"):
        return "GPT-4o"
    if model.startswith("claude"):
        return "Claude Sonnet"
    if "deepseek" in model.lower():
        return "DeepSeek V4 Pro"
    return model


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = ["-" * max(len(h), 3) for h in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def generate_report(
    summary: dict,
    routing: dict | None,
    findings: dict,
    meta: dict | None,
) -> str:
    """
    Generate a one-page Markdown benchmark report.

    Parameters
    ----------
    summary  : dict  — contents of summary.json
    routing  : dict  — contents of routing.json (may be None)
    findings : dict  — output of findings.compute_findings()
    meta     : dict  — contents of meta.json (may be None)

    Returns
    -------
    str — Markdown report ready for download
    """
    lines: list[str] = []

    models_data: dict = summary.get("models", {})
    policy: dict = (routing or {}).get("policy", {})

    # ── Header ────────────────────────────────────────────────────────────────
    run_at = summary.get("run_at", "")
    run_at_display = run_at[:19].replace("T", " ") if run_at else "unknown"
    lines += [
        "# LLM Eval Lab — Benchmark Report",
        "",
        f"**Run date:** {run_at_display}",
    ]

    if meta:
        fp = meta.get("run_fingerprint", "")
        if fp:
            lines.append(f"**Fingerprint:** `{fp}`")
        git = meta.get("code", {})
        if git.get("git_sha"):
            dirty = " (dirty)" if git.get("git_dirty") else ""
            lines.append(
                f"**Git:** `{git['git_sha']}` on `{git.get('git_branch', '?')}`{dirty}"
            )
        prompt_count = meta.get("prompt_count")
        if prompt_count:
            lines.append(f"**Prompts:** {prompt_count} ({prompt_count // 6} per category × 6 categories)")

    lines += ["", "---", ""]

    # ── Key Findings ──────────────────────────────────────────────────────────
    lines += ["## Key Findings", ""]

    if findings:
        w  = findings.get("winner", {})
        ru = findings.get("runner_up", {})
        tr = findings.get("trailing", {})
        bg = findings.get("biggest_gap", {})
        bv = findings.get("best_value", {})
        esc = findings.get("escalation_categories", [])

        if w:
            gap = w["score"] - ru.get("score", 0)
            lines.append(
                f"- **Overall winner:** {w['short_name']} — average quality score "
                f"{w['score']:.3f} across 6 categories, ahead of "
                f"{ru.get('short_name', '?')} ({ru.get('score', 0):.3f}) by {gap:.3f}. "
                f"{tr.get('short_name', '?')} trails at {tr.get('score', 0):.3f}."
            )

        if bg.get("category"):
            cat_label = bg["category"].replace("_", " ").title()
            lines.append(
                f"- **Biggest category gap:** {cat_label} — ±{bg['gap']:.2f} spread "
                f"(best vs worst: {bg['best_short']} {bg['best_score']:.3f} → "
                f"{bg['worst_short']} {bg['worst_score']:.3f}). "
                "Model choice matters most here."
            )

        if bv:
            lines.append(
                f"- **Best quality per dollar:** {bv['short_name']} — "
                f"{bv['quality']:.3f} quality at ${bv['avg_cost']:.4f}/prompt "
                "(measured from this benchmark run, not list prices)."
            )

        if esc:
            esc_names = ", ".join(c.replace("_", " ") for c in esc)
            no_esc = findings.get("no_escalation_categories", [])
            lines.append(
                f"- **Escalation paths:** {len(esc)} of 6 categories have a real "
                f"escalation path ({esc_names}) — the default model and the best model "
                f"differ, so a low-quality live response can be retried against a "
                f"stronger model. The other {len(no_esc)} categories are already routed "
                "to their top model."
            )

    lines += ["", "---", ""]

    # ── Model Leaderboard ─────────────────────────────────────────────────────
    lines += ["## Model Leaderboard", ""]

    ranked = sorted(
        models_data.items(),
        key=lambda x: x[1].get("final_score", 0),
        reverse=True,
    )

    lb_rows = []
    for model, d in ranked:
        avg_cost = sum(d.get("category_costs", {}).values()) / max(len(d.get("category_costs", {})), 1)
        lb_rows.append([
            _short(model),
            f"{d.get('final_score', 0):.4f}",
            f"${avg_cost:.4f}",
            f"{d.get('avg_latency', 0):.2f}s",
        ])

    lines.append(_md_table(
        ["Model", "Avg Quality Score", "Avg Cost/prompt", "Avg Latency"],
        lb_rows,
    ))
    lines += [
        "",
        "_Quality score = average of per-category scores (0–1). "
        "Each response graded by the other two models on correctness, "
        "reasoning, clarity, hallucination-free._",
        "", "---", "",
    ]

    # ── Per-Category Scores ───────────────────────────────────────────────────
    lines += ["## Per-Category Scores", ""]

    model_names = [m for m, _ in ranked]
    short_names = [_short(m) for m in model_names]
    cat_headers = ["Category"] + short_names
    cat_rows = []

    all_cats = ["architecture", "coding", "debugging", "summarization", "business_writing", "agent_planning"]
    for cat in all_cats:
        row = [cat.replace("_", " ").title()]
        for model in model_names:
            score = models_data[model].get("category_scores", {}).get(cat, 0)
            row.append(f"{score:.3f}")
        cat_rows.append(row)

    lines.append(_md_table(cat_headers, cat_rows))
    lines += ["", "---", ""]

    # ── Routing Policy ────────────────────────────────────────────────────────
    lines += ["## Routing Policy", ""]
    lines.append(
        "For each category: the **default** model is the cheapest (or fastest for "
        "summarization) among those that clear the quality bar. The **escalation** "
        "model is the highest-quality — a live response below the bar is retried "
        "against it."
    )
    lines.append("")

    if policy:
        rt_rows = []
        for cat in all_cats:
            rule = policy.get(cat, {})
            default   = _short(rule.get("default", "—"))
            escalate  = _short(rule.get("escalate_to", "—"))
            bar       = rule.get("quality_bar", 0)
            selection = rule.get("selection", "—")
            same = "✓ already best" if rule.get("default") == rule.get("escalate_to") else "↑ escalation active"
            rt_rows.append([
                cat.replace("_", " ").title(),
                default,
                escalate,
                f"{bar:.3f}",
                selection,
                same,
            ])
        lines.append(_md_table(
            ["Category", "Default", "Escalate To", "Quality Bar", "Selection", "Status"],
            rt_rows,
        ))
    else:
        lines.append("_No routing policy found — run `python main.py` to generate one._")

    lines += ["", "---", ""]

    # ── Category Winners ──────────────────────────────────────────────────────
    cat_winners = findings.get("category_winners", {})
    if cat_winners:
        lines += ["## Category Winners", ""]
        cw_rows = [
            [cat.replace("_", " ").title(), info["short_name"], f"{info['score']:.3f}"]
            for cat, info in sorted(cat_winners.items())
        ]
        lines.append(_md_table(["Category", "Best Model", "Score"], cw_rows))
        lines += ["", "---", ""]

    # ── Footer ────────────────────────────────────────────────────────────────
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines += [
        f"_Report generated {generated_at}. "
        "Scores reflect this benchmark run only — models are updated regularly "
        "and results may differ on a fresh run. "
        "See METHODOLOGY.md for scoring details and limitations._",
    ]

    return "\n".join(lines) + "\n"
