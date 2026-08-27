"""Report assembly - the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 unit_economics: dict | None = None,
                 insights: list | None = None,
                 priorities: list | None = None,
                 allocation: dict | None = None,
                 extensions: list | None = None,
                 region_table: list | None = None,
                 caveats: list | None = None) -> str:
    """Return a markdown cost-optimization report.

    Only the first three arguments are required - everything after them is an
    optional section, so the original three-argument call still produces the
    original report.
    """
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI - GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
    ]

    if unit_economics:
        lines += [
            "## Unit economics - the number that actually matters",
            "",
            "A GPU-hour is an input, not an outcome. The bill per unit of work served:",
            "",
            "| Metric | Baseline | Optimized | Change |",
            "|---|---:|---:|---:|",
        ]
        for row in unit_economics:
            lines.append(
                f"| {row['metric']} | {row['baseline']} | {row['optimized']} | {row['change']} |")
        lines.append("")

    lines += [
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) | Share of savings | What it is |",
        "|---|---:|---:|---|",
    ]
    total_lever = sum(levers.values()) or 1.0
    lever_notes = {
        "Inference (cascade/cache/batch)": "route 80% of traffic to the small model, "
                                           "cache the shared prefix, batch the offline work",
        "Purchasing (spot/reserved)": "checkpointed jobs to spot, 24x7 services to 3yr reserved",
        "Right-size util-lies": "move the memory-bound box to a part with more HBM bandwidth per dollar",
        "Kill idle GPUs": "stop paying for instances left up after the job finished",
    }
    for name, amount in levers.items():
        note = lever_notes.get(name, "")
        lines.append(f"| {name} | ${amount:,.0f} | {amount / total_lever:.0%} | {note} |")
    lines.append("")

    if insights:
        lines += ["## What the numbers mean", ""]
        for block in insights:
            lines += [f"### {block['title']}", "", block["body"], ""]

    if priorities:
        lines += [
            "## Recommended sequence (by return on effort)",
            "",
            "| # | Action | Savings/mo | Effort | Risk | Why this position |",
            "|---:|---|---:|---|---|---|",
        ]
        for i, p in enumerate(priorities, 1):
            lines.append(f"| {i} | {p['action']} | ${p['savings']:,.0f} | {p['effort']} | "
                         f"{p['risk']} | {p['why']} |")
        lines.append("")

    if allocation:
        lines += [
            "## Allocation readiness",
            "",
            f"- Tag coverage: **{allocation['coverage']:.0%}** "
            f"(chargeback gate at {allocation.get('threshold', 0.8):.0%}: "
            f"**{'OPEN' if allocation['ready'] else 'CLOSED'}**)",
            f"- FOCUS export: `{allocation.get('focus_path', 'outputs/focus_export.csv')}`",
            "",
            "| Team | Daily inference cost | Share |",
            "|---|---:|---:|",
        ]
        total_team = sum(allocation["by_team"].values()) or 1.0
        for team, cost in sorted(allocation["by_team"].items(), key=lambda x: -x[1]):
            lines.append(f"| {team} | ${cost:,.2f} | {cost / total_team:.0%} |")
        lines.append("")

    if sustainability:
        lines += [
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Cleanest grid: {sustainability.get('best_region', 'n/a')}"
            + (f"  |  cheapest electricity: {sustainability['cheapest_region']}"
               if sustainability.get("cheapest_region") else "")
            + (f"  |  balanced (carbon-priced): {sustainability['balanced_region']}"
               if sustainability.get("balanced_region") else ""),
        ]
        for extra in sustainability.get("notes", []):
            lines.append(f"- {extra}")
        lines.append("")

    if region_table:
        lines += [
            "| Region | $/kWh | gCO2e/kWh | Electricity $/mo | tCO2e/mo | "
            "Blended $ @ $100/t | Latency |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in region_table:
            lines.append(
                f"| {r['region']} | {r['usd_per_kwh']:.3f} | {r['gco2_per_kwh']:.0f} | "
                f"${r['energy_usd']:,.0f} | {r['carbon_kg'] / 1000:.2f} | "
                f"${r['blended_usd']:,.0f} | {r['latency_ms']}ms |")
        lines.append("")

    if extensions:
        lines += ["## Your-Turn extensions (measured)", ""]
        for e in extensions:
            lines += [f"### {e['title']}", "", f"**Result:** {e['result']}", "",
                      e["insight"], ""]

    if caveats:
        lines += ["## Caveats", ""] + [f"- {c}" for c in caveats] + [""]

    lines += ["_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str, baseline_usd: float | None = None,
                      optimized_usd: float | None = None) -> str:
    """Write the savings chart PNG. Returns the path. No-op if matplotlib absent.

    With baseline/optimized supplied this draws a real waterfall (baseline bar,
    one falling step per lever, optimized bar). Without them it falls back to the
    original plain bar chart so old callers keep working.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    names = list(levers.keys())
    vals = [levers[n] for n in names]

    if baseline_usd is None or optimized_usd is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(names, vals, color="#2e548a")
        ax.set_ylabel("Savings (USD / month)")
        ax.set_title("GPU cost savings by FinOps lever")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    labels = ["Baseline"] + [n.split(" (")[0] for n in names] + ["Optimized"]
    fig, ax = plt.subplots(figsize=(10, 5.5))

    running = baseline_usd
    ax.bar(0, baseline_usd, color="#8c2f39", width=0.62)
    ax.text(0, baseline_usd, f"${baseline_usd:,.0f}", ha="center", va="bottom", fontsize=9)

    for i, (name, amount) in enumerate(zip(names, vals), start=1):
        bottom = running - amount
        ax.bar(i, amount, bottom=bottom, color="#2e548a", width=0.62)
        ax.plot([i - 0.69, i - 0.31], [running, running], color="#999", lw=1, ls="--")
        ax.text(i, running, f"-${amount:,.0f}", ha="center", va="bottom", fontsize=9)
        running = bottom

    ax.bar(len(names) + 1, optimized_usd, color="#2f7d4f", width=0.62)
    ax.plot([len(names) + 1 - 0.69, len(names) + 1 - 0.31], [running, running],
            color="#999", lw=1, ls="--")
    ax.text(len(names) + 1, optimized_usd, f"${optimized_usd:,.0f}",
            ha="center", va="bottom", fontsize=9)

    saved_pct = (baseline_usd - optimized_usd) / baseline_usd * 100 if baseline_usd else 0.0
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("GPU spend (USD / month)")
    ax.set_title(f"NimbusAI monthly GPU spend: baseline -> optimized ({saved_pct:.0f}% saved)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, baseline_usd * 1.12)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
