"""M5 - Optimization Report: combine M1-M4 (+M6) into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png

The report is the deliverable, so it carries the analysis, not just the totals:
what each lever is, why the GPU-Util lie happens, the order to execute in, and
the measured result of every Your-Turn extension.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability, pricing
from missions import (m1_efficiency_audit, m2_inference_levers, m3_purchasing,
                      m4_allocation, m6_carbon_aware)

DAYS = 30


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    r4 = m4_allocation.run(verbose=False)
    r6 = m6_carbon_aware.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]
    idle_savings = r1["idle_waste_daily"] * DAYS
    # ext-2 measures right-sizing against the real bandwidth/VRAM floor instead of
    # assuming "one tier down" is always safe.
    rightsize_savings = r1["rightsize_savings_monthly"]

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    rea = r2["reasoning"]
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
        "cheapest_region": r6["cheapest_region"],
        "balanced_region": r6["balanced_region"],
        "notes": [
            f"Reasoning traffic is {rea['traffic_share']:.1%} of requests but "
            f"{rea['cost_share']:.0%} of inference spend and {rea['energy_share']:.0%} of energy "
            f"({rea['wh_per_request']:.0f} Wh vs {rea['baseline_wh_per_request']:.2f} Wh per request).",
            f"Capping the reasoning path at {rea['cap_share']:.0%} of traffic saves "
            f"${rea['usd_saved_monthly']:,.0f}/mo and {rea['kwh_saved_monthly']:,.0f} kWh/mo.",
            f"Moving the {len(r6['movable_jobs'])} checkpointed jobs "
            f"({r6['movable_kwh_monthly']:,.0f} kWh/mo) from {r6['home_region']} to "
            f"{r6['cleanest_region']} cuts {r6['carbon_saved_kg']:,.0f} kg CO2e/mo "
            f"({r6['carbon_cut_pct']:.0f}%); the cheapest grid ({r6['cheapest_region']}) also "
            f"saves ${r6['energy_usd_saved']:,.0f}/mo of electricity.",
        ],
    }

    unit_economics = [
        {"metric": "$ / 1M tokens (inference)",
         "baseline": f"${r2['baseline_per_m']:.3f}",
         "optimized": f"${r2['optimized_per_m']:.3f}",
         "change": f"-{r2['savings_pct']:.0f}%"},
        {"metric": "$ / 1M tokens incl. cache-write premium",
         "baseline": f"${r2['baseline_per_m']:.3f}",
         "optimized": f"${r2['optimized_daily_cache_aware'] / (r2['total_tokens'] / 1e6):.3f}",
         "change": f"-{r2['savings_pct_cache_aware']:.0f}%"},
        {"metric": "Fleet spend / month",
         "baseline": f"${baseline:,.0f}", "optimized": f"${optimized:,.0f}",
         "change": f"-{total_pct:.0f}%"},
        {"metric": "Wh / query (median request)",
         "baseline": f"{wh:.2f}", "optimized": f"{wh:.2f}",
         "change": "unchanged - energy follows tokens, not price"},
    ]

    lie = next((l for l in r1["lies"] if l["gpu_id"] == "gpu-h100-4"), r1["lies"][0])
    healthy = max(r1["summary"], key=lambda s: s["mfu"])
    swap = r1["rightsizing"][0] if r1["rightsizing"] else None

    insights = [
        {"title": "The GPU-Util lie: why 98% busy is not 98% useful",
         "body":
             f"`{lie['gpu_id']}` reports **{lie['gpu_util_pct']}% GPU-Util** and only "
             f"**MFU {lie['mfu']:.2f}** - versus MFU {healthy['mfu']:.2f} on `{healthy['gpu_id']}`, "
             f"the same silicon. `nvidia-smi` GPU-Util answers *\"was at least one kernel resident "
             f"in the last sampling window?\"* It is a duty-cycle counter, not an efficiency "
             f"counter, so a kernel that spends its life stalled on HBM reads - or a stream of "
             f"tiny kernels dominated by launch overhead and pipeline bubbles - pins the number at "
             f"~100% while the tensor cores idle between operands. The roofline confirms the "
             f"mechanism: measured arithmetic intensity is {lie['intensity']:.0f} FLOP/byte against "
             f"a ridge point of {lie['ridge']:.0f}, i.e. the job is memory-bound and starves on "
             f"bandwidth, not on FLOPs. Financially you rent the whole GPU-hour and collect "
             f"{lie['mfu'] / healthy['mfu']:.0%} of the FLOPs your healthy trainers get: "
             f"${num(cat[lie['gpu_type']]['on_demand_hr']):.2f}/h buys about "
             f"{lie['mfu']:.0%} of the part. Bill by MFU/MBU and $/1M-token; treat GPU-Util as a "
             f"liveness probe only."},
        {"title": "Why the cheapest GPU is rarely the cheapest answer",
         "body":
             "Serving is bought in TB/s and GB, not in dollars-per-hour. The catalog prices "
             + ", ".join(f"**{e['gpu_type']}** at ${e['usd_per_tbs_hr']:.2f}/TB-s-hr"
                         for e in r1["unit_economics"][:3])
             + f" ... up to **{r1['unit_economics'][-1]['gpu_type']}** at "
               f"${r1['unit_economics'][-1]['usd_per_tbs_hr']:.2f}. That inversion is why the "
               f"memory-bound A100/A10G servers stay where they are: the cheap parts would need "
               f"two-to-four units to hold the measured bandwidth and end up dearer. "
             + (f"Only `{swap['gpu_id']}` moves - {swap['from']} -> {swap['units']}x{swap['to']} "
                f"at ${swap['to_hr']:.2f}/h, worth ${swap['monthly_savings']:,.0f}/mo - because "
                f"that box needs {swap['need_bw_tbs']:.2f} TB/s and "
                f"{swap['need_vram_gb']:.0f} GB, which one MI300X covers outright."
                if swap else "No downgrade cleared the bandwidth floor.")},
        {"title": "Where the inference bill actually goes",
         "body":
             f"Baseline is the naive deployment: every request on the large model, no cache, no "
             f"batch - ${r2['baseline_per_m']:.3f}/1M-token. Cascading ~80% of traffic to the "
             f"small model, caching the shared system prefix and batching the offline eval work "
             f"lands at ${r2['optimized_per_m']:.3f}/1M-token ({r2['savings_pct']:.0f}% off). "
             f"The discounts multiply rather than add: batch (0.5x) on cached input (0.1x) is "
             f"{pricing.discount_stack(batch=True, cache_hit_frac=1.0):.2f}x of naive. Note the "
             f"honest number is ${r2['optimized_daily_cache_aware']:,.2f}/day rather than "
             f"${r2['optimized_daily']:,.2f}/day once the 1.25x cache-write premium is charged - "
             f"still {r2['savings_pct_cache_aware']:.0f}% off, but do not quote the flattered one."},
        {"title": "Purchasing: the discount you book is not the discount you get",
         "body":
             f"Re-pricing the fleet with reclaim rates per GPU pool, 24x7 commitment billing and a "
             f"1yr-vs-3yr comparison gives ${r3['optimized_monthly']:,.0f}/mo "
             f"({r3['savings_pct']:.1f}% off on-demand). The legacy duty-cycle policy claimed "
             f"${r3['legacy_monthly']:,.0f} ({r3['legacy_savings_pct']:.1f}%) - "
             f"${abs(r3['policy_gap']):,.0f}/mo of savings that do not exist, because it billed "
             f"reservations only for hours used and priced every spot pool at one flat 5%/h. "
             f"A 3yr H100 reservation reads -44% against today's on-demand but only "
             f"-{pricing.risk_adjusted_discount(2.50, 1.40, 3) * 100:.0f}% against the average "
             f"market rate over the term, once you allow for street prices falling ~15%/yr."},
    ]

    priorities = [
        {"action": "Kill idle GPUs (auto-shutdown after job exit)",
         "savings": levers["Kill idle GPUs"], "effort": "hours", "risk": "none",
         "why": "pure waste, one scheduler hook, no product impact - do it today"},
        {"action": "Move checkpointed jobs to spot, commit only the 24x7 services",
         "savings": levers["Purchasing (spot/reserved)"], "effort": "days", "risk": "medium",
         "why": "largest bucket by far; needs checkpointing to be real and a commitment sign-off"},
        {"action": "Ship the inference levers (cascade + cache + batch)",
         "savings": levers["Inference (cascade/cache/batch)"], "effort": "1-2 sprints",
         "risk": "medium",
         "why": "needs an eval gate so the small model does not silently degrade quality"},
        {"action": f"Right-size {swap['gpu_id'] if swap else 'the memory-bound box'} "
                   f"to {swap['to'] if swap else 'a higher-bandwidth part'}",
         "savings": levers["Right-size util-lies"], "effort": "days", "risk": "medium",
         "why": "smallest bucket and a migration; do it after the free money is banked"},
        {"action": f"Cap the reasoning path at {rea['cap_share']:.0%} of traffic",
         "savings": rea["usd_saved_monthly"], "effort": "days", "risk": "low",
         "why": f"only ${rea['usd_saved_monthly']:,.0f}/mo but "
                f"{rea['kwh_saved_monthly']:,.0f} kWh/mo - an energy lever, not a cost lever"},
    ]

    extensions = [
        {"title": "Ext-1 - risk-priced purchasing policy (`pricing.recommend_tier_v2`)",
         "result": f"${r3['optimized_monthly']:,.0f}/mo vs ${r3['legacy_monthly']:,.0f}/mo claimed "
                   f"by the old policy ({r3['savings_pct']:.1f}% vs {r3['legacy_savings_pct']:.1f}% "
                   f"savings on ${r3['on_demand_monthly']:,.0f} of on-demand)",
         "insight": "Per-pool reclaim rates (H100 5%/h vs L4 15%/h), 24x7 commitment billing and a "
                    "1yr-vs-3yr comparison against the declining market price. Same tiers, "
                    f"${abs(r3['policy_gap']):,.0f}/mo less fantasy."},
        {"title": "Ext-2 - MBU right-sizing (`m1.rightsize_candidate`)",
         "result": f"${r1['rightsize_savings_monthly']:,.0f}/mo from "
                   f"{len(r1['rightsizing'])} swap(s); {len(r1['rightsize_keeps'])} GPUs "
                   f"deliberately left alone",
         "insight": "Sizing against measured p95 bandwidth, peak VRAM and p95 FLOPs - and counting "
                    "how many units a cheaper part would take - is what stops a $0.80/h L4 from "
                    "looking like an upgrade over a $1.79/h A100."},
        {"title": "Ext-3 - cache break-even (`pricing.cache_is_worth_it`)",
         "result": f"break-even {r2['cache_breakeven_reads']:.2f} reads/write; measured "
                   + ", ".join(f"{t} {s['avg_reads_per_write']:.2f}"
                               for t, s in sorted(r2["cache_stats"].items()))
                   + f"; honest cost ${r2['optimized_daily_cache_aware']:,.2f}/day vs "
                     f"${r2['optimized_daily']:,.2f}/day naive",
         "insight": "A 1.25x write repaid by 0.1x reads breaks even after 0.28 reads, so a single "
                    "reuse inside the 5-minute TTL already pays - every team clears it here, but "
                    "eval only by 3x, and a thinner-traffic tenant would lose money on caching."},
        {"title": "Ext-4 - reasoning budget",
         "result": f"{rea['traffic_share']:.1%} of requests = {rea['cost_share']:.0%} of spend and "
                   f"{rea['energy_share']:.0%} of energy; cap at {rea['cap_share']:.0%} saves "
                   f"${rea['usd_saved_monthly']:,.0f}/mo + {rea['kwh_saved_monthly']:,.0f} kWh/mo",
         "insight": f"A reasoning request burns {rea['wh_per_request']:.0f} Wh against "
                    f"{rea['baseline_wh_per_request']:.2f} Wh for a normal one: the ~80x energy "
                    "multiplier compounds with ~6x more output tokens. Route on task complexity, "
                    "not on user preference."},
        {"title": "Ext-5 - carbon-aware scheduling (`missions/m6_carbon_aware.py`)",
         "result": f"{r6['carbon_saved_kg']:,.0f} kg CO2e/mo ({r6['carbon_cut_pct']:.0f}%) by moving "
                   f"{r6['movable_kwh_monthly']:,.0f} kWh of checkpointed work to "
                   f"{r6['cleanest_region']}; ${r6['energy_usd_saved']:,.0f}/mo cheaper in "
                   f"{r6['cheapest_region']}",
         "insight": f"Cheapest ({r6['cheapest_region']}), cleanest ({r6['cleanest_region']}) and "
                    f"balanced at $100/tCO2e ({r6['balanced_region']}) are three different answers. "
                    f"Only interruptible jobs move - the cleanest grid costs ~95ms of latency, "
                    f"which is free for training and unacceptable for chat."},
    ]

    caveats = [
        "Prices are June-2026 illustrative snapshots; GPU street prices move monthly.",
        "Savings are modelled from synthetic telemetry (seed 25), not from a production bill.",
        "Reserved buckets assume the commitment is actually consumed - re-check duty cycle "
        "quarterly or the discount reverses.",
        "Right-sizing assumes the workload ports cleanly to the target part (ROCm for MI300X).",
    ]

    md = report.build_report(
        baseline, optimized, levers, sustainability=sust,
        unit_economics=unit_economics, insights=insights, priorities=priorities,
        allocation={"coverage": r4["tag_coverage"], "ready": r4["chargeback_ready"],
                    "by_team": r4["by_team"], "focus_path": "outputs/focus_export.csv"},
        extensions=extensions, region_table=r6["table"], caveats=caveats)

    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"),
                                   baseline_usd=baseline, optimized_usd=optimized)

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print("\nWritten: outputs/report.md" + (" + outputs/savings.png" if png
                                                else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1),
            "unit_economics": unit_economics, "reasoning": rea,
            "carbon": {"saved_kg": r6["carbon_saved_kg"], "cut_pct": r6["carbon_cut_pct"]}}


if __name__ == "__main__":
    run()
