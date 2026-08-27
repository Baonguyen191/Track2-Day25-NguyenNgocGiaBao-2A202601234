"""M3 - Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py

YOUR-TURN EXTENSION 1 lives here: the mission now prices every job twice -
once with the legacy duty-cycle policy (`pricing.recommend_tier`) and once with
the risk-priced policy (`pricing.recommend_tier_v2`) - and prints the delta so
the change is measured, not asserted.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing


def _legacy_cost(job, c, gpu_hours, on_demand_cost):
    """What the original policy would have charged (used-hours billing for reserved)."""
    tier = pricing.recommend_tier(num(job["hours_per_day"]), bool(int(num(job["interruptible"]))))
    if tier == "spot":
        cost = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), num(c["on_demand_hr"]))["spot_cost"]
    elif tier == "reserved":
        cost = gpu_hours * num(c["reserved_3yr_hr"])   # <- optimistic: bills only used hours
    else:
        cost = on_demand_cost
    return tier, cost


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = legacy_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        days = num(j["days"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        # NOTE (ext-1): the original mission billed every job as if it ran 30
        # days/month and ignored workloads.csv's `days` column. Usage hours are
        # hours_per_day x days x GPUs - that is what on-demand, spot AND the
        # legacy comparison are all measured against below.
        gpu_hours = hpd * days * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # --- policy v2: prices reclaim risk, 24x7 commitment billing, 1yr vs 3yr ---
        d = pricing.recommend_tier_v2(
            hours_per_day=hpd, interruptible=interruptible, gpu_type=gtype,
            days_per_month=days, num_gpus=ngpu,
            on_demand_hr=od, spot_hr=num(c["spot_hr"]),
            reserved_1yr_hr=num(c["reserved_1yr_hr"]), reserved_3yr_hr=num(c["reserved_3yr_hr"]),
        )
        opt_cost = d["cost"]

        legacy_tier, legacy_cost = _legacy_cost(j, c, gpu_hours, on_demand_cost)

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        legacy_monthly += legacy_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": d["tier"],
                     "tier_detail": d["tier_detail"], "duty": d["duty"],
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost),
                     "legacy_tier": legacy_tier, "legacy": round(legacy_cost),
                     "reason": d["reason"]})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    legacy_pct = (on_demand_monthly - legacy_monthly) / on_demand_monthly * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% headline reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'duty':>6}{'tier (v2)':>14}{'on-demand':>12}{'optimized':>12}{'legacy':>10}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['duty']:>6.0%}{r['tier_detail']:>14}"
                  f"${r['on_demand']:>11,}${r['optimized']:>11,}${r['legacy']:>9,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}"
              f"  ({savings_pct:.1f}% saved)")
        gap = optimized_monthly - legacy_monthly
        verdict = ("OVERSTATED savings by" if gap > 0 else "was conservative by")
        print(f"legacy policy claimed ${legacy_monthly:,.0f} ({legacy_pct:.1f}% saved) "
              f"-> {verdict} ${abs(gap):,.0f}/mo ({abs(legacy_pct - savings_pct):.1f} pts): it bills "
              f"commitments for used hours only and prices every spot pool at 5%/h.")
        print("\nwhy each job landed where it did:")
        for r in recs:
            print(f"  {r['job_id']:18} {r['reason']}")
        print("\npolicy card - tier by duty cycle (H100 vs L4):")
        print(f"{'duty':>6}{'H100 steady':>16}{'H100 interrupt':>18}{'L4 steady':>14}{'L4 interrupt':>16}")
        h100 = pricing.tier_matrix(cat["H100"])
        l4 = pricing.tier_matrix(cat["L4"])
        for a, b in zip(h100, l4):
            print(f"{a['duty']:>6.0%}{a['steady']:>16}{a['interruptible']:>18}"
                  f"{b['steady']:>14}{b['interruptible']:>16}")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "legacy_monthly": round(legacy_monthly), "legacy_savings_pct": round(legacy_pct, 1),
            "policy_gap": round(optimized_monthly - legacy_monthly)}


if __name__ == "__main__":
    run()
