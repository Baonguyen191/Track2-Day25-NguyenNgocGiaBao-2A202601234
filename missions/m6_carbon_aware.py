"""M6 (YOUR-TURN EXTENSION 5) - Carbon-aware scheduling for interruptible jobs.

Run: python missions/m6_carbon_aware.py

Interruptible jobs are already checkpointed, so they are the ones you can move
across regions without touching the product. This mission prices that move in
both currencies a FinOps team owns: dollars of electricity and grams of CO2e -
then names the "optimal" region under three different corporate priorities,
because there is no single answer.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability

HOME_REGION = "us-east-1"     # where NimbusAI runs everything today
PUE = 1.12                    # datacentre overhead on top of the GPU's own draw
CARBON_PRICE_USD_PER_TON = 100.0  # internal price of carbon used to rank "balanced"

# Round-trip latency penalty from the US east-coast user base (illustrative ms).
REGION_LATENCY_MS = {
    "us-east-1": 15, "us-east-wa": 55, "us-west-2": 70,
    "europe-north1": 110, "europe-central2": 125,
}


def job_energy_kwh(job, cat) -> float:
    """Monthly energy of a job: GPU-hours x board watts x PUE."""
    c = cat[job["gpu_type"]]
    gpu_hours = num(job["hours_per_day"]) * num(job["days"]) * int(num(job["num_gpus"]))
    return gpu_hours * num(c["watts"]) * PUE / 1000.0


def region_table(kwh: float) -> list:
    """Cost / carbon / latency of running `kwh` in every catalogued region."""
    rows = []
    for region in sustainability.REGION_CARBON:
        wh = kwh * 1000.0
        rows.append({
            "region": region,
            "usd_per_kwh": sustainability.REGION_PRICE_KWH.get(region, 0.12),
            "gco2_per_kwh": sustainability.REGION_CARBON[region],
            "energy_usd": sustainability.energy_cost_usd(wh, region),
            "carbon_kg": sustainability.carbon_g(wh, region) / 1000.0,
            "latency_ms": REGION_LATENCY_MS.get(region, 0),
        })
    for r in rows:
        # Balanced = electricity + an internal carbon price on the emissions.
        r["blended_usd"] = r["energy_usd"] + r["carbon_kg"] / 1000.0 * CARBON_PRICE_USD_PER_TON
    return sorted(rows, key=lambda r: r["blended_usd"])


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()

    movable = [j for j in jobs if int(num(j["interruptible"])) == 1]
    kwh = sum(job_energy_kwh(j, cat) for j in movable)
    fixed_kwh = sum(job_energy_kwh(j, cat) for j in jobs if int(num(j["interruptible"])) == 0)

    table = region_table(kwh)
    home = next(r for r in table if r["region"] == HOME_REGION)
    cheapest = min(table, key=lambda r: r["energy_usd"])
    cleanest = min(table, key=lambda r: r["carbon_kg"])
    balanced = min(table, key=lambda r: r["blended_usd"])

    per_job = []
    for j in movable:
        jk = job_energy_kwh(j, cat)
        per_job.append({
            "job_id": j["job_id"], "gpu_type": j["gpu_type"], "kwh": round(jk, 1),
            "home_kg": round(sustainability.carbon_g(jk * 1000, HOME_REGION) / 1000.0, 1),
            "clean_kg": round(sustainability.carbon_g(jk * 1000, cleanest["region"]) / 1000.0, 1),
            "home_usd": round(sustainability.energy_cost_usd(jk * 1000, HOME_REGION), 2),
            "clean_usd": round(sustainability.energy_cost_usd(jk * 1000, cleanest["region"]), 2),
        })

    carbon_saved_kg = home["carbon_kg"] - cleanest["carbon_kg"]
    carbon_cut_pct = carbon_saved_kg / home["carbon_kg"] * 100 if home["carbon_kg"] else 0.0
    usd_saved = home["energy_usd"] - cheapest["energy_usd"]

    result = {
        "movable_jobs": [j["job_id"] for j in movable],
        "movable_kwh_monthly": round(kwh, 1),
        "fixed_kwh_monthly": round(fixed_kwh, 1),
        "home_region": HOME_REGION,
        "home_carbon_kg": round(home["carbon_kg"], 1),
        "home_energy_usd": round(home["energy_usd"], 2),
        "cheapest_region": cheapest["region"], "cleanest_region": cleanest["region"],
        "balanced_region": balanced["region"],
        "carbon_saved_kg": round(carbon_saved_kg, 1),
        "carbon_cut_pct": round(carbon_cut_pct, 1),
        "energy_usd_saved": round(usd_saved, 2),
        "table": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}
                  for r in table],
        "per_job": per_job,
    }

    if verbose:
        print("== M6 Carbon-aware Scheduling (ext5) ==")
        print(f"movable (interruptible) jobs: {', '.join(result['movable_jobs'])}")
        print(f"movable energy: {kwh:,.0f} kWh/month   (locked to users: {fixed_kwh:,.0f} kWh/month)")
        print(f"\n{'region':16}{'$/kWh':>8}{'gCO2/kWh':>10}{'energy $':>11}{'tCO2e':>9}"
              f"{'blended $':>11}{'latency':>9}")
        for r in table:
            print(f"{r['region']:16}{r['usd_per_kwh']:>8.3f}{r['gco2_per_kwh']:>10.0f}"
                  f"{r['energy_usd']:>11,.0f}{r['carbon_kg']/1000:>9.2f}{r['blended_usd']:>11,.0f}"
                  f"{str(r['latency_ms']) + 'ms':>9}")
        print(f"\ncheapest electricity : {cheapest['region']} (${cheapest['energy_usd']:,.0f}/mo, "
              f"saves ${usd_saved:,.0f} vs {HOME_REGION})")
        print(f"cleanest grid        : {cleanest['region']} ({cleanest['carbon_kg']:,.0f} kg CO2e/mo, "
              f"cuts {carbon_saved_kg:,.0f} kg = {carbon_cut_pct:.0f}%)")
        print(f"balanced @ ${CARBON_PRICE_USD_PER_TON:.0f}/tCO2e : {balanced['region']} "
              f"(blended ${balanced['blended_usd']:,.0f}/mo)")
        print("\nper movable job (home vs cleanest):")
        print(f"{'job':18}{'kWh':>9}{'kg CO2e now':>13}{'kg CO2e clean':>15}{'saved kg':>10}")
        for j in per_job:
            print(f"{j['job_id']:18}{j['kwh']:>9,.0f}{j['home_kg']:>13,.0f}{j['clean_kg']:>15,.0f}"
                  f"{j['home_kg'] - j['clean_kg']:>10,.0f}")
        print(f"\ntrade-off: {cleanest['region']} adds ~{REGION_LATENCY_MS[cleanest['region']] - REGION_LATENCY_MS[HOME_REGION]}ms "
              f"of round-trip latency - irrelevant for checkpointed training, fatal for interactive "
              f"serving, which is why only the interruptible {len(movable)} jobs move.")

    return result


if __name__ == "__main__":
    run()
