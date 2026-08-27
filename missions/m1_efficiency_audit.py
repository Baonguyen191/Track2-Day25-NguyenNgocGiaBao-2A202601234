"""M1 - Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py

YOUR-TURN EXTENSION 2 lives here: MBU-driven right-sizing. For every GPU whose
roofline says it is memory-bound we look for the cheapest catalog part that
still clears the measured bandwidth / VRAM / compute floor - and we count how
many of that part it would take, which is exactly why the cheapest $/GPU-hr is
usually the wrong answer.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import math
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics

DAYS = 30
TARGET_MBU = 0.60   # what a well-tuned decode server should hold on HBM bandwidth
TARGET_MFU = 0.45   # a good training/prefill MFU; above this you are pushing the part
VRAM_HEADROOM = 1.10  # KV-cache + fragmentation headroom on top of the observed peak


def _pctl(vals, q):
    """Small percentile helper (no numpy dependency in the graded path)."""
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def unit_economics(cat) -> list:
    """Per-GPU-type price of the two things inference actually buys: VRAM and bandwidth."""
    rows = []
    for gtype, c in cat.items():
        od = num(c["on_demand_hr"])
        rows.append({
            "gpu_type": gtype,
            "on_demand_hr": od,
            "hbm_gb": num(c["hbm_gb"]),
            "peak_bw_tbs": num(c["peak_bw_tbs"]),
            "usd_per_gb_hr": od / num(c["hbm_gb"]) if num(c["hbm_gb"]) else 0.0,
            "usd_per_tbs_hr": od / num(c["peak_bw_tbs"]) if num(c["peak_bw_tbs"]) else 0.0,
        })
    return sorted(rows, key=lambda r: r["usd_per_tbs_hr"])


def rightsize_candidate(need_bw: float, need_vram: float, need_tflops: float, cat) -> dict | None:
    """Cheapest catalog part (possibly N of them) that meets the measured floors.

    A part is only a candidate if it can hold the working set and sustain the
    measured bandwidth at a realistic MBU - not at its marketing peak. When one
    unit cannot, we scale to N units, which is how a "cheap" GPU quietly becomes
    the expensive option.
    """
    best = None
    for gtype, c in cat.items():
        bw, vram, tflops = num(c["peak_bw_tbs"]), num(c["hbm_gb"]), num(c["peak_tflops_fp16"])
        if vram <= 0 or bw <= 0:
            continue
        eps = 1e-9  # keep float dust from demanding a second GPU
        units_bw = math.ceil(need_bw / (bw * TARGET_MBU) - eps) if need_bw > 0 else 1
        units_vram = math.ceil(need_vram / vram - eps) if need_vram > 0 else 1
        units_flops = math.ceil(need_tflops / (tflops * TARGET_MFU) - eps) if need_tflops > 0 else 1
        units = max(1, units_bw, units_vram, units_flops)
        if units > 4:            # more than a 4-way shard is an operational rewrite, not right-sizing
            continue
        cost_hr = units * num(c["on_demand_hr"])
        cand = {"gpu_type": gtype, "units": units, "cost_hr": cost_hr,
                "binding": max((("bandwidth", units_bw), ("vram", units_vram),
                                ("compute", units_flops)), key=lambda t: t[1])[0]}
        if best is None or cost_hr < best["cost_hr"]:
            best = cand
    return best


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0,
                               "tflops": [], "bw": [], "mem": [], "workload": None})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["workload"] = r.get("workload")
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        a["tflops"].append(num(r["achieved_tflops"]))
        a["bw"].append(num(r["achieved_bw_tbs"]))
        a["mem"].append(num(r["mem_used_gb"]))
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        c = cat[a["type"]]
        ridge = num(c["peak_tflops_fp16"]) / num(c["peak_bw_tbs"])  # FLOP/byte where the part turns over
        p95_tflops, p95_bw = _pctl(a["tflops"], 0.95), _pctl(a["bw"], 0.95)
        intensity = metrics.arithmetic_intensity(p95_tflops, p95_bw)  # TFLOP/s per TB/s = FLOP/byte
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"], "workload": a["workload"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
            "p95_tflops": round(p95_tflops, 1), "p95_bw_tbs": round(p95_bw, 3),
            "peak_mem_gb": round(max(a["mem"]), 1),
            "intensity": round(intensity, 1), "ridge": round(ridge, 1),
            "regime": metrics.roofline_regime(intensity, ridge),
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(cat[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # ---- EXTENSION 2: right-size the memory-bound serving GPUs -----------
    # Scope: GPUs that actually serve (infer/embed) plus any util-lie box. A
    # healthy training GPU at MFU ~0.42 is doing its job - the answer there is
    # "leave it alone", not "buy something smaller".
    econ = unit_economics(cat)
    lie_ids = {l["gpu_id"] for l in lies}
    cheapest_small_per_tbs = {e["gpu_type"]: e["usd_per_tbs_hr"] for e in econ}
    rightsizing, keeps = [], []
    rightsize_savings = 0.0
    for s in summary:
        serving = s["workload"] in ("infer", "embed")
        if not (serving or s["gpu_id"] in lie_ids) or s["mbu"] >= TARGET_MBU:
            continue
        cur_hr = num(cat[s["gpu_type"]]["on_demand_hr"])
        cand = rightsize_candidate(s["p95_bw_tbs"], s["peak_mem_gb"] * VRAM_HEADROOM,
                                   s["p95_tflops"], cat)
        row = {"gpu_id": s["gpu_id"], "from": s["gpu_type"], "mbu": s["mbu"],
               "need_bw_tbs": s["p95_bw_tbs"], "need_vram_gb": s["peak_mem_gb"],
               "from_hr": cur_hr}
        if not cand or cand["cost_hr"] >= cur_hr:
            cheaper = [e for e in econ if e["on_demand_hr"] < cur_hr]
            alt = min(cheaper, key=lambda e: e["on_demand_hr"]) if cheaper else None
            row.update({
                "to": cand["gpu_type"] if cand else "-",
                "units": cand["units"] if cand else 0,
                "to_hr": round(cand["cost_hr"], 2) if cand else 0.0,
                "verdict": "keep",
                "why": (("needs %.2f TB/s; the cheapest smaller part (%s at $%.2f/h) costs "
                         "$%.2f per TB/s vs $%.2f here, so you would need several and pay more - "
                         "fix batching, not the box"
                         % (s["p95_bw_tbs"], alt["gpu_type"], alt["on_demand_hr"],
                            alt["usd_per_tbs_hr"],
                            cheapest_small_per_tbs.get(s["gpu_type"], 0.0))) if alt else
                        ("needs %.2f TB/s and is already the cheapest part in the catalog - "
                         "the lever here is batching/quantisation, not hardware"
                         % s["p95_bw_tbs"]))})
            keeps.append(row)
            continue
        monthly = (cur_hr - cand["cost_hr"]) * 24 * DAYS
        rightsize_savings += monthly
        row.update({"to": cand["gpu_type"], "units": cand["units"],
                    "binding_constraint": cand["binding"], "to_hr": round(cand["cost_hr"], 2),
                    "verdict": "swap", "monthly_savings": round(monthly, 2),
                    "why": ("memory-bound at MBU %.0f%%; %dx%s delivers the %.2f TB/s and %.0f GB "
                            "it needs for $%.2f/h instead of $%.2f/h"
                            % (s["mbu"] * 100, cand["units"], cand["gpu_type"], s["p95_bw_tbs"],
                               s["peak_mem_gb"] * VRAM_HEADROOM, cand["cost_hr"], cur_hr))})
        rightsizing.append(row)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}"
              f"{'FLOP/byte':>11}{'ridge':>8}{'regime':>15}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}"
                  f"{s['idle_hours']:>8}{s['intensity']:>11}{s['ridge']:>8}{s['regime']:>15}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*DAYS:,.0f}/month")

        print("\n-- ext2: what a GPU-hour actually buys (sorted by $/TB/s) --")
        print(f"{'type':7}{'$/hr':>8}{'HBM GB':>8}{'TB/s':>7}{'$/GB-hr':>10}{'$/TBs-hr':>10}")
        for e in econ:
            print(f"{e['gpu_type']:7}{e['on_demand_hr']:>8.2f}{e['hbm_gb']:>8.0f}{e['peak_bw_tbs']:>7.2f}"
                  f"{e['usd_per_gb_hr']:>10.4f}{e['usd_per_tbs_hr']:>10.2f}")
        print("\n-- ext2: right-sizing the memory-bound serving GPUs --")
        for r in rightsizing:
            print(f"  SWAP {r['gpu_id']:12} {r['from']} -> {r['units']}x{r['to']}   "
                  f"${r['from_hr']:.2f}/h -> ${r['to_hr']:.2f}/h  = ${r['monthly_savings']:,.0f}/mo")
            print(f"       why: {r['why']}")
        for r in keeps:
            print(f"  KEEP {r['gpu_id']:12} {r['from']} (MBU {r['mbu']:.0%})")
            print(f"       why: {r['why']}")
        print(f"total right-sizing headroom: ${rightsize_savings:,.0f}/month")

    return {"summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
            "unit_economics": econ, "rightsizing": rightsizing, "rightsize_keeps": keeps,
            "rightsize_savings_monthly": round(rightsize_savings, 2)}


if __name__ == "__main__":
    run()
