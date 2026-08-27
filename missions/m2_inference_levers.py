"""M2 - Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py

YOUR-TURN EXTENSION 3 (cache economics) and EXTENSION 4 (reasoning budget) live
here. Ext-3 measures reads-per-write from the traffic itself and only claims the
cache discount for teams that clear the break-even. Ext-4 splits $ and Wh by
is_reasoning and prices a routing cap.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) - illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

CACHE_TTL_MIN = 5          # Anthropic 5-min cache TTL: a write only earns within the window
REASONING_CAP_SHARE = 0.05  # routing target: <=5% of requests may take the reasoning path


def _bucket(ts: str, minutes: int = CACHE_TTL_MIN) -> str:
    """Coarse TTL bucket key from an ISO timestamp: 'HH:MM-slot'."""
    hh, mm = ts[11:13], ts[14:16]
    try:
        slot = int(mm) // minutes
    except ValueError:
        slot = 0
    return f"{hh}:{slot}"


def measure_cache_reads(rows) -> dict:
    """Reads-per-write per team, measured from traffic inside the cache TTL.

    Each team shares one static system prefix. Within a TTL window the first
    request pays the WRITE premium and the rest are reads, so
        reads_per_write = (requests - windows) / windows
    Teams whose traffic is too thin never repay the write - that is exactly what
    pricing.cache_is_worth_it() is for.
    """
    per_team = defaultdict(lambda: defaultdict(int))
    for r in rows:
        per_team[r["team"]][_bucket(r["ts"])] += 1
    out = {}
    for team, buckets in per_team.items():
        reqs = sum(buckets.values())
        windows = len(buckets)
        avg_reads = (reqs - windows) / windows if windows else 0.0
        out[team] = {
            "requests": reqs, "ttl_windows": windows,
            "avg_reads_per_write": round(avg_reads, 3),
            "worth_it": pricing.cache_is_worth_it(avg_reads),
        }
    return out


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    cache_stats = measure_cache_reads(rows)
    breakeven = pricing.cache_breakeven_reads()

    base_cost = opt_cost = opt_cost_cache_aware = 0.0
    total_tokens = 0
    # ext-4 accumulators
    split = {True: {"n": 0, "cost": 0.0, "wh": 0.0, "tokens": 0},
             False: {"n": 0, "cost": 0.0, "wh": 0.0, "tokens": 0}}

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r["is_reasoning"])))
        team = r["team"]
        total_tokens += inp + out

        # BASELINE: naive deployment - everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)

        # OPTIMIZED: cascade (route_tier), prompt caching, batch API.
        # ext-3 gate: only bank the cache discount where reads/write clears break-even.
        pin, pout = MODEL_PRICES[r["route_tier"]]
        cache_ok = cache_stats.get(team, {}).get("worth_it", False)
        eff_cached = cached if cache_ok else 0
        cost = pricing.request_cost(inp, out, pin, pout, cached_in=eff_cached, batch=is_batch)
        opt_cost += cost

        # ext-3, honest variant: charge the 1.25x cache WRITE premium, amortised
        # over the (1 + reads) requests that touch the prefix.
        if cache_ok:
            avg_reads = cache_stats[team]["avg_reads_per_write"]
            in_cost = pricing.cached_cost_with_write(inp, cached, pin, avg_reads)
        else:
            in_cost = (inp / 1e6) * pin
        aware = in_cost + (out / 1e6) * pout
        if is_batch:
            aware *= 0.50
        opt_cost_cache_aware += aware

        # ext-4: what does the reasoning path actually cost in $ and Wh?
        s = split[is_reasoning]
        s["n"] += 1
        s["cost"] += cost
        s["tokens"] += inp + out
        s["wh"] += sustainability.wh_per_query(inp + out, is_reasoning=is_reasoning)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0
    cache_aware_pct = (1 - opt_cost_cache_aware / base_cost) * 100 if base_cost else 0.0

    n_total = len(rows)
    rea, non = split[True], split[False]
    total_wh = rea["wh"] + non["wh"]
    reasoning = {
        "requests": rea["n"],
        "traffic_share": rea["n"] / n_total if n_total else 0.0,
        "cost": round(rea["cost"], 2),
        "cost_share": rea["cost"] / (rea["cost"] + non["cost"]) if (rea["cost"] + non["cost"]) else 0.0,
        "wh": round(rea["wh"], 1),
        "energy_share": rea["wh"] / total_wh if total_wh else 0.0,
        "usd_per_request": round(rea["cost"] / rea["n"], 5) if rea["n"] else 0.0,
        "wh_per_request": round(rea["wh"] / rea["n"], 2) if rea["n"] else 0.0,
        "baseline_usd_per_request": round(non["cost"] / non["n"], 5) if non["n"] else 0.0,
        "baseline_wh_per_request": round(non["wh"] / non["n"], 3) if non["n"] else 0.0,
    }
    # Cap scenario: anything above the cap is re-routed to the non-reasoning path,
    # inheriting the average cost/energy of a normal request.
    cap_n = int(REASONING_CAP_SHARE * n_total)
    moved = max(0, rea["n"] - cap_n)
    reasoning["cap_share"] = REASONING_CAP_SHARE
    reasoning["capped_requests"] = moved
    reasoning["usd_saved_daily"] = round(
        moved * (reasoning["usd_per_request"] - reasoning["baseline_usd_per_request"]), 2)
    reasoning["wh_saved_daily"] = round(
        moved * (reasoning["wh_per_request"] - reasoning["baseline_wh_per_request"]), 1)
    reasoning["usd_saved_monthly"] = round(reasoning["usd_saved_daily"] * 30, 2)
    reasoning["kwh_saved_monthly"] = round(reasoning["wh_saved_daily"] * 30 / 1000.0, 1)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={n_total}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print(f"\n-- ext3: cache economics (TTL {CACHE_TTL_MIN} min) --")
        print(f"break-even reads per write = {breakeven:.2f}  "
              f"(write {pricing.CACHE_WRITE_MULTIPLIER}x, read {pricing.CACHE_READ_DISCOUNT}x)")
        print(f"{'team':12}{'requests':>10}{'TTL windows':>13}{'reads/write':>13}{'cache?':>9}")
        for team, st in sorted(cache_stats.items()):
            print(f"{team:12}{st['requests']:>10}{st['ttl_windows']:>13}"
                  f"{st['avg_reads_per_write']:>13}{('YES' if st['worth_it'] else 'no'):>9}")
        print(f"cache-write-aware cost: ${opt_cost_cache_aware:,.2f}/day "
              f"({cache_aware_pct:.1f}% saved) - the naive number above ignores the write premium "
              f"and flatters the bill by ${opt_cost_cache_aware - opt_cost:,.2f}/day.")

        print(f"\n-- ext4: reasoning budget --")
        print(f"reasoning traffic : {rea['n']}/{n_total} requests ({reasoning['traffic_share']:.1%})")
        print(f"reasoning cost    : ${rea['cost']:,.2f}/day ({reasoning['cost_share']:.1%} of spend)")
        print(f"reasoning energy  : {rea['wh']:,.0f} Wh/day ({reasoning['energy_share']:.1%} of energy)")
        print(f"per request       : ${reasoning['usd_per_request']:.5f} and {reasoning['wh_per_request']:.1f} Wh "
              f"vs ${reasoning['baseline_usd_per_request']:.5f} / {reasoning['baseline_wh_per_request']:.2f} Wh normal "
              f"({reasoning['wh_per_request'] / max(reasoning['baseline_wh_per_request'], 1e-9):.0f}x the energy)")
        print(f"routing cap at {REASONING_CAP_SHARE:.0%} of traffic -> re-route {moved} requests: "
              f"save ${reasoning['usd_saved_monthly']:,.0f}/month and "
              f"{reasoning['kwh_saved_monthly']:,.1f} kWh/month")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_stats": cache_stats, "cache_breakeven_reads": round(breakeven, 3),
        "optimized_daily_cache_aware": round(opt_cost_cache_aware, 2),
        "savings_pct_cache_aware": round(cache_aware_pct, 1),
        "reasoning": reasoning,
    }


if __name__ == "__main__":
    run()
