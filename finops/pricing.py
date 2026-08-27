"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }


# ---------------------------------------------------------------------------
# YOUR-TURN EXTENSION 1 - a purchasing policy that prices RISK, not just rates
# ---------------------------------------------------------------------------
# The simple recommend_tier() above has three blind spots that cost real money:
#   1. It assumes every spot pool is reclaimed at the same rate. It is not: deep
#      H100/A100 pools churn less than the shallow small-GPU pools that every
#      hobby workload bids on.
#   2. It bills a reservation only for the hours you USE. A commitment bills
#      24x7 for the whole term whether you run or not - that is the entire
#      point of break_even_utilization().
#   3. It never compares 1yr vs 3yr. A 3yr lock is cheaper on paper, but GPU
#      street prices fall every year, so part of that "discount" is you paying
#      above the future market in years 2-3.
# The v2 policy below prices all three and returns its reasoning.

# Per-hour spot reclaim probability by GPU type (June-2026 illustrative).
# Small/cheap pools are shallow and get reclaimed far more often than H100/A100.
SPOT_INTERRUPT_RATE = {
    "H100": 0.05, "H200": 0.06, "B200": 0.09,
    "A100": 0.03, "MI300X": 0.04,
    "A10G": 0.12, "L4": 0.15,
}
DEFAULT_INTERRUPT_RATE = 0.05

# Street price of a GPU-hour falls as newer silicon lands (illustrative 15%/yr).
PRICE_DECLINE_PER_YEAR = 0.15

HOURS_PER_MONTH = 24 * 30  # what a reservation actually bills, per GPU


def avg_market_rate(on_demand_hr: float, term_years: int,
                    price_decline_yr: float = PRICE_DECLINE_PER_YEAR) -> float:
    """Average on-demand rate you would pay over `term_years` if you did NOT commit.

    Committing locks today's rate while the market keeps falling, so the honest
    benchmark for a reservation is the AVERAGE future rate, not today's rate.
    """
    if term_years <= 0:
        return on_demand_hr
    path = [on_demand_hr * (1.0 - price_decline_yr) ** t for t in range(int(term_years))]
    return sum(path) / len(path)


def risk_adjusted_discount(on_demand_hr: float, reserved_hr: float, term_years: int,
                           price_decline_yr: float = PRICE_DECLINE_PER_YEAR) -> float:
    """Real discount of a commitment vs. the AVERAGE market rate over the term.

    H100 3yr at $1.40 against $2.50 on-demand looks like -44%; against the
    3-year average market rate (~$2.14) it is only ~-35%. The gap is lock-in risk.
    """
    avg = avg_market_rate(on_demand_hr, term_years, price_decline_yr)
    if avg <= 0:
        return 0.0
    return 1.0 - reserved_hr / avg


def spot_interrupt_rate(gpu_type=None) -> float:
    """Per-hour reclaim probability for a GPU type (falls back to fleet default)."""
    return SPOT_INTERRUPT_RATE.get(gpu_type or "", DEFAULT_INTERRUPT_RATE)


def recommend_tier_v2(
    hours_per_day: float,
    interruptible: bool,
    gpu_type=None,
    days_per_month: float = 30.0,
    num_gpus: int = 1,
    on_demand_hr=None,
    spot_hr=None,
    reserved_1yr_hr=None,
    reserved_3yr_hr=None,
    price_decline_yr: float = PRICE_DECLINE_PER_YEAR,
    reserved_discount: float = 0.45,
) -> dict:
    """Risk-priced purchasing decision: tier, term, costs and the reasoning.

    Ranking happens on a RISK-ADJUSTED effective $/useful-GPU-hour:
      * on-demand  -> the market rate itself
      * spot       -> rate inflated by checkpoint overhead + expected rework at
                      this GPU type's own reclaim rate
      * reserved   -> billed 24x7 for the term (a low duty cycle therefore
                      inflates the per-useful-hour rate), then penalised by the
                      market decline you locked yourself out of.
    Falls back to the documented simple policy when no price data is supplied,
    so recommend_tier() keeps its original behaviour and its unit tests.
    """
    duty_hours = max(0.0, hours_per_day) * max(0.0, days_per_month) * max(1, num_gpus)
    committed_hours = HOURS_PER_MONTH * max(1, num_gpus)
    duty = duty_hours / committed_hours if committed_hours else 0.0

    if not all(v is not None for v in (on_demand_hr, spot_hr, reserved_1yr_hr, reserved_3yr_hr)):
        tier = "spot" if (interruptible and hours_per_day < 24) else (
            "reserved" if duty >= break_even_utilization(reserved_discount) else "on_demand")
        return {"tier": tier, "tier_detail": tier, "duty": round(duty, 3),
                "reason": "no price data - legacy duty-cycle policy"}

    rate = spot_interrupt_rate(gpu_type)
    candidates = {}

    candidates["on_demand"] = {
        "tier": "on_demand", "cost": duty_hours * on_demand_hr,
        "effective_hr": on_demand_hr, "risk_hr": on_demand_hr,
    }

    if interruptible:
        sim = spot_checkpoint_cost(duty_hours, spot_hr, on_demand_hr, interrupt_rate=rate)
        eff = sim["spot_cost"] / duty_hours if duty_hours else spot_hr
        candidates["spot"] = {
            "tier": "spot", "cost": sim["spot_cost"], "effective_hr": eff,
            "risk_hr": eff, "interrupt_rate": rate, "sim": sim,
        }

    for term, res_hr in ((1, reserved_1yr_hr), (3, reserved_3yr_hr)):
        cost = committed_hours * res_hr                      # you pay 24x7, always
        eff = cost / duty_hours if duty_hours else float("inf")
        lockin_penalty = on_demand_hr / avg_market_rate(on_demand_hr, term, price_decline_yr)
        candidates["reserved_%dyr" % term] = {
            "tier": "reserved", "tier_detail": "reserved_%dyr" % term, "cost": cost,
            "effective_hr": eff, "risk_hr": eff * lockin_penalty,
            "term_years": term, "lockin_penalty": round(lockin_penalty, 3),
            "risk_adj_discount": round(
                risk_adjusted_discount(on_demand_hr, res_hr, term, price_decline_yr), 3),
            "naive_discount": round(1.0 - res_hr / on_demand_hr, 3),
        }

    best_key = min(candidates, key=lambda k: candidates[k]["risk_hr"])
    best = candidates[best_key]
    tier = best["tier"]

    if tier == "spot":
        why = ("interruptible; %s spot reclaim ~%.0f%%/h keeps the effective rate at "
               "$%.2f/h vs $%.2f on-demand" % (gpu_type, rate * 100,
                                               best["effective_hr"], on_demand_hr))
    elif tier == "reserved":
        why = ("duty %.0f%% clears the risk-adjusted break-even %.0f%%; %dyr real discount "
               "%.0f%% (headline %.0f%%)" % (duty * 100,
                                             (1 - best["risk_adj_discount"]) * 100,
                                             best["term_years"],
                                             best["risk_adj_discount"] * 100,
                                             best["naive_discount"] * 100))
    else:
        why = ("duty %.0f%% is too low to carry a 24x7 commitment and the job is not "
               "interruptible - stay on-demand" % (duty * 100))

    return {
        "tier": tier, "tier_detail": best.get("tier_detail", best_key), "duty": round(duty, 3),
        "cost": round(best["cost"], 2), "effective_hr": round(best["effective_hr"], 4),
        "interrupt_rate": rate, "reason": why,
        "candidates": {k: round(v["cost"], 2) for k, v in candidates.items()},
        "ranked_risk_hr": {k: round(v["risk_hr"], 4) for k, v in candidates.items()},
    }


def tier_matrix(catalog_row, duty_cycles=(0.15, 0.35, 0.55, 0.75, 1.0)) -> list:
    """Decision matrix for one GPU type: duty cycle x interruptible -> tier.

    Handy as a one-page policy card for platform teams: read off the tier instead
    of re-deriving the economics per job.
    """
    out = []
    for duty in duty_cycles:
        row = {"duty": duty}
        for interruptible in (False, True):
            d = recommend_tier_v2(
                hours_per_day=24 * duty, interruptible=interruptible,
                gpu_type=catalog_row["gpu_type"], days_per_month=30, num_gpus=1,
                on_demand_hr=float(catalog_row["on_demand_hr"]),
                spot_hr=float(catalog_row["spot_hr"]),
                reserved_1yr_hr=float(catalog_row["reserved_1yr_hr"]),
                reserved_3yr_hr=float(catalog_row["reserved_3yr_hr"]),
            )
            row["interruptible" if interruptible else "steady"] = d["tier_detail"]
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# YOUR-TURN EXTENSION 3 - cache economics: a cache WRITE is not free
# ---------------------------------------------------------------------------
CACHE_WRITE_MULTIPLIER = 1.25   # Anthropic 5-min cache write ~1.25x base input
CACHE_READ_DISCOUNT = 0.10      # cached read ~0.1x base input


def cache_breakeven_reads(write_multiplier: float = CACHE_WRITE_MULTIPLIER,
                          read_discount: float = CACHE_READ_DISCOUNT) -> float:
    """Cache READS a written prefix needs before caching beats not caching.

    Serving a prefix once plus N reuses costs (1 + N) uncached. With a cache it
    costs write_multiplier + N * read_discount. Break-even:
        write_multiplier + N*read_discount < 1 + N
        N > (write_multiplier - 1) / (1 - read_discount)
    At 1.25x write / 0.1x read that is N > 0.28 - a single reuse already pays.
    """
    denom = 1.0 - read_discount
    if denom <= 0:
        return float("inf")
    return max(0.0, (write_multiplier - 1.0) / denom)


def cache_is_worth_it(avg_cache_reads: float,
                      write_multiplier: float = CACHE_WRITE_MULTIPLIER,
                      read_discount: float = CACHE_READ_DISCOUNT) -> bool:
    """True when the average prefix is re-read often enough to repay the write premium.

    `avg_cache_reads` is reads per write WITHIN the cache TTL - measure it, do not
    assume it. Bursty traffic on a short TTL can sit below break-even, and then
    prompt caching makes the bill bigger, not smaller.
    """
    return avg_cache_reads > cache_breakeven_reads(write_multiplier, read_discount)


def cached_cost_with_write(input_tok: int, cached_in: int, price_in_per_m: float,
                           avg_cache_reads: float,
                           write_multiplier: float = CACHE_WRITE_MULTIPLIER,
                           read_discount: float = CACHE_READ_DISCOUNT) -> float:
    """Amortised input cost of a cached prefix INCLUDING its share of the write premium.

    One write is amortised over the (1 + avg_cache_reads) requests that touch it.
    """
    cached_in = min(max(0, cached_in), input_tok)
    uncached = input_tok - cached_in
    reads = max(0.0, avg_cache_reads)
    write_share = (write_multiplier - read_discount) / (1.0 + reads)
    unit = read_discount + write_share
    return (uncached / 1e6) * price_in_per_m + (cached_in / 1e6) * price_in_per_m * unit
