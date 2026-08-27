# NimbusAI - GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $18,005  
**Optimized spend:** $9,434  
**Projected savings:** $8,571  (**48%**)

## Unit economics - the number that actually matters

A GPU-hour is an input, not an outcome. The bill per unit of work served:

| Metric | Baseline | Optimized | Change |
|---|---:|---:|---:|
| $ / 1M tokens (inference) | $6.488 | $1.126 | -83% |
| $ / 1M tokens incl. cache-write premium | $6.488 | $1.204 | -81% |
| Fleet spend / month | $18,005 | $9,434 | -48% |
| Wh / query (median request) | 0.24 | 0.24 | unchanged - energy follows tokens, not price |

## Savings by lever

| Lever | Savings (USD) | Share of savings | What it is |
|---|---:|---:|---|
| Inference (cascade/cache/batch) | $1,212 | 14% | route 80% of traffic to the small model, cache the shared prefix, batch the offline work |
| Purchasing (spot/reserved) | $6,363 | 74% | checkpointed jobs to spot, 24x7 services to 3yr reserved |
| Right-size util-lies | $396 | 5% | move the memory-bound box to a part with more HBM bandwidth per dollar |
| Kill idle GPUs | $600 | 7% | stop paying for instances left up after the job finished |

## What the numbers mean

### The GPU-Util lie: why 98% busy is not 98% useful

`gpu-h100-4` reports **98.2% GPU-Util** and only **MFU 0.19** - versus MFU 0.43 on `gpu-h100-3`, the same silicon. `nvidia-smi` GPU-Util answers *"was at least one kernel resident in the last sampling window?"* It is a duty-cycle counter, not an efficiency counter, so a kernel that spends its life stalled on HBM reads - or a stream of tiny kernels dominated by launch overhead and pipeline bubbles - pins the number at ~100% while the tensor cores idle between operands. The roofline confirms the mechanism: measured arithmetic intensity is 245 FLOP/byte against a ridge point of 296, i.e. the job is memory-bound and starves on bandwidth, not on FLOPs. Financially you rent the whole GPU-hour and collect 45% of the FLOPs your healthy trainers get: $2.50/h buys about 19% of the part. Bill by MFU/MBU and $/1M-token; treat GPU-Util as a liveness probe only.

### Why the cheapest GPU is rarely the cheapest answer

Serving is bought in TB/s and GB, not in dollars-per-hour. The catalog prices **MI300X** at $0.37/TB-s-hr, **B200** at $0.64/TB-s-hr, **H100** at $0.75/TB-s-hr ... up to **L4** at $2.67. That inversion is why the memory-bound A100/A10G servers stay where they are: the cheap parts would need two-to-four units to hold the measured bandwidth and end up dearer. Only `gpu-h100-4` moves - H100 -> 1xMI300X at $1.95/h, worth $396/mo - because that box needs 0.90 TB/s and 67 GB, which one MI300X covers outright.

### Where the inference bill actually goes

Baseline is the naive deployment: every request on the large model, no cache, no batch - $6.488/1M-token. Cascading ~80% of traffic to the small model, caching the shared system prefix and batching the offline eval work lands at $1.126/1M-token (83% off). The discounts multiply rather than add: batch (0.5x) on cached input (0.1x) is 0.05x of naive. Note the honest number is $9.07/day rather than $8.48/day once the 1.25x cache-write premium is charged - still 81% off, but do not quote the flattered one.

### Purchasing: the discount you book is not the discount you get

Re-pricing the fleet with reclaim rates per GPU pool, 24x7 commitment billing and a 1yr-vs-3yr comparison gives $10,176/mo (38.5% off on-demand). The legacy duty-cycle policy claimed $9,849 (40.5%) - $327/mo of savings that do not exist, because it billed reservations only for hours used and priced every spot pool at one flat 5%/h. A 3yr H100 reservation reads -44% against today's on-demand but only -35% against the average market rate over the term, once you allow for street prices falling ~15%/yr.

## Recommended sequence (by return on effort)

| # | Action | Savings/mo | Effort | Risk | Why this position |
|---:|---|---:|---|---|---|
| 1 | Kill idle GPUs (auto-shutdown after job exit) | $600 | hours | none | pure waste, one scheduler hook, no product impact - do it today |
| 2 | Move checkpointed jobs to spot, commit only the 24x7 services | $6,363 | days | medium | largest bucket by far; needs checkpointing to be real and a commitment sign-off |
| 3 | Ship the inference levers (cascade + cache + batch) | $1,212 | 1-2 sprints | medium | needs an eval gate so the small model does not silently degrade quality |
| 4 | Right-size gpu-h100-4 to MI300X | $396 | days | medium | smallest bucket and a migration; do it after the free money is banked |
| 5 | Cap the reasoning path at 5% of traffic | $9 | days | low | only $9/mo but 358 kWh/mo - an energy lever, not a cost lever |

## Allocation readiness

- Tag coverage: **92%** (chargeback gate at 80%: **OPEN**)
- FOCUS export: `outputs/focus_export.csv`

| Team | Daily inference cost | Share |
|---|---:|---:|
| assistant | $2.59 | 31% |
| search | $2.49 | 29% |
| eval | $1.79 | 21% |
| rag | $1.60 | 19% |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cleanest grid: europe-north1  |  cheapest electricity: us-east-wa  |  balanced (carbon-priced): us-east-wa
- Reasoning traffic is 8.4% of requests but 16% of inference spend and 94% of energy (148 Wh vs 0.86 Wh per request).
- Capping the reasoning path at 5% of traffic saves $9/mo and 358 kWh/mo.
- Moving the 5 checkpointed jobs (2,004 kWh/mo) from us-east-1 to europe-north1 cuts 701 kg CO2e/mo (92%); the cheapest grid (us-east-wa) also saves $130/mo of electricity.

| Region | $/kWh | gCO2e/kWh | Electricity $/mo | tCO2e/mo | Blended $ @ $100/t | Latency |
|---|---:|---:|---:|---:|---:|---:|
| us-east-wa | 0.060 | 90 | $110 | 0.18 | $128 | 55ms |
| us-west-2 | 0.070 | 120 | $140 | 0.24 | $164 | 70ms |
| europe-north1 | 0.090 | 30 | $180 | 0.06 | $186 | 110ms |
| us-east-1 | 0.120 | 380 | $240 | 0.76 | $317 | 15ms |
| europe-central2 | 0.180 | 660 | $361 | 1.32 | $493 | 125ms |

## Your-Turn extensions (measured)

### Ext-1 - risk-priced purchasing policy (`pricing.recommend_tier_v2`)

**Result:** $10,176/mo vs $9,849/mo claimed by the old policy (38.5% vs 40.5% savings on $16,539 of on-demand)

Per-pool reclaim rates (H100 5%/h vs L4 15%/h), 24x7 commitment billing and a 1yr-vs-3yr comparison against the declining market price. Same tiers, $327/mo less fantasy.

### Ext-2 - MBU right-sizing (`m1.rightsize_candidate`)

**Result:** $396/mo from 1 swap(s); 5 GPUs deliberately left alone

Sizing against measured p95 bandwidth, peak VRAM and p95 FLOPs - and counting how many units a cheaper part would take - is what stops a $0.80/h L4 from looking like an upgrade over a $1.79/h A100.

### Ext-3 - cache break-even (`pricing.cache_is_worth_it`)

**Result:** break-even 0.28 reads/write; measured assistant 1.87, eval 0.90, rag 1.29, search 1.42; honest cost $9.07/day vs $8.48/day naive

A 1.25x write repaid by 0.1x reads breaks even after 0.28 reads, so a single reuse inside the 5-minute TTL already pays - every team clears it here, but eval only by 3x, and a thinner-traffic tenant would lose money on caching.

### Ext-4 - reasoning budget

**Result:** 8.4% of requests = 16% of spend and 94% of energy; cap at 5% saves $9/mo + 358 kWh/mo

A reasoning request burns 148 Wh against 0.86 Wh for a normal one: the ~80x energy multiplier compounds with ~6x more output tokens. Route on task complexity, not on user preference.

### Ext-5 - carbon-aware scheduling (`missions/m6_carbon_aware.py`)

**Result:** 701 kg CO2e/mo (92%) by moving 2,004 kWh of checkpointed work to europe-north1; $130/mo cheaper in us-east-wa

Cheapest (us-east-wa), cleanest (europe-north1) and balanced at $100/tCO2e (us-east-wa) are three different answers. Only interruptible jobs move - the cleanest grid costs ~95ms of latency, which is free for training and unacceptable for chat.

## Caveats

- Prices are June-2026 illustrative snapshots; GPU street prices move monthly.
- Savings are modelled from synthetic telemetry (seed 25), not from a production bill.
- Reserved buckets assume the commitment is actually consumed - re-check duty cycle quarterly or the discount reverses.
- Right-sizing assumes the workload ports cleanly to the target part (ROCm for MI300X).

_Figures are June-2026 as-of snapshots; re-baseline before acting._