"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations
from typing import Optional


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


# Per-GPU-type spot interruption rate (per hour) — H100-class spot is far more
# stable than small inference GPUs; drives the spot-vs-reserved trade-off.
SPOT_INTERRUPT_RATE = {
    "H100": 0.02, "H200": 0.02, "B200": 0.02, "MI300X": 0.03,
    "A100": 0.04, "A10G": 0.08, "L4": 0.10,
}
DEFAULT_INTERRUPT_RATE = 0.05


def _spot_effective_multiplier(interrupt_rate: float, ckpt_overhead_frac: float = 0.03,
                               rework_hours_per_interrupt: float = 0.5) -> float:
    """Effective hours multiplier when running on spot (checkpoint + rework)."""
    return 1.0 + ckpt_overhead_frac + interrupt_rate * rework_hours_per_interrupt


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45,
                   gpu_type: Optional[str] = None, job_days: Optional[float] = None,
                   reserved_1yr_discount: float = 0.20, reserved_3yr_discount: float = 0.45,
                   spot_hr: Optional[float] = None, on_demand_hr: Optional[float] = None) -> str:
    """Pick a purchasing tier from duty cycle, interruptibility, GPU type and job length.

    Extension-1 policy (superset of the documented simple rule — old call sites
    with only (hours_per_day, interruptible) keep the original behaviour):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)

    New considerations:
      - spot interruption rate varies by GPU type (H100-class spot is stable,
        small inference GPUs get reclaimed often). When real prices are given we
        compare spot's *effective* hourly cost (spot price x checkpoint/rework
        multiplier) against 3yr reserved: if spot is no longer cheaper, a
        high-duty interruptible job falls back to 'reserved'.
      - reserved term is chosen by job length: 24/7 permanent services get
        'reserved' (3yr, max discount); high-duty but non-24/7 jobs get
        'reserved_1yr' (above the 1yr break-even, less commitment).
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        rate = SPOT_INTERRUPT_RATE.get(gpu_type, DEFAULT_INTERRUPT_RATE)
        spot_mult = _spot_effective_multiplier(rate)
        if duty >= be and spot_hr is not None and on_demand_hr and on_demand_hr > 0:
            spot_effective_hr = spot_hr * spot_mult
            reserved_hr = on_demand_hr * (1.0 - reserved_3yr_discount)
            if spot_effective_hr >= reserved_hr:
                return "reserved"
        return "spot"
    if duty >= be:
        be_1yr = break_even_utilization(reserved_1yr_discount)
        if duty >= be_1yr and hours_per_day < 24 and job_days is not None and job_days < 365:
            return "reserved_1yr"
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
