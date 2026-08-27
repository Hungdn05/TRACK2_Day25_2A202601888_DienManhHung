"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# Extension 4: the rubric's 10% budget and a stricter 5% policy scenario.
# The dataset is already below 10%; reporting both avoids pretending that the
# required 10% cap creates savings when it does not.
REASONING_CAP_FRACS = (0.10, 0.05)


def _cap_savings(reasoning_share: float, reasoning_cost: float, reasoning_wh: float,
                 target_share: float) -> tuple[float, float]:
    """Savings from removing reasoning traffic above ``target_share``.

    This is intentionally a gross-avoidance estimate: a production policy must
    validate quality and price any replacement non-reasoning route separately.
    """
    if reasoning_share <= target_share or reasoning_share <= 0:
        return 0.0, 0.0
    excess_frac = (reasoning_share - target_share) / reasoning_share
    return reasoning_cost * excess_frac, reasoning_wh * excess_frac


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    reasoning_cost = reasoning_tokens = 0.0
    reasoning_wh = non_reasoning_wh = 0.0
    reasoning_reqs = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r["is_reasoning"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        # Extension 4: reasoning budget — $ and Wh split
        if is_reasoning:
            reasoning_reqs += 1
            reasoning_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
            reasoning_tokens += inp + out
            reasoning_wh += sustainability.wh_per_query(inp + out, is_reasoning=True)
        else:
            non_reasoning_wh += sustainability.wh_per_query(inp + out, is_reasoning=False)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # Extension 4: quantify the required 10% budget and a stricter 5% policy.
    reasoning_share = reasoning_reqs / len(rows) if rows else 0.0
    cap_scenarios = {
        target: _cap_savings(reasoning_share, reasoning_cost, reasoning_wh, target)
        for target in REASONING_CAP_FRACS
    }
    reasoning_cap_10_usd, reasoning_cap_10_wh = cap_scenarios[0.10]
    reasoning_cap_5_usd, reasoning_cap_5_wh = cap_scenarios[0.05]

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print("\n== Extension 4: Reasoning budget ==")
        print(f"reasoning requests: {reasoning_reqs}/{len(rows)} ({reasoning_share:.1%} of traffic)")
        print(f"reasoning cost    : ${reasoning_cost:,.2f}/day  ({reasoning_cost/opt_cost*100:.1f}% of optimized bill)")
        print(f"reasoning energy  : {reasoning_wh:,.0f} Wh/day vs {non_reasoning_wh:,.0f} Wh/day for the rest")
        print(f"10% budget status : {'within budget' if reasoning_share <= 0.10 else 'over budget'}; "
              f"avoid ${reasoning_cap_10_usd:,.2f}/day and {reasoning_cap_10_wh:,.0f} Wh/day")
        print(f"strict 5% policy  : avoid ${reasoning_cap_5_usd:,.2f}/day and {reasoning_cap_5_wh:,.0f} Wh/day")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "reasoning_reqs": reasoning_reqs, "reasoning_share": round(reasoning_share, 4),
        "reasoning_cost_daily": round(reasoning_cost, 2),
        "reasoning_wh_daily": round(reasoning_wh, 1),
        "non_reasoning_wh_daily": round(non_reasoning_wh, 1),
        "reasoning_cap_10_savings_usd": round(reasoning_cap_10_usd, 2),
        "reasoning_cap_10_savings_wh": round(reasoning_cap_10_wh, 1),
        "reasoning_cap_5_savings_usd": round(reasoning_cap_5_usd, 2),
        "reasoning_cap_5_savings_wh": round(reasoning_cap_5_wh, 1),
    }


if __name__ == "__main__":
    run()
