"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def _tier_cost(tier: str, gpu_hours: float, catalog_row: dict, gpu_type: str) -> float:
    """Price a recommendation using the same interruption assumption as policy."""
    on_demand_hr = num(catalog_row["on_demand_hr"])
    if tier == "spot":
        return pricing.spot_checkpoint_cost(
            gpu_hours, num(catalog_row["spot_hr"]), on_demand_hr,
            interrupt_rate=pricing.SPOT_INTERRUPT_RATE.get(gpu_type, pricing.DEFAULT_INTERRUPT_RATE),
        )["spot_cost"]
    if tier in ("reserved", "reserved_3yr"):
        return gpu_hours * num(catalog_row["reserved_3yr_hr"])
    if tier == "reserved_1yr":
        return gpu_hours * num(catalog_row["reserved_1yr_hr"])
    return gpu_hours * on_demand_hr


def recommendation_matrix(cat: dict) -> list[dict]:
    """Extension-1 decision matrix: GPU type × duty cycle × interruptibility."""
    scenarios = (
        ("6h interruptible", 6, True, 30),
        ("16h interruptible", 16, True, 180),
        ("16h reliable", 16, False, 180),
        ("24h reliable", 24, False, 730),
    )
    matrix = []
    for gpu_type, c in cat.items():
        row = {"gpu_type": gpu_type}
        for label, hours, interruptible, job_days in scenarios:
            row[label] = pricing.recommend_tier(
                hours, interruptible, gpu_type=gpu_type, job_days=job_days,
                spot_hr=num(c["spot_hr"]), on_demand_hr=num(c["on_demand_hr"]),
            )
        matrix.append(row)
    return matrix


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = legacy_policy_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        days = num(j["days"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, job_days=days,
                                      spot_hr=num(c["spot_hr"]), on_demand_hr=od)
        opt_cost = _tier_cost(tier, gpu_hours, c, gtype)
        legacy_tier = pricing.recommend_tier(hpd, interruptible)
        legacy_cost = _tier_cost(legacy_tier, gpu_hours, c, gtype)

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        legacy_policy_monthly += legacy_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "legacy_tier": legacy_tier, "on_demand": round(on_demand_cost),
                     "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    legacy_savings_pct = ((on_demand_monthly - legacy_policy_monthly) / on_demand_monthly * 100
                          if on_demand_monthly else 0.0)
    matrix = recommendation_matrix(cat)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        print(f"policy comparison: legacy {legacy_savings_pct:.1f}% saved -> extended {savings_pct:.1f}% saved")
        print("\nExtension 1 decision matrix:")
        print(f"{'GPU':7}{'6h int.':18}{'16h int.':18}{'16h reliable':18}{'24h reliable':18}")
        for row in matrix:
            print(f"{row['gpu_type']:7}{row['6h interruptible']:18}{row['16h interruptible']:18}"
                  f"{row['16h reliable']:18}{row['24h reliable']:18}")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "legacy_policy_monthly": round(legacy_policy_monthly),
            "legacy_savings_pct": round(legacy_savings_pct, 1), "matrix": matrix}


if __name__ == "__main__":
    run()
