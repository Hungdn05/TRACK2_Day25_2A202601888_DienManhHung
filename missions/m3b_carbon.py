"""M3b — Carbon-aware Scheduling (Extension 5).

For every interruptible job in workloads.csv, compare the carbon footprint of
running it in us-east-1 (current) vs the cleanest region (europe-north1), and
report the gCO2e saved. Also prints a 5-region comparison table so the
cheapest / cleanest / balanced region can be chosen explicitly.

Run: python missions/m3b_carbon.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability

DAYS = 30
CURRENT_REGION = "us-east-1"


def _job_wh(job: dict, cat: dict) -> float:
    """Energy (Wh) for one job over its full run: GPU watts x hours."""
    gtype = job["gpu_type"]
    watts = num(cat[gtype]["watts"])
    hours = num(job["hours_per_day"]) * num(job["days"]) * int(num(job["num_gpus"]))
    return watts * hours


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    clean_region = min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)

    rows = []
    total_wh = total_carbon_now = total_carbon_clean = 0.0
    for j in jobs:
        if not bool(int(num(j["interruptible"]))):
            continue
        wh = _job_wh(j, cat)
        c_now = sustainability.carbon_g(wh, CURRENT_REGION)
        c_clean = sustainability.carbon_g(wh, clean_region)
        rows.append({
            "job_id": j["job_id"], "gpu_type": j["gpu_type"],
            "wh": round(wh), "carbon_us_east": round(c_now, 1),
            "carbon_clean": round(c_clean, 1),
            "saved_gco2": round(c_now - c_clean, 1),
        })
        total_wh += wh
        total_carbon_now += c_now
        total_carbon_clean += c_clean

    saved = total_carbon_now - total_carbon_clean
    saved_pct = saved / total_carbon_now * 100 if total_carbon_now else 0.0

    # 5-region comparison table
    regions = []
    for r in sustainability.REGION_CARBON:
        price = sustainability.REGION_PRICE_KWH[r]
        carbon = sustainability.REGION_CARBON[r]
        regions.append({
            "region": r, "price_kwh": price, "carbon_kwh": carbon,
            "energy_cost": round(total_wh / 1000.0 * price, 1),
            "carbon_total": round(sustainability.carbon_g(total_wh, r), 1),
        })
    regions.sort(key=lambda x: x["carbon_kwh"])
    cheapest = min(regions, key=lambda x: x["price_kwh"])
    cleanest = min(regions, key=lambda x: x["carbon_kwh"])
    # Normalize unlike units and weight carbon 75% / energy price 25% for a
    # transparent "balanced" choice. Teams can tune these weights by policy.
    price_min, price_max = min(r["price_kwh"] for r in regions), max(r["price_kwh"] for r in regions)
    carbon_min, carbon_max = min(r["carbon_kwh"] for r in regions), max(r["carbon_kwh"] for r in regions)
    for r in regions:
        price_score = (r["price_kwh"] - price_min) / (price_max - price_min)
        carbon_score = (r["carbon_kwh"] - carbon_min) / (carbon_max - carbon_min)
        r["balanced_score"] = round(0.25 * price_score + 0.75 * carbon_score, 3)
    balanced = min(regions, key=lambda x: x["balanced_score"])

    if verbose:
        print("== Extension 5: Carbon-aware Scheduling ==")
        print(f"interruptible jobs: {len(rows)}  total energy: {total_wh:,.0f} Wh")
        print(f"{'job':18}{'gpu':7}{'Wh':>10}{'us-east-1':>12}{'cleanest':>12}{'saved':>10}")
        for r in rows:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['wh']:>10,}{r['carbon_us_east']:>12,}{r['carbon_clean']:>12,}{r['saved_gco2']:>10,}")
        print(f"\ntotal carbon: {total_carbon_now:,.0f} gCO2e (us-east-1) -> {total_carbon_clean:,.0f} gCO2e ({clean_region})")
        print(f"carbon saved : {saved:,.0f} gCO2e  ({saved_pct:.1f}%)")
        print(f"\n5-region comparison (for {total_wh:,.0f} Wh):")
        print(f"{'region':16}{'$/kWh':>7}{'gCO2/kWh':>10}{'energy $':>10}{'carbon g':>14}{'balance':>9}")
        for r in regions:
            print(f"{r['region']:16}{r['price_kwh']:>7.3f}{r['carbon_kwh']:>10}{r['energy_cost']:>10,.1f}"
                  f"{r['carbon_total']:>14,.1f}{r['balanced_score']:>9.3f}")
        print(f"\ncheapest region : {cheapest['region']} (${cheapest['price_kwh']}/kWh)")
        print(f"cleanest region : {cleanest['region']} ({cleanest['carbon_kwh']} gCO2/kWh)")
        print(f"balanced region : {balanced['region']} (75% carbon / 25% energy-price weighting)")

    return {
        "jobs": rows, "total_wh": round(total_wh),
        "carbon_us_east": round(total_carbon_now, 1), "carbon_clean": round(total_carbon_clean, 1),
        "carbon_saved_gco2": round(saved, 1), "carbon_saved_pct": round(saved_pct, 1),
        "cleanest_region": clean_region, "cheapest_region": cheapest["region"],
        "balanced_region": balanced["region"], "regions": regions,
    }


if __name__ == "__main__":
    run()
