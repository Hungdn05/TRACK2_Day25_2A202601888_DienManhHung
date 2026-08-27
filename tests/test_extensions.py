import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing


def test_recommend_tier_backward_compatible():
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"


def test_recommend_tier_1yr_vs_3yr():
    assert pricing.recommend_tier(20, False, job_days=100) == "reserved_1yr"
    assert pricing.recommend_tier(20, False, job_days=730) == "reserved"
    assert pricing.recommend_tier(24, False, job_days=100) == "reserved"


def test_recommend_tier_spot_falls_back_to_reserved():
    assert pricing.recommend_tier(20, True, gpu_type="A100", spot_hr=1.10, on_demand_hr=1.79) == "reserved"
    assert pricing.recommend_tier(20, True, gpu_type="H100", spot_hr=1.50, on_demand_hr=2.50) == "reserved"
    assert pricing.recommend_tier(20, True, gpu_type="L4", spot_hr=0.35, on_demand_hr=0.80) == "spot"


def test_spot_effective_multiplier():
    assert pricing._spot_effective_multiplier(0.0) == 1.03
    assert pricing._spot_effective_multiplier(0.1) == 1.08


def test_reasoning_budget_reports_required_and_strict_caps():
    from missions import m2_inference_levers
    res = m2_inference_levers.run(verbose=False)
    assert res["reasoning_share"] < 0.10
    assert res["reasoning_cap_10_savings_usd"] == 0.0
    assert res["reasoning_cap_5_savings_usd"] > 0.0


def test_carbon_aware_scheduling():
    from missions import m3b_carbon
    res = m3b_carbon.run(verbose=False)
    assert res["carbon_saved_gco2"] > 0
    assert res["carbon_saved_pct"] > 50
    assert res["cleanest_region"] == "europe-north1"
    assert len(res["jobs"]) == 5
