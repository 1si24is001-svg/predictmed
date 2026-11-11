import numpy as np

from predictmed.features.twin import KidneyTwin, TwinParams, prior_design
from predictmed.models.uncertainty import MondrianConformal, venn_abers_interval
from predictmed.simulator import simulate


def test_twin_recovers_slope_and_resists_inpatient_outlier():
    rng = np.random.default_rng(0)
    p = TwinParams()
    true_slope_m = -6 / 12
    errs = []
    for _ in range(200):
        tw = KidneyTwin(p, prior_design(1, 1, 0, 2.0, 50))
        for t in np.sort(rng.uniform(0, 36, 6)):
            tw.update(t, 50 + true_slope_m * t + rng.normal(0, 4))
        errs.append(tw.summary(36)["twin_slope_yr"] - (-6))
    assert abs(np.mean(errs)) < 1.5
    tw = KidneyTwin(p, prior_design(0, 0, 0, np.nan, 60))
    for t in range(0, 24, 4):
        tw.update(t, 60.0)
    before = tw.summary(24)["twin_egfr_now"]
    tw.update(24, 25.0, inpatient=True)               # acute AKI value should barely move the chronic estimate
    assert before - tw.summary(24)["twin_egfr_now"] < 6


def test_conformal_class_conditional_coverage():
    rng = np.random.default_rng(1)
    y = rng.random(20000) < 0.1
    p = np.clip(np.where(y, rng.beta(4, 3, y.size), rng.beta(1.2, 8, y.size)), 1e-4, 1 - 1e-4)
    c = MondrianConformal(0.1).fit(p[:10000], y[:10000])
    in0, in1 = c.predict_set(p[10000:])
    yt = y[10000:]
    assert in1[yt].mean() >= 0.88 and in0[~yt].mean() >= 0.88


def test_venn_abers_interval_brackets():
    rng = np.random.default_rng(2)
    s = rng.random(2000)
    y = (rng.random(2000) < s).astype(int)
    lo, hi, p = venn_abers_interval(s, y, [0.7])[0]
    assert lo <= p <= hi and 0.5 < p < 0.9


def test_simulator_counterfactual_common_random_numbers():
    a = simulate(600, 48, seed=3)
    b = simulate(600, 48, seed=3, intervention={"month": 24, "actions": {i: ["NO_SGLT2I"] for i in range(60)}})
    treated = {f"M{100000 + i}" for i in range(60)}
    ta, tb = a["oracle_truth"], b["oracle_truth"]
    ua, ub = ta[~ta.member_id.isin(treated)].reset_index(drop=True), tb[~tb.member_id.isin(treated)].reset_index(drop=True)
    assert ua.equals(ub)                               # untreated members unchanged -> causal difference only
