import numpy as np
import pandas as pd
import pytest


def test_groupwise_calibrator_fits_and_transforms():
    from predictmed.models.fairness import GroupwiseCalibrator

    rng = np.random.default_rng(42)
    scores = rng.uniform(0.01, 0.99, 300)
    labels = (scores > 0.5).astype(int)
    groups = rng.choice(["medicaid", "medicare", "commercial"], 300)

    cal = GroupwiseCalibrator(min_samples=20)
    cal.fit(scores, labels, groups)

    out = cal.transform(scores, groups)
    assert out.shape == scores.shape
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_compute_sample_weights():
    from predictmed.models.fairness import compute_sample_weights

    df = pd.DataFrame({"plan_type": ["medicaid"] * 20 + ["medicare"] * 80})
    weights = compute_sample_weights(df, group_col="plan_type")

    # Medicaid is minority class -> higher sample weight
    assert weights[:20].mean() > weights[20:].mean()
    assert np.isclose(weights.mean(), 1.0)


def test_guardian_plan_calibration_gate():
    from predictmed.agents.core import Journal
    from predictmed.agents.guardian import Guardian

    guardian = Guardian(journal=Journal())
    cfg = {
        "promotion_gates": {
            "min_auprc_lift_over_prevalence": 2.0,
            "min_auroc": 0.75,
            "calibration_slope": [0.8, 1.2],
            "max_ece": 0.03,
            "max_subgroup_auroc_gap": 0.08,
            "conformal_coverage_tolerance": 0.03,
            "plan_calibration_oe_min": 0.80,
            "plan_calibration_oe_max": 1.25,
            "plan_calibration_min_events": 30,
        }
    }

    subgroup = pd.DataFrame([
        {"attribute": "race_ethnicity", "group": "White", "events": 100, "auroc": 0.88, "o_e_ratio": 1.0},
        {"attribute": "race_ethnicity", "group": "Black", "events": 50, "auroc": 0.86, "o_e_ratio": 1.0},
        {"attribute": "plan", "group": "Commercial", "events": 80, "auroc": 0.89, "o_e_ratio": 1.05},
        {"attribute": "plan", "group": "Medicaid", "events": 40, "auroc": 0.85, "o_e_ratio": 1.15},
    ])

    valid_metrics = {
        "auprc_lift": 3.0,
        "auroc": 0.88,
        "cal_slope": 1.02,
        "ece": 0.01,
    }
    coverage = {"test": (0.91, 100)}

    res = guardian.gates(valid_metrics, subgroup, coverage, cfg, candidate="test_candidate")
    assert "plan_calibration_gap" in res["detail"]
    assert res["detail"]["plan_calibration_gap"]["pass"] is True
