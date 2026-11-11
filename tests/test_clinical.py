import numpy as np

from predictmed.clinical import (ckd_epi_2021, detect_care_gaps, g_stage, inverse_ckd_epi_2021, kdigo_heatmap,
                                 kfre_4var, modifiable_fraction)


def test_ckd_epi_2021_reference_value():
    # 142 * (1.0/0.9)^-1.2 * 0.9938^50 = 91.7 for a 50-year-old male with Scr 1.0 mg/dL
    assert abs(float(ckd_epi_2021(1.0, 50, False)) - 91.7) < 0.3
    # female at Scr 0.7 (= kappa): 142 * 0.9938^50 * 1.012 = 105.3
    assert abs(ckd_epi_2021(0.7, 50, True) - 105.3) < 0.3


def test_inverse_round_trip():
    eg = np.array([8, 15, 29, 45, 60, 89, 110])
    for female in (True, False):
        cr = inverse_ckd_epi_2021(eg, np.full(7, 63), np.full(7, female))
        assert np.allclose(ckd_epi_2021(cr, np.full(7, 63), np.full(7, female)), eg, atol=1e-6)


def test_staging_and_heatmap():
    assert list(g_stage([95, 75, 50, 35, 20, 10])) == ["G1", "G2", "G3a", "G3b", "G4", "G5"]
    assert kdigo_heatmap(np.array([1.0]), np.array([1.0]))[0] == 0
    assert kdigo_heatmap(np.array([5.0]), np.array([1.0]))[0] == 3


def test_kfre_monotone():
    lo = kfre_4var(65, 1, 50, 30)
    hi = kfre_4var(65, 1, 20, 1000)
    assert 0 < lo < hi < 1


def test_care_gaps_and_modifiable_fraction():
    row = dict(egfr_last=38.0, uacr_last=450.0, dx_diabetes=1, dx_hypertension=1, months_since_uacr=4,
               months_since_egfr=2, rx_sglt2i_active=0, rx_rasi_active=0, rx_nsaid_fills_6m=2,
               months_since_nephrology=99, age=67, rx_statin_active=1)
    g = detect_care_gaps(row)
    assert {"NO_SGLT2I", "NO_RASI", "NSAID_EXPOSURE", "NO_NEPHROLOGY"} <= set(g)
    assert abs(modifiable_fraction(["NO_SGLT2I"]) - 0.37) < 1e-9
    assert modifiable_fraction([]) == 0.0
