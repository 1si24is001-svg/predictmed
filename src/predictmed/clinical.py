"""Clinical knowledge layer: equations, staging, guideline care-gap rules and literature priors.

Everything here is deterministic, versioned and unit-tested, because agents and models
both depend on it. Nothing in this module is a clinical recommendation; it encodes
published equations and guideline statements so the research system can reason with them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KNOWLEDGE_VERSION = "kdigo2024-ckdepi2021-kfre4-v1"

# --------------------------------------------------------------------------------------
# CKD-EPI 2021 race-free creatinine equation (Inker et al., NEJM 2021; NKF reference page)
#   eGFR = 142 * min(Scr/k,1)^a * max(Scr/k,1)^-1.200 * 0.9938^age * 1.012[female]
#   k = 0.7 (F) / 0.9 (M);  a = -0.241 (F) / -0.302 (M)
# --------------------------------------------------------------------------------------

def ckd_epi_2021(scr_mg_dl, age, female):
    scr = np.asarray(scr_mg_dl, dtype=float)
    age = np.asarray(age, dtype=float)
    female = np.asarray(female, dtype=bool)
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    ratio = scr / k
    egfr = (142.0 * np.minimum(ratio, 1.0) ** a * np.maximum(ratio, 1.0) ** -1.200
            * 0.9938 ** age * np.where(female, 1.012, 1.0))
    return egfr


def inverse_ckd_epi_2021(egfr, age, female):
    """Serum creatinine that produces a given eGFR (used by the simulator to emit realistic labs)."""
    egfr = np.asarray(egfr, dtype=float)
    age = np.asarray(age, dtype=float)
    female = np.asarray(female, dtype=bool)
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    base = 142.0 * 0.9938 ** age * np.where(female, 1.012, 1.0)
    r = np.clip(egfr, 1e-3, None) / base
    ratio_high = r ** (-1.0 / 1.200)          # branch Scr/k >= 1
    ratio_low = r ** (1.0 / a)                # branch Scr/k < 1
    ratio = np.where(ratio_high >= 1.0, ratio_high, ratio_low)
    return ratio * k


# --------------------------------------------------------------------------------------
# KDIGO staging
# --------------------------------------------------------------------------------------
G_EDGES = [(90, "G1"), (60, "G2"), (45, "G3a"), (30, "G3b"), (15, "G4"), (-1, "G5")]
G_ORDER = {"G1": 1, "G2": 2, "G3a": 3, "G3b": 4, "G4": 5, "G5": 6}


def g_stage(egfr):
    egfr = np.asarray(egfr, dtype=float)
    out = np.full(egfr.shape, None, dtype=object)
    for thr, lab in reversed(G_EDGES):
        out = np.where(egfr >= thr, lab, out)
    out = np.where(np.isnan(egfr), None, out)
    return out


def g_stage_num(egfr):
    egfr = np.asarray(egfr, dtype=float)
    num = np.select([egfr >= 90, egfr >= 60, egfr >= 45, egfr >= 30, egfr >= 15], [1, 2, 3, 4, 5], 6).astype(float)
    num[np.isnan(egfr)] = np.nan
    return num


def a_stage_num(uacr_mg_g):
    u = np.asarray(uacr_mg_g, dtype=float)
    num = np.select([u < 30, u <= 300], [1, 2], 3).astype(float)
    num[np.isnan(u)] = np.nan
    return num


# KDIGO heat map: 0 low (green), 1 moderate (yellow), 2 high (orange), 3 very high (red)
_HEAT = np.array([
    # A1 A2 A3
    [0, 1, 2],  # G1
    [0, 1, 2],  # G2
    [1, 2, 3],  # G3a
    [2, 3, 3],  # G3b
    [3, 3, 3],  # G4
    [3, 3, 3],  # G5
])


def kdigo_heatmap(g_num, a_num):
    g = np.asarray(g_num, dtype=float)
    a = np.asarray(a_num, dtype=float)
    out = np.full(g.shape, np.nan)
    ok = ~np.isnan(g) & ~np.isnan(a)
    out[ok] = _HEAT[g[ok].astype(int) - 1, a[ok].astype(int) - 1]
    return out


# --------------------------------------------------------------------------------------
# Kidney Failure Risk Equation, 4-variable (Tangri et al.; centred form)
#   L = -0.2201*(age/10-7.036) + 0.2467*male - 0.5567*(eGFR/5-7.222) + 0.4510*(ln ACR-5.137)
#   risk_t = 1 - S0_t ^ exp(L);  S0_2y = 0.9750, S0_5y = 0.9240   (ACR in mg/g)
# Used as the *clinical-equation baseline* that the ML model must beat.
# --------------------------------------------------------------------------------------

def kfre_4var(age, male, egfr, uacr_mg_g, years: int = 2):
    age = np.asarray(age, float)
    male = np.asarray(male, float)
    egfr = np.asarray(egfr, float)
    acr = np.clip(np.asarray(uacr_mg_g, float), 1.0, None)
    lp = (-0.2201 * (age / 10 - 7.036) + 0.2467 * male - 0.5567 * (egfr / 5 - 7.222)
          + 0.4510 * (np.log(acr) - 5.137))
    s0 = {2: 0.9750, 5: 0.9240}[years]
    return 1.0 - s0 ** np.exp(lp)


# --------------------------------------------------------------------------------------
# Literature priors for *impact-weighted* prioritisation.
# These are trial/meta-analysis relative risks for kidney-disease progression, used only to
# estimate how much of a member's risk is plausibly modifiable by closing a care gap.
# They are NOT learned from our observational data and are NOT causal claims about a member.
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class InterventionPrior:
    gap_code: str
    action: str
    relative_risk: float
    evidence: str
    evidence_level: str     # high = RCT/meta-analysis, low = observational/expert


INTERVENTION_PRIORS = {
    "NO_SGLT2I": InterventionPrior(
        "NO_SGLT2I", "Discuss SGLT2 inhibitor eligibility with prescriber", 0.63,
        "SMART-C meta-analysis of 13 SGLT2i trials (Lancet 2022): kidney progression RR 0.63", "high"),
    "NO_RASI": InterventionPrior(
        "NO_RASI", "Discuss ACE inhibitor / ARB for albuminuric CKD", 0.80,
        "RENAAL / IDNT trials in diabetic kidney disease (conservative pooled prior)", "high"),
    "NO_FINERENONE": InterventionPrior(
        "NO_FINERENONE", "Review non-steroidal MRA (finerenone) eligibility", 0.82,
        "FIDELIO-DKD kidney composite HR 0.82", "high"),
    "NSAID_EXPOSURE": InterventionPrior(
        "NSAID_EXPOSURE", "Pharmacist review of chronic NSAID use", 0.90,
        "Observational nephrotoxicity evidence; expert prior", "low"),
    "NO_UACR_TEST": InterventionPrior(
        "NO_UACR_TEST", "Order urine albumin-creatinine ratio (annual for DM/CKD)", 1.00,
        "Diagnostic gap: enables staging, no direct effect assumed", "n/a"),
    "NO_EGFR_TEST": InterventionPrior(
        "NO_EGFR_TEST", "Order serum creatinine / eGFR", 1.00,
        "Diagnostic gap: enables monitoring, no direct effect assumed", "n/a"),
    "NO_NEPHROLOGY": InterventionPrior(
        "NO_NEPHROLOGY", "Nephrology referral (eGFR<30, ACR>300 or rapid decline)", 1.00,
        "Reduces unplanned (crash) dialysis start; progression effect not assumed", "low"),
    "UNDIAGNOSED_CKD": InterventionPrior(
        "UNDIAGNOSED_CKD", "Confirm and code CKD; notify PCP", 1.00,
        "Awareness gap (~87% of US adults with CKD unaware, CDC 2023)", "n/a"),
    "NO_STATIN": InterventionPrior(
        "NO_STATIN", "Statin review (age>=50, eGFR<60)", 1.00,
        "Cardiovascular benefit; no kidney-progression effect assumed", "high"),
}

CRASH_REFERRAL_RR = 0.50   # low-evidence prior: planned care pathway halves crash-start probability


def detect_care_gaps(row: dict) -> list[str]:
    """KDIGO-2024-inspired gap rules on a point-in-time feature snapshot (dict of features)."""
    g = []
    egfr = row.get("egfr_last")
    uacr = row.get("uacr_last")
    dm = bool(row.get("dx_diabetes", 0))
    hf = bool(row.get("dx_heart_failure", 0))
    htn = bool(row.get("dx_hypertension", 0))
    has_egfr = egfr is not None and not np.isnan(egfr)
    ckd = has_egfr and egfr < 60
    albuminuric = uacr is not None and not np.isnan(uacr) and uacr >= 30

    if (dm or ckd) and row.get("months_since_uacr", 99) > 12:
        g.append("NO_UACR_TEST")
    if (dm or ckd or htn) and row.get("months_since_egfr", 99) > 12:
        g.append("NO_EGFR_TEST")
    if has_egfr and egfr >= 20 and (dm and (ckd or albuminuric) or (uacr or 0) >= 200 or (hf and ckd)) \
            and not row.get("rx_sglt2i_active", 0):
        g.append("NO_SGLT2I")
    if albuminuric and (dm or htn or ckd) and not row.get("rx_rasi_active", 0):
        g.append("NO_RASI")
    if dm and albuminuric and has_egfr and egfr >= 25 and row.get("rx_rasi_active", 0) \
            and not row.get("rx_finerenone_active", 0):
        g.append("NO_FINERENONE")
    if ckd and row.get("rx_nsaid_fills_6m", 0) > 0:
        g.append("NSAID_EXPOSURE")
    if has_egfr and (egfr < 30 or (uacr or 0) > 300 or row.get("twin_p_rapid", 0) > 0.5) \
            and row.get("months_since_nephrology", 99) > 12:
        g.append("NO_NEPHROLOGY")
    if row.get("silent_ckd", 0):
        g.append("UNDIAGNOSED_CKD")
    if ckd and row.get("age", 0) >= 50 and not row.get("rx_statin_active", 0):
        g.append("NO_STATIN")
    return g


def modifiable_fraction(gaps: list[str]) -> float:
    """1 - prod(RR) over open gaps with an outcome-modifying prior."""
    rr = 1.0
    for code in gaps:
        p = INTERVENTION_PRIORS.get(code)
        if p is not None:
            rr *= p.relative_risk
    return 1.0 - rr
