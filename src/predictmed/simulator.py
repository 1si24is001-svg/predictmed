"""Literature-calibrated synthetic health-plan cohort for CKD deterioration research.

Why a simulator: real payer data with longitudinal labs + claims + pharmacy is not public.
This generator produces data with the *pathologies* a real payer feed has, because those
pathologies are exactly what the system must be robust to:

* sparse, irregular, informative lab sampling (sicker / engaged members are tested more);
* analytic + biological creatinine noise and acute inpatient lab excursions (AKI);
* under-coding of CKD on claims (the "silent CKD" awareness gap);
* treatment under-use that varies with social deprivation (equity signal);
* a hidden "rapid progressor" phenotype the models must discover without being told;
* competing risk of death, plan disenrollment (censoring), crash vs planned dialysis.

Ground truth (true eGFR per member-month) is written to a separate *oracle* table that no
feature pipeline is allowed to read. It is used only to validate the digital twin and to
quantify how much true deterioration goes unrecorded (ascertainment gap).

Clinical anchors: CKD-EPI 2021 equation for creatinine<->eGFR; KDIGO G/A staging;
SGLT2i effect ~ RR 0.63 on progression (SMART-C); RASi effect in albuminuric CKD;
KDIGO "rapid progression" = sustained decline > 5 mL/min/1.73m2/year.
All rates are plausible research settings, not estimates of any real population.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .clinical import ckd_epi_2021, inverse_ckd_epi_2021

LOINC = {
    "egfr": "98979-8",        # eGFR CKD-EPI 2021 (creatinine)
    "creatinine": "2160-0",
    "uacr": "9318-7",
    "hba1c": "4548-4",
    "potassium": "2823-3",
    "hemoglobin": "718-7",
    "bicarbonate": "1963-8",
    "sbp": "8480-6",
}
UNITS = {"egfr": "mL/min/1.73m2", "creatinine": "mg/dL", "uacr": "mg/g", "hba1c": "%",
         "potassium": "mmol/L", "hemoglobin": "g/dL", "bicarbonate": "mmol/L", "sbp": "mm[Hg]"}

CLAIM_COST = {"PCP": 160, "NEPHRO": 260, "ED": 1900, "IP": 14500, "IP_AKI": 18500,
              "DIALYSIS": 8200, "DIALYSIS_CRASH_START": 52000}
CLAIM_SLOT = {"PCP": 0, "NEPHRO": 0, "ED": 4, "IP": 5, "IP_AKI": 5, "DIALYSIS": 6, "DIALYSIS_CRASH_START": 5}
RX_COST = {"SGLT2I": 520, "RASI": 25, "FINERENONE": 610, "STATIN": 15, "NSAID": 12, "DIURETIC": 10,
           "METFORMIN": 10, "INSULIN": 380}


def _logit(x):
    return 1 / (1 + np.exp(-x))


def simulate(n_members: int = 30000, months: int = 84, start: str = "2019-01-01",
             rapid_share: float = 0.08, seed: int = 7, intervention: dict | None = None) -> dict[str, pd.DataFrame]:
    """`intervention` = {"month": m, "actions": {member_index: [gap_code, ...]}} applies outreach-driven care
    changes at month m.  Two RNG streams with *fixed-size* draws per month (dynamics: `rng`, observation noise:
    `rng_obs`) give common random numbers: a counterfactual run differs from the factual run only through the
    intervention's causal effects — an in-silico randomized outreach trial."""
    rng = np.random.default_rng(seed)
    rng_obs = np.random.default_rng(seed + 1)
    N, T = n_members, months
    start_d = np.datetime64(start, "D")
    month_start = np.array([np.datetime64(pd.Timestamp(start) + pd.DateOffset(months=m), "D")
                            for m in range(T + 1)])

    # ---------------------------------------------------------------- static population
    member_id = np.array([f"M{100000 + i}" for i in range(N)])
    age0 = np.clip(rng.normal(61, 13, N), 25, 92)
    female = rng.random(N) < 0.52
    race = rng.choice(["White", "Black", "Hispanic", "Asian", "Other"], N, p=[0.60, 0.15, 0.17, 0.06, 0.02])
    rural = rng.random(N) < 0.2
    medicare = (age0 >= 65) & (rng.random(N) < 0.88)
    medicaid = (~medicare) & (rng.random(N) < 0.30)
    plan = np.where(medicare, "MedicareAdvantage", np.where(medicaid, "Medicaid", "Commercial"))
    # area-deprivation style SDOH index in [0,1]; structurally higher for some groups
    sdoh = np.clip(rng.beta(2, 3.2, N) + 0.18 * medicaid + 0.10 * np.isin(race, ["Black", "Hispanic"])
                   + 0.05 * rural, 0, 1)

    dm = rng.random(N) < _logit(-1.0 + 0.025 * (age0 - 60) + 1.0 * sdoh + 0.3 * np.isin(race, ["Black", "Hispanic"]))
    htn = rng.random(N) < _logit(0.2 + 0.04 * (age0 - 60) + 0.9 * dm + 0.6 * sdoh)
    hf = rng.random(N) < _logit(-3.3 + 0.05 * (age0 - 60) + 0.6 * dm + 0.8 * htn)
    cvd = rng.random(N) < _logit(-2.2 + 0.04 * (age0 - 60) + 0.5 * dm + 0.5 * htn)
    bp_uncontrolled = htn & (rng.random(N) < np.clip(0.30 + 0.35 * sdoh, 0, 0.9))
    arthritis = rng.random(N) < _logit(-1.4 + 0.03 * (age0 - 60))

    # baseline kidney function (true); prior-injury component creates realistic G3/G4 tail
    egfr0 = rng.normal(104 - 0.85 * (age0 - 30) - 7 * dm - 5 * htn - 8 * hf, 14)
    injured = rng.random(N) < (0.22 + 0.12 * dm + 0.08 * hf)
    egfr0 = egfr0 - injured * rng.gamma(2.0, 9.0, N)
    egfr0 = np.clip(egfr0, 16, 125)

    log_uacr = rng.normal(np.log(10) + 1.2 * dm + 0.5 * htn + 0.9 * (egfr0 < 60) + 0.5 * sdoh
                          + 0.6 * bp_uncontrolled, 1.05)

    rapid = rng.random(N) < np.clip(rapid_share * (0.55 + 0.9 * dm + 0.6 * (log_uacr > np.log(300))
                                                   + 0.3 * bp_uncontrolled), 0, 0.6)
    # individual structural slope (mL/min/1.73m2 per year), untreated
    slope_base = (-0.7 - 1.1 * dm - 0.9 * bp_uncontrolled - 0.4 * hf
                  - 1.3 * np.clip(log_uacr / np.log(10) - 1.5, 0, None)
                  + rng.normal(0, 0.9, N))
    slope_base = slope_base - rapid * rng.uniform(3.0, 7.5, N)
    eskd_threshold = rng.uniform(7.5, 12.5, N)
    # regime switching: members can *enter* an accelerated phase during follow-up (unpredictable onset)
    accel_extra = rng.uniform(2.5, 6.5, N)
    accelerated = np.zeros(N, bool)

    # ---------------------------------------------------------------- dynamic state
    enroll_start = np.where(rng.random(N) < 0.8, 0, rng.integers(1, 36, N))
    enrolled = np.zeros(N, bool)
    disenrolled = np.zeros(N, bool)
    alive = np.ones(N, bool)
    on_dialysis = np.zeros(N, bool)
    egfr_true = egfr0.copy()
    lu_true = log_uacr.copy()

    sglt2 = np.zeros(N, bool)
    rasi = (htn | dm) & (rng.random(N) < np.where(log_uacr > np.log(30), 0.55, 0.35))
    statin = (age0 >= 50) & (rng.random(N) < 0.5)
    finer = np.zeros(N, bool)
    nsaid_chronic = rng.random(N) < (0.06 + 0.12 * arthritis)
    diuretic = (hf | bp_uncontrolled) & (rng.random(N) < 0.5)
    ckd_coded = np.zeros(N, bool)
    nephro_engaged = np.zeros(N, bool)
    last_nephro_month = np.full(N, -999)
    n_low_obs = np.zeros(N, int)
    first_low_obs_month = np.full(N, -999)
    lab_vendor_bias = rng.choice([0.95, 1.0, 1.0, 1.04, 1.07], N)   # inter-laboratory calibration differences
    engagement = np.clip(rng.normal(1.0, 0.25, N) * (1 - 0.35 * sdoh), 0.3, 1.8)

    labs, claims, rx, truth, enroll_rows, events = [], [], [], [], [], []
    claim_seq = 0

    OBS = {}   # per-month fixed-size observation noise (filled at the top of each month)

    def day_in_month(m, idx, slot=0):
        return month_start[m] + OBS["day"][slot, idx].astype("timedelta64[D]")

    def add_claims(idx, m, ctype, dx_list, pos):
        nonlocal claim_seq
        if len(idx) == 0:
            return
        slot = CLAIM_SLOT[ctype]
        d = day_in_month(m, idx, slot)
        ids = np.arange(claim_seq, claim_seq + len(idx))
        claim_seq += len(idx)
        claims.append(pd.DataFrame({
            "claim_id": [f"C{x:09d}" for x in ids], "member_id": member_id[idx], "service_date": d,
            "claim_type": ctype, "dx_codes": dx_list, "place_of_service": pos,
            "allowed_amt": np.round(CLAIM_COST[ctype] * np.exp(0.25 * OBS["g"][7 + slot, idx]), 2)}))

    def dx_for(idx, extra=None):
        out = []
        for i in idx:
            c = []
            if dm[i]:
                c.append("E11.22" if (ckd_coded[i] and OBS["u"][4, i] < 0.6) else "E11.9")
            if htn[i]:
                c.append("I12.9" if ckd_coded[i] else "I10")
            if hf[i]:
                c.append("I50.9")
            if ckd_coded[i]:
                st = 6 if on_dialysis[i] else int(np.digitize(-egfr_true[i], [-90, -60, -45, -30, -15]) + 1)
                c.append({1: "N18.1", 2: "N18.2", 3: "N18.31", 4: "N18.32", 5: "N18.4", 6: "N18.6"}.get(st, "N18.5")
                         if st != 6 else "N18.6")
            if on_dialysis[i]:
                c.append("Z99.2")
            if extra:
                c.insert(0, extra)
            out.append("|".join(c) if c else "Z00.00")
        return out

    for m in range(T):
        OBS["day"] = rng_obs.integers(0, 28, (8, N))
        OBS["g"] = rng_obs.standard_normal((16, N))
        OBS["u"] = rng_obs.random((8, N))
        if intervention and m == intervention["month"]:
            for i, acts in intervention["actions"].items():
                if "NO_SGLT2I" in acts and egfr_true[i] >= 20:
                    sglt2[i] = True
                if "NO_RASI" in acts:
                    rasi[i] = True
                if "NO_FINERENONE" in acts:
                    finer[i] = True
                if "NSAID_EXPOSURE" in acts:
                    nsaid_chronic[i] = False
                if "NO_NEPHROLOGY" in acts:
                    nephro_engaged[i] = True
                    last_nephro_month[i] = m
                if "UNDIAGNOSED_CKD" in acts:
                    ckd_coded[i] = True
                if "NO_STATIN" in acts:
                    statin[i] = True
        cal_year = 2019 + m // 12
        age = age0 + m / 12.0
        newly = (~enrolled) & (~disenrolled) & (enroll_start == m)
        enrolled |= newly
        for i in np.where(newly)[0]:
            enroll_rows.append((member_id[i], month_start[m]))

        act = enrolled & alive & ~disenrolled
        # ---- treatment dynamics -------------------------------------------------
        albuminuric = lu_true > np.log(30)
        sglt2_elig = (dm & ((egfr_true < 60) | albuminuric)) | (lu_true > np.log(200)) | (hf & (egfr_true < 60))
        era = 1.0 if cal_year < 2020 else (2.5 if cal_year < 2022 else 4.5)
        p_sglt2 = 0.0035 * era * (1 - 0.65 * sdoh) * (1 + 1.5 * nephro_engaged) * engagement
        sglt2 |= act & sglt2_elig & (egfr_true >= 20) & (rng.random(N) < p_sglt2)
        sglt2 &= ~(rng.random(N) < 0.008)
        rasi |= act & (albuminuric | htn) & (rng.random(N) < 0.006 * (1 + nephro_engaged) * (1 - 0.4 * sdoh))
        finer |= act & dm & albuminuric & rasi & (cal_year >= 2022) & (rng.random(N) < 0.0015 * (1 + 2 * nephro_engaged))
        statin |= act & (age >= 50) & (rng.random(N) < 0.004)

        nsaid_month = act & (rng.random(N) < np.where(nsaid_chronic, 0.55, 0.015))
        # ---- AKI -----------------------------------------------------------------
        h_aki = 0.0012 * np.exp(0.9 * hf + 0.8 * nsaid_month + 0.9 * (egfr_true < 45) + 0.4 * dm
                                + 0.02 * (age - 60) + 0.3 * diuretic)
        aki = act & ~on_dialysis & (rng.random(N) < h_aki)
        h_onset = 0.0006 * (1 + 1.2 * dm + 1.0 * (lu_true > np.log(300)) + 0.8 * bp_uncontrolled) + 0.08 * aki
        accelerated |= act & ~rapid & (rng.random(N) < h_onset)
        # ---- progression ---------------------------------------------------------
        eff = slope_base - accelerated * accel_extra
        eff = np.where(sglt2, eff * 0.60, eff)
        eff = np.where(rasi & albuminuric, eff * 0.82, eff)
        eff = np.where(finer, eff * 0.85, eff)
        eff = eff - 0.6 * nsaid_chronic
        eff = eff * (1 + 0.35 * (egfr_true < 30))
        step = eff / 12 + rng.normal(0, 0.55, N)
        aki_loss = aki * egfr_true * np.where(rng.random(N) < 0.25, rng.uniform(0.15, 0.40, N), rng.uniform(0.02, 0.12, N))
        egfr_true = np.where(act & ~on_dialysis, np.clip(egfr_true + step - aki_loss, 3, 130), egfr_true)
        lu_true = lu_true + act * (0.04 / 12 + 0.12 / 12 * rapid
                                   - 0.35 / 12 * (sglt2 & (rng.random(N) < 0.5))
                                   - 0.30 / 12 * (rasi & (rng.random(N) < 0.5)) + rng.normal(0, 0.03, N))

        # ---- utilisation ---------------------------------------------------------
        sev = (egfr_true < 30).astype(float)
        p_pcp = np.clip(0.18 + 0.08 * dm + 0.05 * htn + 0.06 * hf, 0, 1) * engagement
        pcp = act & (rng.random(N) < p_pcp)
        p_ed = 0.012 * (1 + 1.2 * hf + 1.5 * sev + 0.6 * sdoh)
        ed = act & (rng.random(N) < p_ed)
        p_ip = 0.005 * (1 + 2.0 * hf + 1.8 * sev + 0.5 * dm)
        ip = act & ~aki & (rng.random(N) < p_ip)
        known_low = ckd_coded & (egfr_true < 45)
        p_neph = (0.0015 + 0.04 * known_low + 0.14 * (ckd_coded & (egfr_true < 30))
                  + 0.06 * (egfr_true < 20) + 0.08 * nephro_engaged) * (1 - 0.45 * sdoh) * engagement
        neph = act & (rng.random(N) < p_neph)
        nephro_engaged |= neph
        last_nephro_month = np.where(neph, m, last_nephro_month)

        # ---- dialysis start -----------------------------------------------------
        start_dial = act & ~on_dialysis & (egfr_true < eskd_threshold)
        planned_path = (m - last_nephro_month) <= 6
        crash = start_dial & (~planned_path | (rng.random(N) < 0.12))
        on_dialysis |= start_dial

        # ---- labs: informative, irregular sampling ------------------------------
        stage_hi = (egfr_true < 60).astype(float)
        p_bmp = (0.05 + 0.07 * dm + 0.04 * htn + 0.05 * stage_hi * ckd_coded + 0.12 * (egfr_true < 30)
                 + 0.25 * neph) * (1 - 0.35 * sdoh) * engagement
        bmp_out = act & ~on_dialysis & (rng.random(N) < np.clip(p_bmp, 0, 0.9))
        bmp_in = act & (aki | ip | (start_dial & crash))           # every admission draws labs
        for setting, mask in (("outpatient", bmp_out & ~bmp_in), ("inpatient", bmp_in)):
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            d = day_in_month(m, idx)
            g = egfr_true[idx]
            if setting == "inpatient":
                acute = np.where(aki[idx], 1.3 + 0.9 * OBS["u"][0, idx], 1.0 + 0.15 * OBS["u"][0, idx])
            else:   # transient pre-renal dips (dehydration, diuretic/NSAID effect) in ~4% of outpatient draws
                acute = np.where(OBS["u"][1, idx] < 0.04, 1.15 + 0.45 * OBS["u"][2, idx], 1.0)
            cr = inverse_ckd_epi_2021(g, age[idx], female[idx]) * acute * lab_vendor_bias[idx] * np.exp(0.075 * OBS["g"][0, idx])
            cr = np.round(cr, 2)
            eg = np.round(ckd_epi_2021(cr, age[idx], female[idx]), 1)
            labs.append(pd.DataFrame({"member_id": member_id[idx], "obs_date": d, "code": LOINC["creatinine"],
                                      "test": "creatinine", "value": cr, "unit": UNITS["creatinine"], "setting": setting}))
            labs.append(pd.DataFrame({"member_id": member_id[idx], "obs_date": d, "code": LOINC["egfr"],
                                      "test": "egfr", "value": eg, "unit": UNITS["egfr"], "setting": setting}))
            k = np.round(np.clip(4.3 + 0.025 * np.clip(45 - g, 0, None) + 0.15 * rasi[idx] + 0.4 * OBS["g"][1, idx], 3, 7), 1)
            labs.append(pd.DataFrame({"member_id": member_id[idx], "obs_date": d, "code": LOINC["potassium"],
                                      "test": "potassium", "value": k, "unit": UNITS["potassium"], "setting": setting}))
            bic = np.round(np.clip(25 - 0.12 * np.clip(45 - g, 0, None) + 1.8 * OBS["g"][2, idx], 12, 32), 0)
            labs.append(pd.DataFrame({"member_id": member_id[idx], "obs_date": d, "code": LOINC["bicarbonate"],
                                      "test": "bicarbonate", "value": bic, "unit": UNITS["bicarbonate"], "setting": setting}))
            cbc = OBS["u"][3, idx] < 0.5
            if cbc.any():
                hg = np.round(np.clip(13.8 - 0.6 * female[idx] - 0.07 * np.clip(60 - g, 0, None) + 1.1 * OBS["g"][3, idx], 6, 18), 1)
                labs.append(pd.DataFrame({"member_id": member_id[idx][cbc], "obs_date": d[cbc], "code": LOINC["hemoglobin"],
                                          "test": "hemoglobin", "value": hg[cbc], "unit": UNITS["hemoglobin"],
                                          "setting": setting}))
            # recognition / coding dynamics for observed low eGFR
            low = eg < 60
            n_low_obs[idx[low]] += 1
            first_low_obs_month[idx[low & (first_low_obs_month[idx] < 0)]] = m
        p_code = 0.035 * (1 + 3 * nephro_engaged) * (1 - 0.4 * sdoh) * (1 + 1.5 * (egfr_true < 30))
        ckd_coded |= act & (n_low_obs >= 1) & (rng.random(N) < p_code)

        p_uacr = (0.02 + 0.045 * dm + 0.03 * ckd_coded + 0.15 * neph) * (1 - 0.4 * sdoh) * engagement
        u = act & ~on_dialysis & (rng.random(N) < p_uacr)
        idx = np.where(u)[0]
        if len(idx):
            labs.append(pd.DataFrame({"member_id": member_id[idx], "obs_date": day_in_month(m, idx, 1), "code": LOINC["uacr"],
                                      "test": "uacr", "value": np.round(np.exp(lu_true[idx] + 0.35 * OBS["g"][4, idx]), 1),
                                      "unit": UNITS["uacr"], "setting": "outpatient"}))
        a1c = act & dm & (rng.random(N) < 0.14 * engagement)
        idx = np.where(a1c)[0]
        if len(idx):
            labs.append(pd.DataFrame({"member_id": member_id[idx], "obs_date": day_in_month(m, idx, 2), "code": LOINC["hba1c"],
                                      "test": "hba1c", "value": np.round(np.clip(7.3 + 0.9 * sdoh[idx] + 0.9 * OBS["g"][5, idx], 5, 13), 1),
                                      "unit": UNITS["hba1c"], "setting": "outpatient"}))
        bp = act & htn & (rng.random(N) < 0.08)
        idx = np.where(bp)[0]
        if len(idx):
            labs.append(pd.DataFrame({"member_id": member_id[idx], "obs_date": day_in_month(m, idx, 3), "code": LOINC["sbp"],
                                      "test": "sbp", "value": np.round(np.where(bp_uncontrolled[idx], 149 + 11 * OBS["g"][6, idx],
                                                                                    127 + 9 * OBS["g"][6, idx]), 0),
                                      "unit": UNITS["sbp"], "setting": "outpatient"}))

        # ---- claims -------------------------------------------------------------
        for ctype, mask, pos, extra in (("PCP", pcp, "11", None), ("ED", ed, "23", None),
                                        ("IP", ip, "21", None), ("IP_AKI", aki, "21", "N17.9"),
                                        ("NEPHRO", neph, "11", None)):
            idx = np.where(mask)[0]
            add_claims(idx, m, ctype, dx_for(idx, extra), pos)
        idx = np.where(start_dial & crash)[0]
        add_claims(idx, m, "DIALYSIS_CRASH_START", dx_for(idx, "N18.6"), "21")
        idx = np.where(act & on_dialysis)[0]
        add_claims(idx, m, "DIALYSIS", dx_for(idx, "N18.6"), "65")
        for i in np.where(start_dial)[0]:
            events.append((member_id[i], month_start[m], "DIALYSIS_START", bool(crash[i])))

        # ---- pharmacy fills -----------------------------------------------------
        for cls, mask in (("SGLT2I", sglt2), ("RASI", rasi), ("FINERENONE", finer), ("STATIN", statin),
                          ("NSAID", nsaid_month), ("DIURETIC", diuretic), ("METFORMIN", dm & (egfr_true >= 30)),
                          ("INSULIN", dm & (egfr_true < 30))):
            adherent = rng.random(N) < (0.82 - 0.25 * sdoh)
            idx = np.where(act & mask & (adherent | (cls == "NSAID")))[0]
            if len(idx):
                rx.append(pd.DataFrame({"member_id": member_id[idx], "fill_date": day_in_month(m, idx, 7),
                                        "drug_class": cls, "days_supply": 30,
                                        "cost": np.round(RX_COST[cls] * np.exp(0.1 * OBS["g"][15, idx]), 2)}))

        # ---- oracle truth -------------------------------------------------------
        idx = np.where(act)[0]
        truth.append(pd.DataFrame({"member_id": member_id[idx], "month": month_start[m], "egfr_true": np.round(egfr_true[idx], 2),
                                   "log_uacr_true": np.round(lu_true[idx], 3), "on_dialysis": on_dialysis[idx]}))

        # ---- exits: death (competing risk) and disenrollment (censoring) ---------
        h_death = 0.00025 * np.exp(0.075 * (age - 60)) * (1 + 1.3 * hf) * (1 + 1.6 * (egfr_true < 30)) * (1 + 2.5 * on_dialysis)
        die = act & (rng.random(N) < h_death)
        for i in np.where(die)[0]:
            events.append((member_id[i], month_start[m], "DEATH", False))
        alive &= ~die
        leave = act & ~die & (rng.random(N) < 0.0075)
        for i in np.where(leave)[0]:
            events.append((member_id[i], month_start[m], "DISENROLL", False))
        disenrolled |= leave

    members = pd.DataFrame({
        "member_id": member_id, "birth_year": (2019 - age0).astype(int), "age_at_2019": np.round(age0, 1),
        "female": female.astype(int), "race_ethnicity": race, "rural": rural.astype(int),
        "plan_type": plan, "sdoh_index": np.round(sdoh, 3),
        "dx_diabetes": dm.astype(int), "dx_hypertension": htn.astype(int), "dx_heart_failure": hf.astype(int),
        "dx_cvd": cvd.astype(int)})
    latent = pd.DataFrame({"member_id": member_id, "rapid_progressor": rapid.astype(int),
                           "accelerated_onset_during_followup": accelerated.astype(int),
                           "slope_base": np.round(slope_base, 3), "egfr0": np.round(egfr0, 2),
                           "bp_uncontrolled": bp_uncontrolled.astype(int), "eskd_threshold": eskd_threshold})
    enrollment = pd.DataFrame(enroll_rows, columns=["member_id", "enroll_date"])
    exits = pd.DataFrame(events, columns=["member_id", "event_month", "event", "crash"])
    out = {
        "members": members,
        "enrollment": enrollment,
        "labs": pd.concat(labs, ignore_index=True),
        "claims": pd.concat(claims, ignore_index=True),
        "rx_fills": pd.concat(rx, ignore_index=True),
        "life_events": exits,
        "oracle_truth": pd.concat(truth, ignore_index=True),
        "oracle_latent": latent,
    }
    for k in ("labs",):
        out[k]["obs_date"] = pd.to_datetime(out[k]["obs_date"])
    out["claims"]["service_date"] = pd.to_datetime(out["claims"]["service_date"])
    out["rx_fills"]["fill_date"] = pd.to_datetime(out["rx_fills"]["fill_date"])
    out["enrollment"]["enroll_date"] = pd.to_datetime(out["enrollment"]["enroll_date"])
    out["life_events"]["event_month"] = pd.to_datetime(out["life_events"]["event_month"])
    out["oracle_truth"]["month"] = pd.to_datetime(out["oracle_truth"]["month"])
    return out
