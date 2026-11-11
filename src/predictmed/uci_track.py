"""Real-data track: UCI Diabetes 130-US Hospitals (1999-2008) — 30-day readmission, with a diabetic
kidney disease (DKD) lens.  Implements the RiskLens-HIL protocol: discharge-time contract, documented
exclusions, patient-grouped 70/15/15 split, development-only tuning/calibration, one-time test report.

This is the honest benchmark: real, messy, historical encounter data where readmission is notoriously
hard to predict (published AUROCs are typically ~0.65-0.70).  It is a separate study from the synthetic
payer CKD model; the two are never concatenated."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .models.metrics import core_metrics, grouped_bootstrap_ci, subgroup_table, threshold_metrics
from .models.uncertainty import choose_calibrator

EXCLUDE_DISPOSITION = {11: "expired", 19: "expired at home (hospice)", 20: "expired in facility (hospice)",
                       21: "expired, place unknown (hospice)", 13: "hospice/home", 14: "hospice/medical facility"}
MEDS = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
        'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
        'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']


def map_target(s: pd.Series) -> pd.Series:
    m = {"<30": 1, ">30": 0, "NO": 0}
    return s.map(m)          # anything else -> NaN (excluded, never coerced to negative)


def icd9_group(code) -> str:
    if pd.isna(code):
        return "missing"
    c = str(code)
    if c.startswith(("V", "E")):
        return "other"
    try:
        x = float(c)
    except ValueError:
        return "other"
    if 390 <= x <= 459 or x == 785: return "circulatory"
    if 460 <= x <= 519 or x == 786: return "respiratory"
    if 520 <= x <= 579 or x == 787: return "digestive"
    if int(x) == 250: return "diabetes"
    if 800 <= x <= 999: return "injury"
    if 710 <= x <= 739: return "musculoskeletal"
    if 580 <= x <= 629 or x == 788: return "genitourinary"
    if 140 <= x <= 239: return "neoplasms"
    return "other"


def kidney_features(d: pd.DataFrame) -> pd.DataFrame:
    codes = d[["diag_1", "diag_2", "diag_3"]].astype(str)
    any_ = lambda pat: codes.apply(lambda c: c.str.match(pat)).any(axis=1).astype(int)
    out = pd.DataFrame(index=d.index)
    out["dx_ckd"] = any_(r"^585")
    out["dx_hypertensive_ckd"] = any_(r"^40[34]")
    out["dx_diabetic_nephropathy"] = any_(r"^250\.4")
    out["dx_aki"] = any_(r"^584")
    out["dx_renal_any"] = any_(r"^58[0-9]")
    stage = codes.apply(lambda c: c.str.extract(r"^585\.([1-6])")[0].astype(float)).max(axis=1)
    out["ckd_stage_icd"] = stage
    out["ckd_esrd"] = (stage == 6).astype(int) | any_(r"^V45\.1|^V56")
    out["dkd_subgroup"] = ((out.dx_ckd + out.dx_hypertensive_ckd + out.dx_diabetic_nephropathy) > 0).astype(int)
    return out


def load(path: str) -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(path, na_values="?", low_memory=False)
    audit = {"raw_rows": len(d), "raw_patients": int(d.patient_nbr.nunique()),
             "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    d["y"] = map_target(d.readmitted)
    audit["invalid_labels_excluded"] = int(d.y.isna().sum())
    d = d[d.y.notna()]
    ex = d.discharge_disposition_id.isin(EXCLUDE_DISPOSITION)
    audit["excluded_death_or_hospice"] = int(ex.sum())
    d = d[~ex].copy()
    audit["eligible_rows"] = len(d)
    audit["eligible_patients"] = int(d.patient_nbr.nunique())
    audit["prevalence"] = float(d.y.mean())
    return d, audit


def featurize(d: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    X = pd.DataFrame(index=d.index)
    num = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
           "number_emergency", "number_inpatient", "number_diagnoses"]
    for c in num:
        X[c] = d[c]
    X["prior_visits_total"] = d.number_outpatient + d.number_emergency + d.number_inpatient
    X["age_mid"] = d.age.str.extract(r"\[(\d+)-")[0].astype(float) + 5
    X["n_med_changes"] = (d[MEDS].isin(["Up", "Down"])).sum(axis=1)
    X["n_meds_active"] = (d[MEDS] != "No").sum(axis=1)
    X = X.join(kidney_features(d).drop(columns=["dkd_subgroup"]))
    cat = {"race": d.race, "gender": d.gender, "admission_type_id": d.admission_type_id.astype(str),
           "discharge_disposition_id": d.discharge_disposition_id.astype(str),
           "admission_source_id": d.admission_source_id.astype(str), "medical_specialty": d.medical_specialty,
           "payer_code": d.payer_code, "max_glu_serum": d.max_glu_serum, "A1Cresult": d.A1Cresult,
           "insulin": d.insulin, "change": d.change, "diabetesMed": d.diabetesMed,
           "diag1_group": d.diag_1.map(icd9_group), "diag2_group": d.diag_2.map(icd9_group),
           "diag3_group": d.diag_3.map(icd9_group)}
    for k, v in cat.items():
        X[k] = v.fillna("missing").astype(str)
    # collapse rare specialties (fit-free rule: frequency in the full file is not label information)
    vc = X.medical_specialty.value_counts()
    X.loc[X.medical_specialty.isin(vc[vc < 100].index), "medical_specialty"] = "rare"
    return X, num + ["prior_visits_total", "age_mid", "n_med_changes", "n_meds_active", "dx_ckd",
                     "dx_hypertensive_ckd", "dx_diabetic_nephropathy", "dx_aki", "dx_renal_any", "ckd_stage_icd",
                     "ckd_esrd"], list(cat.keys())


def split(patients: pd.Series, seed=20261029):
    h = patients.map(lambda p: int(hashlib.sha256(f"{seed}:{p}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
    return np.where(h < 0.70, "train", np.where(h < 0.85, "valid", "test"))


def run(data_path: str, out_dir: str, log=print, tracker=None) -> dict:
    d, audit = load(data_path)
    X, num, cat = featurize(d)
    X_audit_race = X["race"].copy()
    predictors = [c for c in X.columns if c != "race"]           # race kept for audit only
    cat_p = [c for c in cat if c != "race"]
    d["split"] = split(d.patient_nbr)
    for s in ("train", "valid", "test"):
        audit[f"{s}_rows"] = int((d.split == s).sum())
    assert not (set(d[d.split == "train"].patient_nbr) & set(d[d.split == "test"].patient_nbr))
    tr, va, te = (d.split == "train"), (d.split == "valid"), (d.split == "test")
    y = d.y.astype(int)

    Xl = X[predictors].copy()
    for c in cat_p:
        Xl[c] = Xl[c].astype("category")
    pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
                              [c for c in predictors if c not in cat_p]),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat_p)])
    models = {
        "prevalence": (DummyClassifier(strategy="prior"), X[predictors]),
        "logistic": (make_pipeline(pre, LogisticRegression(C=0.05, max_iter=3000)), X[predictors]),
        "random_forest": (make_pipeline(pre, RandomForestClassifier(n_estimators=300, min_samples_leaf=30, n_jobs=2,
                                                                    random_state=0)), X[predictors]),
        "lightgbm": (LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=100,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                                    verbose=-1, random_state=0, n_jobs=2), Xl),
        "catboost": (CatBoostClassifier(iterations=800, depth=6, learning_rate=0.04, verbose=0, random_seed=0,
                                        cat_features=cat_p, thread_count=2), X[predictors]),
    }
    res, preds = {}, {}
    for name, (m, XX) in models.items():
        m.fit(XX[tr], y[tr])
        pv = m.predict_proba(XX[va])[:, 1]
        res[name] = core_metrics(y[va], pv)
        preds[name] = (m, XX)
        log(f"  UCI {name:13s} valid AUROC {res[name]['auroc']:.4f} PR-AUC {res[name]['auprc']:.4f}")
        if tracker:
            with tracker.run(f"uci:{name}", params={"model": name, "track": "uci_readmission"}, tags={"split": "valid"}):
                tracker.metrics(res[name], "valid_")
    champ = max([k for k in res if k != "prevalence"], key=lambda k: res[k]["auprc"])
    m, XX = preds[champ]
    raw_va = m.predict_proba(XX[va])[:, 1]
    half = (d[va].patient_nbr.map(lambda p: int(hashlib.md5(str(p).encode()).hexdigest(), 16) % 2) == 0).to_numpy()
    cal, cal_res = choose_calibrator(raw_va[half], y[va].to_numpy()[half], raw_va[~half], y[va].to_numpy()[~half])
    thr = float(np.quantile(cal.transform(raw_va), 0.80))      # top-20% capacity threshold from validation
    # ---- one-time locked test evaluation
    pt = cal.transform(m.predict_proba(XX[te])[:, 1])
    test = d[te].assign(p=pt, race_audit=X_audit_race[te])
    kf = kidney_features(d[te])
    test["dkd_subgroup"] = kf.dkd_subgroup.map({1: "DKD-coded", 0: "no kidney code"})
    test["age_band"] = pd.cut(X.loc[te, "age_mid"], [0, 50, 65, 80, 200], labels=["<50", "50-64", "65-79", "80+"], right=False)
    rep = {"audit": audit, "exclusions": EXCLUDE_DISPOSITION, "valid_results": res, "champion": champ,
           "calibration": {"chosen": cal.method, "brier_valid_half": cal_res},
           "test": core_metrics(test.y, test.p),
           "test_ci95": grouped_bootstrap_ci(test.y.to_numpy(), test.p.to_numpy(), test.patient_nbr.to_numpy(), 300),
           "test_at_top20_threshold": threshold_metrics(test.y, test.p, thr)}
    dkd = test[test.dkd_subgroup == "DKD-coded"]
    rep["test_dkd_subgroup"] = core_metrics(dkd.y, dkd.p)
    # kidney-feature ablation on the champion family (dev-only refit, evaluated on valid)
    kid = ["dx_ckd", "dx_hypertensive_ckd", "dx_diabetic_nephropathy", "dx_aki", "dx_renal_any", "ckd_stage_icd", "ckd_esrd"]
    if True:   # family-agnostic ablation with LightGBM (fast); validation only
        lg = models["lightgbm"][0]
        p_with = lg.predict_proba(Xl[va])[:, 1]
        abl = LGBMClassifier(**lg.get_params()).fit(Xl[tr].drop(columns=kid), y[tr])
        pa = abl.predict_proba(Xl[va].drop(columns=kid))[:, 1]
        raw_va_l = p_with
        vdkd = kidney_features(d[va]).dkd_subgroup.to_numpy() == 1
        rep["ablation_kidney_features_valid"] = {
            "model": "lightgbm", "with": core_metrics(y[va], p_with), "without": core_metrics(y[va], pa),
            "dkd_with": core_metrics(y[va][vdkd], raw_va_l[vdkd]), "dkd_without": core_metrics(y[va][vdkd], pa[vdkd])}
    sg = subgroup_table(test, "y", "p", ["race_audit", "gender", "age_band", "dkd_subgroup"], thr, 20)
    os.makedirs(out_dir, exist_ok=True)
    sg.to_csv(os.path.join(out_dir, "uci_subgroups_test.csv"), index=False)
    if champ in ("lightgbm", "catboost"):
        imp = (pd.Series(m.booster_.feature_importance("gain"), index=Xl.columns) if champ == "lightgbm"
               else pd.Series(m.get_feature_importance(), index=predictors)).sort_values(ascending=False)
        rep["top_features_gain"] = imp.head(15).round(1).to_dict()
    json.dump(rep, open(os.path.join(out_dir, "uci_report.json"), "w"), indent=2, default=float)
    log(f"  UCI champion {champ}: test AUROC {rep['test']['auroc']:.3f} PR-AUC {rep['test']['auprc']:.3f} "
        f"(prevalence {rep['test']['prevalence']:.3f}); DKD subgroup AUROC {rep['test_dkd_subgroup']['auroc']:.3f}")
    return rep
