"""Cohort eligibility, outcome definitions and the member-grouped + out-of-time split.

Primary target  `y_deterioration_12m` (index date T, window [T, T+12m)):
  * first dialysis claim in window, OR
  * an outpatient eGFR <= 60% of baseline (>= 40% decline, the FDA/EMA-accepted surrogate) that is
    not refuted by a later outpatient value > 70% of baseline within 90 days, OR
  * an outpatient eGFR < 15 under the same refutation rule.
  Baseline = mean of the last two outpatient eGFR values in the 12 months before T (or the last one).
  Inpatient values never define the outcome (acute AKI excursions are not chronic deterioration).

Secondary target `y_crash_24m`: first dialysis claim in [T, T+24m) delivered in an inpatient/ED setting with
  no nephrology claim in the prior 180 days (mirrors the AIR/CMS "crash dialysis" core definition).

Eligibility at T: age >= 18; continuously enrolled >= 12 months before T; alive and enrolled at T; no dialysis
  before T; >= 1 outpatient eGFR in [T-24m, T) with last value >= 15.  Members who die or disenroll before
  T+12m without an event are excluded from the 12m target (documented censoring; see model card).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

COHORT_VERSION = "cohort-ckd-payer-v1"


def eligible_members(tables: dict, T: pd.Timestamp) -> pd.Index:
    T = pd.Timestamp(T)
    m = tables["members"]
    enr = tables["enrollment"].set_index("member_id").enroll_date
    ok = m.set_index("member_id").index.to_series()
    ok = ok[(enr.reindex(ok.index) <= T - pd.DateOffset(months=12)).fillna(False)]
    le = tables["life_events"]
    gone = le[(le.event.isin(["DEATH", "DISENROLL"])) & (le.event_month < T)].member_id
    ok = ok[~ok.isin(gone)]
    cl = tables["claims"]
    dial_before = cl[(cl.claim_type.str.startswith("DIALYSIS")) & (cl.service_date < T)].member_id.unique()
    ok = ok[~ok.isin(dial_before)]
    lb = tables["labs"]
    eo = lb[(lb.test == "egfr") & (lb.setting == "outpatient") & (lb.obs_date < T)
            & (lb.obs_date >= T - pd.DateOffset(months=24))].sort_values("obs_date")
    last = eo.groupby("member_id").value.last()
    has = last[last >= 15].index
    ok = ok[ok.isin(has)]
    age = T.year - m.set_index("member_id").birth_year
    ok = ok[age.reindex(ok.index) >= 18]
    return pd.Index(ok.index, name="member_id")


def _baseline(eo: pd.DataFrame, T) -> pd.Series:
    b = eo[(eo.obs_date < T) & (eo.obs_date >= T - pd.DateOffset(months=12))].sort_values("obs_date")
    b = b.groupby("member_id").value.apply(lambda v: v.iloc[-2:].mean())
    b2 = eo[eo.obs_date < T].sort_values("obs_date").groupby("member_id").value.last()
    return b.reindex(b2.index).fillna(b2)


def labels_at(tables: dict, T: pd.Timestamp, members: pd.Index, horizon=12, crash_horizon=24,
              data_end: pd.Timestamp = pd.Timestamp("2026-01-01")) -> pd.DataFrame:
    T = pd.Timestamp(T)
    end = T + pd.DateOffset(months=horizon)
    lb = tables["labs"]
    eo = lb[(lb.test == "egfr") & (lb.setting == "outpatient") & lb.member_id.isin(members)]
    base = _baseline(eo, T).reindex(members)

    win = eo[(eo.obs_date >= T) & (eo.obs_date < end + pd.Timedelta(days=90))].sort_values(["member_id", "obs_date"])
    win = win.assign(base=win.member_id.map(base))
    win["low"] = (win.value <= 0.6 * win.base) | (win.value < 15)
    win["recovered"] = win.value > 0.7 * win.base
    lab_event = pd.Series(False, index=members)
    first_low_date = {}
    for mid, g in win[win.member_id.isin(win[win.low].member_id.unique())].groupby("member_id"):
        lows = g[g.low & (g.obs_date < end)]
        for _, r in lows.iterrows():
            later = g[(g.obs_date > r.obs_date) & (g.obs_date <= r.obs_date + pd.Timedelta(days=90))]
            if not later.recovered.any():
                lab_event[mid] = True
                first_low_date[mid] = r.obs_date
                break

    cl = tables["claims"]
    cl = cl[cl.member_id.isin(members)]
    dial = cl[cl.claim_type.str.startswith("DIALYSIS")].sort_values("service_date").groupby("member_id").first()
    dial_date = dial.service_date.reindex(members)
    dial_event = (dial_date >= T) & (dial_date < end)

    neph = cl[cl.claim_type == "NEPHRO"][["member_id", "service_date"]]
    crash_flag = pd.Series(False, index=members)
    for mid, r in dial.iterrows():
        if mid not in crash_flag.index or r.place_of_service != "21":
            continue
        prior = neph[(neph.member_id == mid) & (neph.service_date < r.service_date)
                     & (neph.service_date >= r.service_date - pd.Timedelta(days=180))]
        crash_flag[mid] = len(prior) == 0
    crash_end = T + pd.DateOffset(months=crash_horizon)
    crash_event = crash_flag & (dial_date >= T) & (dial_date < crash_end)

    le = tables["life_events"]
    exits = le[le.event.isin(["DEATH", "DISENROLL"])].groupby("member_id").event_month.min().reindex(members)
    y = (lab_event | dial_event.fillna(False)).astype(int)
    observable = exits.isna() | (exits >= end) | (y == 1)
    crash_observable = (crash_end <= data_end) & (exits.isna() | (exits >= crash_end) | crash_event)

    out = pd.DataFrame({"member_id": members, "index_date": T, "baseline_egfr": base.values,
                        "y_deterioration_12m": np.where(observable, y, np.nan),
                        "y_crash_24m": np.where(crash_observable, crash_event.astype(int), np.nan),
                        "y_dialysis_12m": dial_event.fillna(False).astype(int).values})
    return out.reset_index(drop=True)


def oracle_labels(oracle_truth: pd.DataFrame, rows: pd.DataFrame, horizon=12) -> pd.Series:
    """True (latent) 40% decline/ESKD — available only in simulation, used to measure the ascertainment gap."""
    tr = oracle_truth.set_index(["member_id", "month"])
    out = []
    for r in rows[["member_id", "index_date"]].itertuples(index=False):
        t0 = pd.Timestamp(r.index_date) - pd.DateOffset(months=1)
        t1 = pd.Timestamp(r.index_date) + pd.DateOffset(months=horizon - 1)
        try:
            a = tr.loc[(r.member_id, t0)]
            b = tr.loc[(r.member_id, t1)]
            out.append(int(b.on_dialysis or b.egfr_true <= 0.6 * a.egfr_true or b.egfr_true < 15))
        except KeyError:
            out.append(np.nan)
    return pd.Series(out, index=rows.index)


def assign_member_split(member_ids: pd.Series, seed: int, train=0.70, valid=0.15) -> pd.Series:
    """Deterministic hash-based member-level split: stable across reruns and data refreshes."""
    def bucket(mid):
        h = int(hashlib.sha256(f"{seed}:{mid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return "train" if h < train else ("valid" if h < train + valid else "test")
    return member_ids.map(bucket)
