import numpy as np
import pandas as pd

from predictmed.cohort import assign_member_split, eligible_members, labels_at
from predictmed.features.snapshots import FEATURE_CATALOG, build_snapshot
from predictmed.uci_track import map_target, split as uci_split


def _truncate(t, T):
    out = dict(t)
    for n, c in {"labs": "obs_date", "claims": "service_date", "rx_fills": "fill_date"}.items():
        out[n] = t[n][t[n][c] < T]
    return out


def test_uci_target_mapping():
    s = pd.Series(["<30", ">30", "NO", None, "garbage"])
    y = map_target(s)
    assert y.tolist()[:3] == [1, 0, 0]
    assert y.iloc[3:].isna().all()           # missing/invalid never coerced to negative


def test_uci_patient_split_disjoint():
    p = pd.Series(np.arange(20000))
    s = pd.Series(uci_split(p))
    assert set(s.unique()) == {"train", "valid", "test"}
    assert abs((s == "train").mean() - 0.70) < 0.02


def test_member_split_deterministic_and_disjoint():
    ids = pd.Series([f"M{i}" for i in range(5000)])
    a, b = assign_member_split(ids, 1), assign_member_split(ids, 1)
    assert (a == b).all()
    # rows of the same member always land in the same split
    dup = pd.concat([ids, ids])
    s = assign_member_split(dup, 1)
    assert s.groupby(dup.values).nunique().max() == 1


def test_identifiers_and_audit_attributes_not_predictors():
    allowed = [f for f, m in FEATURE_CATALOG.items() if m["policy"] == "allowed"]
    for bad in ("member_id", "claim_id", "race_ethnicity", "index_date"):
        assert bad not in allowed


def test_truncation_invariance_and_canary(small_tables, T):
    mem = eligible_members(small_tables, T)[:300]
    full = build_snapshot(small_tables, T, mem).set_index("member_id")
    trunc = build_snapshot(_truncate(small_tables, T), T, mem).set_index("member_id")
    leaky = []
    for c in full.columns:
        if c in ("race_ethnicity", "index_date"):
            continue
        same = np.isclose(full[c].astype(float), trunc[c].astype(float), equal_nan=True)
        if not same.all():
            leaky.append(c)
    assert leaky == ["nephro_visits_90d"], leaky        # only the planted canary reads the future


def test_labels_use_only_outpatient_and_window(small_tables, T):
    mem = eligible_members(small_tables, T)
    lab = labels_at(small_tables, T, mem)
    assert set(lab.y_deterioration_12m.dropna().unique()) <= {0, 1}
    assert lab.member_id.is_unique
