import os

import pandas as pd
import pytest

from predictmed.ckd_baseline import CSV, understand_and_clean


@pytest.mark.skipif(not os.path.exists(CSV), reason="UCI CKD csv not downloaded")
def test_cleaning_fixes_dirty_labels_and_types():
    raw = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    df, audit = understand_and_clean(raw)
    assert set(df.y.unique()) == {0, 1} and len(df) == 400        # 'ckd\t' repaired, nothing silently dropped
    assert audit["prevalence"] == pytest.approx(250 / 400)
    assert df["htn"].dropna().isin([0.0, 1.0]).all()
    assert pd.api.types.is_float_dtype(df["sc"])


def test_cleaning_on_synthetic_dirty_rows():
    raw = pd.DataFrame({c: ["?"] * 3 for c in ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc",
                                                "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "htn", "dm", "cad", "appet",
                                                "pe", "ane"]})
    raw["class"] = ["ckd\t", "notckd", "weird"]
    raw["dm"] = ["\tyes", " no", "?"]
    raw["sod"] = ["4.5", "140", "?"]                                 # 4.5 mEq/L sodium is implausible -> missing
    df, audit = understand_and_clean(raw)
    assert df.y.tolist() == [1, 0] and audit["rows_dropped_invalid_label"] == 1
    assert df.dm.tolist() == [1.0, 0.0] and pd.isna(df.sod.iloc[0])
