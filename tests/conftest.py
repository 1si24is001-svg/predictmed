import os
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from predictmed.simulator import simulate  # noqa: E402

BUNDLE_DIR = os.path.join(ROOT, "artifacts", "predictmed-ckd")
needs_bundle = pytest.mark.skipif(not os.path.exists(os.path.join(BUNDLE_DIR, "bundle.joblib")),
                                  reason="run `make pipeline` first to produce the model bundle")


@pytest.fixture(scope="session")
def small_tables():
    return simulate(1500, 72, seed=11)


@pytest.fixture(scope="session")
def T():
    return pd.Timestamp("2023-01-01")
