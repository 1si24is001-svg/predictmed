"""Tests for Phase 6 production hardening:
- JWT authentication and role enforcement
- Population Stability Index (PSI) drift monitoring
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_jwt_create_and_verify():
    from predictmed.auth import create_access_token, verify_token

    token = create_access_token(subject="user_123", role="care_manager", expires_in_sec=3600)
    assert isinstance(token, str)

    payload = verify_token(token)
    assert payload["sub"] == "user_123"
    assert payload["role"] == "care_manager"


def test_jwt_tampered_token_rejected():
    from predictmed.auth import create_access_token, verify_token
    from fastapi import HTTPException

    token = create_access_token(subject="user_123", role="viewer")
    parts = token.split(".")
    # Tamper payload
    tampered = f"{parts[0]}.eyJhZG1pbiI6dHJ1ZX0.{parts[2]}"

    with pytest.raises(HTTPException) as exc:
        verify_token(tampered)
    assert exc.value.status_code == 401


def test_psi_identical_distributions():
    from predictmed.monitoring import compute_psi

    rng = np.random.default_rng(42)
    sample = rng.normal(50, 10, 500)
    psi = compute_psi(sample, sample)
    assert psi == 0.0 or psi < 0.01


def test_psi_shifted_distribution():
    from predictmed.monitoring import compute_dataset_drift, compute_psi

    rng = np.random.default_rng(42)
    ref = rng.normal(50, 10, 1000)
    shifted = rng.normal(70, 10, 1000)

    psi = compute_psi(ref, shifted)
    assert psi >= 0.25  # Significant drift detected

    ref_df = pd.DataFrame({"egfr": ref, "age": rng.uniform(30, 80, 1000)})
    cur_df = pd.DataFrame({"egfr": shifted, "age": rng.uniform(30, 80, 1000)})

    drift_report = compute_dataset_drift(ref_df, cur_df, ["egfr", "age"])
    egfr_row = drift_report[drift_report.feature == "egfr"].iloc[0]
    assert egfr_row.status == "SIGNIFICANT_DRIFT"
    age_row = drift_report[drift_report.feature == "age"].iloc[0]
    assert age_row.status == "STABLE"

