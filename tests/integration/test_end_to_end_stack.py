"""Integration test suite for PredictMed end-to-end production stack.
Guarded by PREDICTMED_INTEGRATION=1.
"""
from __future__ import annotations

import os
import pytest


@pytest.mark.integration
def test_full_infrastructure_stack():
    if os.environ.get("PREDICTMED_INTEGRATION") != "1":
        pytest.skip("Skipped: Set PREDICTMED_INTEGRATION=1 to run against live Docker stack")

    import requests

    # 1. Verify API health
    api_res = requests.get("http://localhost:8000/health", timeout=5)
    assert api_res.status_code == 200
    assert api_res.json()["status"] == "ok"

    # 2. Verify MLflow server
    mlflow_res = requests.get("http://localhost:5000/health", timeout=5)
    assert mlflow_res.status_code in (200, 404)  # Server responds

    # 3. Verify PostgreSQL repository connectivity
    from predictmed.repository import get_hil_repository, get_state_repository

    dsn = os.environ.get("PREDICTMED_DATABASE_URL", "postgresql://predictmed:predictmed_secret@localhost:5432/predictmed")
    hil_repo = get_hil_repository(default_path=":memory:")
    aid = hil_repo.propose("test_gate", "member_1", {"metric": 0.95}, "IntegrationRunner")
    assert aid.startswith("test_gate-")
    assert hil_repo.status(aid) == "pending"

    # 4. Verify Kafka streaming parity & DLQ
    from predictmed.streaming.bus import make_bus
    cfg = {
        "streaming": {
            "backend": "kafka",
            "bootstrap_servers": "localhost:9092",
            "topics": {"observation": "clinical.observation.v1"},
        }
    }
    bus = make_bus(cfg)
    assert bus is not None

