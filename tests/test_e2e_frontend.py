"""End-to-End API and Frontend Contract Test.
Verifies that FastAPI serves the React application and all REST contracts.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_e2e_frontend_serves_html():
    from predictmed.serving.api import app

    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "PredictMed" in res.text or "<div id=\"root\">" in res.text


def test_e2e_overview_and_member_endpoints():
    from predictmed.serving.api import app

    client = TestClient(app)

    # Overview contract
    ov = client.get("/v1/overview")
    assert ov.status_code == 200
    ov_data = ov.json()
    assert "total_members" in ov_data
    assert "kdigo_stages" in ov_data
    assert "disclaimer" in ov_data

    # Worklist contract
    wl = client.get("/v1/worklist?capacity=10&strategy=eac")
    assert wl.status_code == 200
    wl_data = wl.json()
    assert len(wl_data["members"]) <= 10

    # Member 360 twin contract
    sample_id = wl_data["members"][0]["member_id"]
    tw = client.get(f"/v1/members/{sample_id}/twin")
    assert tw.status_code == 200
    tw_data = tw.json()
    assert "natural_trajectory" in tw_data
    assert "counterfactual_trajectory" in tw_data
    assert tw_data["member_id"] == sample_id

