import copy
import os

import numpy as np
import pandas as pd
import pytest
from conftest import BUNDLE_DIR, needs_bundle

from predictmed.cohort import eligible_members
from predictmed.features.snapshots import build_snapshot
from predictmed.features.twin import TwinParams, twin_features_at
from predictmed.streaming.bus import InMemoryBus
from predictmed.streaming.envelope import ValidationError, make_envelope, validate
from predictmed.streaming.services import StateStore, StreamingPipeline, events_from_tables, inject_faults

TOPICS = {"snapshot": "features.snapshot.v1", "score_request": "risk.score.request.v1",
          "score_result": "risk.score.result.v1", "dlq": "data.quality.dlq.v1"}


def test_envelope_schema_rules():
    e = make_envelope("observation.recorded", "lab_feed", "S1", "2024-01-01",
                      {"test": "egfr", "code": "98979-8", "value": 45.0, "unit": "mL/min/1.73m2", "setting": "outpatient"})
    validate(e)
    for mutate, code in [(lambda x: x["payload"].update(unit="umol/L"), "UNIT_MISMATCH"),
                         (lambda x: x["payload"].update(value=-3.0), "IMPLAUSIBLE_VALUE"),
                         (lambda x: x.update(schema_version="9.9"), "SCHEMA_VERSION_MISMATCH"),
                         (lambda x: x.update(subject_key=None), "ENVELOPE_MISSING_FIELD"),
                         (lambda x: x["payload"].pop("value"), "PAYLOAD_MISSING_FIELD")]:
        b = copy.deepcopy(e)
        mutate(b)
        with pytest.raises(ValidationError) as ei:
            validate(b)
        assert ei.value.code == code


def test_event_ids_are_deterministic():
    a = make_envelope("claim.adjudicated", "claims", "S1", "2024-01-01", {"claim_id": "C1"}, natural_key=("C1",))
    b = make_envelope("claim.adjudicated", "claims", "S1", "2024-01-01", {"claim_id": "C1"}, natural_key=("C1",))
    assert a["event_id"] == b["event_id"]


def test_inmemory_bus_redelivers_uncommitted():
    bus = InMemoryBus(2)
    for i in range(5):
        bus.produce("t", {"i": i}, key=f"k{i}")
    first = bus.poll("t", "g")
    assert len(first) == 5
    p, off, _, _ = first[0]
    bus.commit("t", "g", p, off)
    assert len(bus.poll("t", "g")) == 4             # uncommitted messages are redelivered


@needs_bundle
def test_streaming_parity_idempotency_dlq(small_tables, T, tmp_path):
    from predictmed.models.bundle import load_bundle
    b = load_bundle(BUNDLE_DIR)
    members = list(eligible_members(small_tables, T)[:60])
    envs, faults = inject_faults(events_from_tables(small_tables, members, T), dup_rate=0.05, bad_rate=0.02, seed=3)
    sp = StreamingPipeline(InMemoryBus(), StateStore(str(tmp_path / "s.db")), b, TOPICS, 3, 0.05, seed=1)
    sp.publish(envs)
    sp.drain_sources()
    sp.publish(envs[-200:])                          # replay storm
    sp.drain_sources()
    assert sp.stats["dlq"] == faults["injected_malformed"]
    assert sp.stats["duplicates_suppressed"] >= faults["injected_duplicates"] + 150
    sp.tick(T, members)
    sp.drain_score_requests()
    sp.drain_results()
    for p_ in range(sp.bus.n):                       # consumer restart before commit -> redelivery
        sp.bus.committed[("audit-sink", TOPICS["score_result"], p_)] = 0
    sp.drain_results()
    res = pd.read_sql_query("SELECT * FROM score_results", sp.store.con)
    assert res.request_id.is_unique and len(res) == len(members)
    # offline path on the same members
    off = build_snapshot(small_tables, T, pd.Index(members, name="member_id"))
    labs = small_tables["labs"]
    tw = twin_features_at(labs[labs.test == "egfr"], labs[labs.test == "uacr"], off[["member_id", "index_date"]],
                          small_tables["members"], TwinParams.from_json(b.twin_params_json))
    off = off.merge(tw, on=["member_id", "index_date"]).set_index("member_id")
    p_off = pd.Series(b.predict(off.reset_index()), index=off.index)
    on = res.dropna(subset=["estimated_probability"]).set_index("member_id").estimated_probability
    assert np.abs(on - p_off.reindex(on.index)).max() < 1e-9


@needs_bundle
def test_api_contract():
    from fastapi.testclient import TestClient
    from predictmed.serving.api import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    sc = pd.read_parquet(os.path.join(os.path.dirname(BUNDLE_DIR), "..", "data", "scored_latest.parquet")).iloc[0]
    feats = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in sc.items()
             if not isinstance(v, pd.Timestamp)}
    r = c.post("/v1/score", json={"features": feats}).json()
    assert r["quality_status"] in ("valid", "warning") and 0 <= r["estimated_probability"] <= 1
    lo, hi = r["probability_interval"]
    assert lo <= hi
    assert "disclaimer" in r and r["model_version"].startswith("predictmed-ckd")
    # missing required field -> rejected, not a fabricated number
    bad = dict(feats, egfr_last=None)
    assert c.post("/v1/score", json={"features": bad}).json()["quality_status"] == "rejected"
    # out-of-range
    assert c.post("/v1/score", json={"features": dict(feats, egfr_last=900)}).json()["reason_code"].startswith("OUT_OF_RANGE")
    # schema mismatch
    assert c.post("/v1/score", json={"features": feats, "schema_version": "features.snapshot.v0"}).status_code == 422
    # unknown fields are ignored and reported
    r = c.post("/v1/score", json={"features": dict(feats, mystery_field=1)}).json()
    assert "mystery_field" in r["ignored_fields"]


@needs_bundle
def test_bundle_checksum_enforced(tmp_path):
    import json
    import shutil
    from predictmed.models.bundle import load_bundle
    d = tmp_path / "b"
    shutil.copytree(BUNDLE_DIR, d)
    m = json.load(open(d / "manifest.json"))
    m["sha256"] = "0" * 64
    json.dump(m, open(d / "manifest.json", "w"))
    with pytest.raises(RuntimeError):
        load_bundle(str(d))
