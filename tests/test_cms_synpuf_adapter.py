import pathlib
import pytest
import pandas as pd

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "synpuf_fixture"


def test_synpuf_fixture_schema():
    from predictmed.ingestion.cms_synpuf import CMSSynPUFAdapter
    from predictmed.ingestion.schema_tests import validate_adapter_output

    adapter = CMSSynPUFAdapter(FIXTURE_DIR, sample_id=1)
    tables = adapter.load()

    assert set(tables.keys()) == {"members", "enrollment", "labs", "claims", "rx_fills", "life_events"}
    assert "test" in tables["labs"].columns
    assert "value" in tables["labs"].columns
    assert len(tables["members"]) > 0

    # All plan_medicare=1
    assert (tables["members"]["plan_medicare"] == 1).all()

    # Schema validation
    errors = validate_adapter_output(tables)
    assert errors == [], f"Validation errors: {errors}"


def test_synpuf_date_parsing():
    from predictmed.ingestion.cms_synpuf import CMSSynPUFAdapter

    adapter = CMSSynPUFAdapter(FIXTURE_DIR, sample_id=1)
    tables = adapter.load()

    if len(tables["claims"]) > 0:
        assert pd.api.types.is_datetime64_any_dtype(tables["claims"]["claim_date"])
    if len(tables["rx_fills"]) > 0:
        assert pd.api.types.is_datetime64_any_dtype(tables["rx_fills"]["fill_date"])


def test_synpuf_no_future_dates():
    from predictmed.ingestion.cms_synpuf import CMSSynPUFAdapter

    adapter = CMSSynPUFAdapter(FIXTURE_DIR, sample_id=1)
    tables = adapter.load()

    now_thresh = pd.Timestamp.now() + pd.Timedelta(days=2)
    for tbl_name, tbl in tables.items():
        for col in ["obs_date", "fill_date", "claim_date", "event_date", "start_date"]:
            if col in tbl.columns and len(tbl) > 0:
                assert (tbl[col] <= now_thresh).all(), f"{tbl_name}.{col} has future dates"


def test_synpuf_available_samples():
    from predictmed.ingestion.cms_synpuf import CMSSynPUFAdapter

    adapter = CMSSynPUFAdapter(FIXTURE_DIR, sample_id=1)
    samples = adapter.available_samples()
    assert 1 in samples

