import pathlib
import pytest
import pandas as pd

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "synthea_fixture"


def test_synthea_adapter_loads_fixture():
    from predictmed.ingestion.synthea import SyntheaAdapter
    from predictmed.ingestion.schema_tests import validate_adapter_output

    adapter = SyntheaAdapter(FIXTURE_DIR)
    tables = adapter.load()

    assert set(tables.keys()) == {"members", "enrollment", "labs", "claims", "rx_fills", "life_events"}
    assert len(tables["members"]) == 3

    # Schema validation
    errors = validate_adapter_output(tables)
    assert errors == [], f"Validation errors: {errors}"

    # Verify eGFR was computed from creatinine
    labs = tables["labs"]
    egfr_rows = labs[labs["test"] == "egfr"]
    assert len(egfr_rows) >= 2, "Expected eGFR to be computed for creatinine observations"
    assert (egfr_rows["value"] > 0).all()


def test_schema_validator_catches_invalid_input():
    from predictmed.ingestion.schema_tests import validate_adapter_output

    bad_tables = {
        "members": pd.DataFrame({"member_id": [1, 2]}),  # missing columns, int id
        "enrollment": pd.DataFrame(),
        "labs": pd.DataFrame({"member_id": ["m1"], "test": ["egfr"], "value": [-10.0]}),  # negative egfr
        "claims": pd.DataFrame(),
        "rx_fills": pd.DataFrame(),
        "life_events": pd.DataFrame(),
    }
    errors = validate_adapter_output(bad_tables)
    assert len(errors) > 0

