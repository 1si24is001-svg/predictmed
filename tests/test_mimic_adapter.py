import pathlib
import pytest
import pandas as pd

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "mimic_fixture"


def test_mimic_fixture_schema():
    from predictmed.ingestion.mimic import MIMICAdapter
    from predictmed.ingestion.schema_tests import validate_adapter_output

    adapter = MIMICAdapter(FIXTURE_DIR)
    tables = adapter.load()

    assert "members" in tables
    assert "labs" in tables
    assert len(tables["members"]) > 0

    # Schema validation
    errors = validate_adapter_output(tables)
    assert errors == [], f"Validation errors: {errors}"

    # All eGFR values positive
    labs = tables["labs"]
    egfr_rows = labs[labs.test == "egfr"]
    assert len(egfr_rows) > 0
    assert (egfr_rows.value > 0).all()

    # Comorbidity checks on fixture
    members = tables["members"]
    assert (members["plan_medicare"] == 1).any()
    assert (members["plan_medicaid"] == 1).any()


def test_mimic_skips_without_env(monkeypatch):
    monkeypatch.delenv("MIMIC_DIR", raising=False)
    from predictmed.ingestion.mimic import load_if_available
    result = load_if_available()
    assert result is None


def test_mimic_loads_with_env(monkeypatch):
    monkeypatch.setenv("MIMIC_DIR", str(FIXTURE_DIR))
    from predictmed.ingestion.mimic import load_if_available
    result = load_if_available()
    assert result is not None
    assert "members" in result
    assert len(result["members"]) > 0

