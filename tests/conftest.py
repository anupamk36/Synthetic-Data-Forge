"""Shared fixtures for Synthetic-Data-Forge tests."""

import os
import sys

import polars as pl
import pytest

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set safe defaults for tests
os.environ.setdefault("FORGE_PHARMA_SAFE_MODE", "true")
os.environ.setdefault("FORGE_LOG_LEVEL", "WARNING")


@pytest.fixture
def simple_schema():
    return {"name": "String", "age": "Int64", "salary": "Float64"}


@pytest.fixture
def date_schema():
    return {"event": "String", "event_date": "Date", "amount": "Float64"}


@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [30, 25, 35],
        "salary": [50000.0, 60000.0, 70000.0],
    })


@pytest.fixture
def tmp_output(tmp_path):
    """Return a temp directory and set FORGE_OUTPUT_ROOT to allow writes there."""
    os.environ["FORGE_OUTPUT_ROOT"] = str(tmp_path)
    # Reload config module to pick up new env var
    import importlib

    import core.config
    importlib.reload(core.config)
    yield tmp_path
    # Cleanup env
    os.environ.pop("FORGE_OUTPUT_ROOT", None)
