"""Tests for core.generator — ForgeEngine."""

import polars as pl
import pytest
from unittest.mock import MagicMock

from core.generator import ForgeEngine, SMART_PROVIDERS


class TestForgeEngine:

    def test_generate_records_basic(self, simple_schema):
        engine = ForgeEngine()
        df = engine.generate_records(simple_schema, 50)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 50
        assert set(df.columns) == {"name", "age", "salary"}

    def test_generate_records_single_row(self, simple_schema):
        engine = ForgeEngine()
        df = engine.generate_records(simple_schema, 1)
        assert len(df) == 1

    def test_smart_provider_email(self):
        engine = ForgeEngine()
        provider = engine._get_provider("email", "String")
        value = provider(engine.fake)
        assert "@" in str(value)

    def test_smart_provider_phone(self):
        engine = ForgeEngine()
        provider = engine._get_provider("phone_number", "String")
        value = provider(engine.fake)
        assert len(str(value)) > 0

    def test_smart_provider_city(self):
        engine = ForgeEngine()
        provider = engine._get_provider("city", "String")
        value = provider(engine.fake)
        assert isinstance(value, str) and len(value) > 0

    def test_dtype_fallback_int(self):
        engine = ForgeEngine()
        provider = engine._get_provider("unknown_column", "Int64")
        value = provider(engine.fake)
        assert isinstance(value, int)

    def test_dtype_fallback_float(self):
        engine = ForgeEngine()
        provider = engine._get_provider("unknown_column", "Float64")
        value = provider(engine.fake)
        assert isinstance(value, float)

    def test_dtype_fallback_date(self):
        engine = ForgeEngine()
        provider = engine._get_provider("unknown_column", "Date")
        from datetime import date
        value = provider(engine.fake)
        assert isinstance(value, date)

    def test_provider_cache(self):
        engine = ForgeEngine()
        p1 = engine._get_provider("email", "String")
        p2 = engine._get_provider("email", "String")
        assert p1 is p2  # Same object from cache

    def test_llm_fallback_on_empty_records(self, simple_schema):
        engine = ForgeEngine()
        mock_llm = MagicMock()
        mock_llm.generate_data.return_value = []

        df = engine.generate_records(simple_schema, 10, use_llm=True, llm_engine=mock_llm, enable_validation=False)
        assert len(df) == 10  # Fell back to Faker
        mock_llm.generate_data.assert_called_once()

    def test_llm_success(self, simple_schema):
        engine = ForgeEngine()
        mock_llm = MagicMock()
        mock_llm.generate_data.return_value = [
            {"name": "Alice", "age": 30, "salary": 50000.0},
            {"name": "Bob", "age": 25, "salary": 60000.0},
        ]

        df = engine.generate_records(simple_schema, 2, use_llm=True, llm_engine=mock_llm)
        assert len(df) == 2
        assert df["name"][0] == "Alice"

    def test_pharma_safe_mode_blocks_ssn(self):
        """PHARMA_SAFE_MODE should NOT match SSN pattern."""
        import core.config
        original = core.config.PHARMA_SAFE_MODE
        try:
            core.config.PHARMA_SAFE_MODE = True
            engine = ForgeEngine()
            # 'ssn' should not get a smart SSN provider in safe mode
            provider = engine._get_provider("ssn", "String")
            value = provider(engine.fake)
            # In safe mode it falls through to generic word — not an SSN format
            assert not (isinstance(value, str) and "-" in value and len(value) == 11)
        finally:
            core.config.PHARMA_SAFE_MODE = original

    def test_generate_with_date_schema(self, date_schema):
        engine = ForgeEngine()
        df = engine.generate_records(date_schema, 20)
        assert len(df) == 20
        assert "event_date" in df.columns
