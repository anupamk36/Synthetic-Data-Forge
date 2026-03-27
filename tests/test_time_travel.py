"""Tests for core.time_travel — TimeTravelEngine."""

import polars as pl
import pytest
from datetime import date

from core.time_travel import TimeTravelEngine
from core.exceptions import ValidationError


class TestTimeTravelEngine:

    def test_basic_monthly(self):
        engine = TimeTravelEngine()
        schema = {"val": "Int64"}
        df = engine.generate_temporal(
            schema, base_count_per_period=10,
            start_date=date(2024, 1, 1), end_date=date(2024, 4, 1),
            frequency="monthly",
        )
        assert len(df) > 0
        assert "_period" in df.columns
        assert "val" in df.columns

    def test_basic_weekly(self):
        engine = TimeTravelEngine()
        schema = {"val": "Int64"}
        df = engine.generate_temporal(
            schema, base_count_per_period=5,
            start_date=date(2024, 1, 1), end_date=date(2024, 2, 1),
            frequency="weekly",
        )
        assert len(df) > 0

    def test_basic_daily(self):
        engine = TimeTravelEngine()
        schema = {"val": "Int64"}
        df = engine.generate_temporal(
            schema, base_count_per_period=2,
            start_date=date(2024, 1, 1), end_date=date(2024, 1, 8),
            frequency="daily",
        )
        # 7 days × 2 records = 14
        assert len(df) == 14

    def test_positive_trend(self):
        engine = TimeTravelEngine()
        schema = {"val": "Int64"}
        df = engine.generate_temporal(
            schema, base_count_per_period=100,
            start_date=date(2024, 1, 1), end_date=date(2024, 6, 1),
            frequency="monthly", trend_pct=10.0,
        )
        # Later periods should have more records than base
        assert len(df) > 500  # 5 months × 100 base

    def test_spike(self):
        engine = TimeTravelEngine()
        schema = {"val": "Int64"}
        spikes = [(date(2024, 3, 15), 5.0)]
        df = engine.generate_temporal(
            schema, base_count_per_period=10,
            start_date=date(2024, 1, 1), end_date=date(2024, 4, 1),
            frequency="monthly", spike_dates=spikes,
        )
        # March period should have ~50 records, others ~10
        assert len(df) > 30  # At least 10+10+50

    def test_invalid_date_range(self):
        engine = TimeTravelEngine()
        with pytest.raises(ValidationError, match="end_date must be after"):
            engine.generate_temporal(
                {"val": "Int64"}, 10,
                start_date=date(2024, 6, 1), end_date=date(2024, 1, 1),
            )

    def test_invalid_frequency(self):
        engine = TimeTravelEngine()
        with pytest.raises(ValidationError, match="Invalid frequency"):
            engine.generate_temporal(
                {"val": "Int64"}, 10,
                start_date=date(2024, 1, 1), end_date=date(2024, 2, 1),
                frequency="biweekly",
            )

    def test_excessive_trend_blocked(self):
        engine = TimeTravelEngine()
        with pytest.raises(ValidationError, match="trend_pct"):
            engine.generate_temporal(
                {"val": "Int64"}, 10,
                start_date=date(2024, 1, 1), end_date=date(2024, 6, 1),
                frequency="monthly", trend_pct=100.0,
            )

    def test_month_end_edge_case(self):
        """Jan 31 → Feb should land on Feb 28/29, not crash."""
        engine = TimeTravelEngine()
        periods = engine._generate_periods(date(2024, 1, 31), date(2024, 4, 1), "monthly")
        assert len(periods) >= 2
        # Second period start should be Feb 29 (2024 is leap year)
        assert periods[1][0] == date(2024, 2, 29)

    def test_volume_preview(self):
        engine = TimeTravelEngine()
        preview = engine.get_volume_preview(
            base_count=100,
            start_date=date(2024, 1, 1), end_date=date(2024, 4, 1),
            frequency="monthly", trend_pct=0.0,
        )
        assert len(preview) == 3
        assert all(p["count"] == 100 for p in preview)

    def test_date_column_in_schema(self):
        engine = TimeTravelEngine()
        schema = {"event_date": "Date", "amount": "Float64"}
        df = engine.generate_temporal(
            schema, base_count_per_period=5,
            start_date=date(2024, 1, 1), end_date=date(2024, 2, 1),
            frequency="monthly",
        )
        assert "event_date" in df.columns
        assert "amount" in df.columns
