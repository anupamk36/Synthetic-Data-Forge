"""
Time-Travel Simulation Engine.

Generates temporal synthetic data with configurable trends, spikes,
and seasonal patterns for pipeline load-testing and anomaly detection.
"""

import polars as pl
from faker import Faker
from datetime import date, timedelta
import math


class TimeTravelEngine:
    """Generates synthetic data with temporal patterns."""

    def __init__(self):
        self.fake = Faker()

    def generate_temporal(
        self,
        schema: dict,
        base_count_per_period: int,
        start_date: date,
        end_date: date,
        frequency: str = "monthly",
        trend_pct: float = 0.0,
        spike_dates: list = None,
        spike_multiplier: float = 3.0,
    ) -> pl.DataFrame:
        """
        Generate data across time periods with trends and spikes.

        Args:
            schema: column name -> type mapping
            base_count_per_period: base number of records per period
            start_date: start of simulation window
            end_date: end of simulation window
            frequency: "daily", "weekly", or "monthly"
            trend_pct: percentage growth/decline per period (e.g., 5.0 = +5% per period)
            spike_dates: list of (date, multiplier) tuples for volume spikes
            spike_multiplier: default multiplier if spike_dates uses simple dates

        Returns:
            Polars DataFrame with an added '_period' column
        """
        if spike_dates is None:
            spike_dates = []

        periods = self._generate_periods(start_date, end_date, frequency)
        all_data = []

        for i, (period_start, period_end) in enumerate(periods):
            # Apply trend: compound growth
            trend_factor = (1 + trend_pct / 100) ** i
            period_count = max(1, int(base_count_per_period * trend_factor))

            # Apply spikes
            for spike_date, multiplier in spike_dates:
                if period_start <= spike_date <= period_end:
                    period_count = int(period_count * multiplier)
                    break

            # Generate records for this period
            for _ in range(period_count):
                row = {"_period": period_start.isoformat()}
                for col, dtype in schema.items():
                    if "Date" in dtype:
                        # Generate dates within the period
                        delta = (period_end - period_start).days
                        random_days = self.fake.random_int(0, max(delta, 1))
                        row[col] = period_start + timedelta(days=random_days)
                    elif "Int" in dtype:
                        row[col] = self.fake.random_int(0, 10000)
                    elif "Float" in dtype:
                        row[col] = self.fake.pyfloat(right_digits=2, positive=True)
                    else:
                        row[col] = self.fake.word()
                all_data.append(row)

        return pl.DataFrame(all_data)

    def _generate_periods(self, start: date, end: date, frequency: str) -> list:
        """Generate list of (period_start, period_end) tuples."""
        periods = []
        current = start

        while current < end:
            if frequency == "daily":
                period_end = current + timedelta(days=1)
            elif frequency == "weekly":
                period_end = current + timedelta(weeks=1)
            else:  # monthly
                # Move to same day next month
                month = current.month + 1
                year = current.year
                if month > 12:
                    month = 1
                    year += 1
                try:
                    period_end = current.replace(year=year, month=month)
                except ValueError:
                    # Handle months with fewer days (e.g., Jan 31 -> Feb 28)
                    period_end = current.replace(year=year, month=month, day=28)

            period_end = min(period_end, end)
            periods.append((current, period_end))
            current = period_end

        return periods

    def get_volume_preview(
        self,
        base_count: int,
        start_date: date,
        end_date: date,
        frequency: str,
        trend_pct: float,
        spike_dates: list = None,
    ) -> list:
        """
        Preview the expected volume distribution without generating data.

        Returns list of {"period": str, "count": int} dicts for charting.
        """
        if spike_dates is None:
            spike_dates = []

        periods = self._generate_periods(start_date, end_date, frequency)
        preview = []

        for i, (period_start, period_end) in enumerate(periods):
            trend_factor = (1 + trend_pct / 100) ** i
            count = max(1, int(base_count * trend_factor))

            for spike_date, multiplier in spike_dates:
                if period_start <= spike_date <= period_end:
                    count = int(count * multiplier)
                    break

            preview.append({
                "period": period_start.isoformat(),
                "count": count,
            })

        return preview
