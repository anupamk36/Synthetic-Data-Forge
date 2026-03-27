"""
Time-Travel Simulation Engine.

Generates temporal synthetic data with configurable trends, spikes,
and seasonal patterns for pipeline load-testing and anomaly detection.
"""

import logging
import polars as pl
from faker import Faker
from datetime import date, timedelta
import calendar

from core.exceptions import ValidationError
from core.validation import validate_temporal_params

logger = logging.getLogger(__name__)

# Safety cap to prevent runaway generation
MAX_TOTAL_RECORDS = 10_000_000


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

        validate_temporal_params(start_date, end_date, frequency, trend_pct, spike_dates)

        periods = self._generate_periods(start_date, end_date, frequency)
        all_data = []
        total_count = 0

        for i, (period_start, period_end) in enumerate(periods):
            trend_factor = (1 + trend_pct / 100) ** i
            period_count = max(1, int(base_count_per_period * trend_factor))

            for spike_date, multiplier in spike_dates:
                if period_start <= spike_date <= period_end:
                    period_count = int(period_count * multiplier)
                    break

            if total_count + period_count > MAX_TOTAL_RECORDS:
                period_count = MAX_TOTAL_RECORDS - total_count
                logger.warning(
                    "Capping generation at %d records (safety limit)", MAX_TOTAL_RECORDS
                )

            for _ in range(period_count):
                row = {"_period": period_start.isoformat()}
                for col, dtype in schema.items():
                    if "Date" in dtype:
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

            total_count += period_count
            if total_count >= MAX_TOTAL_RECORDS:
                break

        logger.info(
            "TimeTravelEngine generated %d records across %d periods",
            len(all_data), len(periods),
        )
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
                month = current.month + 1
                year = current.year
                if month > 12:
                    month = 1
                    year += 1
                last_day = calendar.monthrange(year, month)[1]
                day = min(current.day, last_day)
                period_end = date(year, month, day)

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
        """Preview the expected volume distribution without generating data."""
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
