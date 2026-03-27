"""Tests for core.privacy — PrivacyScorecard."""

import polars as pl
import numpy as np
import pytest
from datetime import date

from core.privacy import PrivacyScorecard


class TestPrivacyScorecard:

    def test_basic_dcr(self, sample_df):
        scorecard = PrivacyScorecard()
        # Synthetic data identical to real → high risk
        result = scorecard.compute_dcr(sample_df, sample_df)
        assert result["error"] is None
        assert result["risk_level"] == "High"
        assert result["pct_exact_matches"] > 0

    def test_different_data_low_risk(self):
        real = pl.DataFrame({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
        syn = pl.DataFrame({"x": [100, 200, 300], "y": [1000.0, 2000.0, 3000.0]})
        scorecard = PrivacyScorecard()
        result = scorecard.compute_dcr(real, syn)
        assert result["error"] is None
        assert result["risk_level"] == "Low"

    def test_no_shared_columns(self):
        real = pl.DataFrame({"a": [1, 2]})
        syn = pl.DataFrame({"b": [1, 2]})
        scorecard = PrivacyScorecard()
        result = scorecard.compute_dcr(real, syn)
        assert result["error"] is not None

    def test_string_columns(self):
        real = pl.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        syn = pl.DataFrame({"name": ["Dave", "Eve", "Frank"]})
        scorecard = PrivacyScorecard()
        result = scorecard.compute_dcr(real, syn)
        assert result["error"] is None
        assert isinstance(result["min_dcr"], float)

    def test_mixed_types(self):
        real = pl.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [30, 25],
            "score": [9.5, 8.0],
        })
        syn = pl.DataFrame({
            "name": ["Charlie", "Dave"],
            "age": [35, 40],
            "score": [7.0, 6.5],
        })
        scorecard = PrivacyScorecard()
        result = scorecard.compute_dcr(real, syn)
        assert result["error"] is None
        assert len(result["dcr_values"]) == 2

    def test_null_handling(self):
        real = pl.DataFrame({"x": [1, None, 3], "y": [None, 2.0, 3.0]})
        syn = pl.DataFrame({"x": [10, 20, None], "y": [100.0, None, 300.0]})
        scorecard = PrivacyScorecard()
        result = scorecard.compute_dcr(real, syn)
        assert result["error"] is None

    def test_sampling_large(self):
        """Ensure large DataFrames are sampled."""
        n = 6000
        real = pl.DataFrame({"x": list(range(n))})
        syn = pl.DataFrame({"x": list(range(n, 2 * n))})
        scorecard = PrivacyScorecard()
        result = scorecard.compute_dcr(real, syn)
        assert result["error"] is None
        # DCR values should be fewer than n (sampled)
        assert len(result["dcr_values"]) <= 5000

    def test_prepare_matrix_numeric(self):
        df = pl.DataFrame({"a": [0, 5, 10], "b": [100.0, 200.0, 300.0]})
        matrix = PrivacyScorecard._prepare_matrix(df)
        assert matrix.shape == (3, 2)
        # Check normalization: min=0, max=1
        assert np.isclose(matrix[:, 0].min(), 0.0)
        assert np.isclose(matrix[:, 0].max(), 1.0)
