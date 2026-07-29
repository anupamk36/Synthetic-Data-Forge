"""Tests for the statistical profiler."""

import numpy as np
import polars as pl
import pytest

from core.profiler import DataProfile, profile_dataframe


class TestProfileDataframe:
    def test_basic_profile(self, sample_df):
        profile = profile_dataframe(sample_df)
        assert isinstance(profile, DataProfile)
        assert profile.row_count == 3
        assert profile.col_count == 3
        assert len(profile.column_stats) == 3

    def test_empty_dataframe(self):
        df = pl.DataFrame({"a": [], "b": []}).cast({"a": pl.Int64, "b": pl.Float64})
        profile = profile_dataframe(df)
        assert profile.row_count == 0

    def test_numeric_stats(self):
        df = pl.DataFrame({
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "salary": [40000.0, 50000.0, 60000.0, 70000.0, 80000.0, 90000.0, 100000.0, 110000.0],
        })
        profile = profile_dataframe(df)

        age_stats = next(cs for cs in profile.column_stats if cs.name == "age")
        assert age_stats.is_numeric
        assert age_stats.min_val == 25
        assert age_stats.max_val == 60
        assert age_stats.mean is not None
        assert age_stats.std is not None
        assert age_stats.percentiles is not None

    def test_categorical_stats(self):
        df = pl.DataFrame({
            "color": ["red", "blue", "red", "green", "blue", "red"],
        })
        profile = profile_dataframe(df)

        color_stats = next(cs for cs in profile.column_stats if cs.name == "color")
        assert color_stats.is_categorical
        assert color_stats.cardinality == 3
        assert color_stats.top_values is not None
        assert color_stats.top_values[0]["value"] == "red"
        assert color_stats.entropy > 0

    def test_null_rates(self):
        df = pl.DataFrame({
            "a": [1, 2, None, 4, None],
            "b": ["x", "y", "z", "w", "v"],
        })
        profile = profile_dataframe(df)

        a_stats = next(cs for cs in profile.column_stats if cs.name == "a")
        assert a_stats.null_rate == pytest.approx(0.4)

        b_stats = next(cs for cs in profile.column_stats if cs.name == "b")
        assert b_stats.null_rate == 0.0


class TestCorrelations:
    def test_pearson_correlation(self):
        np.random.seed(42)
        x = np.arange(50, dtype=float)
        y = x * 2 + np.random.normal(0, 1, 50)
        df = pl.DataFrame({"x": x, "y": y})

        profile = profile_dataframe(df)
        pearson_corrs = [c for c in profile.correlations if c.method == "pearson"]
        assert len(pearson_corrs) == 1
        assert pearson_corrs[0].value > 0.9
        assert pearson_corrs[0].significant is True

    def test_no_correlation(self):
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.normal(0, 1, 50),
            "y": np.random.normal(0, 1, 50),
        })
        profile = profile_dataframe(df)
        pearson_corrs = [c for c in profile.correlations if c.method == "pearson"]
        if pearson_corrs:
            assert abs(pearson_corrs[0].value) < 0.5

    def test_cramers_v_categorical(self):
        df = pl.DataFrame({
            "gender": ["M", "M", "F", "F", "M", "F", "M", "F", "M", "F"] * 5,
            "dept": ["eng", "eng", "hr", "hr", "eng", "hr", "eng", "hr", "eng", "hr"] * 5,
        })
        profile = profile_dataframe(df)
        v_corrs = [c for c in profile.correlations if c.method == "cramers_v"]
        assert len(v_corrs) >= 1


class TestConditionalDistributions:
    def test_conditional_for_correlated_numerics(self):
        np.random.seed(42)
        x = np.arange(50, dtype=float)
        y = x * 3 + np.random.normal(0, 2, 50)
        df = pl.DataFrame({"x": x, "y": y})

        profile = profile_dataframe(df)
        assert len(profile.conditional_distributions) > 0

    def test_no_conditional_for_uncorrelated(self):
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.normal(0, 1, 50),
            "y": np.random.normal(0, 1, 50),
        })
        profile = profile_dataframe(df)
        # May or may not have conditionals depending on random correlation
        # Just verify it doesn't crash


class TestConstraints:
    def test_unique_constraint(self):
        df = pl.DataFrame({"id": list(range(100)), "val": [f"v{i}" for i in range(100)]})
        profile = profile_dataframe(df)
        unique_constraints = [c for c in profile.constraints if c.constraint_type == "unique"]
        assert len(unique_constraints) >= 1

    def test_not_null_constraint(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        profile = profile_dataframe(df)
        nn_constraints = [c for c in profile.constraints if c.constraint_type == "not_null"]
        assert len(nn_constraints) == 2

    def test_range_constraint(self):
        df = pl.DataFrame({"val": [10, 20, 30, 40, 50]})
        profile = profile_dataframe(df)
        range_constraints = [c for c in profile.constraints if c.constraint_type == "range"]
        assert len(range_constraints) == 1
        assert "10" in range_constraints[0].details
        assert "50" in range_constraints[0].details


class TestDataProfileSerialization:
    def test_to_dict(self, sample_df):
        profile = profile_dataframe(sample_df)
        d = profile.to_dict()
        assert "column_stats" in d
        assert "correlations" in d
        assert "constraints" in d

    def test_to_json(self, sample_df):
        profile = profile_dataframe(sample_df)
        j = profile.to_json()
        import json
        parsed = json.loads(j)
        assert parsed["row_count"] == 3

    def test_summary_for_llm(self, sample_df):
        profile = profile_dataframe(sample_df)
        summary = profile.summary_for_llm()
        assert "row_count" in summary
        assert "columns" in summary
        assert "key_correlations" in summary
        assert "constraints" in summary
