"""Tests for core.privacy — PrivacyScorecard."""

import numpy as np
import polars as pl
import pytest

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
        real = pl.DataFrame(
            {
                "name": ["Alice", "Bob"],
                "age": [30, 25],
                "score": [9.5, 8.0],
            }
        )
        syn = pl.DataFrame(
            {
                "name": ["Charlie", "Dave"],
                "age": [35, 40],
                "score": [7.0, 6.5],
            }
        )
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

    def test_prepare_matrices_numeric(self):
        df = pl.DataFrame({"a": [0, 5, 10], "b": [100.0, 200.0, 300.0]})
        real_matrix, syn_matrix = PrivacyScorecard._prepare_matrices(df, df)
        assert real_matrix.shape == (3, 2)
        assert np.isclose(real_matrix[:, 0].min(), 0.0)
        assert np.isclose(real_matrix[:, 0].max(), 1.0)
        # Identical data should produce identical matrices
        assert np.allclose(real_matrix, syn_matrix)


class TestKAnonymity:
    """Tests for PrivacyScorecard.compute_k_anonymity."""

    def test_k2_pairs(self):
        """Each quasi-identifier group has exactly 2 members → min_k = 2."""
        df = pl.DataFrame(
            {
                "age": [25, 25, 30, 30, 35, 35],
                "city": ["NYC", "NYC", "LA", "LA", "SF", "SF"],
                "salary": [50000, 55000, 60000, 65000, 70000, 75000],
            }
        )
        scorecard = PrivacyScorecard()
        result = scorecard.compute_k_anonymity(df, ["age", "city"])
        assert result["min_k"] == 2
        assert result["total_groups"] == 3
        assert result["vulnerable_groups"] == 0
        assert result["mean_group_size"] == 2.0

    def test_k1_all_unique(self):
        """Every record is unique → min_k = 1, all groups vulnerable."""
        df = pl.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "Dave"],
                "age": [25, 30, 35, 40],
            }
        )
        scorecard = PrivacyScorecard()
        result = scorecard.compute_k_anonymity(df, ["name", "age"])
        assert result["min_k"] == 1
        assert result["vulnerable_groups"] == 4
        assert result["total_groups"] == 4

    def test_missing_columns_raises(self):
        """Requesting non-existent quasi-identifiers raises PrivacyError."""
        from core.exceptions import PrivacyError

        df = pl.DataFrame({"age": [25, 30], "city": ["NYC", "LA"]})
        scorecard = PrivacyScorecard()
        with pytest.raises(PrivacyError):
            scorecard.compute_k_anonymity(df, ["age", "nonexistent_col"])

    def test_histogram_top10(self):
        """Histogram should contain at most 10 entries."""
        # Create 15 distinct group sizes
        rows = []
        for size in range(1, 16):
            for _ in range(size):
                rows.append({"group_id": f"g{size}", "val": size})
        df = pl.DataFrame(rows)
        scorecard = PrivacyScorecard()
        result = scorecard.compute_k_anonymity(df, ["group_id"])
        assert len(result["group_size_histogram"]) <= 10


class TestLDiversity:
    """Tests for PrivacyScorecard.compute_l_diversity."""

    def test_diverse_groups(self):
        """Each group has multiple distinct sensitive values."""
        df = pl.DataFrame(
            {
                "age": [25, 25, 25, 30, 30, 30],
                "disease": ["flu", "cold", "covid", "flu", "cold", "asthma"],
            }
        )
        scorecard = PrivacyScorecard()
        result = scorecard.compute_l_diversity(df, ["age"], "disease")
        assert result["min_l"] == 3
        assert result["mean_l"] == 3.0
        assert result["vulnerable_groups"] == 0

    def test_homogeneous_group(self):
        """One group has only one distinct sensitive value → vulnerable."""
        df = pl.DataFrame(
            {
                "age": [25, 25, 30, 30],
                "disease": ["flu", "flu", "cold", "asthma"],
            }
        )
        scorecard = PrivacyScorecard()
        result = scorecard.compute_l_diversity(df, ["age"], "disease")
        assert result["min_l"] == 1
        assert result["vulnerable_groups"] == 1

    def test_missing_sensitive_col_raises(self):
        """Requesting a non-existent sensitive column raises PrivacyError."""
        from core.exceptions import PrivacyError

        df = pl.DataFrame({"age": [25, 30], "city": ["NYC", "LA"]})
        scorecard = PrivacyScorecard()
        with pytest.raises(PrivacyError):
            scorecard.compute_l_diversity(df, ["age"], "nonexistent")


class TestEpsilonEstimation:
    """Tests for PrivacyScorecard.estimate_epsilon."""

    def test_identical_distributions(self):
        """Identical real and synthetic data should yield epsilon near 0."""
        np.random.seed(42)
        data = np.random.randn(500).tolist()
        df = pl.DataFrame({"x": data, "y": [v * 2 for v in data]})
        scorecard = PrivacyScorecard()
        result = scorecard.estimate_epsilon(df, df)
        assert result["estimated_epsilon"] < 0.01
        assert result["interpretation"] == "Strong"
        assert "x" in result["per_column"]
        assert "y" in result["per_column"]

    def test_divergent_distributions(self):
        """Very different distributions should yield high epsilon."""
        real = pl.DataFrame({"x": list(range(100))})
        syn = pl.DataFrame({"x": list(range(1000, 1100))})
        scorecard = PrivacyScorecard()
        result = scorecard.estimate_epsilon(real, syn)
        assert result["estimated_epsilon"] > 3.0
        assert result["interpretation"] == "Weak"

    def test_no_shared_numeric_columns(self):
        """No shared numeric columns returns epsilon 0 with Strong."""
        real = pl.DataFrame({"name": ["Alice", "Bob"]})
        syn = pl.DataFrame({"name": ["Charlie", "Dave"]})
        scorecard = PrivacyScorecard()
        result = scorecard.estimate_epsilon(real, syn)
        assert result["estimated_epsilon"] == 0.0
        assert result["interpretation"] == "Strong"
        assert result["per_column"] == {}


class TestComplianceReport:
    """Integration tests for PrivacyScorecard.generate_compliance_report."""

    def test_full_report_low_risk(self):
        """Data with good privacy properties should be compliant."""
        np.random.seed(123)
        real = pl.DataFrame(
            {
                "age": [25, 25, 30, 30, 35, 35],
                "city": ["NYC", "NYC", "LA", "LA", "SF", "SF"],
                "salary": [50000.0, 52000.0, 60000.0, 62000.0, 70000.0, 72000.0],
                "disease": ["flu", "cold", "covid", "asthma", "flu", "cold"],
            }
        )
        # Synthetic with similar distributions but different values
        syn = pl.DataFrame(
            {
                "age": [25, 25, 30, 30, 35, 35],
                "city": ["NYC", "NYC", "LA", "LA", "SF", "SF"],
                "salary": [51000.0, 53000.0, 61000.0, 63000.0, 71000.0, 73000.0],
                "disease": ["cold", "flu", "asthma", "covid", "cold", "flu"],
            }
        )
        scorecard = PrivacyScorecard()
        report = scorecard.generate_compliance_report(
            real,
            syn,
            quasi_identifiers=["age", "city"],
            sensitive_col="disease",
        )
        assert "dcr" in report
        assert "epsilon" in report
        assert "k_anonymity" in report
        assert "l_diversity" in report
        assert "overall_risk" in report
        assert "recommendations" in report
        assert "compliant" in report
        assert report["k_anonymity"]["min_k"] == 2
        assert isinstance(report["compliant"], bool)

    def test_report_without_optional_params(self):
        """Report works without quasi_identifiers and sensitive_col."""
        real = pl.DataFrame({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
        syn = pl.DataFrame({"x": [100, 200, 300], "y": [1000.0, 2000.0, 3000.0]})
        scorecard = PrivacyScorecard()
        report = scorecard.generate_compliance_report(real, syn)
        assert report["k_anonymity"] is None
        assert report["l_diversity"] is None
        assert report["dcr"] is not None
        assert report["epsilon"] is not None

    def test_report_high_risk_from_identical_data(self):
        """Identical data should produce high risk assessment."""
        df = pl.DataFrame(
            {
                "age": [25, 30, 35, 40],
                "salary": [50000.0, 60000.0, 70000.0, 80000.0],
                "name": ["Alice", "Bob", "Charlie", "Dave"],
            }
        )
        scorecard = PrivacyScorecard()
        report = scorecard.generate_compliance_report(
            df,
            df,
            quasi_identifiers=["age", "name"],
        )
        assert report["overall_risk"] == "High"
        assert report["compliant"] is False
