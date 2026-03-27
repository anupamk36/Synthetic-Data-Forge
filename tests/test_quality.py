"""Tests for core/quality.py — Data Quality Assessment Engine."""

import polars as pl
import pytest

from core.quality import assess_quality


@pytest.fixture
def sample_original():
    return pl.DataFrame({
        "id": list(range(1, 21)),
        "name": [f"Name_{i}" for i in range(20)],
        "age": [25 + i for i in range(20)],
        "salary": [50000.0 + i * 1000 for i in range(20)],
    })


@pytest.fixture
def sample_generated():
    return pl.DataFrame({
        "id": list(range(100, 120)),
        "name": [f"Gen_{i}" for i in range(20)],
        "age": [26 + i for i in range(20)],
        "salary": [51000.0 + i * 1000 for i in range(20)],
    })


class TestQualityAssessment:
    def test_basic_assessment(self, sample_generated):
        report = assess_quality(sample_generated)
        assert report.overall_score > 0
        assert report.completeness == 100.0
        assert len(report.column_details) == 4

    def test_with_original(self, sample_generated, sample_original):
        report = assess_quality(sample_generated, original_df=sample_original)
        assert report.distribution_score > 0
        assert report.overall_score > 0

    def test_with_schema(self, sample_generated):
        schema = {"id": "Int64", "name": "String", "age": "Int64", "salary": "Float64"}
        report = assess_quality(sample_generated, expected_schema=schema)
        assert report.schema_match > 0

    def test_empty_dataframe(self):
        df = pl.DataFrame({"a": []}).cast({"a": pl.Int64})
        report = assess_quality(df)
        assert len(report.warnings) > 0

    def test_completeness_with_nulls(self):
        df = pl.DataFrame({
            "a": [1, 2, None, 4, 5],
            "b": ["x", None, None, "y", "z"],
        })
        report = assess_quality(df)
        assert report.completeness < 100

    def test_column_details(self, sample_generated):
        report = assess_quality(sample_generated)
        for cd in report.column_details:
            assert "column" in cd
            assert "type" in cd
            assert "unique_count" in cd

    def test_report_to_dict(self, sample_generated):
        report = assess_quality(sample_generated)
        d = report.to_dict()
        assert "overall_score" in d
        assert "column_details" in d
        assert isinstance(d["column_details"], list)

    def test_single_value_warning(self):
        df = pl.DataFrame({"status": ["active"] * 20})
        report = assess_quality(df)
        assert any("1 unique value" in w for w in report.warnings)
