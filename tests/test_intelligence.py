"""Tests for core.test_intelligence — AI Test Intelligence Engine."""

import pytest
import polars as pl

from core.test_intelligence import (
    TestIntelligenceEngine,
    SECURITY_STRINGS,
    UNICODE_STRINGS,
    BOUNDARY_INT,
    BOUNDARY_FLOAT,
    BOUNDARY_STRING,
    BOUNDARY_DATE_STRS,
)


@pytest.fixture
def simple_schema():
    return {"name": "String", "age": "Int64", "email": "String", "salary": "Float64"}


@pytest.fixture
def engine():
    return TestIntelligenceEngine(provider_name="ollama")


class TestFallbackAnalysis:

    def test_identifies_email_semantic(self, engine, simple_schema):
        analysis = engine._fallback_analysis(simple_schema)
        assert analysis["columns"]["email"]["semantic"] == "email"
        assert "security" in analysis["columns"]["email"]["categories"]

    def test_identifies_name_semantic(self, engine, simple_schema):
        analysis = engine._fallback_analysis(simple_schema)
        assert analysis["columns"]["name"]["semantic"] == "name"

    def test_identifies_number_types(self, engine, simple_schema):
        analysis = engine._fallback_analysis(simple_schema)
        assert analysis["columns"]["age"]["semantic"] == "number"
        assert analysis["columns"]["salary"]["semantic"] == "price"

    def test_all_columns_have_categories(self, engine, simple_schema):
        analysis = engine._fallback_analysis(simple_schema)
        for col, info in analysis["columns"].items():
            assert "categories" in info
            assert len(info["categories"]) > 0

    def test_domain_detected(self, engine, simple_schema):
        analysis = engine._fallback_analysis(simple_schema)
        assert "domain" in analysis


class TestEdgeCaseGeneration:

    def test_happy_path_generated(self, engine, simple_schema):
        analysis = engine._fallback_analysis(simple_schema)
        result = engine.generate_edge_cases(simple_schema, analysis)
        assert "happy_path" in result
        assert len(result["happy_path"]) > 0

    def test_boundary_values_for_int(self, engine):
        schema = {"count": "Int64"}
        analysis = engine._fallback_analysis(schema)
        result = engine.generate_edge_cases(schema, analysis)
        boundary_values = [r["count"] for r in result["boundary"]]
        for bv in BOUNDARY_INT:
            assert bv in boundary_values

    def test_boundary_values_for_float(self, engine):
        schema = {"price": "Float64"}
        analysis = engine._fallback_analysis(schema)
        result = engine.generate_edge_cases(schema, analysis)
        boundary_values = [r["price"] for r in result["boundary"]]
        for bv in BOUNDARY_FLOAT:
            assert bv in boundary_values

    def test_boundary_values_for_string(self, engine):
        schema = {"description": "String"}
        analysis = engine._fallback_analysis(schema)
        result = engine.generate_edge_cases(schema, analysis)
        assert len(result["boundary"]) > 0

    def test_security_strings_for_text_fields(self, engine):
        schema = {"comment": "String"}
        analysis = engine._fallback_analysis(schema)
        result = engine.generate_edge_cases(schema, analysis)
        security_values = [r["comment"] for r in result["security"]]
        for ss in SECURITY_STRINGS:
            assert ss in security_values

    def test_unicode_strings_for_name_fields(self, engine):
        schema = {"full_name": "String"}
        analysis = engine._fallback_analysis(schema)
        result = engine.generate_edge_cases(schema, analysis)
        unicode_values = [r["full_name"] for r in result["unicode"]]
        for us in UNICODE_STRINGS:
            assert us in unicode_values

    def test_null_patterns_generated(self, engine, simple_schema):
        analysis = engine._fallback_analysis(simple_schema)
        result = engine.generate_edge_cases(simple_schema, analysis)
        assert len(result["nulls"]) >= len(simple_schema)
        single_null_rows = [r for r in result["nulls"] if "single null" in r.get("_scenario", "")]
        assert len(single_null_rows) == len(simple_schema)

    def test_all_rows_have_metadata(self, engine, simple_schema):
        analysis = engine._fallback_analysis(simple_schema)
        result = engine.generate_edge_cases(simple_schema, analysis)
        for category, rows in result.items():
            for row in rows:
                assert "_category" in row, f"Missing _category in {category}"
                assert "_scenario" in row, f"Missing _scenario in {category}"
                assert row["_category"] == category

    def test_seven_categories_returned(self, engine, simple_schema):
        analysis = engine._fallback_analysis(simple_schema)
        result = engine.generate_edge_cases(simple_schema, analysis)
        expected = {"original", "happy_path", "boundary", "invalid", "security", "unicode", "nulls"}
        assert set(result.keys()) == expected

    def test_invalid_emails_generated(self, engine):
        schema = {"email": "String"}
        analysis = engine._fallback_analysis(schema)
        result = engine.generate_edge_cases(schema, analysis)
        assert len(result["invalid"]) > 0
        scenarios = [r["_scenario"] for r in result["invalid"]]
        assert any("malformed email" in s for s in scenarios)


class TestCoverageScoring:

    def test_fallback_scoring(self, engine, simple_schema):
        analysis = engine._fallback_analysis(simple_schema)
        test_data = engine.generate_edge_cases(simple_schema, analysis)
        coverage = engine._fallback_coverage(test_data, analysis)
        assert "score" in coverage
        assert 0 <= coverage["score"] <= 100
        assert "gaps" in coverage
        assert "total_rows" in coverage
        assert coverage["total_rows"] > 0

    def test_empty_data_low_score(self, engine, simple_schema):
        analysis = engine._fallback_analysis(simple_schema)
        empty_data = {"happy_path": [], "boundary": [], "invalid": [],
                      "security": [], "unicode": [], "nulls": []}
        coverage = engine._fallback_coverage(empty_data, analysis)
        assert coverage["score"] < 50
        assert len(coverage["gaps"]) > 0

    def test_full_data_higher_score(self, engine, simple_schema):
        analysis = engine._fallback_analysis(simple_schema)
        test_data = engine.generate_edge_cases(simple_schema, analysis)
        coverage = engine._fallback_coverage(test_data, analysis)
        assert coverage["score"] >= 70


class TestGapFixing:

    def test_fallback_fix_gaps(self, engine, simple_schema):
        analysis = engine._fallback_analysis(simple_schema)
        gaps = [
            {"category": "security", "description": "No SQL injection tests", "severity": "high"},
            {"category": "unicode", "description": "No CJK character tests", "severity": "medium"},
        ]
        additional = engine._fallback_fix_gaps(simple_schema, gaps, analysis)
        assert len(additional["security"]) > 0
        assert len(additional["unicode"]) > 0

    def test_gap_fix_rows_have_metadata(self, engine, simple_schema):
        analysis = engine._fallback_analysis(simple_schema)
        gaps = [{"category": "nulls", "description": "Missing null tests", "severity": "high"}]
        additional = engine._fallback_fix_gaps(simple_schema, gaps, analysis)
        for row in additional["nulls"]:
            assert "_category" in row
            assert "_scenario" in row


class TestMakeRow:

    def test_target_column_gets_value(self, engine):
        schema = {"name": "String", "age": "Int64"}
        row = engine._make_row(schema, "name", "TEST", "boundary", "test scenario")
        assert row["name"] == "TEST"
        assert row["age"] == 1
        assert row["_category"] == "boundary"
        assert row["_scenario"] == "test scenario"

    def test_default_values_correct(self, engine):
        assert engine._default_value("Int64") == 1
        assert engine._default_value("Float64") == 1.0
        assert engine._default_value("Date") == "2024-01-15"
        assert engine._default_value("String") == "test_value"
