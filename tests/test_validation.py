"""Tests for core.validation — Input validation utilities."""

import pytest
from datetime import date

from core.validation import (
    validate_schema,
    sanitize_field_description,
    sanitize_field_descriptions,
    validate_temporal_params,
    validate_relationship,
    validate_file_upload,
    sanitize_partition_value,
)
from core.exceptions import ValidationError


class TestValidateSchema:

    def test_valid_schema(self):
        schema = {"name": "String", "age": "Int64"}
        assert validate_schema(schema) == schema

    def test_empty_schema_raises(self):
        with pytest.raises(ValidationError, match="at least one column"):
            validate_schema({})

    def test_bad_column_name(self):
        with pytest.raises(ValidationError, match="Invalid column name"):
            validate_schema({"'; DROP TABLE --": "String"})

    def test_bad_dtype(self):
        with pytest.raises(ValidationError, match="Unsupported dtype"):
            validate_schema({"name": "VARCHAR"})


class TestSanitizeFieldDescription:

    def test_normal_text(self):
        assert sanitize_field_description("Name of user") == "Name of user"

    def test_strips_control_chars(self):
        assert sanitize_field_description("hello\x00world") == "helloworld"

    def test_truncates_long(self):
        long = "x" * 600
        assert len(sanitize_field_description(long)) == 500

    def test_empty(self):
        assert sanitize_field_description("") == ""

    def test_none_through_dict(self):
        assert sanitize_field_descriptions(None) is None

    def test_dict_sanitization(self):
        result = sanitize_field_descriptions({"col": "hello\x00"})
        assert result == {"col": "hello"}


class TestValidateTemporalParams:

    def test_valid(self):
        validate_temporal_params(
            date(2024, 1, 1), date(2024, 6, 1), "monthly", 5.0, None
        )

    def test_end_before_start(self):
        with pytest.raises(ValidationError, match="end_date"):
            validate_temporal_params(date(2024, 6, 1), date(2024, 1, 1), "monthly", 0, None)

    def test_invalid_frequency(self):
        with pytest.raises(ValidationError, match="frequency"):
            validate_temporal_params(date(2024, 1, 1), date(2024, 6, 1), "hourly", 0, None)

    def test_excessive_trend(self):
        with pytest.raises(ValidationError, match="trend_pct"):
            validate_temporal_params(date(2024, 1, 1), date(2024, 6, 1), "monthly", 99, None)

    def test_spike_validation(self):
        with pytest.raises(ValidationError, match="multiplier"):
            validate_temporal_params(
                date(2024, 1, 1), date(2024, 6, 1), "monthly", 0,
                [(date(2024, 3, 1), 999)]
            )


class TestValidateRelationship:

    def test_valid(self):
        tables = {"a": {"id": "Int64"}, "b": {"a_id": "Int64"}}
        validate_relationship(tables, "a", "id", "b", "a_id")

    def test_missing_parent_table(self):
        tables = {"b": {"a_id": "Int64"}}
        with pytest.raises(ValidationError, match="Parent table"):
            validate_relationship(tables, "a", "id", "b", "a_id")

    def test_missing_column(self):
        tables = {"a": {"id": "Int64"}, "b": {"a_id": "Int64"}}
        with pytest.raises(ValidationError, match="Column"):
            validate_relationship(tables, "a", "nonexistent", "b", "a_id")


class TestValidateFileUpload:

    def test_valid_csv(self):
        validate_file_upload("data.csv", b"name,age\n", 1000, 200)

    def test_valid_parquet(self):
        validate_file_upload("data.parquet", b"PAR1xxxxx", 1000, 200)

    def test_invalid_extension(self):
        with pytest.raises(ValidationError, match="Unsupported"):
            validate_file_upload("data.exe", b"\x00", 1000, 200)

    def test_bad_parquet_magic(self):
        with pytest.raises(ValidationError, match="magic bytes"):
            validate_file_upload("data.parquet", b"NOT_PAR", 1000, 200)

    def test_file_too_large(self):
        with pytest.raises(ValidationError, match="exceeds limit"):
            validate_file_upload("data.csv", b"x", 300 * 1024 * 1024, 200)


class TestSanitizePartitionValue:

    def test_clean_value(self):
        assert sanitize_partition_value("US") == "US"

    def test_special_chars(self):
        assert sanitize_partition_value("US/East") == "US_East"

    def test_spaces(self):
        assert sanitize_partition_value("New York") == "New_York"
