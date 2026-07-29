"""
Input validation utilities for Synthetic-Data-Forge.

Centralises all boundary-level validation so that core engines
can trust their inputs.
"""

import re

from core.exceptions import ValidationError

# ---------------------------------------------------------------------------
# Column / schema validation
# ---------------------------------------------------------------------------
ALLOWED_DTYPES = {"Int64", "Float64", "String", "Date"}

_SAFE_COL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_ .\-]{0,127}$")


def validate_schema(schema: dict) -> dict:
    """Validate and return *schema*; raises on bad column names or types."""
    if not schema:
        raise ValidationError("Schema must contain at least one column.")
    for col, dtype in schema.items():
        if not _SAFE_COL_RE.match(col):
            raise ValidationError(
                f"Invalid column name '{col}'. Must be alphanumeric/underscore, ≤128 chars."
            )
        if dtype not in ALLOWED_DTYPES:
            raise ValidationError(
                f"Unsupported dtype '{dtype}' for column '{col}'. Choose from {ALLOWED_DTYPES}."
            )
    return schema


# ---------------------------------------------------------------------------
# LLM field descriptions
# ---------------------------------------------------------------------------
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_FIELD_DESC_LEN = 500


def sanitize_field_description(desc: str) -> str:
    """Strip control characters and enforce max length."""
    if not desc:
        return ""
    cleaned = _CONTROL_CHAR_RE.sub("", desc)
    return cleaned[:MAX_FIELD_DESC_LEN]


def sanitize_field_descriptions(descs: dict | None) -> dict | None:
    """Sanitize all field descriptions."""
    if descs is None:
        return None
    return {k: sanitize_field_description(v) for k, v in descs.items()}


# ---------------------------------------------------------------------------
# Relational parameters
# ---------------------------------------------------------------------------

def validate_relationship(
    tables: dict,
    parent_table: str,
    parent_col: str,
    child_table: str,
    child_col: str,
) -> None:
    """Ensure the FK relationship references valid tables and columns."""
    if parent_table not in tables:
        raise ValidationError(f"Parent table '{parent_table}' not registered.")
    if child_table not in tables:
        raise ValidationError(f"Child table '{child_table}' not registered.")
    if parent_col not in tables[parent_table]:
        raise ValidationError(
            f"Column '{parent_col}' not found in parent table '{parent_table}'."
        )
    if child_col not in tables[child_table]:
        raise ValidationError(
            f"Column '{child_col}' not found in child table '{child_table}'."
        )


# ---------------------------------------------------------------------------
# File-upload validation
# ---------------------------------------------------------------------------
MAGIC_BYTES = {
    "csv": None,  # text-based, no magic bytes
    "parquet": b"PAR1",
    "json": None,
    "jsonl": None,
}


def validate_file_upload(name: str, header_bytes: bytes, size_bytes: int, max_mb: int) -> None:
    """Validate uploaded file by extension, magic bytes, and size."""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ("csv", "parquet", "json", "jsonl"):
        raise ValidationError(f"Unsupported file type '.{ext}'.")
    if ext == "parquet" and not header_bytes.startswith(b"PAR1"):
        raise ValidationError("File claims to be Parquet but has invalid magic bytes.")
    max_bytes = max_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise ValidationError(f"File size ({size_bytes / 1e6:.1f} MB) exceeds limit ({max_mb} MB).")


# ---------------------------------------------------------------------------
# Partition key sanitization
# ---------------------------------------------------------------------------
_SAFE_PARTITION_RE = re.compile(r"[^A-Za-z0-9_.\-]")


def sanitize_partition_value(value) -> str:
    """Remove unsafe characters from a Hive-style partition value."""
    return _SAFE_PARTITION_RE.sub("_", str(value))
