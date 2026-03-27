"""
Data Quality Assessment Engine.

Computes statistical fidelity metrics comparing generated data against
an original sample: distribution similarity, cardinality, completeness,
and per-column diagnostics.
"""

import logging
import math
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class QualityReport:
    """Holds the results of a data quality assessment."""

    def __init__(self):
        self.overall_score: float = 0.0   # 0-100
        self.completeness: float = 0.0    # % non-null cells
        self.uniqueness: float = 0.0      # avg uniqueness across columns
        self.schema_match: float = 0.0    # % columns matching expected types
        self.distribution_score: float = 0.0  # avg distribution similarity
        self.column_details: list[dict] = []
        self.warnings: list[str] = []

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 1),
            "completeness": round(self.completeness, 1),
            "uniqueness": round(self.uniqueness, 1),
            "schema_match": round(self.schema_match, 1),
            "distribution_score": round(self.distribution_score, 1),
            "column_details": self.column_details,
            "warnings": self.warnings,
        }


def assess_quality(
    generated_df: pl.DataFrame,
    original_df: pl.DataFrame | None = None,
    expected_schema: dict | None = None,
) -> QualityReport:
    """
    Assess the quality of a generated DataFrame.

    Parameters
    ----------
    generated_df : pl.DataFrame
        The synthetic dataset to evaluate.
    original_df : pl.DataFrame, optional
        Original sample to compare distributions against.
    expected_schema : dict, optional
        Expected column → type mapping.

    Returns
    -------
    QualityReport
    """
    report = QualityReport()
    n_rows = len(generated_df)
    n_cols = len(generated_df.columns)

    if n_rows == 0:
        report.warnings.append("Generated dataset is empty.")
        return report

    # ── Completeness ──
    total_cells = n_rows * n_cols
    null_cells = sum(generated_df[col].null_count() for col in generated_df.columns)
    report.completeness = ((total_cells - null_cells) / total_cells * 100) if total_cells > 0 else 0

    # ── Per-column analysis ──
    uniqueness_scores = []
    distribution_scores = []

    for col in generated_df.columns:
        gen_series = generated_df[col]
        col_info: dict[str, Any] = {
            "column": col,
            "type": str(gen_series.dtype),
            "null_pct": round(gen_series.null_count() / n_rows * 100, 1) if n_rows > 0 else 0,
            "unique_count": gen_series.n_unique(),
            "unique_pct": round(gen_series.n_unique() / n_rows * 100, 1) if n_rows > 0 else 0,
        }

        uniqueness_scores.append(col_info["unique_pct"])

        # Distribution comparison with original
        if original_df is not None and col in original_df.columns:
            orig_series = original_df[col]
            dist_score = _compare_distributions(gen_series, orig_series)
            col_info["distribution_similarity"] = round(dist_score * 100, 1)
            distribution_scores.append(dist_score)

        report.column_details.append(col_info)

    report.uniqueness = sum(uniqueness_scores) / len(uniqueness_scores) if uniqueness_scores else 0

    # ── Schema match ──
    if expected_schema:
        matches = 0
        for col, expected_type in expected_schema.items():
            if col in generated_df.columns:
                actual = str(generated_df[col].dtype)
                if _types_compatible(actual, expected_type):
                    matches += 1
        report.schema_match = (matches / len(expected_schema) * 100) if expected_schema else 100
    else:
        report.schema_match = 100.0  # No schema to check against

    # ── Distribution score ──
    if distribution_scores:
        report.distribution_score = sum(distribution_scores) / len(distribution_scores) * 100
    else:
        report.distribution_score = 100.0  # No original to compare

    # ── Overall score (weighted) ──
    report.overall_score = (
        report.completeness * 0.25
        + report.schema_match * 0.25
        + report.distribution_score * 0.30
        + min(report.uniqueness, 100) * 0.20
    )

    # ── Warnings ──
    if report.completeness < 95:
        report.warnings.append(f"Low completeness: {report.completeness:.0f}% (target ≥95%)")
    if report.distribution_score < 70:
        report.warnings.append(f"Distribution divergence detected: {report.distribution_score:.0f}%")
    for cd in report.column_details:
        if cd["unique_count"] == 1 and n_rows > 10:
            report.warnings.append(f"Column '{cd['column']}' has only 1 unique value")

    return report


# ──────────────────────────────────────────────────────────
# Distribution comparison helpers
# ──────────────────────────────────────────────────────────

def _compare_distributions(gen: pl.Series, orig: pl.Series) -> float:
    """
    Compare two series distributions. Returns 0.0 (dissimilar) to 1.0 (identical).

    For numeric columns: uses overlapping histogram bins.
    For categorical/string columns: uses Jaccard similarity of value sets + frequency similarity.
    """
    try:
        if gen.dtype in (pl.Int64, pl.Int32, pl.Int16, pl.Int8,
                         pl.Float64, pl.Float32, pl.UInt64, pl.UInt32):
            return _compare_numeric(gen, orig)
        else:
            return _compare_categorical(gen, orig)
    except Exception:
        return 0.5  # Unknown/mixed → neutral score


def _compare_numeric(gen: pl.Series, orig: pl.Series) -> float:
    """Compare numeric distributions using histogram overlap."""
    gen_clean = gen.drop_nulls().cast(pl.Float64).to_numpy()
    orig_clean = orig.drop_nulls().cast(pl.Float64).to_numpy()

    if len(gen_clean) == 0 or len(orig_clean) == 0:
        return 0.0

    # Shared bins
    lo = min(gen_clean.min(), orig_clean.min())
    hi = max(gen_clean.max(), orig_clean.max())
    if lo == hi:
        return 1.0  # Both constant

    bins = np.linspace(lo, hi, 21)
    gen_hist, _ = np.histogram(gen_clean, bins=bins, density=True)
    orig_hist, _ = np.histogram(orig_clean, bins=bins, density=True)

    # Normalize to probability distributions
    gen_sum = gen_hist.sum()
    orig_sum = orig_hist.sum()
    if gen_sum > 0:
        gen_hist = gen_hist / gen_sum
    if orig_sum > 0:
        orig_hist = orig_hist / orig_sum

    # Histogram intersection (overlap)
    overlap = np.minimum(gen_hist, orig_hist).sum()
    return float(min(overlap, 1.0))


def _compare_categorical(gen: pl.Series, orig: pl.Series) -> float:
    """Compare categorical distributions using value-set and frequency similarity."""
    gen_vals = set(gen.drop_nulls().cast(pl.Utf8).to_list())
    orig_vals = set(orig.drop_nulls().cast(pl.Utf8).to_list())

    if not gen_vals and not orig_vals:
        return 1.0
    if not gen_vals or not orig_vals:
        return 0.0

    # Jaccard similarity of value sets
    intersection = len(gen_vals & orig_vals)
    union = len(gen_vals | orig_vals)
    jaccard = intersection / union if union > 0 else 0

    # Frequency distribution similarity for shared values
    shared = gen_vals & orig_vals
    if not shared:
        return jaccard * 0.5

    gen_counts = gen.drop_nulls().cast(pl.Utf8).value_counts()
    orig_counts = orig.drop_nulls().cast(pl.Utf8).value_counts()

    gen_freq = {}
    for row in gen_counts.to_dicts():
        val = row.get(gen.name, row.get(gen_counts.columns[0]))
        cnt = row.get("count", row.get("counts", 0))
        gen_freq[str(val)] = cnt

    orig_freq = {}
    for row in orig_counts.to_dicts():
        val = row.get(orig.name, row.get(orig_counts.columns[0]))
        cnt = row.get("count", row.get("counts", 0))
        orig_freq[str(val)] = cnt

    gen_total = sum(gen_freq.values()) or 1
    orig_total = sum(orig_freq.values()) or 1

    freq_sim = 0.0
    for val in shared:
        gp = gen_freq.get(val, 0) / gen_total
        op = orig_freq.get(val, 0) / orig_total
        freq_sim += min(gp, op)

    return jaccard * 0.4 + freq_sim * 0.6


def _types_compatible(actual: str, expected: str) -> bool:
    """Check if actual polars type is compatible with expected type string."""
    actual_lower = actual.lower()
    expected_lower = expected.lower()

    type_map = {
        "int64": ["int", "i64"],
        "float64": ["float", "f64"],
        "string": ["str", "utf8", "string", "categorical"],
        "date": ["date", "datetime"],
    }

    for canonical, aliases in type_map.items():
        if any(a in expected_lower for a in aliases) and any(a in actual_lower for a in aliases):
            return True

    return expected_lower in actual_lower
