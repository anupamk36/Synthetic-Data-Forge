"""
Data Quality Assessment Engine.

Computes statistical fidelity metrics comparing generated data against
an original sample: distribution similarity, cardinality, completeness,
per-column diagnostics, and a realism grade (A-F).
"""

import logging
from typing import Any

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def _sanitize_for_json(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class QualityReport:
    """Holds the results of a data quality assessment."""

    def __init__(self):
        self.overall_score: float = 0.0   # 0-100
        self.completeness: float = 0.0    # % non-null cells
        self.uniqueness: float = 0.0      # avg uniqueness across columns
        self.schema_match: float = 0.0    # % columns matching expected types
        self.distribution_score: float = 0.0  # avg distribution similarity
        self.correlation_preservation: float = 0.0  # 0-100
        self.dependency_score: float = 0.0  # 0-100
        self.realism_grade: str = "N/A"
        self.column_details: list[dict] = []
        self.statistical_tests: list[dict] = []
        self.warnings: list[str] = []

    def to_dict(self) -> dict:
        return {
            "overall_score": round(float(self.overall_score), 1),
            "realism_grade": self.realism_grade,
            "completeness": round(float(self.completeness), 1),
            "uniqueness": round(float(self.uniqueness), 1),
            "schema_match": round(float(self.schema_match), 1),
            "distribution_score": round(float(self.distribution_score), 1),
            "correlation_preservation": round(float(self.correlation_preservation), 1),
            "dependency_score": round(float(self.dependency_score), 1),
            "column_details": _sanitize_for_json(self.column_details),
            "statistical_tests": _sanitize_for_json(self.statistical_tests),
            "warnings": self.warnings,
        }


def assess_quality(
    generated_df: pl.DataFrame,
    original_df: pl.DataFrame | None = None,
    expected_schema: dict | None = None,
) -> QualityReport:
    """Assess the quality of a generated DataFrame."""
    report = QualityReport()
    n_rows = len(generated_df)

    if n_rows == 0:
        report.warnings.append("Generated dataset is empty.")
        return report

    report.completeness = _compute_completeness(generated_df)
    uniqueness_scores, distribution_scores = _analyze_columns(
        generated_df, original_df, report,
    )
    report.uniqueness = sum(uniqueness_scores) / len(uniqueness_scores) if uniqueness_scores else 0
    report.schema_match = _compute_schema_match(generated_df, expected_schema)
    report.distribution_score = (
        sum(distribution_scores) / len(distribution_scores) * 100
        if distribution_scores else 100.0
    )

    if original_df is not None:
        report.correlation_preservation = _compute_correlation_preservation(generated_df, original_df)
        report.dependency_score = _compute_dependency_score(generated_df, original_df)

    report.overall_score = (
        report.distribution_score * 0.30
        + report.correlation_preservation * 0.30
        + report.completeness * 0.10
        + report.schema_match * 0.10
        + min(report.uniqueness, 100) * 0.10
        + report.dependency_score * 0.10
    )
    report.realism_grade = _score_to_grade(report.overall_score)
    _add_warnings(report, n_rows)
    return report


def _compute_completeness(df: pl.DataFrame) -> float:
    total_cells = len(df) * len(df.columns)
    if total_cells == 0:
        return 0.0
    null_cells = sum(df[col].null_count() for col in df.columns)
    return (total_cells - null_cells) / total_cells * 100


def _analyze_columns(
    generated_df: pl.DataFrame,
    original_df: pl.DataFrame | None,
    report: QualityReport,
) -> tuple[list[float], list[float]]:
    uniqueness_scores: list[float] = []
    distribution_scores: list[float] = []
    n_rows = len(generated_df)

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

        if original_df is not None and col in original_df.columns:
            orig_series = original_df[col]
            dist_score = _compare_distributions(gen_series, orig_series)
            col_info["distribution_similarity"] = round(dist_score * 100, 1)
            distribution_scores.append(dist_score)

            test_result = _run_statistical_test(gen_series, orig_series, col)
            if test_result:
                col_info["statistical_test"] = test_result
                report.statistical_tests.append(test_result)

        report.column_details.append(col_info)

    return uniqueness_scores, distribution_scores


def _compute_schema_match(df: pl.DataFrame, expected_schema: dict | None) -> float:
    if not expected_schema:
        return 100.0
    matches = sum(
        1 for col, expected_type in expected_schema.items()
        if col in df.columns and _types_compatible(str(df[col].dtype), expected_type)
    )
    return matches / len(expected_schema) * 100


def _add_warnings(report: QualityReport, n_rows: int):
    if report.completeness < 95:
        report.warnings.append(f"Low completeness: {report.completeness:.0f}% (target >= 95%)")
    if report.distribution_score < 70:
        report.warnings.append(f"Distribution divergence detected: {report.distribution_score:.0f}%")
    if report.correlation_preservation < 60:
        report.warnings.append(f"Correlation structure poorly preserved: {report.correlation_preservation:.0f}%")
    for cd in report.column_details:
        if cd["unique_count"] == 1 and n_rows > 10:
            report.warnings.append(f"Column '{cd['column']}' has only 1 unique value")


def _score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    return "F"


def _run_statistical_test(gen: pl.Series, orig: pl.Series, col_name: str) -> dict | None:
    """Run KS test (numeric) or chi-squared test (categorical) between two series."""
    try:
        if gen.dtype in (pl.Int64, pl.Int32, pl.Int16, pl.Int8,
                         pl.Float64, pl.Float32, pl.UInt64, pl.UInt32):
            return _ks_test(gen, orig, col_name)
        else:
            return _chi_squared_test(gen, orig, col_name)
    except Exception:
        return None


def _ks_test(gen: pl.Series, orig: pl.Series, col_name: str) -> dict | None:
    gen_arr = gen.drop_nulls().cast(pl.Float64).to_numpy()
    orig_arr = orig.drop_nulls().cast(pl.Float64).to_numpy()
    if len(gen_arr) < 5 or len(orig_arr) < 5:
        return None

    statistic, p_value = scipy_stats.ks_2samp(gen_arr, orig_arr)
    return {
        "column": col_name,
        "test": "ks_2samp",
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "pass": bool(p_value > 0.05),
    }


def _chi_squared_test(gen: pl.Series, orig: pl.Series, col_name: str) -> dict | None:
    gen_counts = gen.drop_nulls().cast(pl.Utf8).value_counts()
    orig_counts = orig.drop_nulls().cast(pl.Utf8).value_counts()

    gen_freq = _series_to_freq(gen_counts, gen.name)
    orig_freq = _series_to_freq(orig_counts, orig.name)

    all_categories = sorted(set(gen_freq.keys()) | set(orig_freq.keys()))
    if len(all_categories) < 2:
        return None

    observed = [gen_freq.get(c, 0) for c in all_categories]
    expected = [orig_freq.get(c, 0) for c in all_categories]

    total_obs = sum(observed) or 1
    total_exp = sum(expected) or 1
    expected_scaled = [e / total_exp * total_obs for e in expected]

    if any(e < 1 for e in expected_scaled):
        return None

    statistic, p_value = scipy_stats.chisquare(observed, f_exp=expected_scaled)
    return {
        "column": col_name,
        "test": "chi_squared",
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "pass": bool(p_value > 0.05),
    }


def _series_to_freq(vc_df, series_name: str) -> dict:
    freq = {}
    for row in vc_df.to_dicts():
        val = row.get(series_name, row.get(vc_df.columns[0]))
        cnt = row.get("count", row.get("counts", 0))
        freq[str(val)] = cnt
    return freq


def _compute_correlation_preservation(gen_df: pl.DataFrame, orig_df: pl.DataFrame) -> float:
    """Compare correlation matrices of numeric columns. Returns 0-100."""
    numeric_cols = [
        col for col in gen_df.columns
        if col in orig_df.columns
        and gen_df[col].dtype in (pl.Int64, pl.Int32, pl.Float64, pl.Float32)
        and orig_df[col].dtype in (pl.Int64, pl.Int32, pl.Float64, pl.Float32)
    ]
    if len(numeric_cols) < 2:
        return 100.0

    try:
        gen_arr = gen_df.select(numeric_cols).drop_nulls().to_numpy().astype(float)
        orig_arr = orig_df.select(numeric_cols).drop_nulls().to_numpy().astype(float)
        if len(gen_arr) < 3 or len(orig_arr) < 3:
            return 100.0

        gen_corr = np.corrcoef(gen_arr, rowvar=False)
        orig_corr = np.corrcoef(orig_arr, rowvar=False)

        diff = gen_corr - orig_corr
        frobenius = np.linalg.norm(diff, "fro")
        max_frobenius = np.sqrt(2 * len(numeric_cols) ** 2)
        score = max(0, (1 - frobenius / max_frobenius)) * 100
        return float(score)
    except Exception:
        return 100.0


def _compute_dependency_score(gen_df: pl.DataFrame, orig_df: pl.DataFrame) -> float:
    """Measure preservation of categorical conditional distributions. Returns 0-100."""
    cat_cols = [
        col for col in gen_df.columns
        if col in orig_df.columns
        and gen_df[col].dtype in (pl.Utf8, pl.Categorical)
        and orig_df[col].dtype in (pl.Utf8, pl.Categorical)
    ]
    if len(cat_cols) < 2:
        return 100.0

    scores = []
    for i in range(min(len(cat_cols), 5)):
        for j in range(i + 1, min(len(cat_cols), 5)):
            score = _conditional_distribution_similarity(
                gen_df, orig_df, cat_cols[i], cat_cols[j]
            )
            if score is not None:
                scores.append(score)

    return (sum(scores) / len(scores) * 100) if scores else 100.0


def _conditional_distribution_similarity(gen_df: pl.DataFrame, orig_df: pl.DataFrame,
                                         col_a: str, col_b: str) -> float | None:
    try:
        gen_sub = gen_df.select([col_a, col_b]).drop_nulls()
        orig_sub = orig_df.select([col_a, col_b]).drop_nulls()
        if len(gen_sub) < 10 or len(orig_sub) < 10:
            return None

        shared_vals = (
            set(gen_sub[col_a].cast(pl.Utf8).unique().to_list())
            & set(orig_sub[col_a].cast(pl.Utf8).unique().to_list())
        )
        if not shared_vals:
            return 0.0

        similarities = []
        for val in list(shared_vals)[:20]:
            gen_b = gen_sub.filter(pl.col(col_a).cast(pl.Utf8) == val)[col_b].cast(pl.Utf8)
            orig_b = orig_sub.filter(pl.col(col_a).cast(pl.Utf8) == val)[col_b].cast(pl.Utf8)
            if len(gen_b) < 2 or len(orig_b) < 2:
                continue

            gen_vals = set(gen_b.to_list())
            orig_vals = set(orig_b.to_list())
            intersection = len(gen_vals & orig_vals)
            union = len(gen_vals | orig_vals)
            similarities.append(intersection / union if union > 0 else 0)

        return sum(similarities) / len(similarities) if similarities else None
    except Exception:
        return None


# Distribution comparison helpers

def _compare_distributions(gen: pl.Series, orig: pl.Series) -> float:
    try:
        if gen.dtype in (pl.Int64, pl.Int32, pl.Int16, pl.Int8,
                         pl.Float64, pl.Float32, pl.UInt64, pl.UInt32):
            return _compare_numeric(gen, orig)
        else:
            return _compare_categorical(gen, orig)
    except Exception:
        return 0.5


def _compare_numeric(gen: pl.Series, orig: pl.Series) -> float:
    gen_clean = gen.drop_nulls().cast(pl.Float64).to_numpy()
    orig_clean = orig.drop_nulls().cast(pl.Float64).to_numpy()
    if len(gen_clean) == 0 or len(orig_clean) == 0:
        return 0.0

    lo = min(gen_clean.min(), orig_clean.min())
    hi = max(gen_clean.max(), orig_clean.max())
    if lo == hi:
        return 1.0

    bins = np.linspace(lo, hi, 21)
    gen_hist, _ = np.histogram(gen_clean, bins=bins, density=True)
    orig_hist, _ = np.histogram(orig_clean, bins=bins, density=True)

    gen_sum = gen_hist.sum()
    orig_sum = orig_hist.sum()
    if gen_sum > 0:
        gen_hist = gen_hist / gen_sum
    if orig_sum > 0:
        orig_hist = orig_hist / orig_sum

    overlap = np.minimum(gen_hist, orig_hist).sum()
    return float(min(overlap, 1.0))


def _compare_categorical(gen: pl.Series, orig: pl.Series) -> float:
    gen_vals = set(gen.drop_nulls().cast(pl.Utf8).to_list())
    orig_vals = set(orig.drop_nulls().cast(pl.Utf8).to_list())

    if not gen_vals and not orig_vals:
        return 1.0
    if not gen_vals or not orig_vals:
        return 0.0

    intersection = len(gen_vals & orig_vals)
    union = len(gen_vals | orig_vals)
    jaccard = intersection / union if union > 0 else 0

    shared = gen_vals & orig_vals
    if not shared:
        return jaccard * 0.5

    gen_counts = gen.drop_nulls().cast(pl.Utf8).value_counts()
    orig_counts = orig.drop_nulls().cast(pl.Utf8).value_counts()

    gen_freq = _series_to_freq(gen_counts, gen.name)
    orig_freq = _series_to_freq(orig_counts, orig.name)

    gen_total = sum(gen_freq.values()) or 1
    orig_total = sum(orig_freq.values()) or 1

    freq_sim = sum(
        min(gen_freq.get(val, 0) / gen_total, orig_freq.get(val, 0) / orig_total)
        for val in shared
    )

    return jaccard * 0.4 + freq_sim * 0.6


def _types_compatible(actual: str, expected: str) -> bool:
    actual_lower = actual.lower()
    expected_lower = expected.lower()

    type_map = {
        "int64": ["int", "i64"],
        "float64": ["float", "f64"],
        "string": ["str", "utf8", "string", "categorical"],
        "date": ["date", "datetime"],
    }

    for aliases in type_map.values():
        if any(a in expected_lower for a in aliases) and any(a in actual_lower for a in aliases):
            return True

    return expected_lower in actual_lower
