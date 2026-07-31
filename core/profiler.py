"""
Statistical Profiler for Synthetic-Data-Forge.

Analyzes uploaded real data to extract per-column statistics,
cross-column correlations, conditional distributions, and constraints.
The resulting DataProfile guides correlated data generation.
"""

import json
import logging
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

NUMERIC_TYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)

CORRELATION_THRESHOLD = 0.5
CRAMERS_V_THRESHOLD = 0.3


@dataclass
class ColumnStats:
    name: str
    dtype: str
    null_rate: float = 0.0
    unique_rate: float = 0.0
    unique_count: int = 0
    is_numeric: bool = False
    is_categorical: bool = False
    is_date: bool = False
    # Numeric stats
    min_val: float | None = None
    max_val: float | None = None
    mean: float | None = None
    std: float | None = None
    percentiles: dict | None = None
    distribution_type: str = "unknown"
    # Categorical stats
    top_values: list[dict] | None = None
    cardinality: int = 0
    entropy: float = 0.0


@dataclass
class CorrelationEntry:
    col_a: str
    col_b: str
    method: str  # "pearson", "cramers_v", "mutual_info"
    value: float
    significant: bool = True


@dataclass
class ConditionalDistribution:
    condition_col: str
    target_col: str
    table: dict = field(default_factory=dict)


@dataclass
class Constraint:
    constraint_type: str  # "unique", "functional_dep", "range", "not_null"
    columns: list[str] = field(default_factory=list)
    details: str = ""


@dataclass
class DataProfile:
    row_count: int = 0
    col_count: int = 0
    column_stats: list[ColumnStats] = field(default_factory=list)
    correlations: list[CorrelationEntry] = field(default_factory=list)
    conditional_distributions: list[ConditionalDistribution] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    def summary_for_llm(self) -> dict:
        """Compact summary suitable for including in LLM prompts."""
        key_correlations = []
        for c in self.correlations:
            if c.significant and abs(c.value) > 0.3:
                key_correlations.append(f"{c.col_a} and {c.col_b} are correlated ({c.method}={c.value:.2f})")

        constraint_strs = []
        for c in self.constraints:
            constraint_strs.append(f"{c.constraint_type}: {', '.join(c.columns)} — {c.details}")

        col_summaries = {}
        for cs in self.column_stats:
            summary = {"type": cs.dtype}
            if cs.is_numeric and cs.min_val is not None:
                summary["range"] = f"{cs.min_val:.2f} to {cs.max_val:.2f}"
                summary["mean"] = f"{cs.mean:.2f}"
            elif cs.is_categorical and cs.top_values:
                summary["top_values"] = [v["value"] for v in cs.top_values[:5]]
            col_summaries[cs.name] = summary

        return {
            "row_count": self.row_count,
            "columns": col_summaries,
            "key_correlations": key_correlations,
            "constraints": constraint_strs,
        }


def profile_dataframe(df: pl.DataFrame) -> DataProfile:
    """Analyze a DataFrame and return a DataProfile."""
    profile = DataProfile(row_count=len(df), col_count=len(df.columns))

    if len(df) == 0:
        return profile

    numeric_cols = []
    categorical_cols = []

    for col_name in df.columns:
        series = df[col_name]
        cs = _profile_column(series)
        profile.column_stats.append(cs)

        if cs.is_numeric:
            numeric_cols.append(col_name)
        elif cs.is_categorical:
            categorical_cols.append(col_name)

    profile.correlations = _compute_correlations(df, numeric_cols, categorical_cols)

    profile.conditional_distributions = _compute_conditional_distributions(df, profile.correlations)

    profile.constraints = _detect_constraints(df, profile.column_stats)

    logger.info(
        "Profiled %d rows x %d cols: %d correlations, %d conditionals, %d constraints",
        profile.row_count,
        profile.col_count,
        len(profile.correlations),
        len(profile.conditional_distributions),
        len(profile.constraints),
    )
    return profile


def _profile_column(series: pl.Series) -> ColumnStats:
    n = len(series)
    cs = ColumnStats(
        name=series.name,
        dtype=str(series.dtype),
        null_rate=series.null_count() / n if n > 0 else 0,
        unique_count=series.n_unique(),
        unique_rate=series.n_unique() / n if n > 0 else 0,
    )

    if series.dtype in NUMERIC_TYPES:
        cs.is_numeric = True
        clean = series.drop_nulls().cast(pl.Float64)
        if len(clean) > 0:
            arr = clean.to_numpy()
            cs.min_val = float(arr.min())
            cs.max_val = float(arr.max())
            cs.mean = float(arr.mean())
            cs.std = float(arr.std()) if len(arr) > 1 else 0.0
            cs.percentiles = {
                "p5": float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            }
            cs.distribution_type = _detect_distribution_type(arr)

    elif series.dtype in (pl.Date, pl.Datetime):
        cs.is_date = True
        clean = series.drop_nulls()
        if len(clean) > 0:
            cs.min_val = str(clean.min())
            cs.max_val = str(clean.max())

    else:
        cs.is_categorical = True
        clean = series.drop_nulls().cast(pl.Utf8)
        if len(clean) > 0:
            cs.cardinality = clean.n_unique()
            vc = clean.value_counts().sort("count", descending=True)
            top_n = min(20, len(vc))
            cs.top_values = []
            for row in vc.head(top_n).to_dicts():
                val = row.get(series.name, row.get(vc.columns[0]))
                cnt = row.get("count", row.get("counts", 0))
                cs.top_values.append({"value": str(val), "count": int(cnt), "pct": round(cnt / n * 100, 1)})
            cs.entropy = _compute_entropy(clean)

    return cs


def _detect_distribution_type(arr: np.ndarray) -> str:
    if len(arr) < 8:
        return "unknown"
    _, p_normal = scipy_stats.normaltest(arr)
    if p_normal > 0.05:
        return "normal"
    skewness = scipy_stats.skew(arr)
    if abs(skewness) < 0.5:
        return "symmetric"
    elif skewness > 0:
        return "right_skewed"
    else:
        return "left_skewed"


def _compute_entropy(series: pl.Series) -> float:
    vc = series.value_counts()
    counts = []
    for row in vc.to_dicts():
        counts.append(row.get("count", row.get("counts", 0)))
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _compute_correlations(
    df: pl.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> list[CorrelationEntry]:
    correlations = []

    # Pearson for numeric pairs
    if len(numeric_cols) >= 2:
        numeric_df = df.select(numeric_cols).drop_nulls()
        if len(numeric_df) > 2:
            arr = numeric_df.to_numpy().astype(float)
            corr_matrix = np.corrcoef(arr, rowvar=False)
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    r = corr_matrix[i, j]
                    if not np.isnan(r):
                        correlations.append(
                            CorrelationEntry(
                                col_a=numeric_cols[i],
                                col_b=numeric_cols[j],
                                method="pearson",
                                value=round(float(r), 3),
                                significant=bool(abs(r) >= CORRELATION_THRESHOLD),
                            )
                        )

    # Cramer's V for categorical pairs
    if len(categorical_cols) >= 2:
        for i in range(len(categorical_cols)):
            for j in range(i + 1, len(categorical_cols)):
                v = _cramers_v(df, categorical_cols[i], categorical_cols[j])
                if v is not None:
                    correlations.append(
                        CorrelationEntry(
                            col_a=categorical_cols[i],
                            col_b=categorical_cols[j],
                            method="cramers_v",
                            value=round(v, 3),
                            significant=v >= CRAMERS_V_THRESHOLD,
                        )
                    )

    return correlations


def _cramers_v(df: pl.DataFrame, col_a: str, col_b: str) -> float | None:
    """Compute Cramer's V between two categorical columns."""
    try:
        sub = df.select([col_a, col_b]).drop_nulls()
        if len(sub) < 5:
            return None

        a_vals = sub[col_a].cast(pl.Utf8).to_list()
        b_vals = sub[col_b].cast(pl.Utf8).to_list()

        a_cats = sorted(set(a_vals))
        b_cats = sorted(set(b_vals))
        if len(a_cats) < 2 or len(b_cats) < 2:
            return None

        a_idx = {v: i for i, v in enumerate(a_cats)}
        b_idx = {v: i for i, v in enumerate(b_cats)}

        contingency = np.zeros((len(a_cats), len(b_cats)), dtype=int)
        for av, bv in zip(a_vals, b_vals, strict=False):
            contingency[a_idx[av], b_idx[bv]] += 1

        chi2, _, _, _ = scipy_stats.chi2_contingency(contingency)
        n = len(sub)
        min_dim = min(len(a_cats), len(b_cats)) - 1
        if min_dim == 0 or n == 0:
            return None
        return float(np.sqrt(chi2 / (n * min_dim)))
    except Exception:
        return None


def _compute_conditional_distributions(
    df: pl.DataFrame, correlations: list[CorrelationEntry]
) -> list[ConditionalDistribution]:
    """Build conditional probability tables for significantly correlated column pairs."""
    conditionals = []

    significant = [c for c in correlations if c.significant]
    for corr in significant[:20]:  # cap to avoid explosion
        try:
            sub = df.select([corr.col_a, corr.col_b]).drop_nulls()
            if len(sub) < 10:
                continue

            series_a = sub[corr.col_a]
            series_b = sub[corr.col_b]

            # For numeric pairs, bin both columns
            if series_a.dtype in NUMERIC_TYPES and series_b.dtype in NUMERIC_TYPES:
                a_arr = series_a.cast(pl.Float64).to_numpy()
                b_arr = series_b.cast(pl.Float64).to_numpy()
                a_bins = np.percentile(a_arr, [0, 25, 50, 75, 100])
                a_labels = [f"{a_bins[i]:.1f}-{a_bins[i+1]:.1f}" for i in range(4)]
                a_binned = np.digitize(a_arr, a_bins[1:-1])

                table = {}
                for bin_idx in range(4):
                    mask = a_binned == bin_idx
                    if mask.sum() > 0:
                        b_subset = b_arr[mask]
                        table[a_labels[bin_idx]] = {
                            "mean": round(float(b_subset.mean()), 2),
                            "std": round(float(b_subset.std()), 2),
                            "min": round(float(b_subset.min()), 2),
                            "max": round(float(b_subset.max()), 2),
                            "count": int(mask.sum()),
                        }

                conditionals.append(
                    ConditionalDistribution(
                        condition_col=corr.col_a,
                        target_col=corr.col_b,
                        table=table,
                    )
                )

            else:
                # Categorical: count frequencies of target given condition
                a_str = sub[corr.col_a].cast(pl.Utf8)
                b_str = sub[corr.col_b].cast(pl.Utf8)

                a_values = a_str.unique().to_list()
                if len(a_values) > 50:
                    continue

                table = {}
                for a_val in a_values:
                    mask = a_str == a_val
                    b_subset = b_str.filter(mask)
                    vc = b_subset.value_counts().sort("count", descending=True)
                    total = len(b_subset)
                    dist = {}
                    for row in vc.head(10).to_dicts():
                        val = str(row.get(corr.col_b, row.get(vc.columns[0])))
                        cnt = row.get("count", row.get("counts", 0))
                        dist[val] = round(cnt / total, 3) if total > 0 else 0
                    table[str(a_val)] = dist

                conditionals.append(
                    ConditionalDistribution(
                        condition_col=corr.col_a,
                        target_col=corr.col_b,
                        table=table,
                    )
                )

        except Exception as e:
            logger.debug("Skipping conditional for %s/%s: %s", corr.col_a, corr.col_b, e)

    return conditionals


def _detect_constraints(_df: pl.DataFrame, col_stats: list[ColumnStats]) -> list[Constraint]:
    constraints = []

    for cs in col_stats:
        # Unique constraint (potential PK)
        if cs.unique_rate > 0.99 and cs.unique_count > 10:
            constraints.append(
                Constraint(
                    constraint_type="unique",
                    columns=[cs.name],
                    details=f"{cs.unique_rate:.1%} unique values — possible primary key",
                )
            )

        # Not-null constraint
        if cs.null_rate < 1e-9:
            constraints.append(
                Constraint(
                    constraint_type="not_null",
                    columns=[cs.name],
                    details="No null values in source data",
                )
            )

        # Range constraint for numerics
        if cs.is_numeric and cs.min_val is not None:
            constraints.append(
                Constraint(
                    constraint_type="range",
                    columns=[cs.name],
                    details=f"Values range from {cs.min_val:.2f} to {cs.max_val:.2f}",
                )
            )

    return constraints
