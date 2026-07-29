"""
Privacy Scorecard — privacy metrics for synthetic data evaluation.

Includes Distance to Closest Record (DCR), k-anonymity, l-diversity,
epsilon estimation, and compliance reporting.
"""

import logging
import polars as pl
import numpy as np
from scipy.spatial.distance import cdist

from core import config
from core.exceptions import PrivacyError

logger = logging.getLogger(__name__)


class PrivacyScorecard:
    """Privacy metrics: DCR, k-anonymity, l-diversity, epsilon estimation."""

    @staticmethod
    def _prepare_matrices(real_df: pl.DataFrame, syn_df: pl.DataFrame):
        """Convert both DataFrames to numeric numpy matrices using shared encoding.

        String columns are label-encoded using the union of values from both
        DataFrames so the same string always maps to the same number.
        Numeric columns are normalized using the combined min/max range.
        """
        real_arrays = []
        syn_arrays = []

        for col in real_df.columns:
            dtype = str(real_df[col].dtype)

            if "Int" in dtype or "Float" in dtype:
                r_vals = real_df[col].cast(pl.Float64).fill_null(0.0).to_numpy()
                s_vals = syn_df[col].cast(pl.Float64).fill_null(0.0).to_numpy()
                combined_min = min(r_vals.min(), s_vals.min())
                combined_max = max(r_vals.max(), s_vals.max())
                if combined_max > combined_min:
                    r_vals = (r_vals - combined_min) / (combined_max - combined_min)
                    s_vals = (s_vals - combined_min) / (combined_max - combined_min)
                else:
                    r_vals = np.zeros_like(r_vals)
                    s_vals = np.zeros_like(s_vals)
                real_arrays.append(r_vals)
                syn_arrays.append(s_vals)

            elif "Date" in dtype or "Datetime" in dtype:
                try:
                    r_dates = real_df[col].cast(pl.Date)
                    s_dates = syn_df[col].cast(pl.Date)
                    r_ord = np.array([
                        d.toordinal() if d is not None else np.nan
                        for d in r_dates.to_list()
                    ], dtype=np.float64)
                    s_ord = np.array([
                        d.toordinal() if d is not None else np.nan
                        for d in s_dates.to_list()
                    ], dtype=np.float64)

                    all_ord = np.concatenate([r_ord, s_ord])
                    valid = all_ord[~np.isnan(all_ord)]
                    if len(valid) > 0:
                        median_val = np.median(valid)
                        r_ord[np.isnan(r_ord)] = median_val
                        s_ord[np.isnan(s_ord)] = median_val
                        omin, omax = valid.min(), valid.max()
                        if omax > omin:
                            r_ord = (r_ord - omin) / (omax - omin)
                            s_ord = (s_ord - omin) / (omax - omin)
                        else:
                            r_ord = np.zeros_like(r_ord)
                            s_ord = np.zeros_like(s_ord)
                        real_arrays.append(r_ord)
                        syn_arrays.append(s_ord)
                except Exception:
                    logger.debug("Skipping unparseable date column '%s'", col)

            else:
                r_strs = real_df[col].fill_null("__NULL__").cast(pl.Utf8).to_list()
                s_strs = syn_df[col].fill_null("__NULL__").cast(pl.Utf8).to_list()
                all_vals = sorted(set(r_strs) | set(s_strs))
                val_map = {v: i for i, v in enumerate(all_vals)}
                emax = max(len(all_vals) - 1, 1)

                r_enc = np.array([val_map[v] for v in r_strs], dtype=np.float64) / emax
                s_enc = np.array([val_map[v] for v in s_strs], dtype=np.float64) / emax
                real_arrays.append(r_enc)
                syn_arrays.append(s_enc)

        if not real_arrays:
            return np.zeros((len(real_df), 1)), np.zeros((len(syn_df), 1))

        return np.column_stack(real_arrays), np.column_stack(syn_arrays)

    def compute_dcr(self, real_df: pl.DataFrame, synthetic_df: pl.DataFrame) -> dict:
        """Compute Distance to Closest Record metrics."""
        shared_cols = [c for c in real_df.columns if c in synthetic_df.columns]
        if not shared_cols:
            return {
                "min_dcr": None,
                "mean_dcr": None,
                "median_dcr": None,
                "std_dcr": None,
                "pct_exact_matches": None,
                "risk_level": "Unknown",
                "dcr_values": [],
                "error": "No shared columns between real and synthetic data.",
            }

        real_sub = real_df.select(shared_cols)
        syn_sub = synthetic_df.select(shared_cols)

        max_rows = config.DCR_MAX_ROWS
        if len(real_sub) > max_rows:
            real_sub = real_sub.sample(max_rows, seed=42)
        if len(syn_sub) > max_rows:
            syn_sub = syn_sub.sample(max_rows, seed=42)

        real_matrix, syn_matrix = self._prepare_matrices(real_sub, syn_sub)

        distances = cdist(syn_matrix, real_matrix, metric="euclidean")
        min_distances = distances.min(axis=1)

        min_dcr = float(np.min(min_distances))
        mean_dcr = float(np.mean(min_distances))
        median_dcr = float(np.median(min_distances))
        std_dcr = float(np.std(min_distances))

        exact_threshold = config.DCR_EXACT_THRESHOLD
        n_exact = int(np.sum(min_distances < exact_threshold))
        pct_exact = round(100 * n_exact / len(min_distances), 2) if len(min_distances) > 0 else 0

        if pct_exact > config.DCR_HIGH_RISK_PCT or min_dcr < config.DCR_HIGH_RISK_MIN:
            risk_level = "High"
        elif pct_exact > config.DCR_MEDIUM_RISK_PCT or min_dcr < config.DCR_MEDIUM_RISK_MIN:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        logger.info(
            "DCR computed: risk=%s, min=%.6f, mean=%.6f, exact_match_pct=%.2f%%",
            risk_level, min_dcr, mean_dcr, pct_exact,
        )

        return {
            "min_dcr": round(min_dcr, 6),
            "mean_dcr": round(mean_dcr, 6),
            "median_dcr": round(median_dcr, 6),
            "std_dcr": round(std_dcr, 6),
            "pct_exact_matches": pct_exact,
            "risk_level": risk_level,
            "dcr_values": min_distances.tolist(),
            "error": None,
        }

    # ──────────────────────────────────────────────────────────
    # k-Anonymity
    # ──────────────────────────────────────────────────────────
    def compute_k_anonymity(
        self, df: pl.DataFrame, quasi_identifiers: list[str]
    ) -> dict:
        """Compute k-anonymity metrics for *df* over the given quasi-identifiers.

        Returns a dict with min_k, mean_group_size, vulnerable_groups (size 1),
        total_groups, and a group_size_histogram (top 10 sizes by frequency).
        """
        missing = [c for c in quasi_identifiers if c not in df.columns]
        if missing:
            raise PrivacyError(f"Columns not found in DataFrame: {missing}")

        grouped = df.group_by(quasi_identifiers).agg(pl.len().alias("_count"))
        counts = grouped["_count"].to_numpy()

        min_k = int(counts.min())
        mean_group_size = float(counts.mean())
        vulnerable_groups = int(np.sum(counts == 1))
        total_groups = len(counts)

        # Histogram: size -> number of groups with that size (top 10)
        unique_sizes, size_counts = np.unique(counts, return_counts=True)
        order = np.argsort(-size_counts)[:10]
        group_size_histogram = {
            int(unique_sizes[i]): int(size_counts[i]) for i in order
        }

        logger.info(
            "k-anonymity: min_k=%d, mean_group=%.2f, vulnerable=%d/%d",
            min_k, mean_group_size, vulnerable_groups, total_groups,
        )

        return {
            "min_k": min_k,
            "mean_group_size": round(mean_group_size, 4),
            "vulnerable_groups": vulnerable_groups,
            "total_groups": total_groups,
            "group_size_histogram": group_size_histogram,
        }

    # ──────────────────────────────────────────────────────────
    # l-Diversity
    # ──────────────────────────────────────────────────────────
    def compute_l_diversity(
        self,
        df: pl.DataFrame,
        quasi_identifiers: list[str],
        sensitive_col: str,
    ) -> dict:
        """Compute l-diversity for equivalence classes defined by *quasi_identifiers*.

        For each group, counts distinct values of *sensitive_col*.
        Returns min_l, mean_l, and vulnerable_groups (groups with l == 1).
        """
        missing = [c for c in quasi_identifiers if c not in df.columns]
        if sensitive_col not in df.columns:
            missing.append(sensitive_col)
        if missing:
            raise PrivacyError(f"Columns not found in DataFrame: {missing}")

        grouped = df.group_by(quasi_identifiers).agg(
            pl.col(sensitive_col).n_unique().alias("_l")
        )
        l_values = grouped["_l"].to_numpy()

        min_l = int(l_values.min())
        mean_l = float(l_values.mean())
        vulnerable_groups = int(np.sum(l_values == 1))

        logger.info(
            "l-diversity: min_l=%d, mean_l=%.2f, vulnerable=%d",
            min_l, mean_l, vulnerable_groups,
        )

        return {
            "min_l": min_l,
            "mean_l": round(mean_l, 4),
            "vulnerable_groups": vulnerable_groups,
        }

    # ──────────────────────────────────────────────────────────
    # Epsilon estimation
    # ──────────────────────────────────────────────────────────
    def estimate_epsilon(
        self, real_df: pl.DataFrame, syn_df: pl.DataFrame
    ) -> dict:
        """Estimate differential-privacy epsilon via histogram density comparison.

        For each shared numeric column, builds 50-bin normalised histograms and
        computes epsilon_col = max(|log(P_syn / P_real)|) with Laplace smoothing.
        Overall epsilon = max across columns.
        """
        shared_numeric = [
            c for c in real_df.columns
            if c in syn_df.columns
            and ("Int" in str(real_df[c].dtype) or "Float" in str(real_df[c].dtype))
        ]

        if not shared_numeric:
            return {
                "estimated_epsilon": 0.0,
                "interpretation": "Strong",
                "per_column": {},
            }

        per_column: dict[str, float] = {}
        smoothing = 1e-10

        for col in shared_numeric:
            r_vals = real_df[col].cast(pl.Float64).fill_null(0.0).to_numpy()
            s_vals = syn_df[col].cast(pl.Float64).fill_null(0.0).to_numpy()

            combined_min = min(r_vals.min(), s_vals.min())
            combined_max = max(r_vals.max(), s_vals.max())

            bins = np.linspace(combined_min, combined_max, 51)

            r_hist, _ = np.histogram(r_vals, bins=bins, density=False)
            s_hist, _ = np.histogram(s_vals, bins=bins, density=False)

            # Normalise to probability distributions
            r_prob = r_hist / r_hist.sum() + smoothing
            s_prob = s_hist / s_hist.sum() + smoothing

            eps_col = float(np.max(np.abs(np.log(s_prob / r_prob))))
            per_column[col] = round(eps_col, 6)

        estimated_epsilon = max(per_column.values())

        if estimated_epsilon < 1.0:
            interpretation = "Strong"
        elif estimated_epsilon <= 3.0:
            interpretation = "Moderate"
        else:
            interpretation = "Weak"

        logger.info(
            "Epsilon estimate: %.4f (%s)", estimated_epsilon, interpretation,
        )

        return {
            "estimated_epsilon": round(estimated_epsilon, 6),
            "interpretation": interpretation,
            "per_column": per_column,
        }

    # ──────────────────────────────────────────────────────────
    # Compliance report
    # ──────────────────────────────────────────────────────────
    def generate_compliance_report(
        self,
        real_df: pl.DataFrame,
        syn_df: pl.DataFrame,
        quasi_identifiers: list[str] | None = None,
        sensitive_col: str | None = None,
    ) -> dict:
        """Generate a full privacy compliance report.

        Combines DCR, k-anonymity, l-diversity, and epsilon estimation into a
        single assessment with an overall risk level and recommendations.
        """
        dcr = self.compute_dcr(real_df, syn_df)
        epsilon = self.estimate_epsilon(real_df, syn_df)

        k_anonymity = None
        if quasi_identifiers:
            k_anonymity = self.compute_k_anonymity(syn_df, quasi_identifiers)

        l_diversity = None
        if quasi_identifiers and sensitive_col:
            l_diversity = self.compute_l_diversity(
                syn_df, quasi_identifiers, sensitive_col
            )

        # Determine overall risk
        dcr_risk = dcr.get("risk_level", "Unknown")
        eps_val = epsilon.get("estimated_epsilon", 0.0)
        min_k = k_anonymity["min_k"] if k_anonymity else None

        if (
            dcr_risk == "High"
            or eps_val > 3.0
            or (min_k is not None and min_k < 2)
        ):
            overall_risk = "High"
        elif dcr_risk == "Medium" or eps_val > 1.0:
            overall_risk = "Medium"
        else:
            overall_risk = "Low"

        # Recommendations
        recommendations: list[str] = []
        if dcr_risk == "High":
            recommendations.append(
                "High DCR risk — synthetic records are too close to real data. "
                "Add noise or increase diversity."
            )
        if eps_val > 3.0:
            recommendations.append(
                "Weak epsilon — distributions diverge significantly. "
                "Consider retraining the generator with tighter constraints."
            )
        elif eps_val > 1.0:
            recommendations.append(
                "Moderate epsilon — some distributional leakage detected. "
                "Review numeric column distributions."
            )
        if min_k is not None and min_k < 2:
            recommendations.append(
                "k-anonymity violated (k < 2) — some records are uniquely "
                "identifiable. Generalise or suppress quasi-identifiers."
            )
        if l_diversity and l_diversity["vulnerable_groups"] > 0:
            recommendations.append(
                "l-diversity concern — some equivalence classes have "
                "homogeneous sensitive values. Diversify sensitive attributes."
            )
        if not recommendations:
            recommendations.append("All privacy metrics are within acceptable bounds.")

        # Strip non-serialisable dcr_values from DCR result
        dcr_clean = {k: v for k, v in dcr.items() if k != "dcr_values"}

        return {
            "dcr": dcr_clean,
            "k_anonymity": k_anonymity,
            "l_diversity": l_diversity,
            "epsilon": epsilon,
            "overall_risk": overall_risk,
            "recommendations": recommendations,
            "compliant": overall_risk == "Low",
        }
