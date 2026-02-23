"""
Privacy Scorecard — Distance to Closest Record (DCR) metric.

Measures how close synthetic records are to real records to flag
potential privacy leaks (exact or near-exact matches).
"""

import polars as pl
import numpy as np
from scipy.spatial.distance import cdist


class PrivacyScorecard:
    """Computes DCR between real and synthetic DataFrames."""

    @staticmethod
    def _prepare_matrix(df: pl.DataFrame) -> np.ndarray:
        """
        Convert a Polars DataFrame to a numeric numpy matrix.
        - Numeric columns: normalized to [0, 1]
        - String/categorical columns: label-encoded then normalized
        - Date columns: converted to ordinal integers then normalized
        """
        arrays = []
        for col in df.columns:
            dtype = str(df[col].dtype)

            if "Int" in dtype or "Float" in dtype:
                vals = df[col].cast(pl.Float64).fill_null(0.0).to_numpy()
                vmin, vmax = vals.min(), vals.max()
                if vmax > vmin:
                    vals = (vals - vmin) / (vmax - vmin)
                else:
                    vals = np.zeros_like(vals)
                arrays.append(vals)

            elif "Date" in dtype or "Datetime" in dtype:
                # Convert dates to ordinal
                try:
                    date_series = df[col].cast(pl.Date)
                    ordinals = np.array([
                        d.toordinal() if d is not None else 0
                        for d in date_series.to_list()
                    ], dtype=np.float64)
                    omin, omax = ordinals.min(), ordinals.max()
                    if omax > omin:
                        ordinals = (ordinals - omin) / (omax - omin)
                    else:
                        ordinals = np.zeros_like(ordinals)
                    arrays.append(ordinals)
                except Exception:
                    pass  # Skip unparseable date columns

            else:
                # String/categorical: label encode
                unique_vals = df[col].fill_null("__NULL__").unique().to_list()
                val_map = {v: i for i, v in enumerate(unique_vals)}
                encoded = np.array([
                    val_map.get(v, 0)
                    for v in df[col].fill_null("__NULL__").to_list()
                ], dtype=np.float64)
                emax = max(len(unique_vals) - 1, 1)
                encoded = encoded / emax
                arrays.append(encoded)

        if not arrays:
            return np.zeros((len(df), 1))

        return np.column_stack(arrays)

    def compute_dcr(self, real_df: pl.DataFrame, synthetic_df: pl.DataFrame) -> dict:
        """
        Compute Distance to Closest Record metrics.

        Returns a dict with:
        - min_dcr: minimum DCR across all synthetic records
        - mean_dcr: average DCR
        - median_dcr: median DCR
        - std_dcr: standard deviation of DCR
        - pct_exact_matches: % of synthetic records with DCR ≈ 0
        - risk_level: "High" / "Medium" / "Low"
        - dcr_values: array of all DCR values (for histogram)
        """
        # Use only shared columns
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

        # Sample if too large (performance guard)
        max_rows = 5000
        if len(real_sub) > max_rows:
            real_sub = real_sub.sample(max_rows, seed=42)
        if len(syn_sub) > max_rows:
            syn_sub = syn_sub.sample(max_rows, seed=42)

        real_matrix = self._prepare_matrix(real_sub)
        syn_matrix = self._prepare_matrix(syn_sub)

        # Compute pairwise Euclidean distances
        distances = cdist(syn_matrix, real_matrix, metric="euclidean")

        # For each synthetic record, find the closest real record
        min_distances = distances.min(axis=1)

        min_dcr = float(np.min(min_distances))
        mean_dcr = float(np.mean(min_distances))
        median_dcr = float(np.median(min_distances))
        std_dcr = float(np.std(min_distances))

        # "Exact match" threshold = DCR < 0.01 (nearly identical)
        exact_threshold = 0.01
        n_exact = int(np.sum(min_distances < exact_threshold))
        pct_exact = round(100 * n_exact / len(min_distances), 2) if len(min_distances) > 0 else 0

        # Risk assessment
        if pct_exact > 5 or min_dcr < 0.005:
            risk_level = "High"
        elif pct_exact > 1 or min_dcr < 0.02:
            risk_level = "Medium"
        else:
            risk_level = "Low"

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
