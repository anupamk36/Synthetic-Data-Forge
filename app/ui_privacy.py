"""
Privacy Scorecard Dashboard UI.

Displays DCR metrics, risk assessment, and distribution histogram.
"""

import streamlit as st
import polars as pl
import numpy as np
from core.privacy import PrivacyScorecard


def render_privacy_scorecard(real_df: pl.DataFrame, synthetic_df: pl.DataFrame):
    """Render the full privacy scorecard dashboard."""
    scorecard = PrivacyScorecard()

    with st.spinner("Computing DCR metrics..."):
        results = scorecard.compute_dcr(real_df, synthetic_df)

    if results.get("error"):
        st.error(f"⚠️ {results['error']}")
        return

    # Risk badge
    risk = results["risk_level"]
    if risk == "Low":
        st.success("🟢 Privacy Risk: **LOW** — Synthetic data is well-differentiated from real data.")
    elif risk == "Medium":
        st.warning("🟡 Privacy Risk: **MEDIUM** — Some synthetic records are close to real data.")
    else:
        st.error("🔴 Privacy Risk: **HIGH** — Synthetic data contains near-copies of real records!")

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min DCR", f"{results['min_dcr']:.4f}")
    col2.metric("Mean DCR", f"{results['mean_dcr']:.4f}")
    col3.metric("Median DCR", f"{results['median_dcr']:.4f}")
    col4.metric("Exact Matches", f"{results['pct_exact_matches']}%")

    # Histogram
    st.subheader("DCR Distribution")
    dcr_values = results.get("dcr_values", [])
    if dcr_values:
        hist_values, bin_edges = np.histogram(dcr_values, bins=30)
        chart_data = pl.DataFrame({
            "DCR Range": [f"{bin_edges[i]:.3f}" for i in range(len(hist_values))],
            "Count": hist_values.tolist(),
        })
        st.bar_chart(chart_data, x="DCR Range", y="Count")

    # Details expander
    with st.expander("📊 Detailed Metrics"):
        st.json({
            "min_dcr": results["min_dcr"],
            "mean_dcr": results["mean_dcr"],
            "median_dcr": results["median_dcr"],
            "std_dcr": results["std_dcr"],
            "pct_exact_matches": results["pct_exact_matches"],
            "risk_level": results["risk_level"],
            "real_rows_analyzed": len(real_df),
            "synthetic_rows_analyzed": len(synthetic_df),
        })
