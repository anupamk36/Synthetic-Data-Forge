"""Privacy Audit page — DCR metrics and risk assessment."""

import logging

import numpy as np
import polars as pl
import streamlit as st

from app.components import empty_state, hero, metric_row, section, status_pill
from app.ui_schema import read_full_dataframe
from core.privacy import PrivacyScorecard

logger = logging.getLogger(__name__)


def render_privacy_page():
    """Full privacy audit page with upload + scorecard."""

    hero(
        "🛡️  Privacy Audit",
        "Compare synthetic data against the original to measure re-identification risk using Distance to Closest Record (DCR).",
    )

    # ---- Upload section ----
    section("📂", "Upload Datasets")
    col_real, col_syn = st.columns(2)

    with col_real:
        with st.container(border=True):
            st.markdown("**Real (original) data**")
            real_file = st.file_uploader(
                "Upload real data",
                type=["csv", "parquet", "json", "jsonl"],
                key="priv_real",
                label_visibility="collapsed",
            )
            real_df = None
            if real_file:
                real_df = read_full_dataframe(real_file)
                st.caption(f"{len(real_df):,} rows × {len(real_df.columns)} columns")

    with col_syn:
        with st.container(border=True):
            st.markdown("**Synthetic data**")
            syn_source = st.radio(
                "Source",
                ["Upload file", "Use last generated"],
                horizontal=True,
                key="priv_source",
            )
            syn_df = None
            if syn_source == "Upload file":
                syn_file = st.file_uploader(
                    "Upload synthetic data",
                    type=["csv", "parquet", "json", "jsonl"],
                    key="priv_syn",
                    label_visibility="collapsed",
                )
                if syn_file:
                    syn_df = read_full_dataframe(syn_file)
                    st.caption(f"{len(syn_df):,} rows × {len(syn_df.columns)} columns")
            else:
                if "generated_df" in st.session_state:
                    syn_df = st.session_state.generated_df
                    st.caption(
                        f"{len(syn_df):,} rows × {len(syn_df.columns)} columns (from last generation)"
                    )
                else:
                    st.info("No data generated yet. Generate data in the Single Table page first.")

    # ---- Check readiness ----
    if real_df is None or syn_df is None:
        if real_df is not None or syn_df is not None:
            st.info("Upload both real and synthetic datasets to run the audit.")
        else:
            empty_state(
                "🛡️",
                "Upload real & synthetic datasets to begin",
                "We will compute DCR metrics and assess re-identification risk.",
            )
        return

    # ---- Run analysis ----
    st.divider()
    section("📊", "Analysis Results")

    scorecard = PrivacyScorecard()
    with st.spinner("Computing DCR metrics…"):
        results = scorecard.compute_dcr(real_df, syn_df)

    if results.get("error"):
        st.error(f"⚠️ {results['error']}")
        return

    # Risk badge
    risk = results["risk_level"]
    if risk == "Low":
        status_pill("🟢  Privacy Risk: LOW — synthetic data is well-differentiated", "green")
    elif risk == "Medium":
        status_pill("🟡  Privacy Risk: MEDIUM — some records are close to real data", "amber")
    else:
        status_pill("🔴  Privacy Risk: HIGH — near-copies detected!", "red")

    st.write("")  # spacer

    # Metric cards
    metric_row([
        (f"{results['min_dcr']:.4f}", "Min DCR"),
        (f"{results['mean_dcr']:.4f}", "Mean DCR"),
        (f"{results['median_dcr']:.4f}", "Median DCR"),
        (f"{results['pct_exact_matches']}%", "Exact Matches"),
    ])

    # Histogram
    st.write("")
    section("📈", "DCR Distribution")
    dcr_values = results.get("dcr_values", [])
    if dcr_values:
        hist_values, bin_edges = np.histogram(dcr_values, bins=30)
        chart_data = pl.DataFrame({
            "DCR Range": [f"{bin_edges[i]:.3f}" for i in range(len(hist_values))],
            "Count": hist_values.tolist(),
        })
        st.bar_chart(chart_data, x="DCR Range", y="Count")

    # Detailed metrics
    with st.expander("📋  Detailed Metrics"):
        st.json({
            "min_dcr": results["min_dcr"],
            "mean_dcr": results["mean_dcr"],
            "median_dcr": results["median_dcr"],
            "std_dcr": results["std_dcr"],
            "pct_exact_matches": results["pct_exact_matches"],
            "risk_level": results["risk_level"],
            "real_rows_analyzed": len(real_df),
            "synthetic_rows_analyzed": len(syn_df),
        })
