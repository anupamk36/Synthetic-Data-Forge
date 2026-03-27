"""Data Quality Dashboard — statistical fidelity assessment of generated data."""

import logging

import polars as pl
import streamlit as st

from app.components import empty_state, hero, metric_row, section, status_pill
from app.ui_schema import read_full_dataframe
from core.quality import assess_quality

logger = logging.getLogger(__name__)


def render_quality_page():
    """Full data quality assessment page."""

    hero(
        "📈  Data Quality",
        "Measure the statistical fidelity, completeness, and consistency of your synthetic data.",
    )

    # ── Upload ──
    section("📂", "Upload Datasets")
    col_orig, col_gen = st.columns(2)

    with col_orig:
        with st.container(border=True):
            st.markdown("**Original (sample) data** *(optional)*")
            orig_file = st.file_uploader(
                "Upload original data for comparison",
                type=["csv", "parquet", "json", "jsonl"],
                key="qual_orig",
                label_visibility="collapsed",
            )
            orig_df = None
            if orig_file:
                orig_df = read_full_dataframe(orig_file)
                st.caption(f"{len(orig_df):,} rows × {len(orig_df.columns)} columns")

    with col_gen:
        with st.container(border=True):
            st.markdown("**Synthetic data**")
            gen_source = st.radio(
                "Source",
                ["Upload file", "Use last generated"],
                horizontal=True,
                key="qual_source",
            )
            gen_df = None
            if gen_source == "Upload file":
                gen_file = st.file_uploader(
                    "Upload synthetic data",
                    type=["csv", "parquet", "json", "jsonl"],
                    key="qual_gen",
                    label_visibility="collapsed",
                )
                if gen_file:
                    gen_df = read_full_dataframe(gen_file)
                    st.caption(f"{len(gen_df):,} rows × {len(gen_df.columns)} columns")
            else:
                if "generated_df" in st.session_state:
                    gen_df = st.session_state.generated_df
                    st.caption(f"{len(gen_df):,} rows × {len(gen_df.columns)} columns (from last generation)")
                else:
                    st.info("No data generated yet. Generate data in the Single Table page first.")

    if gen_df is None:
        empty_state("📈", "Provide synthetic data to assess", "Upload a file or generate data first.")
        return

    # ── Run assessment ──
    st.divider()
    section("📊", "Quality Assessment")

    with st.spinner("Analyzing data quality…"):
        report = assess_quality(gen_df, original_df=orig_df)

    # ── Overall score ──
    score = report.overall_score
    if score >= 80:
        status_pill(f"🟢  Overall Quality: {score:.0f}/100 — Excellent", "green")
    elif score >= 60:
        status_pill(f"🟡  Overall Quality: {score:.0f}/100 — Good", "amber")
    else:
        status_pill(f"🔴  Overall Quality: {score:.0f}/100 — Needs Improvement", "red")

    st.write("")

    metric_row([
        (f"{report.completeness:.0f}%", "Completeness"),
        (f"{report.uniqueness:.0f}%", "Avg Uniqueness"),
        (f"{report.schema_match:.0f}%", "Schema Match"),
        (f"{report.distribution_score:.0f}%", "Distribution Fidelity"),
    ])

    # ── Warnings ──
    if report.warnings:
        st.write("")
        section("⚠️", "Warnings")
        for w in report.warnings:
            st.warning(w)

    # ── Column details ──
    st.write("")
    section("🔎", "Column-Level Details")

    col_data = []
    for cd in report.column_details:
        row = {
            "Column": cd["column"],
            "Type": cd["type"],
            "Null %": cd["null_pct"],
            "Unique Count": cd["unique_count"],
            "Unique %": cd["unique_pct"],
        }
        if "distribution_similarity" in cd:
            row["Distribution Match %"] = cd["distribution_similarity"]
        col_data.append(row)

    if col_data:
        st.dataframe(pl.DataFrame(col_data), use_container_width=True, hide_index=True)

    # ── Distribution comparison charts ──
    if orig_df is not None:
        st.write("")
        section("📉", "Distribution Comparisons")
        st.caption("Comparing value distributions between original and synthetic data.")

        numeric_cols = [
            col for col in gen_df.columns
            if col in orig_df.columns
            and gen_df[col].dtype in (pl.Int64, pl.Int32, pl.Float64, pl.Float32)
        ]

        if numeric_cols:
            selected = st.selectbox("Select column", numeric_cols, key="qual_dist_col")
            if selected:
                _render_distribution_chart(orig_df[selected], gen_df[selected], selected)
        else:
            st.info("No shared numeric columns for distribution comparison.")


def _render_distribution_chart(orig: pl.Series, gen: pl.Series, col_name: str):
    """Render overlapping histogram for a numeric column."""
    import numpy as np

    orig_vals = orig.drop_nulls().cast(pl.Float64).to_numpy()
    gen_vals = gen.drop_nulls().cast(pl.Float64).to_numpy()

    if len(orig_vals) == 0 or len(gen_vals) == 0:
        st.info("Not enough data for comparison.")
        return

    lo = min(orig_vals.min(), gen_vals.min())
    hi = max(orig_vals.max(), gen_vals.max())
    bins = np.linspace(lo, hi, 25)

    orig_hist, _ = np.histogram(orig_vals, bins=bins)
    gen_hist, _ = np.histogram(gen_vals, bins=bins)

    chart_df = pl.DataFrame({
        "Bin": [f"{bins[i]:.1f}" for i in range(len(orig_hist))],
        "Original": orig_hist.tolist(),
        "Synthetic": gen_hist.tolist(),
    })

    st.bar_chart(chart_df, x="Bin", y=["Original", "Synthetic"])
