"""
Time-Travel Simulation UI.

Lets users configure temporal patterns (trends, spikes)
and preview volume distribution before generating.
"""

import streamlit as st
import polars as pl
from datetime import date, timedelta
from app.ui_schema import infer_schema, render_schema_editor
from core.time_travel import TimeTravelEngine
from core.sinks import LocalSink
import os


def render_time_travel_tab():
    """Render the time-travel simulation interface."""

    st.markdown("Generate synthetic data with realistic temporal patterns — growth trends, seasonal spikes, and more.")

    # --- File Upload ---
    uploaded_file = st.file_uploader(
        "Upload a sample file to infer schema",
        type=["csv", "parquet"],
        key="tt_upload",
    )

    if uploaded_file:
        schema, sample_df = infer_schema(uploaded_file)
        st.session_state.tt_schema = schema
        st.session_state.tt_sample = sample_df

    if "tt_schema" not in st.session_state or not st.session_state.tt_schema:
        st.info("📂 Upload a sample file to get started with time-travel simulation.")
        return

    # --- Schema Editor ---
    st.subheader("📋 Schema")
    edited_schema = render_schema_editor(st.session_state.tt_schema, key_prefix="tt")

    # --- Temporal Configuration ---
    st.divider()
    st.subheader("⏰ Temporal Configuration")

    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", value=date(2024, 1, 1), key="tt_start")
    end_date = col2.date_input("End Date", value=date(2024, 12, 31), key="tt_end")

    col3, col4, col5 = st.columns(3)
    frequency = col3.selectbox("Period Frequency", ["monthly", "weekly", "daily"], key="tt_freq")
    base_count = col4.number_input("Base Records Per Period", value=100, min_value=1, key="tt_base")
    trend_pct = col5.slider("Trend % Per Period", min_value=-20.0, max_value=20.0, value=0.0, step=0.5, key="tt_trend")

    # --- Spike Configuration ---
    st.subheader("📈 Volume Spikes")
    st.markdown("Add date-specific volume multipliers (e.g., Black Friday = 3× volume).")

    if "tt_spikes" not in st.session_state:
        st.session_state.tt_spikes = []

    spike_col1, spike_col2 = st.columns(2)
    spike_date = spike_col1.date_input("Spike Date", value=date(2024, 11, 29), key="tt_spike_date")
    spike_mult = spike_col2.number_input("Multiplier", value=3.0, min_value=1.0, step=0.5, key="tt_spike_mult")

    if st.button("➕ Add Spike", key="tt_add_spike"):
        st.session_state.tt_spikes.append((spike_date, spike_mult))

    if st.session_state.tt_spikes:
        for i, (sd, sm) in enumerate(st.session_state.tt_spikes):
            col_a, col_b = st.columns([4, 1])
            col_a.markdown(f"📌 **{sd.isoformat()}** — {sm}× volume")
            if col_b.button("🗑️", key=f"del_spike_{i}"):
                st.session_state.tt_spikes.pop(i)
                st.rerun()

    # --- Volume Preview Chart ---
    st.divider()
    st.subheader("📊 Volume Preview")

    engine = TimeTravelEngine()
    preview = engine.get_volume_preview(
        base_count=base_count,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        trend_pct=trend_pct,
        spike_dates=st.session_state.tt_spikes,
    )

    if preview:
        preview_df = pl.DataFrame(preview)
        total_records = sum(p["count"] for p in preview)
        st.metric("Total Records Across All Periods", f"{total_records:,}")
        st.bar_chart(preview_df, x="period", y="count")

    # --- Generation Settings ---
    st.divider()
    col_fmt, col_rpp = st.columns(2)
    output_format = col_fmt.selectbox("Output Format", ["parquet", "csv", "json"], key="tt_fmt")
    records_per_file = col_rpp.number_input("Records Per File", value=500, min_value=1, key="tt_rpp")

    output_path = st.text_input("Output Directory", value="./output_temporal", key="tt_output")

    # --- Generate ---
    if st.button("🚀 Generate Temporal Data", key="tt_gen"):
        with st.spinner("Simulating time-travel data..."):
            df = engine.generate_temporal(
                schema=edited_schema,
                base_count_per_period=base_count,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                trend_pct=trend_pct,
                spike_dates=st.session_state.tt_spikes,
            )

        resolved_path = os.path.abspath(os.path.expanduser(output_path))
        sink = LocalSink()
        # Partition by period
        written = sink.push(df, resolved_path, output_format, records_per_file, partitions=["_period"])

        st.success(f"✅ Generated {len(df):,} records across {len(preview)} periods to `{resolved_path}`")

        with st.expander("📊 Preview (first 20 rows)"):
            st.dataframe(df.head(20), use_container_width=True)
