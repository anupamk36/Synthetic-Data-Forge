"""Time-Travel Simulation page — temporal patterns, trends, and spikes."""

import logging
import os
from datetime import date

import polars as pl
import streamlit as st

from app.components import (
    GenerationJob,
    download_buttons,
    empty_state,
    hero,
    metric_row,
    section,
)
from app.ui_schema import infer_schema, render_schema_editor
from core.config import validate_output_path
from core.exceptions import ForgeError
from core.sinks import LocalSink
from core.time_travel import TimeTravelEngine

logger = logging.getLogger(__name__)

_JOB_KEY = "tt_gen_job"


def render_time_travel_page():
    """Full time-travel simulation page."""

    hero(
        "⏰  Time Travel Simulator",
        "Generate synthetic data with realistic temporal patterns — growth trends, seasonal spikes, and more.",
    )

    # ---- Upload ----
    section("📂", "Upload Sample Data")
    uploaded_file = st.file_uploader(
        "Upload a sample file to infer schema",
        type=["csv", "parquet", "json", "jsonl"],
        key="tt_upload",
        label_visibility="collapsed",
    )

    if uploaded_file:
        schema, sample_df = infer_schema(uploaded_file)
        st.session_state.tt_schema = schema
        st.session_state.tt_sample = sample_df

    if "tt_schema" not in st.session_state or not st.session_state.tt_schema:
        empty_state(
            "⏰",
            "Upload a sample file to get started",
            "We will generate data across time periods with configurable trends.",
        )
        return

    # ---- Schema ----
    section("📋", "Schema")
    with st.container(border=True):
        edited_schema = render_schema_editor(st.session_state.tt_schema, key_prefix="tt")

    # ---- Temporal Configuration ----
    section("⏰", "Temporal Configuration")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        start_date = c1.date_input("Start Date", value=date(2024, 1, 1), key="tt_start")
        end_date = c2.date_input("End Date", value=date(2024, 12, 31), key="tt_end")

        c3, c4, c5 = st.columns(3)
        frequency = c3.selectbox("Frequency", ["monthly", "weekly", "daily"], key="tt_freq")
        base_count = c4.number_input("Base Records / Period", value=100, min_value=1, key="tt_base")
        trend_pct = c5.slider("Trend % / Period", -20.0, 20.0, 0.0, 0.5, key="tt_trend")

    # ---- Spikes ----
    section("📈", "Volume Spikes")
    if "tt_spikes" not in st.session_state:
        st.session_state.tt_spikes = []

    with st.container(border=True):
        sc1, sc2, sc3 = st.columns([2, 2, 1])
        spike_date = sc1.date_input("Spike Date", value=date(2024, 11, 29), key="tt_spike_date")
        spike_mult = sc2.number_input("Multiplier", value=3.0, min_value=1.0, step=0.5, key="tt_spike_mult")
        with sc3:
            st.write("")  # vertical spacer
            if st.button("➕ Add Spike", key="tt_add_spike", use_container_width=True):
                st.session_state.tt_spikes.append((spike_date, spike_mult))
                st.rerun()

    if st.session_state.tt_spikes:
        for i, (sd, sm) in enumerate(st.session_state.tt_spikes):
            ca, cb = st.columns([5, 1])
            ca.markdown(f"📌 **{sd.isoformat()}** — {sm}× volume")
            if cb.button("🗑️", key=f"del_spike_{i}"):
                st.session_state.tt_spikes.pop(i)
                st.rerun()

    # ---- Volume Preview ----
    st.divider()
    section("📊", "Volume Preview")
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

        metric_row([
            (f"{total_records:,}", "Total Records"),
            (str(len(preview)), "Periods"),
            (frequency.title(), "Frequency"),
        ])
        st.bar_chart(preview_df, x="period", y="count")

    # ---- Output Settings ----
    section("⚙️", "Output Settings")
    with st.container(border=True):
        cf, cr = st.columns(2)
        output_format = cf.selectbox("Output Format", ["parquet", "csv", "json"], key="tt_fmt")
        records_per_file = cr.number_input("Records Per File", value=500, min_value=1, key="tt_rpp")
        output_path = st.text_input("Output Directory", value="./output_temporal", key="tt_output")

    # ---- Generate / Stop ----
    st.divider()

    if _JOB_KEY not in st.session_state:
        st.session_state[_JOB_KEY] = None

    job: GenerationJob | None = st.session_state[_JOB_KEY]
    is_running = job is not None and job.status == "running"

    btn_col, stop_col, reset_col = st.columns([2, 1, 1])
    with btn_col:
        if st.button(
            "🚀  Generate Temporal Data",
            type="primary",
            disabled=is_running,
            use_container_width=True,
        ):
            job = GenerationJob()
            st.session_state[_JOB_KEY] = job
            job.start(
                _generate_temporal,
                engine,
                dict(edited_schema),
                base_count,
                start_date,
                end_date,
                frequency,
                trend_pct,
                list(st.session_state.tt_spikes),
            )
            st.session_state["_tt_write_settings"] = dict(
                output_path=output_path, output_format=output_format,
                records_per_file=records_per_file,
            )
            st.session_state["_tt_preview"] = preview
            st.rerun()

    with stop_col:
        if is_running and st.button("⏹  Stop", use_container_width=True):
            job.request_stop()

    with reset_col:
        if job and job.status in ("complete", "stopped", "error"):
            if st.button("🔄  Reset", use_container_width=True):
                st.session_state[_JOB_KEY] = None
                st.rerun()

    # Progress monitor (polling loop)
    if is_running:
        import time
        progress_bar = st.empty()
        while job.status == "running":
            progress_bar.progress(job.progress, text=job.status_text)
            time.sleep(0.5)
        progress_bar.empty()
        st.rerun()

    # ---- Results ----
    if job and job.status in ("complete", "stopped"):
        df = job.result_df
        if df is not None and len(df) > 0:
            st.session_state.tt_generated_df = df
            prev = st.session_state.get("_tt_preview", [])
            st.success(
                f"✅  Generated {len(df):,} records across {len(prev)} periods",
                icon="🎉",
            )

            metric_row([
                (f"{len(df):,}", "Records"),
                (str(len(df.columns)), "Columns"),
                (str(len(prev)), "Periods"),
            ])

            with st.expander("📊  Data Preview (first 20 rows)", expanded=True):
                st.dataframe(df.head(20), use_container_width=True)

            section("📥", "Download")
            download_buttons(df, prefix="temporal_data")

            section("💾", "Save to Disk")
            if st.button("💾  Write to Output Directory", use_container_width=True):
                _write_tt_output(df, st.session_state.get("_tt_write_settings", {}))
        else:
            st.warning("Generation produced no records.")

    elif job and job.status == "error":
        st.error(f"❌  {job.error_msg}", icon="🚨")


def _generate_temporal(engine, schema, base_count, start_date, end_date, frequency, trend_pct, spikes, **kwargs):
    """Runs in background thread."""
    progress_callback = kwargs.get("progress_callback")
    df = engine.generate_temporal(
        schema=schema,
        base_count_per_period=base_count,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        trend_pct=trend_pct,
        spike_dates=spikes,
    )
    if progress_callback:
        progress_callback(1, 1)
    return df


def _write_tt_output(df, settings: dict):
    try:
        resolved = validate_output_path(settings.get("output_path", "./output_temporal"))
        sink = LocalSink()
        written = sink.push(
            df, resolved,
            settings.get("output_format", "parquet"),
            settings.get("records_per_file", 500),
            partitions=["_period"],
        )
        st.success(f"✅  Wrote {len(written)} file(s) to `{resolved}`")
    except ForgeError as e:
        st.error(f"❌  {e}")
    except Exception as e:
        st.error(f"❌  Write failed: {e}")
