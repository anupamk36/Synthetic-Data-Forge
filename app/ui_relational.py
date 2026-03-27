"""Multi-Table (Hydra) page — FK-consistent multi-table generation."""

import logging
import os

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
from core.relational import RelationalEngine
from core.sinks import LocalSink

logger = logging.getLogger(__name__)

_JOB_KEY = "multi_gen_job"


def render_relational_page():
    """Full multi-table generation page."""

    hero(
        "🔗  Multi-Table Generator (Hydra)",
        "Upload related tables, define FK relationships, and generate referentially-consistent synthetic datasets.",
    )

    # ---- File Upload ----
    section("📂", "Upload Related Tables")
    uploaded_files = st.file_uploader(
        "Upload 2+ related CSVs or Parquets",
        type=["csv", "parquet", "json", "jsonl"],
        accept_multiple_files=True,
        key="multi_upload",
        label_visibility="collapsed",
    )

    if not uploaded_files:
        empty_state(
            "📂",
            "Upload 2 or more files to define relationships",
            "Parent tables are generated first; child tables inherit FK values.",
        )
        return

    # ---- Infer schemas ----
    if "multi_schemas" not in st.session_state:
        st.session_state.multi_schemas = {}
        st.session_state.multi_samples = {}

    for f in uploaded_files:
        table_name = os.path.splitext(f.name)[0]
        if table_name not in st.session_state.multi_schemas:
            schema, sample_df = infer_schema(f)
            st.session_state.multi_schemas[table_name] = schema
            st.session_state.multi_samples[table_name] = sample_df

    table_names = list(st.session_state.multi_schemas.keys())

    # ---- Schema editors ----
    section("📋", "Table Schemas")
    for tname in table_names:
        with st.expander(f"📄  {tname}", expanded=True):
            schema = st.session_state.multi_schemas[tname]
            edited = render_schema_editor(schema, key_prefix=f"multi_{tname}")
            st.session_state.multi_schemas[tname] = edited
            if tname in st.session_state.multi_samples:
                st.dataframe(st.session_state.multi_samples[tname], use_container_width=True)

    # ---- Relationships ----
    section("🔗", "Define Relationships")
    if "relationships" not in st.session_state:
        st.session_state.relationships = []

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        parent_table = c1.selectbox("Parent Table", table_names, key="rel_parent")
        parent_col = c2.selectbox(
            "Parent Column",
            list(st.session_state.multi_schemas.get(parent_table, {}).keys()),
            key="rel_pcol",
        )
        child_table = c3.selectbox(
            "Child Table",
            [t for t in table_names if t != parent_table] or table_names,
            key="rel_child",
        )
        child_col = c4.selectbox(
            "Child Column",
            list(st.session_state.multi_schemas.get(child_table, {}).keys()) if child_table else [],
            key="rel_ccol",
        )
        if st.button("➕  Add Relationship"):
            rel = (parent_table, parent_col, child_table, child_col)
            if rel not in st.session_state.relationships:
                st.session_state.relationships.append(rel)
                st.rerun()

    if st.session_state.relationships:
        for i, (pt, pc, ct, cc) in enumerate(st.session_state.relationships):
            col_a, col_b = st.columns([5, 1])
            col_a.markdown(f"`{pt}.{pc}` → `{ct}.{cc}`")
            if col_b.button("🗑️", key=f"del_rel_{i}"):
                st.session_state.relationships.pop(i)
                st.rerun()

        with st.expander("🗺️  Relationship Map (Mermaid)"):
            lines = ["graph TD"]
            for pt, pc, ct, cc in st.session_state.relationships:
                lines.append(f'    {pt}["{pt}"] -->|{pc} = {cc}| {ct}["{ct}"]')
            st.code("\n".join(lines), language="mermaid")

    # ---- Generation Settings ----
    section("⚙️", "Generation Settings")
    with st.container(border=True):
        counts = {}
        cols = st.columns(min(len(table_names), 4))
        for i, tname in enumerate(table_names):
            with cols[i % len(cols)]:
                counts[tname] = st.number_input(
                    f"Rows for {tname}", value=100, min_value=1, key=f"count_{tname}"
                )

        col_fmt, col_rpp = st.columns(2)
        output_format = col_fmt.selectbox("Output Format", ["parquet", "csv", "json"], key="multi_fmt")
        records_per_file = col_rpp.number_input("Records Per File", value=250, min_value=1, key="multi_rpp")

        output_path = st.text_input("Output Directory", value="./output_multi", key="multi_output")

    # ---- Generate / Stop ----
    st.divider()

    if _JOB_KEY not in st.session_state:
        st.session_state[_JOB_KEY] = None

    job: GenerationJob | None = st.session_state[_JOB_KEY]
    is_running = job is not None and job.status == "running"

    btn_col, stop_col, reset_col = st.columns([2, 1, 1])
    with btn_col:
        if st.button("🚀  Generate All Tables", type="primary", disabled=is_running, use_container_width=True):
            job = GenerationJob()
            st.session_state[_JOB_KEY] = job
            job.start(
                _generate_multi,
                dict(st.session_state.multi_schemas),
                list(st.session_state.relationships),
                dict(counts),
            )
            st.session_state["_multi_write_settings"] = dict(
                output_path=output_path, output_format=output_format, records_per_file=records_per_file,
            )
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
        result = job.result_df  # dict[str, DataFrame]
        if result and isinstance(result, dict):
            st.session_state.multi_results = result
            total = sum(len(df) for df in result.values())
            st.success(f"✅  Generated {len(result)} tables ({total:,} total records)", icon="🎉")

            metric_row([
                (str(len(result)), "Tables"),
                (f"{total:,}", "Total Records"),
                (job.status.title(), "Status"),
            ])

            for tname, df in result.items():
                with st.expander(f"📊  {tname} ({len(df):,} rows)", expanded=False):
                    st.dataframe(df.head(15), use_container_width=True)
                    download_buttons(df, prefix=f"synthetic_{tname}")

            # Save all
            section("💾", "Save All Tables to Disk")
            if st.button("💾  Write All to Output Directory", use_container_width=True):
                _write_multi_output(result, st.session_state.get("_multi_write_settings", {}))
        else:
            st.warning("Generation produced no results.")

    elif job and job.status == "error":
        st.error(f"❌  {job.error_msg}", icon="🚨")


def _generate_multi(schemas, relationships, counts, **kwargs):
    """Generate all tables (runs in background thread)."""
    progress_callback = kwargs.get("progress_callback")
    stop_check = kwargs.get("stop_check")

    engine = RelationalEngine()
    for tname, schema in schemas.items():
        engine.add_table(tname, schema)
    for pt, pc, ct, cc in relationships:
        engine.add_relationship(pt, pc, ct, cc)

    results = engine.generate_all(counts)
    if progress_callback:
        progress_callback(1, 1)
    return results


def _write_multi_output(results: dict, settings: dict):
    try:
        resolved = validate_output_path(settings.get("output_path", "./output_multi"))
        sink = LocalSink()
        for tname, df in results.items():
            table_dir = os.path.join(resolved, tname)
            sink.push(df, table_dir, settings.get("output_format", "parquet"), settings.get("records_per_file", 250))
        st.success(f"✅  Wrote {len(results)} tables to `{resolved}`")
    except ForgeError as e:
        st.error(f"❌  {e}")
    except Exception as e:
        st.error(f"❌  Write failed: {e}")
