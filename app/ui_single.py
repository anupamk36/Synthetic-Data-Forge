"""Single-Table Generation page — the main workhorse of ForgeFlow AI."""

import logging
import math
import os
import time as _time

import polars as pl
import streamlit as st

from app.components import (
    GenerationJob,
    download_buttons,
    empty_state,
    hero,
    metric_row,
    section,
    status_pill,
)
from app.ui_schema import infer_schema, render_schema_editor
from core import audit
from core.config import validate_output_path
from core.exceptions import ForgeError
from core.generator import ForgeEngine
from core.llm_logic import LLMLogicEngine
from core.sinks import get_sink

logger = logging.getLogger(__name__)

_JOB_KEY = "single_gen_job"
_SETTINGS_KEY = "single_gen_settings"
_START_TIME_KEY = "single_gen_start_ts"
_RUN_ID_KEY = "single_gen_run_id"


# ══════════════════════════════════════════════════════════════════════
# Public entry-point
# ══════════════════════════════════════════════════════════════════════
def render_single_table_page():
    """Render the full single-table generation page."""

    hero(
        "📊  Single Table Generator",
        "Upload a sample file, customise the schema, and generate realistic synthetic data in seconds.",
    )

    # ------------------------------------------------------------------
    # 1 ▸ File Upload
    # ------------------------------------------------------------------
    section("📂", "Upload Sample Data")
    uploaded_file = st.file_uploader(
        "Drag a CSV, Parquet, or JSON file here to infer schema",
        type=["csv", "parquet", "json", "jsonl"],
        key="single_upload",
        label_visibility="collapsed",
    )

    if uploaded_file:
        schema, sample_df = infer_schema(uploaded_file)
        st.session_state.single_schema = schema
        st.session_state.single_file = uploaded_file
        with st.expander("📄  Sample rows (first 5)", expanded=False):
            st.dataframe(sample_df, use_container_width=True)

    if "single_schema" not in st.session_state or not st.session_state.single_schema:
        empty_state("📂", "Upload a file to get started", "We'll auto-detect the schema for you.")
        return

    # ------------------------------------------------------------------
    # 2 ▸ Schema Editor
    # ------------------------------------------------------------------
    section("📋", "Schema Editor")
    with st.container(border=True):
        edited_schema = render_schema_editor(st.session_state.single_schema, key_prefix="single")

    # ------------------------------------------------------------------
    # 3 ▸ Generation Settings  (partitioning moved to post-gen)
    # ------------------------------------------------------------------
    section("⚙️", "Generation Settings")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        total_rec = c1.number_input("Total Records", value=1_000, min_value=1, key="single_total")
        output_format = c2.selectbox("Output Format", ["parquet", "csv", "json"], key="single_fmt")
        seed_val = c3.number_input(
            "Random Seed", value=0, min_value=0, key="single_seed",
            help="Set to 0 for random. Any other value produces reproducible output.",
        )

    # ------------------------------------------------------------------
    # 4 ▸ LLM Configuration
    # ------------------------------------------------------------------
    section("🧠", "LLM Smart Mode")
    llm_engine = LLMLogicEngine()
    ollama_ok = llm_engine.is_available()

    with st.container(border=True):
        use_llm = st.toggle(
            "Enable LLM-powered generation",
            help="Uses Ollama to produce semantically coherent records. Best for small batches.",
            key="single_use_llm",
        )

        if ollama_ok:
            status_pill("🟢  Ollama connected", "green")
            models = llm_engine.get_available_models()
            if models:
                selected_model = st.selectbox("Model", models, key="llm_model")
                llm_engine.model = selected_model
        else:
            status_pill("🔴  Ollama offline — LLM rules will be skipped", "red")

        field_descriptions = None
        if use_llm:
            st.caption("Add semantic hints per field (e.g. *'Sex: M or F'*). Empty fields use column names.")
            field_descriptions = {}
            cols = st.columns(min(len(edited_schema), 3))
            for i, col_name in enumerate(edited_schema):
                with cols[i % len(cols)]:
                    field_descriptions[col_name] = st.text_input(
                        col_name, key=f"desc_{col_name}", placeholder=f"Describe {col_name}…"
                    )

    # ------------------------------------------------------------------
    # 5 ▸ Generate / Stop Controls
    # ------------------------------------------------------------------
    st.divider()

    if _JOB_KEY not in st.session_state:
        st.session_state[_JOB_KEY] = None

    job: GenerationJob | None = st.session_state[_JOB_KEY]
    is_running = job is not None and job.status == "running"

    btn_col, stop_col, reset_col = st.columns([2, 1, 1])
    with btn_col:
        if st.button(
            "🚀  Generate Data",
            type="primary",
            disabled=is_running,
            use_container_width=True,
        ):
            job = GenerationJob()
            st.session_state[_JOB_KEY] = job
            st.session_state[_START_TIME_KEY] = _time.time()

            engine_type = "faker"
            engine = ForgeEngine(seed=seed_val if seed_val > 0 else None)
            gen_kwargs = {}
            if use_llm:
                gen_kwargs.update(
                    use_llm=True,
                    llm_engine=llm_engine if ollama_ok else None,
                    field_descriptions=field_descriptions,
                )
                engine_type = "llm"

            run_id = audit.start_run(
                "single", edited_schema,
                {"count": total_rec, "format": output_format, "seed": seed_val, "use_llm": use_llm},
                engine=engine_type,
            )
            st.session_state[_RUN_ID_KEY] = run_id

            job.start(engine.generate_records, edited_schema, total_rec, **gen_kwargs)
            st.rerun()

    with stop_col:
        if is_running:
            if st.button("⏹  Stop", use_container_width=True):
                job.request_stop()

    with reset_col:
        if job and job.status in ("complete", "stopped", "error"):
            if st.button("🔄  Reset", use_container_width=True):
                st.session_state[_JOB_KEY] = None
                st.session_state.pop(_START_TIME_KEY, None)
                st.rerun()

    # ------------------------------------------------------------------
    # 6 ▸ Progress Monitor (polling loop)
    # ------------------------------------------------------------------
    if is_running:
        progress_bar = st.empty()
        status_area = st.empty()
        while job.status == "running":
            pct = job.progress
            elapsed = _time.time() - st.session_state.get(_START_TIME_KEY, _time.time())
            rps = job.records_done / elapsed if elapsed > 0.1 else 0
            progress_bar.progress(
                min(pct, 1.0),
                text=f"{job.status_text}  |  ⏱ {elapsed:.1f}s  |  ⚡ {rps:,.0f} rec/s",
            )
            _time.sleep(0.5)
        progress_bar.empty()
        status_area.empty()
        st.rerun()

    # ------------------------------------------------------------------
    # 7 ▸ Results
    # ------------------------------------------------------------------
    if job and job.status in ("complete", "stopped"):
        df = job.result_df
        if df is not None and len(df) > 0:
            st.session_state.generated_df = df

            # ----- audit finish -----
            elapsed = _time.time() - st.session_state.get(_START_TIME_KEY, _time.time())
            run_id = st.session_state.get(_RUN_ID_KEY)
            if run_id:
                audit.finish_run(
                    run_id, status=job.status,
                    record_count=len(df), columns=len(df.columns),
                    elapsed_sec=round(elapsed, 2),
                )
                st.session_state[_RUN_ID_KEY] = None  # don't log twice

            rps = len(df) / elapsed if elapsed > 0.5 else 0

            st.success(f"✅  {job.status_text}", icon="🎉")

            metric_row([
                (f"{len(df):,}", "Records"),
                (str(len(df.columns)), "Columns"),
                (f"{elapsed:.1f}s", "Elapsed"),
                (f"{rps:,.0f}", "Rec / sec"),
                (job.status.title(), "Status"),
            ])

            # ----- data preview -----
            with st.expander("📊  Data Preview (first 20 rows)", expanded=True):
                st.dataframe(df.head(20), use_container_width=True)

            # ----- column profiler -----
            _render_column_profiler(df)

            # ----- download -----
            section("📥", "Download")
            download_buttons(df)

            # ----- post-gen write settings -----
            _render_write_section(df, edited_schema, output_format)
        else:
            st.warning("Generation produced no records.")

    elif job and job.status == "error":
        run_id = st.session_state.get(_RUN_ID_KEY)
        if run_id:
            audit.finish_run(run_id, status="error", error_msg=job.error_msg or "")
            st.session_state[_RUN_ID_KEY] = None
        st.error(f"❌  {job.error_msg}", icon="🚨")


# ══════════════════════════════════════════════════════════════════════
# Column profiler
# ══════════════════════════════════════════════════════════════════════
def _render_column_profiler(df: pl.DataFrame):
    """Quick column-level profiling: type, unique, nulls, sample values."""
    section("🔎", "Column Profiler")
    with st.expander("View column statistics", expanded=False):
        rows = []
        for col in df.columns:
            s = df[col]
            n_unique = s.n_unique()
            n_null = s.null_count()
            completeness = ((len(s) - n_null) / len(s) * 100) if len(s) > 0 else 0
            sample = ", ".join(str(v) for v in s.drop_nulls().head(3).to_list())
            rows.append({
                "Column": col,
                "Type": str(s.dtype),
                "Unique": n_unique,
                "Nulls": n_null,
                "Completeness": f"{completeness:.0f}%",
                "Sample Values": sample,
            })
        st.dataframe(pl.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# Post-generation write section (partitioning + folder tree + write)
# ══════════════════════════════════════════════════════════════════════
def _render_write_section(df: pl.DataFrame, schema: dict, output_format: str):
    """Output destination, partition picker, folder tree preview, and write button."""

    section("💾", "Save to Disk")
    with st.container(border=True):
        # --- Sink ---
        sink_type = st.radio(
            "Sink", ["Local Filesystem", "Amazon S3"],
            horizontal=True, key="single_sink",
        )

        if sink_type == "Local Filesystem":
            output_path = st.text_input(
                "Output Directory", value="./output_data", key="single_output",
            )
        else:
            s3c1, s3c2, s3c3 = st.columns(3)
            s3_bucket = s3c1.text_input("S3 Bucket", key="s3_bucket")
            s3_prefix = s3c2.text_input("S3 Prefix", value="synthetic-data", key="s3_prefix")
            s3_region = s3c3.text_input("Region", value="us-east-1", key="s3_region")
            output_path = ""

            with st.expander("🔑  AWS Credentials", expanded=False):
                st.caption("Leave blank to use environment variables or IAM role.")
                aws_access_key = st.text_input("Access Key ID", key="s3_access_key", type="password")
                aws_secret_key = st.text_input("Secret Access Key", key="s3_secret_key", type="password")
                aws_session_token = st.text_input("Session Token (optional)", key="s3_session_token", type="password")

        # --- Partition + records per file ---
        p1, p2 = st.columns(2)
        rec_per_file = p1.number_input(
            "Records Per File", value=250, min_value=1, key="single_rpp",
        )
        partition_on = p2.multiselect(
            "Partition Columns (Hive-style)",
            options=list(schema.keys()),
            key="single_partitions",
        )

        # --- Folder tree preview ---
        _render_folder_tree(
            df, output_format, rec_per_file, partition_on,
            output_path if sink_type == "Local Filesystem" else f"s3://{locals().get('s3_bucket', 'bucket')}/{locals().get('s3_prefix', '')}",
        )

        # --- Write button ---
        if st.button("💾  Write to Output Destination", type="primary", use_container_width=True):
            settings = dict(
                sink_type=sink_type,
                output_path=output_path if sink_type == "Local Filesystem" else "",
                output_format=output_format,
                records_per_file=rec_per_file,
                partition_on=partition_on or None,
                s3_bucket=locals().get("s3_bucket", ""),
                s3_prefix=locals().get("s3_prefix", ""),
                s3_region=locals().get("s3_region", ""),
                aws_access_key_id=locals().get("aws_access_key", ""),
                aws_secret_access_key=locals().get("aws_secret_key", ""),
                aws_session_token=locals().get("aws_session_token", ""),
            )
            _write_output(df, settings)


# ══════════════════════════════════════════════════════════════════════
# Folder tree preview
# ══════════════════════════════════════════════════════════════════════
def _render_folder_tree(
    df: pl.DataFrame,
    fmt: str,
    rec_per_file: int,
    partitions: list[str],
    root_label: str,
):
    """Show what the output folder structure will look like."""
    ext = {"parquet": "parquet", "csv": "csv", "json": "json"}.get(fmt, "parquet")
    lines: list[str] = [f"📁 {root_label}/"]

    if partitions:
        # Build unique combos (sample up to 6 for preview)
        try:
            combos = (
                df.select(partitions)
                .unique()
                .sort(partitions)
                .head(6)
                .to_dicts()
            )
        except Exception:
            combos = []

        for ci, combo in enumerate(combos):
            is_last_combo = ci == len(combos) - 1
            parts_path = "/".join(f"{k}={v}" for k, v in combo.items())
            branch = "└── " if is_last_combo and len(combos) <= 6 else "├── "

            # Count rows for this partition combo
            mask = pl.lit(True)
            for k, v in combo.items():
                mask = mask & (pl.col(k) == v)
            n_rows = df.filter(mask).height
            n_files = max(1, math.ceil(n_rows / rec_per_file))

            lines.append(f"  {branch}📂 {parts_path}/")
            for fi in range(min(n_files, 3)):
                fb = "└── " if fi == min(n_files, 3) - 1 and n_files <= 3 else "├── "
                lines.append(f"      {fb}📄 part_{fi}.{ext}")
            if n_files > 3:
                lines.append(f"      └── … ({n_files} files total)")

        if len(combos) == 6:
            actual = df.select(partitions).unique().height
            if actual > 6:
                lines.append(f"  └── … ({actual} partition groups total)")
    else:
        n_files = max(1, math.ceil(len(df) / rec_per_file))
        for fi in range(min(n_files, 5)):
            fb = "└── " if fi == min(n_files, 5) - 1 and n_files <= 5 else "├── "
            lines.append(f"  {fb}📄 part_{fi}.{ext}")
        if n_files > 5:
            lines.append(f"  └── … ({n_files} files total)")

    st.markdown("**Output structure preview:**")
    st.code("\n".join(lines), language=None)


# ══════════════════════════════════════════════════════════════════════
# Helper: write generated data to sink
# ══════════════════════════════════════════════════════════════════════
def _write_output(df: pl.DataFrame, settings: dict):
    """Write the generated DataFrame to the configured sink."""
    try:
        if settings.get("sink_type") == "Local Filesystem":
            resolved = validate_output_path(settings["output_path"])
            sink = get_sink("local")
            written = sink.push(
                df, resolved,
                settings["output_format"],
                settings["records_per_file"],
                settings.get("partition_on"),
            )
            st.success(f"✅  Wrote {len(written)} file(s) → `{resolved}`")
            logger.info("Wrote %d records to %s", len(df), resolved)
        else:
            sink = get_sink(
                "s3",
                bucket=settings["s3_bucket"],
                prefix=settings["s3_prefix"],
                region=settings["s3_region"],
                aws_access_key_id=settings.get("aws_access_key_id", ""),
                aws_secret_access_key=settings.get("aws_secret_access_key", ""),
                aws_session_token=settings.get("aws_session_token", ""),
            )
            written = sink.push(
                df, "",
                settings["output_format"],
                settings["records_per_file"],
                settings.get("partition_on"),
            )
            st.success(f"✅  Pushed {len(written)} file(s) to S3")
    except ForgeError as e:
        st.error(f"❌  {e}")
    except Exception as e:
        st.error(f"❌  Write failed: {e}")
