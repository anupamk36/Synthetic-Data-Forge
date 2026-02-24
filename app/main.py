import streamlit as st
import polars as pl
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.generator import ForgeEngine
from core.llm_logic import LLMLogicEngine
from core.sinks import get_sink, LocalSink
from app.ui_schema import infer_schema, render_schema_editor, read_full_dataframe
from app.ui_privacy import render_privacy_scorecard
from app.ui_relational import render_relational_tab
from app.ui_time_travel import render_time_travel_tab

# --- Page Config ---
st.set_page_config(
    page_title="ForgeFlow AI — Synthetic Data Forge",
    page_icon="🛠️",
    layout="wide",
)

st.title("🛠️ ForgeFlow: Synthetic Data Forge")
st.caption("Generate realistic, privacy-safe synthetic data with business logic, relational integrity, and temporal patterns.")

# --- Tab Layout ---
tab_single, tab_multi, tab_time, tab_privacy = st.tabs([
    "📊 Single Table",
    "🔗 Multi-Table (Hydra)",
    "⏰ Time Travel",
    "🛡️ Privacy Scorecard",
])

# ======================================================================
# TAB 1: Single Table Generation
# ======================================================================
with tab_single:

    # --- File Upload ---
    uploaded_file = st.file_uploader(
        "Drop a CSV, Parquet, or JSON file to infer schema",
        type=["csv", "parquet", "json", "jsonl"],
        key="single_upload",
    )

    if uploaded_file:
        schema, sample_df = infer_schema(uploaded_file)
        st.session_state.single_schema = schema
        st.session_state.single_file = uploaded_file

        with st.expander("📄 Sample Data (first 5 rows)", expanded=False):
            st.dataframe(sample_df, use_container_width=True)

    if "single_schema" in st.session_state and st.session_state.single_schema:
        # --- Schema Editor ---
        st.subheader("📋 Verify & Edit Schema")
        edited_schema = render_schema_editor(st.session_state.single_schema, key_prefix="single")

        # --- Generation Settings ---
        st.divider()
        st.subheader("⚙️ Generation Settings")

        col1, col2, col3 = st.columns(3)
        total_rec = col1.number_input("Total Records", value=1000, min_value=1, key="single_total")
        rec_per_file = col2.number_input("Records Per File", value=250, min_value=1, key="single_rpp")
        output_format = col3.selectbox("Output Format", ["parquet", "csv", "json"], key="single_fmt")

        partition_on = st.multiselect(
            "Partition Columns (Hive-style nesting)",
            options=list(edited_schema.keys()),
            key="single_partitions",
        )

        use_llm = st.checkbox(
            "Use Smart LLM Generation",
            help="Directs Ollama to generate semantically coherent data. Slower but higher quality. Recommended for small batches.",
            key="single_use_llm",
        )


        # --- LLM Business Logic ---
        st.divider()
        st.subheader("🧠 Business Logic Rules (LLM-Powered)")

        llm_engine = LLMLogicEngine()
        ollama_available = llm_engine.is_available()

        if ollama_available:
            st.success("🟢 Ollama is running — LLM rules are available.")
            models = llm_engine.get_available_models()
            if models:
                selected_model = st.selectbox("LLM Model", models, key="llm_model")
                llm_engine.model = selected_model
        else:
            st.warning(
                "⚠️ Ollama is not running. Start it with `docker compose up -d` "
                "and pull a model with `docker exec forge-ollama ollama pull llama3`. "
                "Rules below will be skipped."
            )

        if use_llm:
            with st.expander("🧠 Smart Mode: Field Descriptions", expanded=True):
                st.info("Provide semantic hints for fields (e.g., 'Sex: M or F'). Empty fields will use their names as hints.")
                field_descriptions = {}
                for col in edited_schema:
                    field_descriptions[col] = st.text_input(f"Description for `{col}`", key=f"desc_{col}")
        else:
            field_descriptions = None

        # --- Sink Selection ---
        st.divider()
        st.subheader("📤 Output Destination")

        sink_type = st.radio("Sink", ["Local Filesystem", "Amazon S3"], horizontal=True, key="single_sink")

        if sink_type == "Local Filesystem":
            output_path = st.text_input(
                "Output Directory (use ~/Desktop/... for Desktop)",
                value="./output_data",
                key="single_output",
            )
        else:
            s3_col1, s3_col2, s3_col3 = st.columns(3)
            s3_bucket = s3_col1.text_input("S3 Bucket", key="s3_bucket")
            s3_prefix = s3_col2.text_input("S3 Prefix", value="synthetic-data", key="s3_prefix")
            s3_region = s3_col3.text_input("Region", value="us-east-1", key="s3_region")
            output_path = ""

        # --- Generate Button ---
        st.divider()
        if st.button("🚀 Generate Data", key="single_gen", type="primary"):

            engine = ForgeEngine()
            # rules = [r.strip() for r in rules_text.strip().split("\n") if r.strip()] # Removed rules_text

            with st.spinner("Forging synthetic data..."):
                if use_llm:
                    df = engine.generate_records(
                        edited_schema,
                        total_rec,
                        use_llm=True,
                        llm_engine=llm_engine if ollama_available else None,
                        field_descriptions=field_descriptions
                    )
                    # Check if it actually used LLM or fell back
                    if ollama_available and len(df) > 0:
                        st.info("🧠 Generated using Smart LLM Mode.")
                    else:
                        st.warning("⚠️ LLM generation failed or was unavailable. Fell back to standard Faker generation (field descriptions ignored).")
                else:
                    df = engine.generate_records(edited_schema, total_rec)

            # Write output
            with st.spinner("Writing output files..."):
                if sink_type == "Local Filesystem":
                    sink = get_sink("local")
                    resolved = os.path.abspath(os.path.expanduser(output_path))
                    written = sink.push(df, resolved, output_format, rec_per_file, partition_on or None)
                    st.success(f"✅ Generated {len(df):,} records → `{resolved}` ({len(written)} files)")
                else:
                    try:
                        sink = get_sink("s3", bucket=s3_bucket, prefix=s3_prefix, region=s3_region)
                        written = sink.push(df, "", output_format, rec_per_file, partition_on or None)
                        st.success(f"✅ Pushed {len(df):,} records to S3 ({len(written)} files)")
                    except Exception as e:
                        st.error(f"❌ S3 push failed: {str(e)}")

            # Store for privacy tab
            st.session_state.generated_df = df

            with st.expander("📊 Preview (first 20 rows)"):
                st.dataframe(df.head(20), use_container_width=True)

# ======================================================================
# TAB 2: Multi-Table (Hydra)
# ======================================================================
with tab_multi:
    render_relational_tab()

# ======================================================================
# TAB 3: Time Travel
# ======================================================================
with tab_time:
    render_time_travel_tab()

# ======================================================================
# TAB 4: Privacy Scorecard
# ======================================================================
with tab_privacy:
    st.markdown("Compare your synthetic data against the original to measure privacy risk using Distance to Closest Record (DCR).")

    col_real, col_syn = st.columns(2)

    with col_real:
        st.subheader("📁 Real Data")
        real_file = st.file_uploader("Upload original (real) data", type=["csv", "parquet", "json", "jsonl"], key="priv_real")
        real_df = None
        if real_file:
            real_df = read_full_dataframe(real_file)
            st.caption(f"{len(real_df):,} rows × {len(real_df.columns)} columns")

    with col_syn:
        st.subheader("🔬 Synthetic Data")
        syn_source = st.radio("Source", ["Upload file", "Use last generated"], horizontal=True, key="priv_source")

        syn_df = None
        if syn_source == "Upload file":
            syn_file = st.file_uploader("Upload synthetic data", type=["csv", "parquet", "json", "jsonl"], key="priv_syn")
            if syn_file:
                syn_df = read_full_dataframe(syn_file)
                st.caption(f"{len(syn_df):,} rows × {len(syn_df.columns)} columns")
        else:
            if "generated_df" in st.session_state:
                syn_df = st.session_state.generated_df
                st.caption(f"{len(syn_df):,} rows × {len(syn_df.columns)} columns (from last generation)")
            else:
                st.info("No data generated yet. Generate data in the Single Table tab first.")

    if real_df is not None and syn_df is not None:
        st.divider()
        render_privacy_scorecard(real_df, syn_df)
    elif real_df is not None or syn_df is not None:
        st.info("Upload both real and synthetic data to compute the privacy scorecard.")