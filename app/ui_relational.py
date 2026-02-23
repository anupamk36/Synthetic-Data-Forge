"""
Multi-Table Relational Map UI.

Lets users upload multiple files, define FK relationships,
and generate referentially-consistent synthetic data.
"""

import streamlit as st
import polars as pl
from app.ui_schema import infer_schema, render_schema_editor
from core.relational import RelationalEngine
from core.sinks import LocalSink
import os


def render_relational_tab():
    """Render the Hydra multi-table generation interface."""

    st.markdown("Upload multiple CSV/Parquet files to generate related synthetic datasets with FK integrity.")

    # --- File Upload ---
    uploaded_files = st.file_uploader(
        "Upload related tables",
        type=["csv", "parquet", "json", "jsonl"],
        accept_multiple_files=True,
        key="multi_upload",
    )

    if not uploaded_files:
        st.info("📂 Upload 2 or more files to define relationships between tables.")
        return

    # --- Infer schemas ---
    if "multi_schemas" not in st.session_state:
        st.session_state.multi_schemas = {}
        st.session_state.multi_samples = {}

    for f in uploaded_files:
        table_name = os.path.splitext(f.name)[0]
        if table_name not in st.session_state.multi_schemas:
            schema, sample_df = infer_schema(f)
            st.session_state.multi_schemas[table_name] = schema
            st.session_state.multi_samples[table_name] = sample_df

    # --- Display schemas ---
    st.subheader("📋 Table Schemas")
    table_names = list(st.session_state.multi_schemas.keys())

    for tname in table_names:
        with st.expander(f"📄 {tname}", expanded=True):
            schema = st.session_state.multi_schemas[tname]
            edited = render_schema_editor(schema, key_prefix=f"multi_{tname}")
            st.session_state.multi_schemas[tname] = edited

            # Show sample
            if tname in st.session_state.multi_samples:
                st.dataframe(st.session_state.multi_samples[tname], use_container_width=True)

    # --- Define Relationships ---
    st.subheader("🔗 Define Relationships")
    st.markdown("Link tables via foreign key columns. Parent tables are generated first.")

    if "relationships" not in st.session_state:
        st.session_state.relationships = []

    col1, col2, col3, col4 = st.columns(4)
    parent_table = col1.selectbox("Parent Table", table_names, key="rel_parent")
    parent_col = col2.selectbox(
        "Parent Column",
        list(st.session_state.multi_schemas.get(parent_table, {}).keys()),
        key="rel_pcol",
    )
    child_table = col3.selectbox(
        "Child Table",
        [t for t in table_names if t != parent_table],
        key="rel_child",
    ) if len(table_names) > 1 else col3.selectbox("Child Table", table_names, key="rel_child")
    child_col = col4.selectbox(
        "Child Column",
        list(st.session_state.multi_schemas.get(child_table, {}).keys()) if child_table else [],
        key="rel_ccol",
    )

    if st.button("➕ Add Relationship"):
        rel = (parent_table, parent_col, child_table, child_col)
        if rel not in st.session_state.relationships:
            st.session_state.relationships.append(rel)

    # Show existing relationships
    if st.session_state.relationships:
        st.markdown("**Current Relationships:**")
        for i, (pt, pc, ct, cc) in enumerate(st.session_state.relationships):
            col_a, col_b = st.columns([4, 1])
            col_a.markdown(f"`{pt}.{pc}` → `{ct}.{cc}`")
            if col_b.button("🗑️", key=f"del_rel_{i}"):
                st.session_state.relationships.pop(i)
                st.rerun()

        # DAG visualization
        with st.expander("🗺️ Relationship Map (Mermaid)"):
            mermaid_lines = ["graph TD"]
            for pt, pc, ct, cc in st.session_state.relationships:
                mermaid_lines.append(f'    {pt}["{pt}"] -->|{pc} = {cc}| {ct}["{ct}"]')
            st.code("\n".join(mermaid_lines), language="mermaid")

    # --- Generation Settings ---
    st.divider()
    st.subheader("⚙️ Generation Settings")

    counts = {}
    cols = st.columns(min(len(table_names), 4))
    for i, tname in enumerate(table_names):
        with cols[i % len(cols)]:
            counts[tname] = st.number_input(
                f"Rows for `{tname}`", value=100, min_value=1, key=f"count_{tname}"
            )

    col_fmt, col_rpp = st.columns(2)
    output_format = col_fmt.selectbox("Output Format", ["parquet", "csv", "json"], key="multi_fmt")
    records_per_file = col_rpp.number_input("Records Per File", value=250, min_value=1, key="multi_rpp")

    output_path = st.text_input(
        "Output Directory",
        value="./output_multi",
        key="multi_output",
    )

    # --- Generate ---
    if st.button("🚀 Generate All Tables", key="multi_gen"):
        engine = RelationalEngine()

        for tname, schema in st.session_state.multi_schemas.items():
            engine.add_table(tname, schema)

        for pt, pc, ct, cc in st.session_state.relationships:
            engine.add_relationship(pt, pc, ct, cc)

        try:
            with st.spinner("Building DAG and generating tables..."):
                results = engine.generate_all(counts)

            sink = LocalSink()
            resolved = os.path.abspath(os.path.expanduser(output_path))

            for tname, df in results.items():
                table_dir = os.path.join(resolved, tname)
                sink.push(df, table_dir, output_format, records_per_file)

            st.success(f"✅ Generated {len(results)} tables to `{resolved}`")

            # Show previews
            for tname, df in results.items():
                with st.expander(f"📊 Preview: {tname} ({len(df)} rows)"):
                    st.dataframe(df.head(10), use_container_width=True)

        except ValueError as e:
            st.error(f"❌ {str(e)}")
