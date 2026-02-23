"""
Reusable Schema Editor UI Component.

Provides schema inference and interactive editing for uploaded files.
"""

import streamlit as st
import polars as pl


def infer_schema(uploaded_file) -> dict:
    """Infer schema from an uploaded CSV or Parquet file."""
    if uploaded_file.name.endswith("csv"):
        df = pl.read_csv(uploaded_file, n_rows=5)
    else:
        df = pl.read_parquet(uploaded_file)
        if len(df) > 5:
            df = df.head(5)
    return {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}, df


def render_schema_editor(schema: dict, key_prefix: str = "") -> dict:
    """
    Render an interactive schema editor.

    Returns the edited schema dict.
    """
    TYPE_OPTIONS = ["Int64", "Float64", "String", "Date"]

    edited_schema = {}
    num_cols = min(len(schema), 4)
    cols = st.columns(num_cols) if num_cols > 0 else []

    for i, (col_name, dtype) in enumerate(schema.items()):
        with cols[i % num_cols] if cols else st.container():
            # Determine default index
            if "Int" in dtype:
                default_idx = 0
            elif "Float" in dtype:
                default_idx = 1
            elif "Date" in dtype or "Datetime" in dtype:
                default_idx = 3
            else:
                default_idx = 2

            new_type = st.selectbox(
                f"📋 `{col_name}`",
                TYPE_OPTIONS,
                index=default_idx,
                key=f"{key_prefix}_schema_{col_name}",
            )
            edited_schema[col_name] = new_type

    return edited_schema


def read_full_dataframe(uploaded_file) -> pl.DataFrame:
    """Read the full uploaded file into a DataFrame."""
    uploaded_file.seek(0)
    if uploaded_file.name.endswith("csv"):
        return pl.read_csv(uploaded_file)
    else:
        return pl.read_parquet(uploaded_file)
