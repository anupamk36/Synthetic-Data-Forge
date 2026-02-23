"""
Reusable Schema Editor UI Component.

Provides schema inference and interactive editing for uploaded files.
"""

import streamlit as st
import polars as pl


def infer_schema(uploaded_file) -> dict:
    """Infer schema from an uploaded CSV, Parquet, or JSON file."""
    name = uploaded_file.name.lower()
    if name.endswith("csv"):
        df = pl.read_csv(uploaded_file, n_rows=5)
    elif name.endswith("parquet"):
        df = pl.read_parquet(uploaded_file)
        if len(df) > 5:
            df = df.head(5)
    elif name.endswith("json") or name.endswith("jsonl"):
        uploaded_file.seek(0)
        try:
            # Try standard JSON first (it needs to be a list of objects)
            df = pl.read_json(uploaded_file)
        except Exception:
            # Fallback to NDJSON
            uploaded_file.seek(0)
            df = pl.read_ndjson(uploaded_file)
        
        if len(df) > 5:
            df = df.head(5)
    else:
        # Fallback
        df = pl.read_csv(uploaded_file, n_rows=5)

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
    name = uploaded_file.name.lower()
    if name.endswith("csv"):
        return pl.read_csv(uploaded_file)
    elif name.endswith("parquet"):
        return pl.read_parquet(uploaded_file)
    elif name.endswith("json") or name.endswith("jsonl"):
        try:
            return pl.read_json(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pl.read_ndjson(uploaded_file)
    else:
        return pl.read_csv(uploaded_file)
