"""Generation History page — browse past runs from the audit trail."""

import json
import logging

import streamlit as st

from app.components import empty_state, hero, metric_row, section, status_pill
from core import audit

logger = logging.getLogger(__name__)

_FEATURE_LABELS = {
    "single": "📊 Single Table",
    "relational": "🔗 Multi-Table",
    "time_travel": "⏰ Time Travel",
}


def render_history_page():
    """Full generation history page."""

    hero(
        "📜  Generation History",
        "Browse and inspect past generation runs — who generated what, when, with which settings.",
    )

    # ── Filters ──
    section("🔍", "Filters")
    fc1, fc2 = st.columns(2)
    feature_filter = fc1.selectbox(
        "Feature", ["All", "single", "relational", "time_travel"], key="hist_feature",
    )
    limit = fc2.number_input("Show last N runs", value=50, min_value=1, max_value=500, key="hist_limit")

    feat = feature_filter if feature_filter != "All" else None
    runs = audit.list_runs(limit=limit, feature=feat)

    if not runs:
        empty_state("📜", "No generation runs recorded yet", "Generate data on any page to start building history.")
        return

    # ── Summary metrics ──
    total_records = sum(r.get("record_count", 0) for r in runs)
    completed = sum(1 for r in runs if r.get("status") == "complete")
    errors = sum(1 for r in runs if r.get("status") == "error")

    metric_row([
        (str(len(runs)), "Total Runs"),
        (f"{total_records:,}", "Total Records"),
        (str(completed), "Completed"),
        (str(errors), "Errors"),
    ])

    # ── Run list ──
    st.write("")
    section("📋", "Run Details")

    for run in runs:
        feature_label = _FEATURE_LABELS.get(run.get("feature", ""), run.get("feature", "?"))
        status = run.get("status", "unknown")
        rec = run.get("record_count", 0)
        elapsed = run.get("elapsed_sec", 0)
        engine = run.get("engine", "faker")
        created = run.get("created_at", "")[:19].replace("T", " ")

        # Status color
        if status == "complete":
            badge = "🟢"
        elif status == "running":
            badge = "🔵"
        elif status == "stopped":
            badge = "🟡"
        else:
            badge = "🔴"

        title = f"{badge} {feature_label}  —  {rec:,} records  —  {engine}  —  {created}"

        with st.expander(title, expanded=False):
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Records", f"{rec:,}")
            mc2.metric("Elapsed", f"{elapsed:.1f}s" if elapsed else "—")
            mc3.metric("Engine", engine.upper())
            mc4.metric("Status", status.title())

            # Schema
            schema_json = run.get("schema_json", "{}")
            try:
                schema = json.loads(schema_json) if isinstance(schema_json, str) else schema_json
            except json.JSONDecodeError:
                schema = {}

            if schema:
                st.caption("**Schema:**")
                st.json(schema)

            # Settings
            settings_json = run.get("settings_json", "{}")
            try:
                settings = json.loads(settings_json) if isinstance(settings_json, str) else settings_json
            except json.JSONDecodeError:
                settings = {}

            if settings:
                st.caption("**Settings:**")
                st.json(settings)

            # Output path
            if run.get("output_path"):
                st.caption(f"**Output:** `{run['output_path']}`")

            # Error
            if run.get("error_msg"):
                st.error(f"Error: {run['error_msg']}")

            # Run ID
            st.caption(f"Run ID: `{run['id']}`")
