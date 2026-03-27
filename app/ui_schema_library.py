"""Schema Library page — save, browse, and reuse schemas across teams."""

import json
import logging

import streamlit as st

from app.components import empty_state, hero, metric_row, section, status_pill
from core import audit

logger = logging.getLogger(__name__)


def render_schema_library_page():
    """Full schema library page."""

    hero(
        "📚  Schema Library",
        "Save, browse, and share schema templates across your team. Stop redefining the same columns every time.",
    )

    # ── Save new schema ──
    section("💾", "Save Current Schema")
    with st.container(border=True):
        current_schema = st.session_state.get("single_schema")
        if current_schema:
            st.caption(f"Schema from Single Table page: **{len(current_schema)} columns**")
            st.json(current_schema)

            sc1, sc2 = st.columns(2)
            schema_name = sc1.text_input("Schema Name", placeholder="e.g. Clinical Trial Subjects", key="lib_name")
            schema_tags = sc2.text_input("Tags (comma-separated)", placeholder="e.g. clinical, phase3", key="lib_tags")
            schema_desc = st.text_area("Description", placeholder="Describe what this schema represents…", key="lib_desc", height=80)

            if st.button("💾  Save to Library", type="primary", use_container_width=True):
                if not schema_name.strip():
                    st.error("Schema name is required.")
                else:
                    schema_id = audit.save_schema(
                        name=schema_name.strip(),
                        schema=current_schema,
                        description=schema_desc.strip(),
                        tags=schema_tags.strip(),
                    )
                    st.success(f"✅  Saved as **{schema_name}** (ID: {schema_id})")
                    st.rerun()
        else:
            st.info("Upload a file on the **Single Table** page first to populate a schema here.")

    # ── Import schema from JSON ──
    section("📥", "Import Schema")
    with st.container(border=True):
        uploaded = st.file_uploader("Upload a JSON schema file", type=["json"], key="lib_import")
        if uploaded:
            try:
                imported = json.loads(uploaded.read().decode("utf-8"))
                if isinstance(imported, dict):
                    # Could be {col: type} or {name, schema, ...}
                    schema_data = imported.get("schema", imported)
                    imp_name = imported.get("name", uploaded.name.replace(".json", ""))
                    imp_desc = imported.get("description", "")
                    imp_tags = imported.get("tags", "")
                    imp_fd = imported.get("field_descriptions")

                    st.json(schema_data)
                    if st.button("💾  Import to Library", use_container_width=True):
                        sid = audit.save_schema(
                            name=imp_name,
                            schema=schema_data,
                            description=imp_desc,
                            field_descriptions=imp_fd,
                            tags=imp_tags,
                        )
                        st.success(f"✅  Imported as **{imp_name}** (ID: {sid})")
                        st.rerun()
                else:
                    st.error("Invalid JSON structure. Expected a dictionary.")
            except json.JSONDecodeError:
                st.error("Invalid JSON file.")

    # ── Browse saved schemas ──
    section("📖", "Saved Schemas")

    search = st.text_input("🔍  Search by name or tag", key="lib_search", placeholder="Type to filter…")
    schemas = audit.list_schemas(search=search)

    if not schemas:
        empty_state("📚", "No schemas saved yet", "Save a schema from the Single Table page or import a JSON file.")
        return

    metric_row([(str(len(schemas)), "Schemas Saved")])

    for s in schemas:
        schema_data = json.loads(s["schema_json"]) if isinstance(s.get("schema_json"), str) else s.get("schema_json", {})
        fd = json.loads(s.get("field_descriptions_json", "{}")) if isinstance(s.get("field_descriptions_json"), str) else {}
        tags = s.get("tags", "")
        tag_pills = ""
        if tags:
            tag_pills = " ".join(f"`{t.strip()}`" for t in tags.split(",") if t.strip())

        with st.expander(f"📄  **{s['name']}** — {len(schema_data)} columns  {tag_pills}", expanded=False):
            if s.get("description"):
                st.caption(s["description"])
            st.json(schema_data)

            bc1, bc2, bc3 = st.columns(3)

            # Load into single table
            with bc1:
                if st.button("📋  Use in Generator", key=f"use_{s['id']}", use_container_width=True):
                    st.session_state.single_schema = schema_data
                    st.success("Schema loaded! Switch to **Single Table** page.")

            # Export as JSON
            with bc2:
                export_data = json.dumps({
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "tags": tags,
                    "schema": schema_data,
                    "field_descriptions": fd,
                }, indent=2)
                st.download_button(
                    "📥  Export JSON",
                    data=export_data.encode("utf-8"),
                    file_name=f"{s['name'].replace(' ', '_').lower()}_schema.json",
                    mime="application/json",
                    key=f"export_{s['id']}",
                    use_container_width=True,
                )

            # Delete
            with bc3:
                if st.button("🗑️  Delete", key=f"del_{s['id']}", use_container_width=True):
                    audit.delete_schema(s["id"])
                    st.rerun()
