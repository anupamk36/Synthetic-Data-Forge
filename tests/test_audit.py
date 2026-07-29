"""Tests for core/audit.py — SQLite audit trail and schema registry."""

import os
import tempfile

# Point audit DB to a temp file before importing
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["FORGE_AUDIT_DB"] = _tmp_db.name
_tmp_db.close()

from core import audit


class TestGenerationRuns:
    def test_start_and_finish(self):
        run_id = audit.start_run("single", {"name": "String"}, {"count": 100}, engine="faker")
        assert run_id
        audit.finish_run(run_id, status="complete", record_count=100, columns=1, elapsed_sec=1.5)
        run = audit.get_run(run_id)
        assert run["status"] == "complete"
        assert run["record_count"] == 100

    def test_list_runs(self):
        audit.start_run("single", {"a": "Int64"})
        audit.start_run("relational", {"b": "String"})
        runs = audit.list_runs(limit=10)
        assert len(runs) >= 2

    def test_list_runs_by_feature(self):
        audit.start_run("time_travel", {"c": "Date"})
        runs = audit.list_runs(feature="time_travel")
        assert all(r["feature"] == "time_travel" for r in runs)

    def test_error_run(self):
        run_id = audit.start_run("single", {"x": "String"})
        audit.finish_run(run_id, status="error", error_msg="test error")
        run = audit.get_run(run_id)
        assert run["status"] == "error"
        assert run["error_msg"] == "test error"

    def test_get_nonexistent_run(self):
        assert audit.get_run("nonexistent") is None


class TestSchemaRegistry:
    def test_save_and_get(self):
        sid = audit.save_schema("Test Schema", {"id": "Int64", "name": "String"}, description="A test schema")
        s = audit.get_schema(sid)
        assert s["name"] == "Test Schema"
        assert "id" in s["schema_json"]

    def test_list_schemas(self):
        audit.save_schema("Alpha", {"x": "Int64"}, tags="clinical")
        audit.save_schema("Beta", {"y": "String"}, tags="financial")
        schemas = audit.list_schemas()
        assert len(schemas) >= 2

    def test_search_schemas(self):
        audit.save_schema("Gamma Search", {"z": "Float64"}, tags="pharma,trial")
        results = audit.list_schemas(search="pharma")
        assert any("Gamma" in s["name"] for s in results)

    def test_update_schema(self):
        sid = audit.save_schema("Update Me", {"a": "String"})
        audit.update_schema(sid, name="Updated Name")
        s = audit.get_schema(sid)
        assert s["name"] == "Updated Name"

    def test_delete_schema(self):
        sid = audit.save_schema("Delete Me", {"a": "Int64"})
        audit.delete_schema(sid)
        assert audit.get_schema(sid) is None

    def test_get_nonexistent_schema(self):
        assert audit.get_schema("nonexistent") is None
