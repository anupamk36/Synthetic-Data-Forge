"""Tests for the REST API — api/server.py."""

import os
import tempfile

# Point audit DB to a temp file
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["FORGE_AUDIT_DB"] = _tmp_db.name
_tmp_db.close()

import pytest
from fastapi.testclient import TestClient

from api.server import app

client = TestClient(app)


class TestHealth:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "ollama_available" in data


class TestGenerate:
    def test_generate_faker(self):
        r = client.post("/api/v1/generate", json={
            "schema": {"name": "String", "age": "Int64"},
            "count": 10,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["run_id"]
        assert len(data["data"]) == 10

    def test_generate_csv_format(self):
        r = client.post("/api/v1/generate", json={
            "schema": {"id": "Int64"},
            "count": 5,
            "output_format": "csv",
        })
        assert r.status_code == 200
        assert r.json()["format"] == "csv"

    def test_generate_with_seed(self):
        payload = {"schema": {"val": "Int64"}, "count": 5, "seed": 42}
        r1 = client.post("/api/v1/generate", json=payload)
        r2 = client.post("/api/v1/generate", json=payload)
        assert r1.json()["data"] == r2.json()["data"]


class TestAsyncGenerate:
    def test_async_flow(self):
        r = client.post("/api/v1/generate/async", json={
            "schema": {"x": "Int64"},
            "count": 20,
        })
        assert r.status_code == 200
        job_id = r.json()["job_id"]

        # Poll until done (with timeout)
        import time
        for _ in range(20):
            status = client.get(f"/api/v1/jobs/{job_id}").json()
            if status["status"] in ("complete", "error"):
                break
            time.sleep(0.3)

        assert status["status"] == "complete"

        # Fetch data
        data_r = client.get(f"/api/v1/jobs/{job_id}/data")
        assert data_r.status_code == 200
        assert len(data_r.json()["data"]) == 20


class TestSchemaAPI:
    def test_crud(self):
        # Create
        r = client.post("/api/v1/schemas", json={
            "name": "API Test Schema",
            "schema": {"id": "Int64", "name": "String"},
            "tags": "test",
        })
        assert r.status_code == 201
        sid = r.json()["id"]

        # Read
        r = client.get(f"/api/v1/schemas/{sid}")
        assert r.status_code == 200
        assert r.json()["name"] == "API Test Schema"

        # List
        r = client.get("/api/v1/schemas")
        assert r.status_code == 200
        assert any(s["id"] == sid for s in r.json())

        # Update
        r = client.put(f"/api/v1/schemas/{sid}", json={"name": "Updated"})
        assert r.status_code == 200

        # Delete
        r = client.delete(f"/api/v1/schemas/{sid}")
        assert r.status_code == 200
        assert client.get(f"/api/v1/schemas/{sid}").status_code == 404

    def test_404(self):
        assert client.get("/api/v1/schemas/nonexistent").status_code == 404


class TestHistory:
    def test_list_history(self):
        # Generate something first to populate history
        client.post("/api/v1/generate", json={"schema": {"a": "Int64"}, "count": 5})
        r = client.get("/api/v1/history")
        assert r.status_code == 200
        assert len(r.json()) > 0

    def test_filter_by_feature(self):
        r = client.get("/api/v1/history?feature=single")
        assert r.status_code == 200
