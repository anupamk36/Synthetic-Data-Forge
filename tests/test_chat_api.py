"""Tests for chat API endpoints — api/chat_routes.py."""

import json
from io import BytesIO
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.server import app

client = TestClient(app)


class TestChatStream:
    @patch("api.chat_routes._agent")
    def test_stream_basic(self, mock_agent):
        mock_agent.stream_response.return_value = iter([
            {"event": "token", "data": {"content": "Hello"}},
            {"event": "done", "data": {}},
        ])

        response = client.post(
            "/api/v1/chat/stream",
            json={"session_id": "test-sess", "message": "Hi"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = response.text.strip().split("\n")
        events = []
        current_event = {}
        for line in lines:
            if line.startswith("event:"):
                current_event["event"] = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                current_event["data"] = line.split(":", 1)[1].strip()
                events.append(current_event)
                current_event = {}

        assert len(events) >= 2
        assert events[0]["event"] == "token"

    @patch("api.chat_routes._agent")
    def test_stream_with_tool_call(self, mock_agent):
        mock_agent.stream_response.return_value = iter([
            {"event": "tool_call", "data": {"tool": "generate_schema", "args": {"description": "test"}}},
            {"event": "tool_result", "data": {"tool": "generate_schema", "result": {"schema": {}}}},
            {"event": "token", "data": {"content": "Done"}},
            {"event": "done", "data": {}},
        ])

        response = client.post(
            "/api/v1/chat/stream",
            json={"session_id": "test-sess", "message": "Generate a schema"},
        )
        assert response.status_code == 200


class TestChatUpload:
    def test_upload_csv(self):
        csv_content = b"name,age\nAlice,30\nBob,25\n"
        response = client.post(
            "/api/v1/chat/upload",
            data={"session_id": "upload-test"},
            files={"file": ("test.csv", BytesIO(csv_content), "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data_key"] == "uploaded"
        assert data["rows"] == 2
        assert "name" in data["columns"]

    def test_upload_invalid_file(self):
        response = client.post(
            "/api/v1/chat/upload",
            data={"session_id": "bad-upload"},
            files={"file": ("test.csv", BytesIO(b"not,valid\x00\x01\x02"), "text/csv")},
        )
        # Should either parse it or return 400
        assert response.status_code in (200, 400)


class TestChatClear:
    def test_clear_session(self):
        response = client.post(
            "/api/v1/chat/clear",
            json={"session_id": "clear-me"},
        )
        assert response.status_code == 200
        assert response.json()["cleared"] is True


class TestChatModels:
    @patch("core.llm_providers.get_provider_models")
    def test_list_models(self, mock_models):
        mock_models.return_value = ["gemini-2.5-flash", "gpt-4o"]
        response = client.get("/api/v1/chat/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "default" in data


class TestChatDownload:
    def test_download_nonexistent_session(self):
        response = client.get("/api/v1/chat/download/fake-sess/generated")
        assert response.status_code == 404
