"""Tests for the chat agent — core/chat_agent.py."""

import json
import time
from unittest.mock import MagicMock, patch

import polars as pl

from core.chat_agent import TOOL_DEFINITIONS, ChatAgent, SessionStore, ToolExecutor


class TestSessionStore:
    def test_create_and_get(self):
        store = SessionStore(max_sessions=10, ttl_seconds=60)
        session = store.create("test-1")
        assert session["id"] == "test-1"
        assert session["messages"] == []
        assert session["data"] == {}

        retrieved = store.get("test-1")
        assert retrieved is session

    def test_get_nonexistent_returns_none(self):
        store = SessionStore(max_sessions=10, ttl_seconds=60)
        assert store.get("nonexistent") is None

    def test_get_or_create(self):
        store = SessionStore(max_sessions=10, ttl_seconds=60)
        session = store.get_or_create("auto-1")
        assert session["id"] == "auto-1"

        same = store.get_or_create("auto-1")
        assert same is session

    def test_ttl_eviction(self):
        store = SessionStore(max_sessions=10, ttl_seconds=1)
        store.create("old")
        store._store["old"]["created_at"] = time.time() - 2
        assert store.get("old") is None

    def test_lru_eviction_at_max(self):
        store = SessionStore(max_sessions=3, ttl_seconds=60)
        store.create("a")
        store.create("b")
        store.create("c")
        assert store.size == 3

        store.create("d")
        assert store.size == 3
        assert store.get("a") is None
        assert store.get("d") is not None

    def test_clear_session(self):
        store = SessionStore(max_sessions=10, ttl_seconds=60)
        store.create("x")
        store.clear("x")
        assert store.get("x") is None

    def test_auto_generate_id(self):
        store = SessionStore(max_sessions=10, ttl_seconds=60)
        session = store.create()
        assert len(session["id"]) == 32


class TestToolExecutor:
    def _make_session(self, **kwargs):
        session = {
            "id": "test",
            "messages": [],
            "data": {},
            "schema": None,
            "field_descriptions": None,
            "reports": {},
            "created_at": time.time(),
        }
        session.update(kwargs)
        return session

    def test_unknown_tool(self):
        session = self._make_session()
        executor = ToolExecutor(session, MagicMock())
        result = executor.execute("nonexistent_tool", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    @patch("core.chat_agent.ForgeEngine")
    def test_generate_data_no_schema(self, mock_engine):
        session = self._make_session()
        executor = ToolExecutor(session, MagicMock())
        result = executor.execute("generate_data", {"num_records": 100})
        assert "error" in result
        assert "No schema" in result["error"]

    @patch("core.chat_agent.ForgeEngine")
    def test_generate_data_with_schema(self, mock_engine_cls):
        mock_engine = MagicMock()
        mock_df = pl.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        mock_engine.generate_records.return_value = mock_df
        mock_engine_cls.return_value = mock_engine

        session = self._make_session(
            schema={"name": "String", "age": "Int64"},
            field_descriptions={"name": "First name"},
        )
        executor = ToolExecutor(session, MagicMock())
        result = executor.execute("generate_data", {"num_records": 2})

        assert result["record_count"] == 2
        assert "name" in result["columns"]
        assert session["data"]["generated"] is mock_df

    def test_generate_schema(self):
        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "schema": {"id": "Int64", "name": "String"},
            "field_descriptions": {"id": "Unique ID", "name": "Full name"},
        })
        mock_provider.chat_complete.return_value = mock_response

        session = self._make_session()
        executor = ToolExecutor(session, mock_provider)
        result = executor.execute("generate_schema", {"description": "employee data"})

        assert result["column_count"] == 2
        assert "id" in result["schema"]
        assert session["schema"] == {"id": "Int64", "name": "String"}

    def test_profile_data_no_data(self):
        session = self._make_session()
        executor = ToolExecutor(session, MagicMock())
        result = executor.execute("profile_data", {"data_source": "uploaded"})
        assert "error" in result

    @patch("core.chat_agent.PrivacyScorecard")
    def test_run_privacy_audit(self, mock_scorecard_cls):
        mock_scorecard = MagicMock()
        mock_scorecard.compute_dcr.return_value = {
            "risk_level": "Low",
            "min_dcr": 0.42,
            "mean_dcr": 0.78,
            "median_dcr": 0.65,
            "pct_exact_matches": 0.0,
        }
        mock_scorecard_cls.return_value = mock_scorecard

        real_df = pl.DataFrame({"a": [1, 2, 3]})
        synth_df = pl.DataFrame({"a": [4, 5, 6]})
        session = self._make_session()
        session["data"]["uploaded"] = real_df
        session["data"]["generated"] = synth_df

        executor = ToolExecutor(session, MagicMock())
        result = executor.execute("run_privacy_audit", {
            "real_source": "uploaded",
            "synthetic_source": "generated",
        })

        assert result["risk_level"] == "Low"
        assert result["min_dcr"] == 0.42

    @patch("core.chat_agent.assess_quality")
    def test_run_quality_check(self, mock_assess):
        mock_assess.return_value = {
            "overall_score": 85,
            "realism_grade": "B",
            "completeness": 1.0,
            "uniqueness": 0.9,
            "distribution_score": 0.8,
            "correlation_preservation": 0.75,
            "warnings": [],
        }

        real_df = pl.DataFrame({"a": [1, 2, 3]})
        synth_df = pl.DataFrame({"a": [4, 5, 6]})
        session = self._make_session()
        session["data"]["uploaded"] = real_df
        session["data"]["generated"] = synth_df

        executor = ToolExecutor(session, MagicMock())
        result = executor.execute("run_quality_check", {
            "original_source": "uploaded",
            "synthetic_source": "generated",
        })

        assert result["grade"] == "B"
        assert result["overall_score"] == 85


class TestChatAgent:
    def _mock_stream_chunks(self, text: str):
        """Create mock streaming chunks that yield text token by token."""
        chunks = []
        for char in text:
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = char
            delta.tool_calls = None
            choice = MagicMock()
            choice.delta = delta
            choice.finish_reason = None
            chunk.choices = [choice]
            chunks.append(chunk)

        final_chunk = MagicMock()
        final_delta = MagicMock()
        final_delta.content = None
        final_delta.tool_calls = None
        final_choice = MagicMock()
        final_choice.delta = final_delta
        final_choice.finish_reason = "stop"
        final_chunk.choices = [final_choice]
        chunks.append(final_chunk)
        return chunks

    @patch("core.chat_agent.get_provider")
    def test_stream_response_simple(self, mock_get_provider):
        mock_provider = MagicMock()
        chunks = self._mock_stream_chunks("Hello!")
        mock_provider.chat_stream.return_value = iter(chunks)
        mock_get_provider.return_value = mock_provider

        agent = ChatAgent()
        events = list(agent.stream_response("sess-1", "Hi"))

        token_events = [e for e in events if e["event"] == "token"]
        done_events = [e for e in events if e["event"] == "done"]

        assert len(token_events) == 6  # H, e, l, l, o, !
        assert len(done_events) == 1
        assert "".join(e["data"]["content"] for e in token_events) == "Hello!"

    @patch("core.chat_agent.get_provider")
    def test_session_persistence_across_calls(self, mock_get_provider):
        mock_provider = MagicMock()
        chunks = self._mock_stream_chunks("Reply")
        mock_provider.chat_stream.return_value = iter(chunks)
        mock_get_provider.return_value = mock_provider

        agent = ChatAgent()
        list(agent.stream_response("sess-2", "First message"))

        session = agent.sessions.get("sess-2")
        assert len(session["messages"]) == 2  # user + assistant
        assert session["messages"][0]["role"] == "user"
        assert session["messages"][1]["role"] == "assistant"
        assert session["messages"][1]["content"] == "Reply"

    @patch("core.chat_agent.get_provider")
    def test_stream_error_handling(self, mock_get_provider):
        mock_provider = MagicMock()
        mock_provider.chat_stream.side_effect = Exception("Connection failed")
        mock_get_provider.return_value = mock_provider

        agent = ChatAgent()
        events = list(agent.stream_response("sess-3", "Hello"))

        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        assert "Connection failed" in error_events[0]["data"]["message"]


class TestToolDefinitions:
    def test_all_tools_have_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func

    def test_tool_count(self):
        assert len(TOOL_DEFINITIONS) == 6
