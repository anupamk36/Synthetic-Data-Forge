"""
Forge AI Chat Agent.

Manages conversational sessions, tool definitions, tool execution,
and streaming response generation via the configured LLM provider.
"""

import json
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Generator

from core import config
from core.generator import ForgeEngine
from core.llm_logic import LLMLogicEngine
from core.llm_providers import get_provider
from core.privacy import PrivacyScorecard
from core.profiler import profile_dataframe
from core.quality import assess_quality

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the Forge AI Assistant, an expert in synthetic data generation. "
    "You help users create realistic, privacy-safe synthetic datasets using "
    "the Synthetic Data Forge platform.\n\n"
    "Your capabilities:\n"
    "- Generate database schemas from natural language descriptions\n"
    "- Run data generation pipelines (statistical + LLM-enhanced)\n"
    "- Profile uploaded datasets for statistics and correlations\n"
    "- Run privacy audits (Distance-to-Closest-Record analysis)\n"
    "- Assess data quality (distribution fidelity, correlation preservation)\n"
    "- Explain metrics and suggest improvements\n\n"
    "Guidelines:\n"
    "- When generating a schema, always show it to the user and ask for "
    "confirmation before generating data\n"
    "- Explain technical metrics in plain language\n"
    "- Suggest next steps after each action\n"
    "- If a tool fails, explain what went wrong and suggest alternatives\n"
    "- Be concise but thorough in explanations"
)

SCHEMA_GEN_PROMPT = (
    "Generate a database schema for the following data description. "
    "Return ONLY a JSON object with two keys:\n"
    '1. "schema": an object mapping column names to Polars dtypes '
    "(String, Int64, Float64, Date, Boolean)\n"
    '2. "field_descriptions": an object mapping column names to short '
    "semantic descriptions for data generation hints.\n\n"
    "Description: {description}\n\n"
    "Return ONLY the JSON, no markdown or explanation."
)

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "generate_schema",
            "description": "Create a data schema from a natural language description of the desired dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What kind of data the user wants",
                    },
                    "num_columns_hint": {
                        "type": "integer",
                        "description": "Approximate number of columns to generate",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_data",
            "description": "Generate synthetic data using the Forge pipeline",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_records": {
                        "type": "integer",
                        "description": "How many records to generate",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["csv", "json", "parquet"],
                        "description": "Output format",
                    },
                    "use_llm": {
                        "type": "boolean",
                        "description": "Whether to use LLM for generation",
                    },
                },
                "required": ["num_records"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "profile_data",
            "description": "Analyze a dataset for statistics, correlations, and constraints",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_source": {
                        "type": "string",
                        "enum": ["uploaded", "generated"],
                        "description": "Which dataset to profile",
                    },
                },
                "required": ["data_source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_privacy_audit",
            "description": "Compute DCR metrics between real and synthetic data",
            "parameters": {
                "type": "object",
                "properties": {
                    "real_source": {
                        "type": "string",
                        "enum": ["uploaded", "generated"],
                        "description": "Source of real data",
                    },
                    "synthetic_source": {
                        "type": "string",
                        "enum": ["uploaded", "generated"],
                        "description": "Source of synthetic data",
                    },
                },
                "required": ["real_source", "synthetic_source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_quality_check",
            "description": "Assess quality and statistical fidelity of synthetic data",
            "parameters": {
                "type": "object",
                "properties": {
                    "original_source": {
                        "type": "string",
                        "enum": ["uploaded", "generated"],
                        "description": "Source of original data",
                    },
                    "synthetic_source": {
                        "type": "string",
                        "enum": ["uploaded", "generated"],
                        "description": "Source of synthetic data",
                    },
                },
                "required": ["original_source", "synthetic_source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_improvements",
            "description": "Analyze reports and recommend parameter adjustments",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "enum": ["quality", "privacy", "both"],
                        "description": "Which area to focus suggestions on",
                    },
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Session Store
# ---------------------------------------------------------------------------


class SessionStore:
    """In-memory LRU session store with TTL eviction."""

    def __init__(self, max_sessions: int | None = None, ttl_seconds: int | None = None):
        self._max = max_sessions or config.CHAT_MAX_SESSIONS
        self._ttl = ttl_seconds or config.CHAT_SESSION_TTL
        self._store: OrderedDict[str, dict] = OrderedDict()

    def get(self, session_id: str) -> dict | None:
        self._evict_expired()
        if session_id in self._store:
            self._store.move_to_end(session_id)
            return self._store[session_id]
        return None

    def get_or_create(self, session_id: str) -> dict:
        session = self.get(session_id)
        if session is None:
            session = self.create(session_id)
        return session

    def create(self, session_id: str | None = None) -> dict:
        self._evict_expired()
        if len(self._store) >= self._max:
            self._store.popitem(last=False)
        sid = session_id or uuid.uuid4().hex
        session = {
            "id": sid,
            "messages": [],
            "data": {},
            "schema": None,
            "field_descriptions": None,
            "reports": {},
            "created_at": time.time(),
        }
        self._store[sid] = session
        return session

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [sid for sid, s in self._store.items() if now - s["created_at"] > self._ttl]
        for sid in expired:
            del self._store[sid]

    @property
    def size(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Tool Executor
# ---------------------------------------------------------------------------


class ToolExecutor:
    """Executes chat tool calls against core Forge modules."""

    def __init__(self, session: dict, provider):
        self._session = session
        self._provider = provider

    def execute(self, tool_name: str, arguments: dict) -> dict:
        handler = getattr(self, f"_tool_{tool_name}", None)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(**arguments)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e)
            return {"error": str(e)}

    def _tool_generate_schema(self, description: str, num_columns_hint: int | None = None) -> dict:
        hint = f" Aim for approximately {num_columns_hint} columns." if num_columns_hint else ""
        prompt = SCHEMA_GEN_PROMPT.format(description=description + hint)

        response = self._provider.chat_complete(
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()

        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse schema JSON: {e}"}

        schema = result.get("schema", {})
        field_descriptions = result.get("field_descriptions", {})

        self._session["schema"] = schema
        self._session["field_descriptions"] = field_descriptions

        preview = []
        if schema:
            preview = [{"column": col, "type": dtype} for col, dtype in schema.items()]

        return {
            "schema": schema,
            "field_descriptions": field_descriptions,
            "preview": preview,
            "column_count": len(schema),
        }

    def _tool_generate_data(self, num_records: int, format: str = "csv", use_llm: bool = True) -> dict:
        schema = self._session.get("schema")
        if not schema:
            return {"error": "No schema available. Generate or upload a schema first."}

        field_descriptions = self._session.get("field_descriptions") or {}

        llm_engine = None
        if use_llm:
            llm_engine = LLMLogicEngine(
                provider_name=config.DEFAULT_LLM_PROVIDER,
            )
            if not llm_engine.is_available():
                llm_engine = None

        engine = ForgeEngine()
        df = engine.generate_records(
            schema,
            num_records,
            field_descriptions=field_descriptions,
            use_llm=use_llm,
            llm_engine=llm_engine,
        )

        self._session["data"]["generated"] = df

        quality_report = None
        if "uploaded" in self._session["data"]:
            try:
                quality_report = assess_quality(self._session["data"]["uploaded"], df)
            except Exception as e:
                logger.warning("Quality assessment failed: %s", e)

        preview_rows = df.head(5).to_dicts()
        for row in preview_rows:
            for k, v in row.items():
                if not isinstance(v, str | int | float | bool | type(None)):
                    row[k] = str(v)

        result = {
            "record_count": len(df),
            "columns": df.columns,
            "format": format,
            "session_id": self._session["id"],
            "data_key": "generated",
            "download_formats": ["csv", "json", "parquet"],
            "preview": preview_rows,
        }
        if quality_report:
            result["quality_grade"] = quality_report.get("realism_grade", "N/A")
            result["quality_score"] = quality_report.get("overall_score", 0)

        return result

    def _tool_profile_data(self, data_source: str = "uploaded") -> dict:
        df = self._session["data"].get(data_source)
        if df is None:
            return {"error": f"No {data_source} data available in this session."}

        profile = profile_dataframe(df)
        summary = profile.summary_for_llm() if hasattr(profile, "summary_for_llm") else {}
        self._session["reports"]["profile"] = summary

        return summary

    def _tool_run_privacy_audit(self, real_source: str = "uploaded", synthetic_source: str = "generated") -> dict:
        real_df = self._session["data"].get(real_source)
        synth_df = self._session["data"].get(synthetic_source)

        if real_df is None:
            return {"error": f"No {real_source} data available."}
        if synth_df is None:
            return {"error": f"No {synthetic_source} data available."}

        scorecard = PrivacyScorecard()
        report = scorecard.compute_dcr(real_df, synth_df)
        self._session["reports"]["privacy"] = report

        return {
            "risk_level": report.get("risk_level", "Unknown"),
            "min_dcr": round(report.get("min_dcr", 0), 4),
            "mean_dcr": round(report.get("mean_dcr", 0), 4),
            "median_dcr": round(report.get("median_dcr", 0), 4),
            "pct_exact_matches": round(report.get("pct_exact_matches", 0), 2),
        }

    def _tool_run_quality_check(self, original_source: str = "uploaded", synthetic_source: str = "generated") -> dict:
        original_df = self._session["data"].get(original_source)
        synthetic_df = self._session["data"].get(synthetic_source)

        if original_df is None:
            return {"error": f"No {original_source} data available."}
        if synthetic_df is None:
            return {"error": f"No {synthetic_source} data available."}

        report = assess_quality(original_df, synthetic_df)
        self._session["reports"]["quality"] = report

        return {
            "overall_score": report.get("overall_score", 0),
            "grade": report.get("realism_grade", "N/A"),
            "completeness": report.get("completeness", 0),
            "uniqueness": report.get("uniqueness", 0),
            "distribution_score": report.get("distribution_score", 0),
            "correlation_preservation": report.get("correlation_preservation", 0),
            "warnings": report.get("warnings", []),
        }

    def _tool_suggest_improvements(self, focus: str = "both") -> dict:
        reports_context = ""
        if focus in ("quality", "both") and "quality" in self._session["reports"]:
            reports_context += f"Quality Report: {json.dumps(self._session['reports']['quality'])}\n\n"
        if focus in ("privacy", "both") and "privacy" in self._session["reports"]:
            reports_context += f"Privacy Report: {json.dumps(self._session['reports']['privacy'])}\n\n"

        if not reports_context:
            return {"error": "No reports available. Run a quality check or privacy audit first."}

        prompt = (
            "Based on these synthetic data reports, provide 3-5 specific, actionable "
            "suggestions to improve the data quality or privacy. Be concise.\n\n"
            f"{reports_context}"
            "Return a JSON object with keys: "
            '"suggestions" (array of strings) and "priority" (high/medium/low).'
        )

        response = self._provider.chat_complete(
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()

        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()

        try:
            result = json.loads(raw)
            return result
        except json.JSONDecodeError:
            return {"suggestions": [raw], "priority": "medium"}


# ---------------------------------------------------------------------------
# Chat Agent
# ---------------------------------------------------------------------------


class ChatAgent:
    """Orchestrates conversation flow with streaming and tool calling."""

    def __init__(self, session_store: SessionStore | None = None):
        self._sessions = session_store or SessionStore()
        self._provider = None
        self._provider_key: tuple[str, str] = ("", "")

    @property
    def sessions(self) -> SessionStore:
        return self._sessions

    def _get_provider(self, provider_name: str | None = None, model: str | None = None):
        name = provider_name or config.CHAT_PROVIDER
        mdl = model or config.CHAT_MODEL
        key = (name, mdl)
        if self._provider is None or self._provider_key != key:
            self._provider = get_provider(name, model=mdl)
            self._provider_key = key
        return self._provider

    def stream_response(
        self, session_id: str, message: str, provider: str | None = None, model: str | None = None
    ) -> Generator[dict, None, None]:
        """Process user message and yield SSE event dicts.

        Yields dicts with keys: event (str), data (dict).
        """
        session = self._sessions.get_or_create(session_id)
        llm = self._get_provider(provider, model)

        session["messages"].append({"role": "user", "content": message})

        if len(session["messages"]) > config.CHAT_MAX_TURNS * 2:
            session["messages"] = session["messages"][-config.CHAT_MAX_TURNS :]

        yield from self._run_completion(session, llm)

    def _run_completion(self, session: dict, llm) -> Generator[dict, None, None]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session["messages"]

        try:
            stream = llm.chat_stream(messages=messages, tools=TOOL_DEFINITIONS)
        except Exception as e:
            logger.error("Chat stream failed: %s", e)
            yield {"event": "error", "data": {"message": str(e)}}
            yield {"event": "done", "data": {}}
            return

        content_parts = []
        tool_calls_acc: dict[int, dict] = {}
        finish_reason = None

        try:
            for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue

                delta = choice.delta
                finish_reason = choice.finish_reason

                if delta.content:
                    content_parts.append(delta.content)
                    yield {"event": "token", "data": {"content": delta.content}}

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc.function:
                            if tc.function.name:
                                tool_calls_acc[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_acc[idx]["arguments"] += tc.function.arguments
        except Exception as e:
            logger.error("Error during stream consumption: %s", e)
            yield {"event": "error", "data": {"message": str(e)}}
            yield {"event": "done", "data": {}}
            return

        if content_parts:
            full_content = "".join(content_parts)
            session["messages"].append({"role": "assistant", "content": full_content})

        if finish_reason == "tool_calls" and tool_calls_acc:
            assistant_msg = {
                "role": "assistant",
                "content": "".join(content_parts) if content_parts else None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in tool_calls_acc.values()
                ],
            }
            if not content_parts:
                session["messages"].pop() if content_parts == [] and session["messages"] and session["messages"][
                    -1
                ].get("role") == "assistant" else None
            session["messages"].append(assistant_msg)

            executor = ToolExecutor(session, llm)
            for _idx, tc in tool_calls_acc.items():
                tool_name = tc["name"]
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}

                yield {"event": "tool_call", "data": {"tool": tool_name, "args": args}}

                result = executor.execute(tool_name, args)
                yield {"event": "tool_result", "data": {"tool": tool_name, "result": result}}

                session["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result),
                    }
                )

            yield from self._run_completion(session, llm)
            return

        yield {"event": "done", "data": {}}
