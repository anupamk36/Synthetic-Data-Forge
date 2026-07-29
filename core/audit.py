"""
Audit trail — persistent generation history backed by SQLite.

Every generation run, schema save, or privacy audit is logged here so
teams can answer "who generated what, when, with which settings?"
"""

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_DB_PATH = os.environ.get("FORGE_AUDIT_DB", os.path.join(os.getcwd(), "forge_audit.db"))
_LOCAL = threading.local()

# ──────────────────────────────────────────────────────────
# Connection helper (thread-safe)
# ──────────────────────────────────────────────────────────


def _conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection, creating tables on first call."""
    if not hasattr(_LOCAL, "conn") or _LOCAL.conn is None:
        _LOCAL.conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _LOCAL.conn.row_factory = sqlite3.Row
        _init_tables(_LOCAL.conn)
    return _LOCAL.conn


def _init_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS generation_runs (
            id          TEXT PRIMARY KEY,
            created_at  TEXT NOT NULL,
            feature     TEXT NOT NULL,          -- single | relational | time_travel
            status      TEXT NOT NULL DEFAULT 'running', -- running | complete | stopped | error
            schema_json TEXT,
            settings_json TEXT,
            record_count INTEGER DEFAULT 0,
            columns     INTEGER DEFAULT 0,
            elapsed_sec REAL DEFAULT 0,
            output_path TEXT,
            error_msg   TEXT,
            engine      TEXT DEFAULT 'faker',   -- faker | llm
            model_name  TEXT
        );

        CREATE TABLE IF NOT EXISTS saved_schemas (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            description TEXT DEFAULT '',
            schema_json TEXT NOT NULL,
            field_descriptions_json TEXT,
            tags        TEXT DEFAULT '',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_runs_created ON generation_runs(created_at);
        CREATE INDEX IF NOT EXISTS idx_runs_feature ON generation_runs(feature);
        CREATE INDEX IF NOT EXISTS idx_schemas_name ON saved_schemas(name);
    """)
    conn.commit()


# ──────────────────────────────────────────────────────────
# Generation run tracking
# ──────────────────────────────────────────────────────────


def start_run(
    feature: str,
    schema: dict,
    settings: dict | None = None,
    engine: str = "faker",
    model_name: str | None = None,
) -> str:
    """Record the start of a generation run. Returns the run ID."""
    run_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    conn = _conn()
    conn.execute(
        """INSERT INTO generation_runs
           (id, created_at, feature, status, schema_json, settings_json, engine, model_name)
           VALUES (?, ?, ?, 'running', ?, ?, ?, ?)""",
        (
            run_id,
            now,
            feature,
            json.dumps(schema, default=str),
            json.dumps(settings or {}, default=str),
            engine,
            model_name or "",
        ),
    )
    conn.commit()
    logger.info("Audit: started run %s [%s / %s]", run_id, feature, engine)
    return run_id


def finish_run(
    run_id: str,
    *,
    status: str = "complete",
    record_count: int = 0,
    columns: int = 0,
    elapsed_sec: float = 0,
    output_path: str = "",
    error_msg: str = "",
):
    """Mark a generation run as finished (complete / stopped / error)."""
    conn = _conn()
    conn.execute(
        """UPDATE generation_runs
           SET status = ?, record_count = ?, columns = ?, elapsed_sec = ?,
               output_path = ?, error_msg = ?
           WHERE id = ?""",
        (status, record_count, columns, elapsed_sec, output_path, error_msg, run_id),
    )
    conn.commit()
    logger.info("Audit: finished run %s → %s (%d records)", run_id, status, record_count)


def list_runs(limit: int = 50, feature: str | None = None) -> list[dict]:
    """Return recent generation runs, newest first."""
    conn = _conn()
    if feature:
        rows = conn.execute(
            "SELECT * FROM generation_runs WHERE feature = ? ORDER BY created_at DESC LIMIT ?",
            (feature, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM generation_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_run(run_id: str) -> dict | None:
    """Get a single run by ID."""
    conn = _conn()
    row = conn.execute("SELECT * FROM generation_runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


# ──────────────────────────────────────────────────────────
# Schema registry
# ──────────────────────────────────────────────────────────


def save_schema(
    name: str,
    schema: dict,
    description: str = "",
    field_descriptions: dict | None = None,
    tags: str = "",
) -> str:
    """Save a named schema to the registry. Returns the schema ID."""
    schema_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    conn = _conn()
    conn.execute(
        """INSERT INTO saved_schemas
           (id, name, description, schema_json, field_descriptions_json, tags, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            schema_id,
            name,
            description,
            json.dumps(schema, default=str),
            json.dumps(field_descriptions or {}, default=str),
            tags,
            now,
            now,
        ),
    )
    conn.commit()
    logger.info("Audit: saved schema '%s' (id=%s)", name, schema_id)
    return schema_id


def update_schema(
    schema_id: str,
    name: str | None = None,
    schema: dict | None = None,
    description: str | None = None,
    field_descriptions: dict | None = None,
    tags: str | None = None,
):
    """Update fields of an existing saved schema."""
    conn = _conn()
    now = datetime.now(timezone.utc).isoformat()
    row = conn.execute("SELECT * FROM saved_schemas WHERE id = ?", (schema_id,)).fetchone()
    if not row:
        return

    conn.execute(
        """UPDATE saved_schemas
           SET name = ?, description = ?, schema_json = ?,
               field_descriptions_json = ?, tags = ?, updated_at = ?
           WHERE id = ?""",
        (
            name if name is not None else row["name"],
            description if description is not None else row["description"],
            json.dumps(schema, default=str) if schema is not None else row["schema_json"],
            json.dumps(field_descriptions, default=str)
            if field_descriptions is not None
            else row["field_descriptions_json"],
            tags if tags is not None else row["tags"],
            now,
            schema_id,
        ),
    )
    conn.commit()


def delete_schema(schema_id: str):
    """Delete a schema from the registry."""
    conn = _conn()
    conn.execute("DELETE FROM saved_schemas WHERE id = ?", (schema_id,))
    conn.commit()
    logger.info("Audit: deleted schema id=%s", schema_id)


def list_schemas(search: str = "") -> list[dict]:
    """Return all saved schemas, optionally filtered by name/tag search."""
    conn = _conn()
    if search:
        pattern = f"%{search}%"
        rows = conn.execute(
            "SELECT * FROM saved_schemas WHERE name LIKE ? OR tags LIKE ? ORDER BY updated_at DESC",
            (pattern, pattern),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM saved_schemas ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_schema(schema_id: str) -> dict | None:
    """Get a single saved schema by ID."""
    conn = _conn()
    row = conn.execute("SELECT * FROM saved_schemas WHERE id = ?", (schema_id,)).fetchone()
    return dict(row) if row else None
