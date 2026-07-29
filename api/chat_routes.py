"""
Chat API endpoints — SSE streaming + supporting operations.
"""

import json
import logging
import uuid
from io import BytesIO

import polars as pl
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from core.chat_agent import ChatAgent, SessionStore
from core.config import MAX_UPLOAD_SIZE_MB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

_session_store = SessionStore()
_agent = ChatAgent(session_store=_session_store)


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    message: str
    provider: str | None = None
    model: str | None = None


class ClearRequest(BaseModel):
    session_id: str


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """Stream chat response as SSE events."""
    def event_generator():
        for event in _agent.stream_response(
            session_id=req.session_id,
            message=req.message,
            provider=req.provider,
            model=req.model,
        ):
            yield {
                "event": event["event"],
                "data": json.dumps(event["data"]),
            }

    return EventSourceResponse(event_generator())


@router.post("/upload")
async def chat_upload(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a file into the chat session data store."""
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_UPLOAD_SIZE_MB}MB limit.",
        )

    filename = file.filename or "upload"
    buf = BytesIO(contents)

    try:
        if filename.endswith(".parquet"):
            df = pl.read_parquet(buf)
        elif filename.endswith(".json") or filename.endswith(".jsonl"):
            df = pl.read_json(buf)
        else:
            df = pl.read_csv(buf)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    session = _session_store.get_or_create(session_id)
    session["data"]["uploaded"] = df

    return {
        "data_key": "uploaded",
        "rows": len(df),
        "columns": df.columns,
    }


@router.post("/clear")
async def chat_clear(req: ClearRequest):
    """Clear conversation history for a session."""
    _session_store.clear(req.session_id)
    return {"cleared": True}


@router.get("/models")
async def chat_models():
    """List available models for the chat provider."""
    from core.llm_providers import get_provider_models
    from core import config

    models = get_provider_models(config.CHAT_PROVIDER)
    return {
        "models": models,
        "default": config.CHAT_MODEL,
        "provider": config.CHAT_PROVIDER,
    }


@router.get("/download/{session_id}/{data_key}")
async def chat_download(
    session_id: str,
    data_key: str,
    format: str = Query("csv"),
):
    """Download data from the chat session store."""
    session = _session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    df = session["data"].get(data_key)
    if df is None:
        raise HTTPException(status_code=404, detail=f"No '{data_key}' data in session.")

    buf = BytesIO()
    if format == "parquet":
        df.write_parquet(buf)
        media_type = "application/octet-stream"
        ext = "parquet"
    elif format == "json":
        buf.write(df.write_json().encode())
        media_type = "application/json"
        ext = "json"
    else:
        df.write_csv(buf)
        media_type = "text/csv"
        ext = "csv"

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename=forge_data.{ext}"},
    )
