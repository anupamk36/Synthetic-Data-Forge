"""
ForgeFlow AI — REST API.

FastAPI backend that exposes synthetic data generation, privacy audits,
schema management, and job tracking as HTTP endpoints for programmatic use.
"""

import json
import logging
import os
import sys
import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logging_config import setup_logging
setup_logging()

from core import audit
from core.config import PHARMA_SAFE_MODE
from core.exceptions import ForgeError
from core.generator import ForgeEngine
from core.llm_logic import LLMLogicEngine
from core.privacy import PrivacyScorecard

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────
app = FastAPI(
    title="ForgeFlow AI API",
    description="Programmatic access to synthetic data generation, privacy audits, and schema management.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (run_id → status dict)
_JOBS: dict[str, dict] = {}


# ──────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    model_config = {"populate_by_name": True}
    schema_def: dict[str, str] = Field(..., alias="schema", description="Column name → data type (Int64, Float64, String, Date)")
    count: int = Field(100, ge=1, le=10_000_000, description="Number of records to generate")
    use_llm: bool = Field(False, description="Use LLM for semantically coherent generation")
    field_descriptions: dict[str, str] | None = Field(None, description="Column → semantic hint for LLM")
    seed: int | None = Field(None, description="Random seed for reproducible output")
    output_format: str = Field("json", description="Return format: json, csv, or ndjson")


class SchemaPayload(BaseModel):
    model_config = {"populate_by_name": True}
    name: str = Field(..., min_length=1, max_length=128)
    schema_def: dict[str, str] = Field(..., alias="schema")
    description: str = ""
    field_descriptions: dict[str, str] | None = None
    tags: str = ""


class SchemaUpdatePayload(BaseModel):
    model_config = {"populate_by_name": True}
    name: str | None = None
    schema_def: dict[str, str] | None = Field(None, alias="schema")
    description: str | None = None
    field_descriptions: dict[str, str] | None = None
    tags: str | None = None


# ──────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    llm = LLMLogicEngine()
    return {
        "status": "ok",
        "ollama_available": llm.is_available(),
        "pharma_safe_mode": PHARMA_SAFE_MODE,
    }


# ──────────────────────────────────────────────────────────
# Generate (sync for small, async job for large)
# ──────────────────────────────────────────────────────────
@app.post("/api/v1/generate")
def generate_data(req: GenerateRequest):
    """Generate synthetic data synchronously (for ≤10k records) and return inline."""
    engine = ForgeEngine(seed=req.seed)

    gen_kwargs: dict[str, Any] = {}
    engine_type = "faker"

    if req.use_llm:
        llm = LLMLogicEngine()
        if llm.is_available():
            gen_kwargs["use_llm"] = True
            gen_kwargs["llm_engine"] = llm
            gen_kwargs["field_descriptions"] = req.field_descriptions
            engine_type = "llm"

    run_id = audit.start_run("single", req.schema_def, req.model_dump(by_alias=True), engine=engine_type)

    t0 = time.time()
    try:
        df = engine.generate_records(req.schema_def, req.count, **gen_kwargs)
        elapsed = time.time() - t0

        audit.finish_run(
            run_id,
            status="complete",
            record_count=len(df),
            columns=len(df.columns),
            elapsed_sec=round(elapsed, 2),
        )

        if req.output_format == "csv":
            return {"run_id": run_id, "format": "csv", "data": df.write_csv()}
        elif req.output_format == "ndjson":
            return {"run_id": run_id, "format": "ndjson", "data": df.write_ndjson()}
        else:
            return {"run_id": run_id, "format": "json", "data": df.to_dicts()}

    except Exception as e:
        audit.finish_run(run_id, status="error", error_msg=str(e), elapsed_sec=time.time() - t0)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate/async")
def generate_data_async(req: GenerateRequest):
    """Submit a generation job that runs in the background. Returns a job ID to poll."""
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {"status": "running", "progress": 0, "records_done": 0, "total": req.count}

    def _run():
        engine = ForgeEngine(seed=req.seed)
        gen_kwargs: dict[str, Any] = {}
        engine_type = "faker"

        if req.use_llm:
            llm = LLMLogicEngine()
            if llm.is_available():
                gen_kwargs["use_llm"] = True
                gen_kwargs["llm_engine"] = llm
                gen_kwargs["field_descriptions"] = req.field_descriptions
                engine_type = "llm"

        def _progress(done, total):
            _JOBS[job_id]["records_done"] = done
            _JOBS[job_id]["progress"] = done / total if total else 0

        gen_kwargs["progress_callback"] = _progress
        run_id = audit.start_run("single", req.schema_def, req.model_dump(by_alias=True), engine=engine_type)
        _JOBS[job_id]["run_id"] = run_id

        t0 = time.time()
        try:
            df = engine.generate_records(req.schema_def, req.count, **gen_kwargs)
            elapsed = time.time() - t0
            _JOBS[job_id].update(status="complete", records_done=len(df), data=df.to_dicts())
            audit.finish_run(run_id, status="complete", record_count=len(df),
                             columns=len(df.columns), elapsed_sec=round(elapsed, 2))
        except Exception as e:
            _JOBS[job_id].update(status="error", error=str(e))
            audit.finish_run(run_id, status="error", error_msg=str(e), elapsed_sec=time.time() - t0)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@app.get("/api/v1/jobs/{job_id}")
def get_job_status(job_id: str):
    """Poll an async generation job for status and progress."""
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    result = {k: v for k, v in job.items() if k != "data"}
    if job.get("status") == "complete" and "data" in job:
        result["record_count"] = len(job["data"])
    return result


@app.get("/api/v1/jobs/{job_id}/data")
def get_job_data(job_id: str, format: str = "json"):
    """Retrieve the data from a completed async job."""
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}, not complete")
    data = job.get("data", [])
    if format == "csv":
        import polars as pl
        df = pl.DataFrame(data)
        return {"format": "csv", "data": df.write_csv()}
    return {"format": "json", "data": data}


# ──────────────────────────────────────────────────────────
# Privacy audit
# ──────────────────────────────────────────────────────────
class PrivacyAuditRequest(BaseModel):
    real_data: list[dict]
    synthetic_data: list[dict]


@app.post("/api/v1/privacy/audit")
def privacy_audit(req: PrivacyAuditRequest):
    """Compute DCR privacy metrics between real & synthetic datasets."""
    import polars as pl

    try:
        real_df = pl.DataFrame(req.real_data)
        syn_df = pl.DataFrame(req.synthetic_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {e}")

    scorecard = PrivacyScorecard()
    results = scorecard.compute_dcr(real_df, syn_df)

    if results.get("error"):
        raise HTTPException(status_code=400, detail=results["error"])

    # Remove numpy arrays (not JSON serializable)
    results.pop("dcr_values", None)
    return results


# ──────────────────────────────────────────────────────────
# Schema registry
# ──────────────────────────────────────────────────────────
@app.get("/api/v1/schemas")
def list_schemas(search: str = ""):
    """List all saved schemas, optionally filtered by search term."""
    schemas = audit.list_schemas(search)
    for s in schemas:
        s["schema"] = json.loads(s.pop("schema_json", "{}"))
        s["field_descriptions"] = json.loads(s.pop("field_descriptions_json", "{}"))
    return schemas


@app.get("/api/v1/schemas/{schema_id}")
def get_schema(schema_id: str):
    """Get a saved schema by ID."""
    s = audit.get_schema(schema_id)
    if not s:
        raise HTTPException(status_code=404, detail="Schema not found")
    s["schema"] = json.loads(s.pop("schema_json", "{}"))
    s["field_descriptions"] = json.loads(s.pop("field_descriptions_json", "{}"))
    return s


@app.post("/api/v1/schemas", status_code=201)
def create_schema(req: SchemaPayload):
    """Save a new schema to the registry."""
    schema_id = audit.save_schema(
        name=req.name,
        schema=req.schema_def,
        description=req.description,
        field_descriptions=req.field_descriptions,
        tags=req.tags,
    )
    return {"id": schema_id, "name": req.name}


@app.put("/api/v1/schemas/{schema_id}")
def update_schema(schema_id: str, req: SchemaUpdatePayload):
    """Update an existing saved schema."""
    existing = audit.get_schema(schema_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Schema not found")
    audit.update_schema(
        schema_id,
        name=req.name,
        schema=req.schema_def,
        description=req.description,
        field_descriptions=req.field_descriptions,
        tags=req.tags,
    )
    return {"id": schema_id, "updated": True}


@app.delete("/api/v1/schemas/{schema_id}")
def delete_schema(schema_id: str):
    """Delete a schema from the registry."""
    existing = audit.get_schema(schema_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Schema not found")
    audit.delete_schema(schema_id)
    return {"id": schema_id, "deleted": True}


# ──────────────────────────────────────────────────────────
# Generation history
# ──────────────────────────────────────────────────────────
@app.get("/api/v1/history")
def generation_history(limit: int = Query(50, ge=1, le=500), feature: str | None = None):
    """List recent generation runs from the audit trail."""
    return audit.list_runs(limit=limit, feature=feature)


@app.get("/api/v1/history/{run_id}")
def get_run_detail(run_id: str):
    """Get details of a specific generation run."""
    run = audit.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    run["schema"] = json.loads(run.pop("schema_json", "{}"))
    run["settings"] = json.loads(run.pop("settings_json", "{}"))
    return run
