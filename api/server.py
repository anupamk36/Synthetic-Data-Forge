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
from io import BytesIO
from typing import Any

import polars as pl
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logging_config import setup_logging

setup_logging()

from api.chat_routes import router as chat_router
from api.medical_routes import router as medical_router
from api.test_intelligence_routes import router as test_intelligence_router
from core import audit
from core.config import PHARMA_SAFE_MODE, validate_output_path
from core.generator import ForgeEngine
from core.llm_logic import LLMLogicEngine
from core.llm_providers import AVAILABLE_PROVIDERS, get_provider_models
from core.privacy import PrivacyScorecard
from core.profiler import profile_dataframe
from core.quality import assess_quality
from core.relational import RelationalEngine
from core.sinks import get_sink

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

app.include_router(chat_router)
app.include_router(medical_router)
app.include_router(test_intelligence_router)

# In-memory job store (run_id → status dict)
_JOBS: dict[str, dict] = {}


def _serialize_rows(df: pl.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of dicts with JSON-safe types.
    Polars to_dicts() preserves date/datetime as Python objects which
    aren't JSON-serializable — cast them to strings first."""
    date_cols = [
        col
        for col, dtype in zip(df.columns, df.dtypes, strict=False)
        if "Date" in str(dtype) or "Datetime" in str(dtype)
    ]
    if date_cols:
        df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in date_cols])
    return df.to_dicts()


# ──────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    model_config = {"populate_by_name": True}
    schema_def: dict[str, str] = Field(
        ..., alias="schema", description="Column name → data type (Int64, Float64, String, Date)"
    )
    count: int = Field(100, ge=1, le=10_000_000, description="Number of records to generate")
    use_llm: bool = Field(False, description="Use LLM for semantically coherent generation")
    field_descriptions: dict[str, str] | None = Field(None, description="Column → semantic hint for LLM")
    seed: int | None = Field(None, description="Random seed for reproducible output")
    output_format: str = Field("json", description="Return format: json, csv, or ndjson")
    provider: str = Field("ollama", description="LLM provider: ollama, claude, openai, or gemini")
    model: str | None = Field(None, description="Model name (provider-specific)")
    api_key: str | None = Field(None, description="API key for cloud providers (not logged)")
    llm_validation: bool = Field(True, description="Enable LLM semantic validation pass")
    validation_sample_rate: float = Field(1.0, ge=0.0, le=1.0, description="Fraction of rows to validate")
    token_budget_usd: float = Field(1.0, ge=0.0, description="Max API spend in USD (0 = unlimited)")


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


class RelationalRequest(BaseModel):
    tables: dict[str, dict[str, str]] = Field(..., description="Table name -> {col: type}")
    relationships: list[dict] = Field(default_factory=list, description="FK relationships")
    counts: dict[str, int] = Field(..., description="Table name -> row count")
    source_data: dict[str, list[dict]] | None = Field(None, description="Optional source data per table")


class QualityRequest(BaseModel):
    generated_data: list[dict] = Field(..., description="Generated rows")
    original_data: list[dict] | None = Field(None, description="Original rows for comparison")
    expected_schema: dict[str, str] | None = Field(None, description="Expected column types")


class ExportRequest(BaseModel):
    data: list[dict] = Field(..., description="Row data to export")
    sink_type: str = Field("local", description="local or s3")
    output_path: str = Field("./output_data", description="Destination path")
    output_format: str = Field("parquet", description="parquet, csv, or json")
    records_per_file: int = Field(250, ge=1, description="Rows per output file")
    partition_on: list[str] | None = Field(None, description="Partition columns")
    s3_bucket: str | None = Field(None, description="S3 bucket name")
    s3_prefix: str | None = Field(None, description="S3 key prefix")
    s3_region: str | None = Field(None, description="AWS region")
    s3_access_key: str | None = Field(None, description="AWS access key ID")
    s3_secret_key: str | None = Field(None, description="AWS secret access key")
    s3_session_token: str | None = Field(None, description="AWS session token (optional)")


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

    gen_kwargs: dict[str, Any] = {
        "enable_validation": req.llm_validation,
        "validation_sample_rate": req.validation_sample_rate,
    }
    engine_type = "faker"

    if req.use_llm or req.llm_validation:
        llm = LLMLogicEngine(
            provider_name=req.provider,
            api_key=req.api_key,
            model=req.model,
        )
        if llm.is_available():
            if req.use_llm:
                gen_kwargs["use_llm"] = True
                gen_kwargs["field_descriptions"] = req.field_descriptions
                engine_type = "llm"
            gen_kwargs["llm_engine"] = llm

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
            return {"run_id": run_id, "format": "json", "data": _serialize_rows(df)}

    except Exception as e:
        audit.finish_run(run_id, status="error", error_msg=str(e), elapsed_sec=time.time() - t0)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/generate/async")
def generate_data_async(req: GenerateRequest):
    """Submit a generation job that runs in the background. Returns a job ID to poll."""
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {
        "status": "running",
        "progress": 0,
        "records_done": 0,
        "total": req.count,
        "stop_requested": False,
        "partial_data": [],
    }

    def _run():
        engine = ForgeEngine(seed=req.seed)
        gen_kwargs: dict[str, Any] = {
            "enable_validation": req.llm_validation,
            "validation_sample_rate": req.validation_sample_rate,
        }
        engine_type = "faker"

        if req.use_llm or req.llm_validation:
            llm = LLMLogicEngine(
                provider_name=req.provider,
                api_key=req.api_key,
                model=req.model,
            )
            if llm.is_available():
                if req.use_llm:
                    gen_kwargs["use_llm"] = True
                    gen_kwargs["field_descriptions"] = req.field_descriptions
                    engine_type = "llm"
                gen_kwargs["llm_engine"] = llm

        def _progress(done, total):
            if _JOBS[job_id].get("stop_requested"):
                _JOBS[job_id]["status"] = "stopped"
                raise InterruptedError("Job stopped by user")
            _JOBS[job_id]["records_done"] = done
            _JOBS[job_id]["progress"] = done / total if total else 0

        def _batch_callback(records: list[dict]):
            _JOBS[job_id]["partial_data"].extend(records)

        gen_kwargs["batch_callback"] = _batch_callback

        gen_kwargs["progress_callback"] = _progress
        run_id = audit.start_run("single", req.schema_def, req.model_dump(by_alias=True), engine=engine_type)
        _JOBS[job_id]["run_id"] = run_id

        t0 = time.time()
        try:
            df = engine.generate_records(req.schema_def, req.count, **gen_kwargs)
            elapsed = time.time() - t0
            _JOBS[job_id].update(status="complete", records_done=len(df), data=_serialize_rows(df))
            audit.finish_run(
                run_id, status="complete", record_count=len(df), columns=len(df.columns), elapsed_sec=round(elapsed, 2)
            )
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
    result = {k: v for k, v in job.items() if k not in ("data", "partial_data")}
    if job.get("status") == "complete" and "data" in job:
        result["record_count"] = len(job["data"])
    partial = job.get("partial_data")
    if partial and job.get("status") == "running":
        result["partial_data"] = partial[-20:]
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
        raise HTTPException(status_code=400, detail=f"Invalid data: {e}") from e

    scorecard = PrivacyScorecard()
    results = scorecard.compute_dcr(real_df, syn_df)

    if results.get("error"):
        raise HTTPException(status_code=400, detail=results["error"])

    # Remove numpy arrays (not JSON serializable)
    results.pop("dcr_values", None)
    return results


class PrivacyReportRequest(BaseModel):
    real_data: list[dict]
    synthetic_data: list[dict]
    quasi_identifiers: list[str] | None = None
    sensitive_column: str | None = None


@app.post("/api/v1/privacy/report")
def privacy_report(req: PrivacyReportRequest):
    """Generate a full privacy compliance report with DCR, k-anonymity, l-diversity, and epsilon."""
    try:
        real_df = pl.DataFrame(req.real_data)
        syn_df = pl.DataFrame(req.synthetic_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {e}") from e

    scorecard = PrivacyScorecard()
    try:
        report = scorecard.generate_compliance_report(
            real_df,
            syn_df,
            quasi_identifiers=req.quasi_identifiers,
            sensitive_col=req.sensitive_column,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return report


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


# ──────────────────────────────────────────────────────────
# Providers
# ──────────────────────────────────────────────────────────
@app.get("/api/v1/providers")
def list_providers():
    """List available LLM providers and their models."""
    result = []
    for name in AVAILABLE_PROVIDERS:
        models = get_provider_models(name)
        entry = {"name": name, "models": models}
        if name in ("ollama", "alchemy"):
            llm = LLMLogicEngine(provider_name=name)
            entry["available"] = llm.is_available()
        else:
            entry["available"] = None  # requires API key to check
        result.append(entry)
    return result


# ──────────────────────────────────────────────────────────
# Data Profiling
# ──────────────────────────────────────────────────────────
class ProfileRequest(BaseModel):
    data: list[dict] = Field(..., description="Array of row objects to profile")


@app.post("/api/v1/profile")
def profile_data(req: ProfileRequest):
    """Profile uploaded data and return statistical summary."""
    import polars as pl

    try:
        df = pl.DataFrame(req.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {e}") from e
    profile = profile_dataframe(df)
    return profile.to_dict()


# ──────────────────────────────────────────────────────────
# Cost Estimation
# ──────────────────────────────────────────────────────────
class EstimateRequest(BaseModel):
    model_config = {"populate_by_name": True}
    schema_def: dict[str, str] = Field(..., alias="schema")
    count: int = Field(100, ge=1, le=10_000_000)
    provider: str = Field("ollama")
    model: str | None = None


@app.post("/api/v1/estimate")
def estimate_cost(req: EstimateRequest):
    """Estimate API cost for a generation task."""
    llm = LLMLogicEngine(provider_name=req.provider, model=req.model)
    cost = llm.estimate_cost(req.schema_def, req.count)
    return {"provider": req.provider, "model": req.model, "count": req.count, "estimated_cost_usd": round(cost, 6)}


# ──────────────────────────────────────────────────────────
# File Upload & Schema Inference
# ──────────────────────────────────────────────────────────
_POLARS_DTYPE_MAP = {
    "Int8": "Int64",
    "Int16": "Int64",
    "Int32": "Int64",
    "Int64": "Int64",
    "UInt8": "Int64",
    "UInt16": "Int64",
    "UInt32": "Int64",
    "UInt64": "Int64",
    "Float32": "Float64",
    "Float64": "Float64",
    "Date": "Date",
    "Datetime": "Date",
    "Utf8": "String",
    "String": "String",
    "Categorical": "String",
    "Boolean": "String",
}


def _map_polars_dtype(dtype: pl.DataType) -> str:
    """Map a polars dtype to our canonical type string."""
    dtype_str = str(dtype)
    # Handle parameterised types like Datetime(time_unit='us', ...)
    base = dtype_str.split("(")[0]
    return _POLARS_DTYPE_MAP.get(base, "String")


@app.post("/api/v1/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept a CSV, Parquet, JSON, or JSONL file upload, infer schema, and return a preview."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    name = file.filename.lower()
    content = await file.read()

    try:
        if name.endswith(".parquet"):
            df = pl.read_parquet(BytesIO(content))
        elif name.endswith(".json"):
            try:
                df = pl.read_json(BytesIO(content))
            except Exception:
                df = pl.read_ndjson(BytesIO(content))
        elif name.endswith(".jsonl") or name.endswith(".ndjson"):
            df = pl.read_ndjson(BytesIO(content))
        else:
            # Default to CSV (covers .csv and unknown extensions)
            df = pl.read_csv(BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}") from e

    schema = {col: _map_polars_dtype(dtype) for col, dtype in zip(df.columns, df.dtypes, strict=False)}
    raw_rows = _serialize_rows(df.head(500))

    # Truncate long string values to prevent massive JSON payloads
    sample_rows = []
    for row in raw_rows:
        trimmed = {}
        for k, v in row.items():
            if isinstance(v, str) and len(v) > 200:
                trimmed[k] = v[:200] + "..."
            else:
                trimmed[k] = v
        sample_rows.append(trimmed)

    return {"schema": schema, "sample_rows": sample_rows, "row_count": len(df)}


# ──────────────────────────────────────────────────────────
# Relational (Multi-Table) Generation
# ──────────────────────────────────────────────────────────
@app.post("/api/v1/generate/relational")
def generate_relational(req: RelationalRequest):
    """Generate multi-table synthetic data with FK integrity."""
    try:
        engine = RelationalEngine()

        for table_name, schema in req.tables.items():
            engine.add_table(table_name, schema)

        for rel in req.relationships:
            engine.add_relationship(
                rel["parent_table"],
                rel["parent_col"],
                rel["child_table"],
                rel["child_col"],
            )

        if req.source_data:
            for table_name, rows in req.source_data.items():
                engine.set_source_data(table_name, pl.DataFrame(rows))

        results = engine.generate_all(req.counts)
        return {name: _serialize_rows(df) for name, df in results.items()}

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing relationship field: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ──────────────────────────────────────────────────────────
# Quality Assessment
# ──────────────────────────────────────────────────────────
@app.post("/api/v1/quality/assess")
def quality_assess(req: QualityRequest):
    """Assess the statistical quality and realism of generated data."""
    try:
        generated_df = pl.DataFrame(req.generated_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid generated_data: {e}") from e

    original_df = None
    if req.original_data:
        try:
            original_df = pl.DataFrame(req.original_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid original_data: {e}") from e

    try:
        report = assess_quality(
            generated_df,
            original_df=original_df,
            expected_schema=req.expected_schema,
        )
        return report.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ──────────────────────────────────────────────────────────
# Data Export (Sink)
# ──────────────────────────────────────────────────────────
@app.post("/api/v1/export")
def export_data(req: ExportRequest):
    """Write generated data to a server-side sink (local filesystem or S3)."""
    try:
        df = pl.DataFrame(req.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {e}") from e

    try:
        if req.sink_type == "s3":
            if not req.s3_bucket:
                raise HTTPException(status_code=400, detail="s3_bucket is required for S3 sink")
            sink = get_sink(
                "s3",
                bucket=req.s3_bucket,
                prefix=req.s3_prefix or "",
                region=req.s3_region or "us-east-1",
                aws_access_key_id=req.s3_access_key or "",
                aws_secret_access_key=req.s3_secret_key or "",
                aws_session_token=req.s3_session_token or "",
            )
            files = sink.push(
                df,
                destination=req.output_path,
                file_format=req.output_format,
                records_per_file=req.records_per_file,
                partitions=req.partition_on,
            )
        else:
            validated_path = validate_output_path(req.output_path)
            sink = get_sink("local")
            files = sink.push(
                df,
                destination=validated_path,
                file_format=req.output_format,
                records_per_file=req.records_per_file,
                partitions=req.partition_on,
            )

        return {"files_written": files}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ──────────────────────────────────────────────────────────
# Job Stop
# ──────────────────────────────────────────────────────────
@app.post("/api/v1/jobs/{job_id}/stop")
def stop_job(job_id: str):
    """Signal a running async job to stop."""
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "running":
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job['status']}, only running jobs can be stopped",
        )
    job["stop_requested"] = True
    return {"job_id": job_id, "stop_requested": True}
