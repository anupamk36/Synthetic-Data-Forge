"""Medical data generation API routes — FHIR, HL7v2, Clinical Trials."""

import json
import logging
import threading
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.medical.engine import MedicalEngine, ALL_RESOURCE_TYPES, RESOURCE_DEPENDENCIES, resolve_dependencies
from core.medical.fhir.bundle import build_bundle, bundle_to_ndjson, bundle_stats
from core.medical.fhir.validator import validate_resource, validate_bundle
from core.medical.fhir.hl7v2_converter import convert_registry_to_hl7v2
from core.medical.terminologies.loader import search_codes
from core.llm_providers import get_provider
from core.medical.narrative_engine import ClinicalNarrativeEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/medical", tags=["medical"])

_MEDICAL_JOBS: dict[str, dict] = {}

TERMINOLOGY_FILES = {
    "icd10": "icd10_common.json",
    "loinc": "loinc_common.json",
    "snomed": "snomed_common.json",
    "rxnorm": "rxnorm_common.json",
}


class EncounterRange(BaseModel):
    min: int = 1
    max: int = 5


class FHIRGenerateRequest(BaseModel):
    resource_types: list[str] = Field(default_factory=lambda: list(ALL_RESOURCE_TYPES))
    patient_count: int = Field(default=100, ge=1, le=50000)
    encounters_per_patient: EncounterRange = Field(default_factory=EncounterRange)
    clinical_density: str = Field(default="moderate", pattern="^(low|moderate|high)$")
    output_format: str = Field(default="bundle", pattern="^(bundle|ndjson|individual|tabular)$")
    bundle_type: str = Field(default="collection", pattern="^(collection|transaction)$")
    include_narrative: bool = False
    terminology_focus: str | None = None
    seed: int | None = None
    include_hl7v2: bool = False
    narrative_doc_types: list[str] | None = None
    narrative_provider: str = "ollama"
    narrative_api_key: str | None = None
    narrative_model: str | None = None


class FHIRValidateRequest(BaseModel):
    resource: dict


class HL7v2ConvertRequest(BaseModel):
    bundle: dict
    message_types: list[str] = Field(default_factory=lambda: ["ADT_A01", "ORU_R01"])


class NarrativeGenerateRequest(BaseModel):
    bundle: dict
    doc_types: list[str] | None = None
    provider: str = "ollama"
    api_key: str | None = None
    model: str | None = None
    encounter_ids: list[str] | None = None


@router.get("/fhir/resource-types")
def list_resource_types():
    return [
        {
            "type": rt,
            "dependencies": RESOURCE_DEPENDENCIES.get(rt, []),
        }
        for rt in ALL_RESOURCE_TYPES
    ]


@router.post("/fhir/generate")
def generate_fhir(req: FHIRGenerateRequest):
    for rt in req.resource_types:
        if rt not in ALL_RESOURCE_TYPES:
            raise HTTPException(400, f"Unknown resource type: {rt}")

    engine = MedicalEngine(seed=req.seed, terminology_focus=req.terminology_focus)

    narrative_provider_instance = None
    if req.include_narrative:
        try:
            narrative_provider_instance = get_provider(
                req.narrative_provider,
                api_key=req.narrative_api_key,
                model=req.narrative_model,
            )
        except Exception as e:
            logger.warning("Could not init narrative provider: %s", e)

    registry = engine.generate(
        resource_types=req.resource_types,
        patient_count=req.patient_count,
        encounters_per_patient=req.encounters_per_patient.model_dump(),
        clinical_density=req.clinical_density,
        narrative_provider=narrative_provider_instance,
        narrative_doc_types=req.narrative_doc_types,
    )

    engine.clean_internal_fields(registry)

    output = engine.build_output(
        registry,
        output_format=req.output_format,
        bundle_type=req.bundle_type,
        resource_types=req.resource_types,
    )

    result = {
        "status": "completed",
        "stats": output["stats"],
        "format": output["format"],
        "data": output["data"],
    }

    if req.include_hl7v2:
        hl7v2_messages = convert_registry_to_hl7v2(registry)
        result["hl7v2_messages"] = hl7v2_messages
        result["hl7v2_count"] = len(hl7v2_messages)

    return result


@router.post("/fhir/generate/async")
def generate_fhir_async(req: FHIRGenerateRequest):
    for rt in req.resource_types:
        if rt not in ALL_RESOURCE_TYPES:
            raise HTTPException(400, f"Unknown resource type: {rt}")

    job_id = str(uuid.uuid4())
    _MEDICAL_JOBS[job_id] = {"status": "running", "progress": {}, "started_at": time.time()}

    def _run():
        try:
            engine = MedicalEngine(seed=req.seed, terminology_focus=req.terminology_focus)

            def on_progress(resource_type: str, count: int):
                _MEDICAL_JOBS[job_id]["progress"][resource_type] = count

            narrative_provider_instance = None
            if req.include_narrative:
                try:
                    narrative_provider_instance = get_provider(
                        req.narrative_provider,
                        api_key=req.narrative_api_key,
                        model=req.narrative_model,
                    )
                except Exception as e:
                    logger.warning("Could not init narrative provider: %s", e)

            registry = engine.generate(
                resource_types=req.resource_types,
                patient_count=req.patient_count,
                encounters_per_patient=req.encounters_per_patient.model_dump(),
                clinical_density=req.clinical_density,
                progress_callback=on_progress,
                narrative_provider=narrative_provider_instance,
                narrative_doc_types=req.narrative_doc_types,
            )

            engine.clean_internal_fields(registry)

            output = engine.build_output(
                registry,
                output_format=req.output_format,
                bundle_type=req.bundle_type,
            )

            result = {"stats": output["stats"], "format": output["format"], "data": output["data"]}

            if req.include_hl7v2:
                hl7v2_messages = convert_registry_to_hl7v2(registry)
                result["hl7v2_messages"] = hl7v2_messages
                result["hl7v2_count"] = len(hl7v2_messages)

            _MEDICAL_JOBS[job_id].update({"status": "completed", "result": result, "elapsed": time.time() - _MEDICAL_JOBS[job_id]["started_at"]})

        except Exception as e:
            logger.exception("Medical generation job %s failed", job_id)
            _MEDICAL_JOBS[job_id].update({"status": "failed", "error": str(e)})

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "running"}


@router.get("/fhir/jobs/{job_id}")
def get_medical_job(job_id: str):
    job = _MEDICAL_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    response = {"job_id": job_id, "status": job["status"], "progress": job.get("progress", {})}
    if job["status"] == "completed":
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job.get("error")
    return response


@router.get("/terminologies/search")
def search_terminologies(
    system: str = Query(..., description="Terminology system: icd10, loinc, snomed, rxnorm"),
    query: str = Query(..., min_length=1, description="Search term"),
    limit: int = Query(default=20, ge=1, le=100),
):
    filename = TERMINOLOGY_FILES.get(system)
    if not filename:
        raise HTTPException(400, f"Unknown system: {system}. Use: {list(TERMINOLOGY_FILES.keys())}")

    results = search_codes(filename, query, limit=limit)
    return {"system": system, "query": query, "results": results, "count": len(results)}


@router.post("/fhir/validate")
def validate_fhir_resource(req: FHIRValidateRequest):
    errors = validate_resource(req.resource)
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "resource_type": req.resource.get("resourceType"),
    }


@router.post("/hl7v2/convert")
def convert_to_hl7v2(req: HL7v2ConvertRequest):
    from core.medical.fhir.references import ReferenceRegistry

    registry = ReferenceRegistry()
    entries = req.bundle.get("entry", [])
    for entry in entries:
        resource = entry.get("resource", {})
        rt = resource.get("resourceType")
        rid = resource.get("id")
        if rt and rid:
            registry.register(rt, rid, resource)

    messages = convert_registry_to_hl7v2(registry, message_types=req.message_types)
    return {"messages": messages, "count": len(messages)}


@router.post("/narratives/generate")
def generate_narratives(req: NarrativeGenerateRequest):
    from core.medical.fhir.references import ReferenceRegistry
    import base64

    registry = ReferenceRegistry()
    entries = req.bundle.get("entry", [])
    for entry in entries:
        resource = entry.get("resource", {})
        rt = resource.get("resourceType")
        rid = resource.get("id")
        if rt and rid:
            registry.register(rt, rid, resource)

    try:
        provider = get_provider(req.provider, api_key=req.api_key, model=req.model)
    except Exception as e:
        raise HTTPException(400, f"Failed to initialize provider: {e}")

    engine = ClinicalNarrativeEngine(provider=provider, allowed_types=req.doc_types)
    results = engine.generate_for_all_encounters(registry, register=True)

    narratives = []
    for doc in results:
        text = ""
        content = doc.get("content", [{}])
        if content:
            data = content[0].get("attachment", {}).get("data", "")
            if data:
                text = base64.b64decode(data).decode("utf-8")
        narratives.append({
            "id": doc["id"],
            "type": doc["type"]["coding"][0]["display"],
            "text": text,
            "document_reference": doc,
        })

    return {"status": "completed", "documents": narratives, "count": len(narratives)}


# ──────────────────────────────────────────────────────────────────────
# Clinical Trials endpoints
# ──────────────────────────────────────────────────────────────────────

class TrialGenerateRequest(BaseModel):
    profile: str = Field(default="oncology_phase2")
    num_sites: int = Field(default=5, ge=1, le=50)
    subjects_per_arm: int = Field(default=50, ge=5, le=5000)
    dropout_rate: float = Field(default=0.15, ge=0.0, le=0.8)
    effect_size: float = Field(default=0.3, ge=0.0, le=1.0)
    seed: int | None = None
    output_formats: list[str] = Field(default_factory=lambda: ["sdtm", "fhir"])


@router.get("/trials/profiles")
def list_trial_profiles():
    from core.medical.trial_profiles.profiles import list_profiles
    return list_profiles()


@router.post("/trials/generate")
def generate_trial(req: TrialGenerateRequest):
    from core.medical.trial_engine import TrialEngine

    engine = TrialEngine(seed=req.seed)
    registry = engine.generate(
        profile_id=req.profile,
        num_sites=req.num_sites,
        subjects_per_arm=req.subjects_per_arm,
        dropout_rate=req.dropout_rate,
        effect_size=req.effect_size,
    )

    result: dict = {"status": "completed"}

    if "sdtm" in req.output_formats:
        sdtm = engine.build_sdtm(registry)
        result["sdtm"] = {domain: {"rows": len(rows), "data": rows} for domain, rows in sdtm.items()}

    engine.clean_internal_fields(registry)

    if "fhir" in req.output_formats:
        fhir_output = engine.build_fhir_output(registry)
        result["fhir"] = fhir_output

    from core.medical.fhir.bundle import bundle_stats
    result["stats"] = bundle_stats(registry)

    return result


@router.post("/trials/generate/async")
def generate_trial_async(req: TrialGenerateRequest):
    from core.medical.trial_engine import TrialEngine

    job_id = str(uuid.uuid4())
    _MEDICAL_JOBS[job_id] = {"status": "running", "progress": {}, "started_at": time.time()}

    def _run():
        try:
            engine = TrialEngine(seed=req.seed)

            def on_progress(step: str, count: int):
                _MEDICAL_JOBS[job_id]["progress"][step] = count

            registry = engine.generate(
                profile_id=req.profile,
                num_sites=req.num_sites,
                subjects_per_arm=req.subjects_per_arm,
                dropout_rate=req.dropout_rate,
                effect_size=req.effect_size,
                progress_callback=on_progress,
            )

            result: dict = {}
            if "sdtm" in req.output_formats:
                sdtm = engine.build_sdtm(registry)
                result["sdtm"] = {domain: {"rows": len(rows), "data": rows} for domain, rows in sdtm.items()}

            engine.clean_internal_fields(registry)

            if "fhir" in req.output_formats:
                fhir_output = engine.build_fhir_output(registry)
                result["fhir"] = fhir_output

            from core.medical.fhir.bundle import bundle_stats
            result["stats"] = bundle_stats(registry)

            _MEDICAL_JOBS[job_id].update({
                "status": "completed",
                "result": result,
                "elapsed": time.time() - _MEDICAL_JOBS[job_id]["started_at"],
            })
        except Exception as e:
            logger.exception("Trial generation job %s failed", job_id)
            _MEDICAL_JOBS[job_id].update({"status": "failed", "error": str(e)})

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "running"}


@router.get("/trials/jobs/{job_id}")
def get_trial_job(job_id: str):
    job = _MEDICAL_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    response = {"job_id": job_id, "status": job["status"], "progress": job.get("progress", {})}
    if job["status"] == "completed":
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job.get("error")
    return response


# ──────────────────────────────────────────────────────────────────────
# Imaging / DICOM endpoints
# ──────────────────────────────────────────────────────────────────────

class ImagingGenerateRequest(BaseModel):
    modalities: list[str] = Field(default_factory=lambda: ["CT"])
    body_parts: list[str] | None = None
    num_studies: int = Field(default=50, ge=1, le=10000)
    include_instance_metadata: bool = True
    output_format: str = Field(default="dicom_json", pattern="^(dicom_json|fhir|csv)$")
    seed: int | None = None


@router.get("/imaging/modalities")
def list_imaging_modalities():
    try:
        data = _load_codeset("dicom_modalities.json")
        return {
            "modalities": data.get("modalities", []),
            "body_parts": data.get("body_parts", []),
        }
    except Exception:
        return {"modalities": [], "body_parts": []}


def _load_codeset_safe(name):
    from core.medical.terminologies.loader import _load_codeset as _lc
    return _lc(name)


@router.post("/imaging/generate")
def generate_imaging(req: ImagingGenerateRequest):
    from core.medical.imaging_engine import ImagingEngine

    engine = ImagingEngine(seed=req.seed)
    result = engine.generate(
        modalities=req.modalities,
        body_parts=req.body_parts,
        num_studies=req.num_studies,
        include_instance_metadata=req.include_instance_metadata,
    )
    output = engine.build_output(result, output_format=req.output_format)
    return {"status": "completed", **output}


@router.post("/imaging/generate/async")
def generate_imaging_async(req: ImagingGenerateRequest):
    from core.medical.imaging_engine import ImagingEngine

    job_id = str(uuid.uuid4())
    _MEDICAL_JOBS[job_id] = {"status": "running", "progress": {}, "started_at": time.time()}

    def _run():
        try:
            engine = ImagingEngine(seed=req.seed)

            def on_progress(step: str, count: int):
                _MEDICAL_JOBS[job_id]["progress"][step] = count

            result = engine.generate(
                modalities=req.modalities,
                body_parts=req.body_parts,
                num_studies=req.num_studies,
                include_instance_metadata=req.include_instance_metadata,
                progress_callback=on_progress,
            )
            output = engine.build_output(result, output_format=req.output_format)
            _MEDICAL_JOBS[job_id].update({"status": "completed", "result": output, "elapsed": time.time() - _MEDICAL_JOBS[job_id]["started_at"]})
        except Exception as e:
            logger.exception("Imaging generation job %s failed", job_id)
            _MEDICAL_JOBS[job_id].update({"status": "failed", "error": str(e)})

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "running"}


@router.get("/imaging/jobs/{job_id}")
def get_imaging_job(job_id: str):
    job = _MEDICAL_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    response = {"job_id": job_id, "status": job["status"], "progress": job.get("progress", {})}
    if job["status"] == "completed":
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job.get("error")
    return response
