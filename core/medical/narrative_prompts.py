"""Clinical narrative prompts — system prompts and context assembly for LLM-driven narrative generation."""

from __future__ import annotations

from typing import Any

from core.medical.fhir.references import ReferenceRegistry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_DOC_TYPES: list[str] = [
    "discharge_summary",
    "radiology_report",
    "pathology_report",
    "clinical_note",
    "operative_note",
]

#: LOINC code + display for each document type
DOC_TYPE_LOINC: dict[str, dict[str, str]] = {
    "discharge_summary": {"code": "18842-5", "display": "Discharge summary"},
    "radiology_report": {"code": "18726-0", "display": "Radiology studies (set)"},
    "pathology_report": {"code": "11526-1", "display": "Pathology study"},
    "clinical_note": {"code": "11506-3", "display": "Progress note"},
    "operative_note": {"code": "11504-8", "display": "Surgical operation note"},
}

#: Maximum LLM tokens per document type
DOC_TYPE_MAX_TOKENS: dict[str, int] = {
    "discharge_summary": 1200,
    "radiology_report": 600,
    "pathology_report": 700,
    "clinical_note": 500,
    "operative_note": 800,
}

SYSTEM_PROMPTS: dict[str, str] = {
    "discharge_summary": (
        "You are an experienced attending physician writing a hospital discharge summary. "
        "Write a concise, realistic, clinically accurate discharge summary in plain text. "
        "Include: reason for admission, hospital course, procedures performed, "
        "discharge condition, medications, and follow-up instructions. "
        "Use standard medical terminology. Do not use markdown or headers — write flowing prose."
    ),
    "radiology_report": (
        "You are a board-certified radiologist writing a diagnostic imaging report. "
        "Write a realistic, structured radiology report in plain text. "
        "Include: clinical indication, technique, findings, and impression. "
        "Use precise anatomical language. Do not use markdown — write in standard radiology report style."
    ),
    "pathology_report": (
        "You are a pathologist writing a surgical pathology report. "
        "Write a realistic pathology report in plain text. "
        "Include: clinical history, gross description, microscopic description, and final diagnosis. "
        "Use standard pathology terminology. Do not use markdown."
    ),
    "clinical_note": (
        "You are a clinician writing a progress note (SOAP format). "
        "Write a concise, realistic clinical progress note in plain text. "
        "Include: Subjective (patient complaints), Objective (vitals and exam findings), "
        "Assessment (diagnoses), and Plan (treatments and orders). "
        "Do not use markdown headers — write in plain text SOAP format."
    ),
    "operative_note": (
        "You are a surgeon writing an operative note. "
        "Write a realistic, detailed operative note in plain text. "
        "Include: preoperative diagnosis, postoperative diagnosis, procedure performed, "
        "surgeon name, anesthesia type, findings, complications, and specimen disposition. "
        "Do not use markdown — write in standard operative note format."
    ),
}


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


def _extract_coding_display(codeable_concept: dict | None) -> tuple[str, str]:
    """Extract (code, display) from a CodeableConcept dict. Returns ('', '') if absent."""
    if not codeable_concept:
        return ("", "")
    codings = codeable_concept.get("coding", [])
    if codings:
        c = codings[0]
        return (c.get("code", ""), c.get("display", codeable_concept.get("text", "")))
    return ("", codeable_concept.get("text", ""))


def assemble_clinical_context(registry: ReferenceRegistry, encounter_id: str) -> dict[str, Any]:
    """Extract and summarise all clinical data for an encounter into a flat context dict.

    The returned dict has these keys:
        encounter_id, patient, encounter, conditions, observations,
        medications, procedures, imaging

    Patient lookup strategy:
        1. Check encounter._patient_id (internal field set during generation)
        2. Fall back to parsing encounter.subject.reference ("Patient/<id>")
    """
    encounter = registry.get_resource("Encounter", encounter_id)
    if not encounter:
        raise ValueError(f"Encounter {encounter_id!r} not found in registry")

    # ------------------------------------------------------------------
    # Resolve patient
    # ------------------------------------------------------------------
    patient_id: str = encounter.get("_patient_id", "")
    if not patient_id:
        subject_ref = encounter.get("subject", {}).get("reference", "")
        if "/" in subject_ref:
            patient_id = subject_ref.split("/", 1)[1]

    patient = registry.get_resource("Patient", patient_id) or {}

    # ------------------------------------------------------------------
    # Encounter summary
    # ------------------------------------------------------------------
    enc_class = encounter.get("class", {})
    period = encounter.get("period", {})
    enc_summary = {
        "class_code": enc_class.get("code", ""),
        "class_display": enc_class.get("display", ""),
        "start": period.get("start", ""),
        "end": period.get("end", ""),
        "status": encounter.get("status", ""),
    }

    # ------------------------------------------------------------------
    # Linked clinical resources
    # ------------------------------------------------------------------
    conditions: list[dict] = []
    for cond in registry.resources_by_type("Condition"):
        enc_ref = cond.get("encounter", {}).get("reference", "")
        if encounter_id not in enc_ref:
            continue
        code, display = _extract_coding_display(cond.get("code"))
        conditions.append(
            {
                "code": code,
                "display": display,
                "clinical_status": (
                    cond.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "")
                    if cond.get("clinicalStatus")
                    else ""
                ),
            }
        )

    observations: list[dict] = []
    for obs in registry.resources_by_type("Observation"):
        enc_ref = obs.get("encounter", {}).get("reference", "")
        if encounter_id not in enc_ref:
            continue
        code, display = _extract_coding_display(obs.get("code"))
        vq = obs.get("valueQuantity", {})
        observations.append(
            {
                "code": code,
                "display": display,
                "value": vq.get("value") if vq else obs.get("valueString", ""),
                "unit": vq.get("unit", "") if vq else "",
            }
        )

    medications: list[dict] = []
    for med in registry.resources_by_type("MedicationRequest"):
        enc_ref = med.get("encounter", {}).get("reference", "")
        if encounter_id not in enc_ref:
            continue
        code, display = _extract_coding_display(med.get("medicationCodeableConcept"))
        medications.append(
            {
                "code": code,
                "display": display,
                "status": med.get("status", ""),
            }
        )

    procedures: list[dict] = []
    for proc in registry.resources_by_type("Procedure"):
        enc_ref = proc.get("encounter", {}).get("reference", "")
        if encounter_id not in enc_ref:
            continue
        code, display = _extract_coding_display(proc.get("code"))
        procedures.append(
            {
                "code": code,
                "display": display,
                "status": proc.get("status", ""),
            }
        )

    imaging: list[dict] = []
    for img in registry.resources_by_type("ImagingStudy"):
        enc_ref = img.get("encounter", {}).get("reference", "")
        if encounter_id not in enc_ref:
            continue
        series = img.get("series", [])
        modality = ""
        description = ""
        if series:
            first = series[0]
            modality = first.get("modality", {}).get("code", "") if isinstance(first.get("modality"), dict) else ""
            description = first.get("description", "")
        imaging.append(
            {
                "id": img["id"],
                "modality": modality,
                "description": description,
                "status": img.get("status", ""),
            }
        )

    # ------------------------------------------------------------------
    # Patient summary
    # ------------------------------------------------------------------
    names = patient.get("name", [{}])
    name_obj = names[0] if names else {}
    patient_summary = {
        "id": patient.get("id", ""),
        "family": name_obj.get("family", ""),
        "given": (name_obj.get("given") or [""])[0],
        "gender": patient.get("gender", ""),
        "birthDate": patient.get("birthDate", ""),
        "age": patient.get("_age", ""),
    }

    return {
        "encounter_id": encounter_id,
        "patient": patient_summary,
        "encounter": enc_summary,
        "conditions": conditions,
        "observations": observations,
        "medications": medications,
        "procedures": procedures,
        "imaging": imaging,
    }


# ---------------------------------------------------------------------------
# Doc-type selection
# ---------------------------------------------------------------------------


def determine_doc_types(
    context: dict[str, Any],
    allowed_types: list[str] | None,
) -> list[str]:
    """Decide which document types to generate for a given encounter context.

    Rules:
      IMP (inpatient)  → discharge_summary + clinical_note
                         + operative_note + pathology_report  (if procedures present)
      AMB / EMER       → clinical_note only
      Any encounter    → + radiology_report  (if imaging present)

    ``allowed_types``, when provided, acts as a whitelist filter on the result.
    """
    enc_class = context.get("encounter", {}).get("class_code", "")
    has_procedures = bool(context.get("procedures"))
    has_imaging = bool(context.get("imaging"))

    selected: set[str] = set()

    if enc_class == "IMP":
        selected.add("discharge_summary")
        selected.add("clinical_note")
        if has_procedures:
            selected.add("operative_note")
            selected.add("pathology_report")
    else:
        # AMB, EMER, or unknown
        selected.add("clinical_note")

    if has_imaging:
        selected.add("radiology_report")

    if allowed_types is not None:
        selected = selected.intersection(allowed_types)

    # Return in canonical order
    return [dt for dt in ALL_DOC_TYPES if dt in selected]


# ---------------------------------------------------------------------------
# User-prompt formatting
# ---------------------------------------------------------------------------


def _fmt_list(items: list[dict], key: str) -> str:
    """Format a list of dicts into a readable bullet string using ``key``."""
    if not items:
        return "None documented"
    parts = []
    for item in items:
        display = item.get(key, "")
        if not display:
            display = item.get("code", "unknown")
        extra_parts = []
        if item.get("value") is not None and item.get("value") != "":
            val = item["value"]
            unit = item.get("unit", "")
            extra_parts.append(f"{val} {unit}".strip())
        if extra_parts:
            display = f"{display} ({'; '.join(extra_parts)})"
        parts.append(f"  - {display}")
    return "\n".join(parts)


def format_user_prompt(doc_type: str, context: dict[str, Any]) -> str:
    """Format the LLM user-turn prompt for the given document type and clinical context."""
    patient = context["patient"]
    encounter = context["encounter"]

    patient_line = (
        f"{patient.get('given', '')} {patient.get('family', '')} "
        f"({patient.get('gender', 'unknown')}, DOB: {patient.get('birthDate', 'unknown')}"
        + (f", age {patient['age']}" if patient.get("age") else "")
        + ")"
    ).strip()

    enc_line = (
        f"{encounter.get('class_display', encounter.get('class_code', 'unknown'))} encounter, "
        f"from {encounter.get('start', 'unknown')} to {encounter.get('end', 'unknown')}"
    )

    conditions_text = _fmt_list(context.get("conditions", []), "display")
    observations_text = _fmt_list(context.get("observations", []), "display")
    medications_text = _fmt_list(context.get("medications", []), "display")
    procedures_text = _fmt_list(context.get("procedures", []), "display")
    imaging_text = _fmt_list(context.get("imaging", []), "description")

    base = (
        f"Patient: {patient_line}\n"
        f"Encounter: {enc_line}\n\n"
        f"Diagnoses / Conditions:\n{conditions_text}\n\n"
        f"Observations / Vitals / Labs:\n{observations_text}\n\n"
        f"Medications:\n{medications_text}\n\n"
        f"Procedures:\n{procedures_text}\n\n"
        f"Imaging Studies:\n{imaging_text}\n\n"
    )

    instructions: dict[str, str] = {
        "discharge_summary": (
            "Using the clinical data above, write a complete hospital discharge summary "
            f"for this {encounter.get('class_display', 'inpatient')} encounter. "
            "Be clinically realistic and thorough."
        ),
        "radiology_report": (
            "Using the imaging study data above, write a realistic radiology report. "
            "Include clinical indication, technique, findings, and impression."
        ),
        "pathology_report": (
            "Using the procedure and clinical data above, write a realistic pathology report. "
            "Include clinical history, gross and microscopic description, and final diagnosis."
        ),
        "clinical_note": (
            "Using the clinical data above, write a clinical progress note in SOAP format "
            "(Subjective, Objective, Assessment, Plan)."
        ),
        "operative_note": (
            "Using the procedure data above, write a complete operative note. "
            "Include preoperative and postoperative diagnoses, procedure, findings, "
            "complications (if any), and estimated blood loss."
        ),
    }

    instruction = instructions.get(
        doc_type,
        f"Using the clinical data above, write a realistic {doc_type.replace('_', ' ')}.",
    )

    return base + instruction
