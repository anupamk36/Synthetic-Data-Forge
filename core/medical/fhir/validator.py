"""FHIR R4 resource validation — structural and referential integrity checks."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

REQUIRED_FIELDS: dict[str, list[str]] = {
    "Organization": ["resourceType", "id", "name"],
    "Practitioner": ["resourceType", "id", "name"],
    "Patient": ["resourceType", "id", "name", "gender", "birthDate"],
    "Encounter": ["resourceType", "id", "status", "class", "subject"],
    "Condition": ["resourceType", "id", "clinicalStatus", "code", "subject"],
    "Observation": ["resourceType", "id", "status", "code", "subject"],
    "MedicationRequest": ["resourceType", "id", "status", "intent", "medicationCodeableConcept", "subject"],
    "Procedure": ["resourceType", "id", "status", "code", "subject"],
    "DiagnosticReport": ["resourceType", "id", "status", "code", "subject"],
    "AllergyIntolerance": ["resourceType", "id", "clinicalStatus", "code", "patient"],
    "ImagingStudy": ["resourceType", "id", "status", "subject"],
    "ResearchStudy": ["resourceType", "id", "title", "status"],
    "ResearchSubject": ["resourceType", "id", "status", "study", "individual"],
    "Specimen": ["resourceType", "id", "subject"],
    "DocumentReference": ["resourceType", "id", "status", "type", "subject", "content"],
}

VALID_STATUSES = {
    "Encounter": ["planned", "arrived", "triaged", "in-progress", "onleave", "finished", "cancelled"],
    "Condition": None,  # uses clinicalStatus CodeableConcept
    "Observation": ["registered", "preliminary", "final", "amended", "corrected", "cancelled"],
    "MedicationRequest": ["active", "on-hold", "cancelled", "completed", "stopped", "draft"],
    "Procedure": ["preparation", "in-progress", "not-done", "on-hold", "stopped", "completed"],
    "DiagnosticReport": ["registered", "partial", "preliminary", "final", "amended", "corrected"],
    "AllergyIntolerance": None,
    "ImagingStudy": ["registered", "available", "cancelled"],
    "ResearchStudy": [
        "active",
        "administratively-completed",
        "approved",
        "closed-to-accrual",
        "closed-to-accrual-and-intervention",
        "completed",
        "disapproved",
        "in-review",
        "temporarily-closed-to-accrual",
        "temporarily-closed-to-accrual-and-intervention",
        "withdrawn",
    ],
    "ResearchSubject": [
        "candidate",
        "eligible",
        "follow-up",
        "ineligible",
        "not-registered",
        "off-study",
        "on-study",
        "on-study-intervention",
        "on-study-observation",
        "pending-on-study",
        "potential-candidate",
        "screening",
        "withdrawn",
    ],
    "Specimen": ["available", "unavailable", "unsatisfactory", "entered-in-error"],
    "DocumentReference": ["current", "superseded", "entered-in-error"],
}


def validate_resource(resource: dict) -> list[dict]:
    """Validate a single FHIR resource. Returns list of error dicts."""
    errors = []
    rt = resource.get("resourceType")

    if not rt:
        errors.append({"path": "resourceType", "error": "missing_resource_type"})
        return errors

    if not resource.get("id"):
        errors.append({"path": "id", "error": "missing_id"})

    required = REQUIRED_FIELDS.get(rt, [])
    for field in required:
        if field not in resource or resource[field] is None:
            errors.append({"path": field, "error": f"required_field_missing: {field}"})

    _validate_coding_fields(resource, rt, errors)

    status_field = "status" if rt not in ("Condition", "AllergyIntolerance") else None
    if status_field and rt in VALID_STATUSES and VALID_STATUSES[rt]:
        status = resource.get(status_field)
        if status and status not in VALID_STATUSES[rt]:
            errors.append({"path": status_field, "error": f"invalid_status: {status}"})

    _validate_references(resource, errors)

    return errors


def _validate_coding_fields(resource: dict, resource_type: str, errors: list[dict]):
    """Check that CodeableConcept fields have proper coding structure."""
    codeable_fields = _get_codeable_fields(resource_type)
    for field_path in codeable_fields:
        value = resource.get(field_path)
        if value is None:
            continue
        if not isinstance(value, dict):
            errors.append({"path": field_path, "error": "expected_codeable_concept"})
            continue
        coding = value.get("coding")
        if not coding or not isinstance(coding, list) or len(coding) == 0:
            errors.append({"path": f"{field_path}.coding", "error": "empty_coding"})
            continue
        for i, c in enumerate(coding):
            if not c.get("system"):
                errors.append({"path": f"{field_path}.coding[{i}].system", "error": "missing_system"})
            if not c.get("code"):
                errors.append({"path": f"{field_path}.coding[{i}].code", "error": "missing_code"})


def _get_codeable_fields(resource_type: str) -> list[str]:
    mapping = {
        "Condition": ["code", "clinicalStatus", "verificationStatus", "severity"],
        "Observation": ["code"],
        "MedicationRequest": ["medicationCodeableConcept"],
        "Procedure": ["code"],
        "DiagnosticReport": ["code"],
        "AllergyIntolerance": ["code", "clinicalStatus", "verificationStatus"],
        "DocumentReference": ["type"],
    }
    return mapping.get(resource_type, [])


def _validate_references(resource: dict, errors: list[dict]):
    """Check that Reference fields have proper structure."""
    _walk_for_references(resource, [], errors)


def _walk_for_references(obj, path: list[str], errors: list[dict]):
    if isinstance(obj, dict):
        if "reference" in obj:
            ref = obj["reference"]
            if not isinstance(ref, str) or "/" not in ref:
                errors.append(
                    {
                        "path": ".".join(path + ["reference"]),
                        "error": f"malformed_reference: {ref}",
                    }
                )
        for key, value in obj.items():
            if key in ("text", "div"):
                continue
            _walk_for_references(value, path + [key], errors)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _walk_for_references(item, path + [str(i)], errors)


def validate_bundle(bundle: dict) -> list[dict]:
    """Validate all resources in a FHIR Bundle."""
    errors = []

    if bundle.get("resourceType") != "Bundle":
        errors.append({"path": "resourceType", "error": "not_a_bundle"})
        return errors

    entries = bundle.get("entry", [])
    for i, entry in enumerate(entries):
        resource = entry.get("resource", {})
        resource_errors = validate_resource(resource)
        for err in resource_errors:
            err["entry_index"] = i
            err["resource_type"] = resource.get("resourceType", "Unknown")
            err["resource_id"] = resource.get("id", "unknown")
        errors.extend(resource_errors)

    return errors
