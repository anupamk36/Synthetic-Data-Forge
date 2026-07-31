"""MedicalEngine — orchestrates FHIR resource generation across all types."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from core.medical.fhir.bundle import build_bundle, bundle_stats, bundle_to_ndjson
from core.medical.fhir.generators import (
    FHIRGeneratorContext,
    generate_allergy_intolerances,
    generate_conditions,
    generate_diagnostic_reports,
    generate_encounters,
    generate_imaging_studies,
    generate_medication_requests,
    generate_observations,
    generate_organizations,
    generate_patients,
    generate_practitioners,
    generate_procedures,
)
from core.medical.fhir.references import ReferenceRegistry
from core.medical.fhir.validator import validate_bundle
from core.medical.narrative_engine import ClinicalNarrativeEngine

logger = logging.getLogger(__name__)

GENERATION_ORDER = [
    "Organization",
    "Practitioner",
    "Patient",
    "Encounter",
    "Condition",
    "Observation",
    "MedicationRequest",
    "Procedure",
    "DiagnosticReport",
    "AllergyIntolerance",
    "ImagingStudy",
    "DocumentReference",
]

ALL_RESOURCE_TYPES = set(GENERATION_ORDER)

RESOURCE_DEPENDENCIES = {
    "Organization": [],
    "Practitioner": ["Organization"],
    "Patient": ["Organization"],
    "Encounter": ["Patient", "Practitioner", "Organization"],
    "Condition": ["Encounter"],
    "Observation": ["Encounter"],
    "MedicationRequest": ["Encounter"],
    "Procedure": ["Encounter"],
    "DiagnosticReport": ["Encounter", "Observation"],
    "AllergyIntolerance": ["Patient"],
    "ImagingStudy": ["Encounter"],
    "DocumentReference": ["Encounter", "Condition", "Observation", "MedicationRequest", "Procedure"],
}


def resolve_dependencies(requested: list[str]) -> list[str]:
    """Given a list of requested resource types, add all dependencies and return in generation order."""
    needed = set(requested)
    changed = True
    while changed:
        changed = False
        for rt in list(needed):
            for dep in RESOURCE_DEPENDENCIES.get(rt, []):
                if dep not in needed:
                    needed.add(dep)
                    changed = True
    return [rt for rt in GENERATION_ORDER if rt in needed]


class MedicalEngine:
    """Orchestrates generation of FHIR resources with cross-resource coherence."""

    def __init__(
        self,
        seed: int | None = None,
        terminology_focus: str | None = None,
    ):
        self.ctx = FHIRGeneratorContext(seed=seed, terminology_focus=terminology_focus)

    def generate(
        self,
        resource_types: list[str],
        patient_count: int = 100,
        encounters_per_patient: dict | None = None,
        clinical_density: str = "moderate",
        progress_callback: Callable[[str, int], None] | None = None,
        narrative_provider: Any | None = None,
        narrative_doc_types: list[str] | None = None,
    ) -> ReferenceRegistry:
        """Generate all requested FHIR resources.

        Args:
            resource_types: List of FHIR resource types to generate.
            patient_count: Number of patients.
            encounters_per_patient: {"min": int, "max": int} range.
            clinical_density: "low", "moderate", or "high".
            progress_callback: Called with (resource_type, count) after each type.

        Returns:
            ReferenceRegistry with all generated resources.
        """
        if encounters_per_patient is None:
            encounters_per_patient = {"min": 1, "max": 5}

        ordered = resolve_dependencies(resource_types)
        logger.info("Generation order: %s", ordered)

        start = time.time()
        patient_ids = []
        encounter_ids = []

        for rt in ordered:
            t0 = time.time()

            if rt == "Organization":
                n = max(1, patient_count // 50)
                results = generate_organizations(self.ctx, n)
            elif rt == "Practitioner":
                n = max(2, patient_count // 10)
                results = generate_practitioners(self.ctx, n)
            elif rt == "Patient":
                results = generate_patients(self.ctx, patient_count)
                patient_ids = [r["id"] for r in results]
            elif rt == "Encounter":
                results = generate_encounters(
                    self.ctx,
                    patient_ids,
                    min_per_patient=encounters_per_patient["min"],
                    max_per_patient=encounters_per_patient["max"],
                )
                encounter_ids = [r["id"] for r in results]
            elif rt == "Condition":
                results = generate_conditions(self.ctx, encounter_ids, clinical_density)
            elif rt == "Observation":
                results = generate_observations(self.ctx, encounter_ids, clinical_density)
            elif rt == "MedicationRequest":
                results = generate_medication_requests(self.ctx, encounter_ids, clinical_density)
            elif rt == "Procedure":
                results = generate_procedures(self.ctx, encounter_ids, clinical_density)
            elif rt == "DiagnosticReport":
                results = generate_diagnostic_reports(self.ctx, encounter_ids)
            elif rt == "AllergyIntolerance":
                results = generate_allergy_intolerances(self.ctx, patient_ids)
            elif rt == "ImagingStudy":
                results = generate_imaging_studies(self.ctx, encounter_ids, clinical_density)
            elif rt == "DocumentReference":
                if narrative_provider is None:
                    logger.info("Skipping DocumentReference — no narrative provider configured")
                    continue
                narr_engine = ClinicalNarrativeEngine(
                    provider=narrative_provider,
                    allowed_types=narrative_doc_types,
                )
                results = narr_engine.generate_for_all_encounters(
                    self.ctx.registry,
                    register=True,
                )
            else:
                logger.warning("Unknown resource type: %s", rt)
                continue

            elapsed = time.time() - t0
            logger.info("Generated %d %s resources in %.2fs", len(results), rt, elapsed)

            if progress_callback:
                progress_callback(rt, len(results))

        total_elapsed = time.time() - start
        stats = bundle_stats(self.ctx.registry)
        logger.info("Total generation: %d resources in %.2fs", stats["total"], total_elapsed)

        return self.ctx.registry

    def build_output(
        self,
        registry: ReferenceRegistry,
        output_format: str = "bundle",
        bundle_type: str = "collection",
        resource_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build the final output in the requested format."""
        stats = bundle_stats(registry)

        if output_format == "bundle":
            bundle = build_bundle(registry, bundle_type=bundle_type, resource_types=resource_types)
            return {"format": "bundle", "data": bundle, "stats": stats}

        elif output_format == "ndjson":
            ndjson = bundle_to_ndjson(registry)
            return {"format": "ndjson", "data": ndjson, "stats": stats}

        elif output_format == "individual":
            grouped: dict[str, list[dict]] = {}
            for resource in registry.all_resources():
                rt = resource.get("resourceType", "Unknown")
                if resource_types and rt not in resource_types:
                    continue
                grouped.setdefault(rt, []).append(resource)
            return {"format": "individual", "data": grouped, "stats": stats}

        elif output_format == "tabular":
            tabular = self._flatten_to_tabular(registry, resource_types)
            return {"format": "tabular", "data": tabular, "stats": stats}

        else:
            bundle = build_bundle(registry, bundle_type=bundle_type, resource_types=resource_types)
            return {"format": "bundle", "data": bundle, "stats": stats}

    def validate(self, registry: ReferenceRegistry) -> dict:
        """Validate all generated resources."""
        bundle = build_bundle(registry)
        errors = validate_bundle(bundle)
        ref_errors = registry.verify_integrity()
        return {
            "valid": len(errors) == 0 and len(ref_errors) == 0,
            "structure_errors": errors,
            "reference_errors": ref_errors,
            "total_resources": bundle["total"],
        }

    @staticmethod
    def _flatten_to_tabular(registry: ReferenceRegistry, resource_types: list[str] | None) -> dict[str, list[dict]]:
        """Flatten FHIR resources to tabular rows for CSV/Parquet export."""
        tables: dict[str, list[dict]] = {}

        for resource in registry.all_resources():
            rt = resource.get("resourceType", "Unknown")
            if resource_types and rt not in resource_types:
                continue

            if rt == "Patient":
                names = resource.get("name", [{}])
                name = names[0] if names else {}
                addr = resource.get("address", [{}])
                address = addr[0] if addr else {}
                tables.setdefault("patients", []).append(
                    {
                        "id": resource["id"],
                        "family_name": name.get("family"),
                        "given_name": (name.get("given") or [""])[0],
                        "gender": resource.get("gender"),
                        "birth_date": resource.get("birthDate"),
                        "city": address.get("city"),
                        "state": address.get("state"),
                    }
                )
            elif rt == "Observation":
                code = resource.get("code", {}).get("coding", [{}])[0]
                vq = resource.get("valueQuantity", {})
                tables.setdefault("observations", []).append(
                    {
                        "id": resource["id"],
                        "patient_id": resource.get("subject", {}).get("reference", "").replace("Patient/", ""),
                        "encounter_id": resource.get("encounter", {}).get("reference", "").replace("Encounter/", ""),
                        "loinc_code": code.get("code"),
                        "loinc_display": code.get("display"),
                        "value": vq.get("value") if vq else resource.get("valueString"),
                        "unit": vq.get("unit") if vq else None,
                        "date": resource.get("effectiveDateTime"),
                    }
                )
            elif rt == "Condition":
                code = resource.get("code", {}).get("coding", [{}])[0]
                tables.setdefault("conditions", []).append(
                    {
                        "id": resource["id"],
                        "patient_id": resource.get("subject", {}).get("reference", "").replace("Patient/", ""),
                        "encounter_id": resource.get("encounter", {}).get("reference", "").replace("Encounter/", ""),
                        "icd10_code": code.get("code"),
                        "icd10_display": code.get("display"),
                        "onset_date": resource.get("onsetDateTime"),
                        "status": resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code"),
                    }
                )
            elif rt == "Encounter":
                period = resource.get("period", {})
                tables.setdefault("encounters", []).append(
                    {
                        "id": resource["id"],
                        "patient_id": resource.get("subject", {}).get("reference", "").replace("Patient/", ""),
                        "status": resource.get("status"),
                        "class": resource.get("class", {}).get("code"),
                        "start": period.get("start"),
                        "end": period.get("end"),
                    }
                )
            elif rt == "MedicationRequest":
                code = resource.get("medicationCodeableConcept", {}).get("coding", [{}])[0]
                tables.setdefault("medication_requests", []).append(
                    {
                        "id": resource["id"],
                        "patient_id": resource.get("subject", {}).get("reference", "").replace("Patient/", ""),
                        "rxnorm_code": code.get("code"),
                        "medication": code.get("display"),
                        "status": resource.get("status"),
                        "authored_on": resource.get("authoredOn"),
                    }
                )

        return tables

    @staticmethod
    def clean_internal_fields(registry: ReferenceRegistry):
        """Remove internal fields (prefixed with _) before output."""
        for resource in registry.all_resources():
            keys_to_remove = [k for k in resource if k.startswith("_")]
            for k in keys_to_remove:
                del resource[k]
