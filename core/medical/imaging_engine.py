"""ImagingEngine — orchestrates DICOM metadata generation with optional trial integration."""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Callable

from core.medical.dicom.metadata import (
    generate_full_study,
    to_dicom_json,
    generate_study_metadata,
    generate_series_metadata,
    generate_instance_metadata,
)
from core.medical.dicom.uid_generator import generate_study_uid
from core.medical.fhir.generators import FHIRGeneratorContext, _meta, _make_codeable
from core.medical.fhir.bundle import build_bundle, bundle_stats
from core.medical.fhir.references import ReferenceRegistry
from core.medical.terminologies.loader import _load_codeset

logger = logging.getLogger(__name__)


def _load_body_parts() -> list[dict]:
    try:
        data = _load_codeset("dicom_modalities.json")
        return data.get("body_parts", [])
    except Exception:
        return []


class ImagingEngine:
    """Generates DICOM imaging metadata at study/series/instance level."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.ctx = FHIRGeneratorContext(seed=seed)

    def generate(
        self,
        modalities: list[str] = None,
        body_parts: list[str] = None,
        num_studies: int = 50,
        include_instance_metadata: bool = True,
        trial_registry: ReferenceRegistry | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> dict:
        """Generate DICOM imaging metadata.

        Args:
            modalities: List of modality codes (CT, MR, US, DX, MG, PT).
            body_parts: Body parts to image (auto-selected from modality if None).
            num_studies: Number of imaging studies to generate.
            include_instance_metadata: Generate instance-level tags.
            trial_registry: Link to existing trial data for integration.
            progress_callback: Called with (step_name, count).

        Returns:
            Dict with studies, DICOM JSON, FHIR resources, and stats.
        """
        if modalities is None:
            modalities = ["CT"]

        all_body_parts = _load_body_parts()
        start = time.time()

        studies = []
        fhir_imaging_studies = []

        # If trial integration, get patient/encounter info from trial registry
        trial_patients = []
        trial_encounters = []
        if trial_registry:
            trial_patients = trial_registry.resources_by_type("Patient")
            trial_encounters = trial_registry.resources_by_type("Encounter")

        for i in range(num_studies):
            modality = self.rng.choice(modalities)

            # Select body part compatible with modality
            if body_parts:
                bp = self.rng.choice(body_parts)
            else:
                compatible = [b for b in all_body_parts if modality in b.get("modalities", [])]
                if compatible:
                    bp = self.rng.choice(compatible)["code"]
                else:
                    bp = "CHEST"

            # Get patient info from trial registry if available
            patient_info = None
            encounter_ref = None
            if trial_patients:
                patient = self.rng.choice(trial_patients)
                names = patient.get("name", [{}])
                name = names[0] if names else {}
                patient_info = {
                    "id": patient.get("identifier", [{}])[0].get("value", patient["id"]),
                    "name": f"{name.get('family', 'DOE')}^{(name.get('given') or ['JOHN'])[0]}",
                    "birth_date": (patient.get("birthDate") or "1970-01-01").replace("-", ""),
                    "sex": {"male": "M", "female": "F"}.get(patient.get("gender", ""), "O"),
                }
                # Find an encounter for this patient
                patient_encounters = [
                    e for e in trial_encounters
                    if e.get("subject", {}).get("reference") == f"Patient/{patient['id']}"
                ]
                if patient_encounters:
                    encounter_ref = self.rng.choice(patient_encounters)

            full_study = generate_full_study(
                modality=modality,
                patient_info=patient_info,
                body_part=bp,
                include_instances=include_instance_metadata,
                rng=self.rng,
            )
            studies.append(full_study)

            # Generate FHIR ImagingStudy resource
            fhir_study = self._build_fhir_imaging_study(full_study, patient_info, encounter_ref)
            fhir_imaging_studies.append(fhir_study)

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback("Studies", i + 1)

        if progress_callback:
            progress_callback("Studies", num_studies)

        elapsed = time.time() - start
        total_series = sum(s["total_series"] for s in studies)
        total_instances = sum(s["total_instances"] for s in studies)

        logger.info(
            "Imaging generation: %d studies, %d series, %d instances in %.2fs",
            num_studies, total_series, total_instances, elapsed,
        )

        return {
            "studies": studies,
            "fhir_resources": fhir_imaging_studies,
            "stats": {
                "num_studies": num_studies,
                "total_series": total_series,
                "total_instances": total_instances,
                "modalities": list(set(s["study"]["modality"] for s in studies)),
                "elapsed_seconds": round(elapsed, 2),
            },
        }

    def _build_fhir_imaging_study(self, full_study: dict, patient_info: dict | None, encounter_ref: dict | None) -> dict:
        """Build a FHIR ImagingStudy resource from DICOM metadata."""
        study = full_study["study"]
        series_list = full_study["series"]

        study_id = self.ctx.uid()
        fhir_series = []
        for s in series_list:
            fhir_series.append({
                "uid": s["series_instance_uid"],
                "number": s["series_number"],
                "modality": {"system": "http://dicom.nema.org/resources/ontology/DCM", "code": s["modality"], "display": s["series_description"]},
                "numberOfInstances": s["number_of_instances"],
                "description": s["series_description"],
            })

        resource = {
            "resourceType": "ImagingStudy",
            "id": study_id,
            "meta": _meta(),
            "status": "available",
            "started": f"{study['study_date'][:4]}-{study['study_date'][4:6]}-{study['study_date'][6:]}T{study['study_time'][:2]}:{study['study_time'][2:4]}:{study['study_time'][4:]}Z",
            "numberOfSeries": full_study["total_series"],
            "numberOfInstances": full_study["total_instances"],
            "modality": [{"system": "http://dicom.nema.org/resources/ontology/DCM", "code": study["modality"], "display": study["study_description"]}],
            "series": fhir_series,
            "description": study["study_description"],
        }

        if patient_info:
            resource["subject"] = {"reference": f"Patient/{patient_info['id']}", "display": patient_info["name"].replace("^", " ")}
        else:
            resource["subject"] = {"reference": f"Patient/{study['patient_id']}"}

        if encounter_ref:
            enc_id = encounter_ref.get("id", "")
            resource["encounter"] = {"reference": f"Encounter/{enc_id}"}

        self.ctx.registry.register("ImagingStudy", study_id, resource)
        return resource

    def build_output(self, result: dict, output_format: str = "dicom_json") -> dict:
        """Format the output."""
        if output_format == "dicom_json":
            dicom_jsons = []
            for full_study in result["studies"]:
                dj = to_dicom_json(full_study["study"], full_study["series"], full_study["instances"])
                dicom_jsons.append(dj)
            return {"format": "dicom_json", "data": dicom_jsons, "stats": result["stats"]}

        elif output_format == "fhir":
            bundle = build_bundle(self.ctx.registry)
            return {"format": "fhir", "data": bundle, "stats": result["stats"]}

        elif output_format == "csv":
            studies_csv = []
            series_csv = []
            instances_csv = []
            for full_study in result["studies"]:
                s = full_study["study"]
                studies_csv.append({
                    "study_uid": s["study_instance_uid"],
                    "study_date": s["study_date"],
                    "modality": s["modality"],
                    "body_part": s["body_part_examined"],
                    "patient_id": s["patient_id"],
                    "institution": s["institution_name"],
                    "description": s["study_description"],
                    "accession_number": s["accession_number"],
                })
                for ser in full_study["series"]:
                    series_csv.append({
                        "study_uid": s["study_instance_uid"],
                        "series_uid": ser["series_instance_uid"],
                        "series_number": ser["series_number"],
                        "modality": ser["modality"],
                        "description": ser["series_description"],
                        "num_instances": ser["number_of_instances"],
                    })
                    for inst in full_study["instances"].get(ser["series_instance_uid"], []):
                        instances_csv.append({
                            "series_uid": ser["series_instance_uid"],
                            "instance_uid": inst["sop_instance_uid"],
                            "instance_number": inst["instance_number"],
                            "rows": inst["rows"],
                            "columns": inst["columns"],
                        })
            return {
                "format": "csv",
                "data": {"studies": studies_csv, "series": series_csv, "instances": instances_csv},
                "stats": result["stats"],
            }

        else:
            return {"format": output_format, "data": result["studies"], "stats": result["stats"]}
