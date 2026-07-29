"""TrialEngine — orchestrates clinical trial data generation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from core.medical.fhir.bundle import build_bundle, bundle_stats
from core.medical.fhir.generators import (
    FHIRGeneratorContext,
    generate_organizations,
    generate_patients,
    generate_practitioners,
)
from core.medical.fhir.references import ReferenceRegistry
from core.medical.fhir.sdtm_exporter import export_all
from core.medical.fhir.trial_generators import (
    generate_adverse_events,
    generate_disease_assessments,
    generate_research_study,
    generate_research_subjects,
    generate_study_drug_exposure,
    generate_trial_labs,
    generate_trial_visits,
    generate_trial_vitals,
)
from core.medical.trial_profiles.profiles import get_profile, list_profiles

logger = logging.getLogger(__name__)


class TrialEngine:
    """Orchestrates generation of synthetic clinical trial data."""

    def __init__(self, seed: int | None = None):
        self.ctx = FHIRGeneratorContext(seed=seed)
        self.study_id: str | None = None

    def generate(
        self,
        profile_id: str,
        num_sites: int = 5,
        subjects_per_arm: int = 50,
        dropout_rate: float = 0.15,
        effect_size: float = 0.3,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> ReferenceRegistry:
        """Generate a complete clinical trial dataset.

        Args:
            profile_id: Trial profile ID (e.g., "oncology_phase2").
            num_sites: Number of clinical sites.
            subjects_per_arm: Subjects per treatment arm.
            dropout_rate: Expected attrition rate (0.0-1.0).
            effect_size: Treatment vs control difference (0.0-1.0).
            progress_callback: Called with (step_name, count) after each step.

        Returns:
            ReferenceRegistry with all generated resources.
        """
        profile = get_profile(profile_id)
        start = time.time()

        ratio_str = profile.get("randomization_ratio", "1:1")
        ratios = [int(r) for r in ratio_str.split(":")]
        total_subjects = sum(subjects_per_arm * r // ratios[0] for r in ratios)

        # Step 1: Sites
        orgs = generate_organizations(self.ctx, num_sites)
        if progress_callback:
            progress_callback("Sites", len(orgs))

        # Step 2: Investigators
        n_investigators = max(num_sites, 3)
        practitioners = generate_practitioners(self.ctx, n_investigators)
        if progress_callback:
            progress_callback("Investigators", len(practitioners))

        # Step 3: Patients
        patients = generate_patients(self.ctx, total_subjects)
        patient_ids = [p["id"] for p in patients]
        if progress_callback:
            progress_callback("Patients", len(patients))

        # Step 4: Research Study
        study = generate_research_study(self.ctx, profile)
        self.study_id = study["id"]
        if progress_callback:
            progress_callback("Study", 1)

        # Step 5: Research Subjects (enrollment + randomization)
        subjects = generate_research_subjects(
            self.ctx, profile, patient_ids, self.study_id, subjects_per_arm
        )
        if progress_callback:
            progress_callback("Subjects", len(subjects))

        # Step 6: Visit schedule
        visits_by_subject = generate_trial_visits(self.ctx, subjects, profile, dropout_rate)
        total_visits = sum(len(v) for v in visits_by_subject.values())
        if progress_callback:
            progress_callback("Visits", total_visits)

        # Step 7: Vital signs
        vitals = generate_trial_vitals(self.ctx, visits_by_subject)
        if progress_callback:
            progress_callback("Vitals", len(vitals))

        # Step 8: Labs
        labs = generate_trial_labs(self.ctx, visits_by_subject, profile)
        if progress_callback:
            progress_callback("Labs", len(labs))

        # Step 9: Adverse events
        aes = generate_adverse_events(self.ctx, visits_by_subject, profile)
        if progress_callback:
            progress_callback("Adverse Events", len(aes))

        # Step 10: Study drug exposure
        exposure = generate_study_drug_exposure(self.ctx, visits_by_subject, profile)
        if progress_callback:
            progress_callback("Drug Exposure", len(exposure))

        # Step 11: Disease-specific assessments
        assessments = generate_disease_assessments(self.ctx, subjects, visits_by_subject, profile)
        if progress_callback:
            progress_callback("Disease Assessments", len(assessments))

        elapsed = time.time() - start
        stats = bundle_stats(self.ctx.registry)
        logger.info(
            "Trial generation complete: %d resources in %.2fs (profile=%s, subjects=%d)",
            stats["total"], elapsed, profile_id, len(subjects),
        )

        return self.ctx.registry

    def build_sdtm(self, registry: ReferenceRegistry) -> dict[str, list[dict]]:
        """Export trial data as CDISC SDTM domains."""
        if not self.study_id:
            raise ValueError("No study generated yet — call generate() first")

        profile_id = None
        study = registry.get_resource("ResearchStudy", self.study_id)
        if study:
            desc = study.get("description", "")
            for pid in [p["id"] for p in list_profiles()]:
                if pid in desc.lower():
                    profile_id = pid
                    break

        therapeutic_area = "oncology"
        if profile_id:
            try:
                profile = get_profile(profile_id)
                therapeutic_area = profile.get("therapeutic_area", "oncology")
            except ValueError:
                pass

        return export_all(registry, self.study_id, therapeutic_area)

    def build_fhir_output(
        self,
        registry: ReferenceRegistry,
        bundle_type: str = "collection",
    ) -> dict:
        """Build FHIR Bundle output."""
        stats = bundle_stats(registry)
        bundle = build_bundle(registry, bundle_type=bundle_type)
        return {"format": "bundle", "data": bundle, "stats": stats}

    @staticmethod
    def clean_internal_fields(registry: ReferenceRegistry):
        """Remove internal fields (prefixed with _) before output."""
        for resource in registry.all_resources():
            keys_to_remove = [k for k in resource if k.startswith("_")]
            for k in keys_to_remove:
                del resource[k]
