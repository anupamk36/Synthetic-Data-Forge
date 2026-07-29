"""Tests for the TrialEngine — clinical trial data generation."""

import pytest

from core.medical.trial_engine import TrialEngine
from core.medical.trial_profiles.profiles import list_profiles, get_profile
from core.medical.fhir.bundle import bundle_stats
from core.medical.fhir.sdtm_exporter import export_all
from core.medical.fhir.validator import validate_resource


class TestTrialProfiles:
    def test_list_profiles(self):
        profiles = list_profiles()
        assert len(profiles) >= 3
        ids = [p["id"] for p in profiles]
        assert "oncology_phase2" in ids

    def test_get_profile(self):
        profile = get_profile("oncology_phase2")
        assert profile["therapeutic_area"] == "oncology"
        assert "arms" in profile
        assert "visit_schedule" in profile
        assert len(profile["arms"]) >= 2

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("nonexistent_profile")


class TestTrialEngineGeneration:
    def test_generate_oncology(self):
        engine = TrialEngine(seed=42)
        registry = engine.generate(
            profile_id="oncology_phase2",
            num_sites=2,
            subjects_per_arm=10,
            dropout_rate=0.1,
        )
        assert len(registry.get_ids("ResearchStudy")) == 1
        assert len(registry.get_ids("ResearchSubject")) > 0
        assert len(registry.get_ids("Patient")) > 0
        assert len(registry.get_ids("Encounter")) > 0
        assert len(registry.get_ids("Observation")) > 0

    def test_generate_with_all_profiles(self):
        profiles = list_profiles()
        for profile_info in profiles:
            engine = TrialEngine(seed=42)
            registry = engine.generate(
                profile_id=profile_info["id"],
                num_sites=2,
                subjects_per_arm=5,
                dropout_rate=0.1,
            )
            stats = bundle_stats(registry)
            assert stats["total"] > 0, f"Profile {profile_info['id']} generated no resources"
            assert len(registry.get_ids("ResearchStudy")) == 1
            assert len(registry.get_ids("ResearchSubject")) > 0

    def test_randomization(self):
        engine = TrialEngine(seed=42)
        registry = engine.generate(
            profile_id="oncology_phase2",
            num_sites=2,
            subjects_per_arm=20,
        )
        subjects = registry.resources_by_type("ResearchSubject")
        arms = [s.get("assignedArm") for s in subjects]
        unique_arms = set(arms)
        assert len(unique_arms) >= 2, "Should have at least 2 arms"

    def test_referential_integrity(self):
        engine = TrialEngine(seed=42)
        registry = engine.generate(
            profile_id="oncology_phase2",
            num_sites=3,
            subjects_per_arm=10,
        )
        engine.clean_internal_fields(registry)
        errors = registry.verify_integrity()
        assert len(errors) == 0, f"Reference errors: {errors[:3]}"

    def test_clean_internal_fields(self):
        engine = TrialEngine(seed=42)
        registry = engine.generate("oncology_phase2", num_sites=2, subjects_per_arm=5)
        engine.clean_internal_fields(registry)
        for resource in registry.all_resources():
            internal = [k for k in resource if k.startswith("_")]
            assert len(internal) == 0, f"Internal fields: {internal}"

    def test_dropout(self):
        engine = TrialEngine(seed=42)
        registry = engine.generate(
            profile_id="oncology_phase2",
            num_sites=2,
            subjects_per_arm=30,
            dropout_rate=0.5,
        )
        subjects = registry.resources_by_type("ResearchSubject")
        withdrawn = [s for s in subjects if s.get("status") == "withdrawn"]
        assert len(withdrawn) > 0, "With 50% dropout rate, some subjects should withdraw"


class TestSDTMExport:
    def test_export_dm(self):
        engine = TrialEngine(seed=42)
        registry = engine.generate("oncology_phase2", num_sites=2, subjects_per_arm=10)
        sdtm = engine.build_sdtm(registry)
        assert "DM" in sdtm
        dm = sdtm["DM"]
        assert len(dm) > 0
        row = dm[0]
        assert "STUDYID" in row
        assert "SUBJID" in row
        assert "ARM" in row
        assert "SEX" in row

    def test_export_sv(self):
        engine = TrialEngine(seed=42)
        registry = engine.generate("oncology_phase2", num_sites=2, subjects_per_arm=10)
        sdtm = engine.build_sdtm(registry)
        assert "SV" in sdtm
        sv = sdtm["SV"]
        assert len(sv) > 0
        assert "VISITNUM" in sv[0]
        assert "VISIT" in sv[0]

    def test_export_ae(self):
        engine = TrialEngine(seed=42)
        registry = engine.generate("oncology_phase2", num_sites=2, subjects_per_arm=20)
        sdtm = engine.build_sdtm(registry)
        assert "AE" in sdtm
        ae = sdtm["AE"]
        assert len(ae) > 0
        row = ae[0]
        assert "AETERM" in row
        assert "AESEV" in row
        assert "AEREL" in row

    def test_export_lb(self):
        engine = TrialEngine(seed=42)
        registry = engine.generate("oncology_phase2", num_sites=2, subjects_per_arm=10)
        sdtm = engine.build_sdtm(registry)
        assert "LB" in sdtm
        lb = sdtm["LB"]
        assert len(lb) > 0
        assert "LBTESTCD" in lb[0]
        assert "LBORRES" in lb[0]

    def test_export_vs(self):
        engine = TrialEngine(seed=42)
        registry = engine.generate("oncology_phase2", num_sites=2, subjects_per_arm=10)
        sdtm = engine.build_sdtm(registry)
        assert "VS" in sdtm

    def test_oncology_has_tr_rs(self):
        engine = TrialEngine(seed=42)
        registry = engine.generate("oncology_phase2", num_sites=2, subjects_per_arm=15)
        sdtm = engine.build_sdtm(registry)
        assert "TR" in sdtm, "Oncology trials should have Tumor Results"
        assert "RS" in sdtm, "Oncology trials should have Disease Response"


class TestFHIRValidation:
    def test_research_study_validates(self):
        engine = TrialEngine(seed=42)
        registry = engine.generate("oncology_phase2", num_sites=2, subjects_per_arm=5)
        engine.clean_internal_fields(registry)
        studies = registry.resources_by_type("ResearchStudy")
        assert len(studies) == 1
        errors = validate_resource(studies[0])
        assert len(errors) == 0, f"ResearchStudy errors: {errors}"

    def test_research_subject_validates(self):
        engine = TrialEngine(seed=42)
        registry = engine.generate("oncology_phase2", num_sites=2, subjects_per_arm=5)
        engine.clean_internal_fields(registry)
        subjects = registry.resources_by_type("ResearchSubject")
        assert len(subjects) > 0
        errors = validate_resource(subjects[0])
        assert len(errors) == 0, f"ResearchSubject errors: {errors}"


class TestTNMAndRECIST:
    def test_tnm_staging(self):
        from core.medical.terminologies.tnm import random_tnm_stage
        stage = random_tnm_stage("lung", rng=__import__("random").Random(42))
        assert "t" in stage
        assert "n" in stage
        assert "m" in stage
        assert stage["t"].startswith("T")

    def test_recist_trajectories(self):
        from core.medical.terminologies.tnm import generate_tumor_measurements, classify_recist_response
        measurements = generate_tumor_measurements("responder", 5, rng=__import__("random").Random(42))
        assert len(measurements) == 5
        assert measurements[0]["pct_change"] == 0.0
        last_pct = measurements[-1]["pct_change"]
        assert last_pct < 0, "Responder trajectory should show decrease"

    def test_recist_classification(self):
        from core.medical.terminologies.tnm import classify_recist_response
        assert classify_recist_response(-100) == "CR"
        assert classify_recist_response(-35) == "PR"
        assert classify_recist_response(0) == "SD"
        assert classify_recist_response(25) == "PD"
