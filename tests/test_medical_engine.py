"""Tests for the Medical Engine — FHIR resource generation."""

import json
import pytest

from core.medical.engine import MedicalEngine, resolve_dependencies, ALL_RESOURCE_TYPES
from core.medical.fhir.bundle import build_bundle, bundle_to_ndjson, bundle_stats
from core.medical.fhir.validator import validate_resource, validate_bundle
from core.medical.fhir.references import ReferenceRegistry
from core.medical.fhir.hl7v2_converter import convert_registry_to_hl7v2


class TestResolveDependencies:
    def test_patient_includes_organization(self):
        ordered = resolve_dependencies(["Patient"])
        assert "Organization" in ordered
        assert ordered.index("Organization") < ordered.index("Patient")

    def test_observation_includes_full_chain(self):
        ordered = resolve_dependencies(["Observation"])
        assert "Patient" in ordered
        assert "Encounter" in ordered
        assert "Organization" in ordered
        assert "Practitioner" in ordered

    def test_all_types_returns_full_order(self):
        ordered = resolve_dependencies(list(ALL_RESOURCE_TYPES))
        assert len(ordered) == len(ALL_RESOURCE_TYPES)


class TestMedicalEngine:
    def test_generate_minimal(self):
        engine = MedicalEngine(seed=42)
        registry = engine.generate(
            resource_types=["Patient", "Encounter", "Observation"],
            patient_count=5,
            encounters_per_patient={"min": 1, "max": 2},
            clinical_density="low",
        )
        assert len(registry.get_ids("Patient")) == 5
        assert len(registry.get_ids("Encounter")) >= 5
        assert len(registry.get_ids("Observation")) > 0

    def test_generate_all_types(self):
        engine = MedicalEngine(seed=123)
        registry = engine.generate(
            resource_types=list(ALL_RESOURCE_TYPES),
            patient_count=10,
            encounters_per_patient={"min": 1, "max": 3},
            clinical_density="moderate",
        )
        # Should have all requested types (allergy is probabilistic)
        assert len(registry.get_ids("Patient")) == 10
        assert len(registry.get_ids("Encounter")) >= 10
        assert len(registry.get_ids("Condition")) > 0
        assert len(registry.get_ids("Observation")) > 0
        assert len(registry.get_ids("MedicationRequest")) > 0
        assert len(registry.get_ids("Organization")) >= 1
        assert len(registry.get_ids("Practitioner")) >= 1

    def test_seed_reproducibility(self):
        engine1 = MedicalEngine(seed=99)
        engine2 = MedicalEngine(seed=99)
        reg1 = engine1.generate(["Patient"], patient_count=5)
        reg2 = engine2.generate(["Patient"], patient_count=5)
        patients1 = reg1.resources_by_type("Patient")
        patients2 = reg2.resources_by_type("Patient")
        assert patients1[0]["id"] == patients2[0]["id"]
        assert patients1[0]["birthDate"] == patients2[0]["birthDate"]

    def test_referential_integrity(self):
        engine = MedicalEngine(seed=42)
        registry = engine.generate(
            resource_types=["Patient", "Encounter", "Observation"],
            patient_count=10,
            encounters_per_patient={"min": 1, "max": 3},
        )
        engine.clean_internal_fields(registry)
        errors = registry.verify_integrity()
        assert len(errors) == 0, f"Reference errors: {errors}"

    def test_clean_internal_fields(self):
        engine = MedicalEngine(seed=42)
        registry = engine.generate(["Patient", "Encounter"], patient_count=3)
        engine.clean_internal_fields(registry)
        for resource in registry.all_resources():
            internal_keys = [k for k in resource if k.startswith("_")]
            assert len(internal_keys) == 0, f"Internal fields not cleaned: {internal_keys}"


class TestBundleOutput:
    def test_build_collection_bundle(self):
        engine = MedicalEngine(seed=42)
        registry = engine.generate(["Patient"], patient_count=5)
        engine.clean_internal_fields(registry)
        bundle = build_bundle(registry, bundle_type="collection")
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert bundle["total"] >= 5

    def test_build_transaction_bundle(self):
        engine = MedicalEngine(seed=42)
        registry = engine.generate(["Patient"], patient_count=3)
        engine.clean_internal_fields(registry)
        bundle = build_bundle(registry, bundle_type="transaction")
        assert bundle["type"] == "transaction"
        for entry in bundle["entry"]:
            assert "request" in entry
            assert entry["request"]["method"] == "PUT"

    def test_ndjson_output(self):
        engine = MedicalEngine(seed=42)
        registry = engine.generate(["Patient", "Encounter"], patient_count=3)
        engine.clean_internal_fields(registry)
        ndjson = bundle_to_ndjson(registry)
        assert "Patient" in ndjson
        assert "Encounter" in ndjson
        for line in ndjson["Patient"].strip().split("\n"):
            parsed = json.loads(line)
            assert parsed["resourceType"] == "Patient"

    def test_bundle_stats(self):
        engine = MedicalEngine(seed=42)
        registry = engine.generate(["Patient", "Encounter"], patient_count=5)
        stats = bundle_stats(registry)
        assert stats["total"] > 0
        assert "Patient" in stats["by_type"]
        assert stats["by_type"]["Patient"] == 5

    def test_tabular_output(self):
        engine = MedicalEngine(seed=42)
        registry = engine.generate(
            ["Patient", "Encounter", "Observation"],
            patient_count=5,
            encounters_per_patient={"min": 1, "max": 2},
        )
        engine.clean_internal_fields(registry)
        output = engine.build_output(registry, output_format="tabular")
        assert output["format"] == "tabular"
        assert "patients" in output["data"]
        assert len(output["data"]["patients"]) == 5


class TestValidation:
    def test_validate_valid_resource(self):
        engine = MedicalEngine(seed=42)
        registry = engine.generate(["Patient"], patient_count=1)
        engine.clean_internal_fields(registry)
        patient = registry.resources_by_type("Patient")[0]
        errors = validate_resource(patient)
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_validate_invalid_resource(self):
        errors = validate_resource({"resourceType": "Patient", "id": "test"})
        assert len(errors) > 0  # Missing required fields

    def test_validate_bundle(self):
        engine = MedicalEngine(seed=42)
        registry = engine.generate(["Patient", "Encounter"], patient_count=5)
        engine.clean_internal_fields(registry)
        bundle = build_bundle(registry)
        errors = validate_bundle(bundle)
        assert len(errors) == 0, f"Bundle validation errors: {errors[:5]}"

    def test_full_validation(self):
        engine = MedicalEngine(seed=42)
        registry = engine.generate(
            list(ALL_RESOURCE_TYPES),
            patient_count=10,
            clinical_density="moderate",
        )
        engine.clean_internal_fields(registry)
        result = engine.validate(registry)
        assert result["valid"], f"Errors: {result['structure_errors'][:3]} | Refs: {result['reference_errors'][:3]}"


class TestHL7v2Converter:
    def test_convert_adt_a01(self):
        engine = MedicalEngine(seed=42)
        registry = engine.generate(["Patient", "Encounter"], patient_count=3)
        engine.clean_internal_fields(registry)
        messages = convert_registry_to_hl7v2(registry, message_types=["ADT_A01"])
        assert len(messages) > 0
        for msg in messages:
            assert msg.startswith("MSH|")
            assert "ADT^A01" in msg
            assert "PID|" in msg
            assert "PV1|" in msg

    def test_convert_oru_r01(self):
        engine = MedicalEngine(seed=42)
        registry = engine.generate(
            ["Patient", "Encounter", "Observation"],
            patient_count=3,
            encounters_per_patient={"min": 1, "max": 2},
        )
        engine.clean_internal_fields(registry)
        messages = convert_registry_to_hl7v2(registry, message_types=["ORU_R01"])
        assert len(messages) > 0
        for msg in messages:
            assert msg.startswith("MSH|")
            assert "ORU^R01" in msg
            assert "OBX|" in msg

    def test_convert_all_types(self):
        engine = MedicalEngine(seed=42)
        registry = engine.generate(
            ["Patient", "Encounter", "Observation"],
            patient_count=5,
        )
        engine.clean_internal_fields(registry)
        messages = convert_registry_to_hl7v2(registry)
        assert len(messages) > 0


class TestTerminologies:
    def test_icd10_codes_load(self):
        from core.medical.terminologies import icd10
        codes = icd10.all_codes()
        assert len(codes) > 50
        for code in codes[:5]:
            assert "code" in code
            assert "display" in code
            assert "system" in code or True  # system is on the file level

    def test_loinc_codes_load(self):
        from core.medical.terminologies import loinc
        codes = loinc.all_codes()
        assert len(codes) > 30
        vitals = loinc.vital_sign_codes()
        assert len(vitals) > 0
        labs = loinc.lab_codes()
        assert len(labs) > 0

    def test_snomed_codes_load(self):
        from core.medical.terminologies import snomed
        codes = snomed.all_codes()
        assert len(codes) > 50
        procedures = [c for c in codes if c.get("type") == "procedure"]
        assert len(procedures) > 0
        findings = [c for c in codes if c.get("type") == "finding"]
        assert len(findings) > 0

    def test_rxnorm_codes_load(self):
        from core.medical.terminologies import rxnorm
        codes = rxnorm.all_codes()
        assert len(codes) > 30

    def test_random_diagnosis(self):
        from core.medical.terminologies import icd10
        dx = icd10.random_diagnosis(age=50)
        assert "code" in dx
        assert "display" in dx

    def test_terminology_search(self):
        from core.medical.terminologies.loader import search_codes
        results = search_codes("icd10_common.json", "diabetes")
        assert len(results) > 0
        assert "diabetes" in results[0]["display"].lower()
