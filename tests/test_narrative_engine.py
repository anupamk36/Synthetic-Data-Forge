"""Tests for clinical narrative generation — narrative_prompts + narrative_engine."""

from __future__ import annotations

import base64

import pytest

from core.medical.fhir.references import ReferenceRegistry

# ---------------------------------------------------------------------------
# Shared test helper
# ---------------------------------------------------------------------------


def _build_test_registry() -> tuple[ReferenceRegistry, str, str]:
    """Return (registry, patient_id, encounter_id) with rich clinical data."""
    reg = ReferenceRegistry()

    patient_id = "pat-001"
    patient = {
        "resourceType": "Patient",
        "id": patient_id,
        "name": [{"family": "Smith", "given": ["John"]}],
        "gender": "male",
        "birthDate": "1965-03-15",
        "_age": 59,
    }
    reg.register("Patient", patient_id, patient)

    org_id = "org-001"
    org = {
        "resourceType": "Organization",
        "id": org_id,
        "name": "General Hospital",
    }
    reg.register("Organization", org_id, org)

    prac_id = "prac-001"
    prac = {
        "resourceType": "Practitioner",
        "id": prac_id,
        "name": [{"family": "Jones", "given": ["Dr. Alice"]}],
    }
    reg.register("Practitioner", prac_id, prac)

    encounter_id = "enc-001"
    encounter = {
        "resourceType": "Encounter",
        "id": encounter_id,
        "status": "finished",
        "class": {"code": "IMP", "display": "inpatient encounter"},
        "subject": {"reference": f"Patient/{patient_id}"},
        "period": {"start": "2024-01-10T08:00:00", "end": "2024-01-14T16:00:00"},
        "participant": [{"individual": {"reference": f"Practitioner/{prac_id}"}}],
        "serviceProvider": {"reference": f"Organization/{org_id}"},
        "_patient_id": patient_id,
        "_start_date": "2024-01-10",
    }
    reg.register("Encounter", encounter_id, encounter)

    cond_id = "cond-001"
    condition = {
        "resourceType": "Condition",
        "id": cond_id,
        "code": {
            "coding": [
                {"system": "http://hl7.org/fhir/sid/icd-10", "code": "I21.9", "display": "Acute myocardial infarction"}
            ],
            "text": "Acute myocardial infarction",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "encounter": {"reference": f"Encounter/{encounter_id}"},
        "clinicalStatus": {
            "coding": [{"code": "active"}],
        },
    }
    reg.register("Condition", cond_id, condition)

    obs_id = "obs-001"
    observation = {
        "resourceType": "Observation",
        "id": obs_id,
        "status": "final",
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "8867-4", "display": "Heart rate"}],
            "text": "Heart rate",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "encounter": {"reference": f"Encounter/{encounter_id}"},
        "valueQuantity": {"value": 88, "unit": "bpm"},
    }
    reg.register("Observation", obs_id, observation)

    proc_id = "proc-001"
    procedure = {
        "resourceType": "Procedure",
        "id": proc_id,
        "status": "completed",
        "code": {
            "coding": [{"system": "http://snomed.info/sct", "code": "80146002", "display": "Appendectomy"}],
            "text": "Appendectomy",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "encounter": {"reference": f"Encounter/{encounter_id}"},
    }
    reg.register("Procedure", proc_id, procedure)

    med_id = "med-001"
    medication = {
        "resourceType": "MedicationRequest",
        "id": med_id,
        "status": "active",
        "medicationCodeableConcept": {
            "coding": [
                {
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": "1049502",
                    "display": "Aspirin 325 MG",
                }
            ],
            "text": "Aspirin 325 MG",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "encounter": {"reference": f"Encounter/{encounter_id}"},
    }
    reg.register("MedicationRequest", med_id, medication)

    img_id = "img-001"
    imaging = {
        "resourceType": "ImagingStudy",
        "id": img_id,
        "status": "available",
        "subject": {"reference": f"Patient/{patient_id}"},
        "encounter": {"reference": f"Encounter/{encounter_id}"},
        "series": [{"modality": {"code": "CT"}, "description": "CT Chest"}],
    }
    reg.register("ImagingStudy", img_id, imaging)

    return reg, patient_id, encounter_id


# ===========================================================================
# Task 1 Tests: narrative_prompts module
# ===========================================================================


class TestAssembleClinicalContext:
    """Tests for assemble_clinical_context()."""

    def test_returns_dict_with_required_keys(self):
        from core.medical.narrative_prompts import assemble_clinical_context

        reg, patient_id, enc_id = _build_test_registry()
        ctx = assemble_clinical_context(reg, enc_id)
        required = {
            "encounter_id",
            "patient",
            "encounter",
            "conditions",
            "observations",
            "medications",
            "procedures",
            "imaging",
        }
        assert required.issubset(ctx.keys()), f"Missing keys: {required - ctx.keys()}"

    def test_patient_data_extracted(self):
        from core.medical.narrative_prompts import assemble_clinical_context

        reg, patient_id, enc_id = _build_test_registry()
        ctx = assemble_clinical_context(reg, enc_id)
        patient = ctx["patient"]
        assert patient["id"] == "pat-001"
        assert patient["gender"] == "male"
        assert patient["birthDate"] == "1965-03-15"

    def test_encounter_data_extracted(self):
        from core.medical.narrative_prompts import assemble_clinical_context

        reg, patient_id, enc_id = _build_test_registry()
        ctx = assemble_clinical_context(reg, enc_id)
        enc = ctx["encounter"]
        assert enc["class_code"] == "IMP"
        assert "start" in enc
        assert "end" in enc

    def test_conditions_extracted(self):
        from core.medical.narrative_prompts import assemble_clinical_context

        reg, patient_id, enc_id = _build_test_registry()
        ctx = assemble_clinical_context(reg, enc_id)
        assert len(ctx["conditions"]) >= 1
        cond = ctx["conditions"][0]
        assert "display" in cond
        assert "code" in cond

    def test_observations_extracted(self):
        from core.medical.narrative_prompts import assemble_clinical_context

        reg, patient_id, enc_id = _build_test_registry()
        ctx = assemble_clinical_context(reg, enc_id)
        assert len(ctx["observations"]) >= 1
        obs = ctx["observations"][0]
        assert "display" in obs

    def test_procedures_extracted(self):
        from core.medical.narrative_prompts import assemble_clinical_context

        reg, patient_id, enc_id = _build_test_registry()
        ctx = assemble_clinical_context(reg, enc_id)
        assert len(ctx["procedures"]) >= 1
        proc = ctx["procedures"][0]
        assert "display" in proc

    def test_imaging_extracted(self):
        from core.medical.narrative_prompts import assemble_clinical_context

        reg, patient_id, enc_id = _build_test_registry()
        ctx = assemble_clinical_context(reg, enc_id)
        assert len(ctx["imaging"]) >= 1

    def test_patient_lookup_via_subject_reference(self):
        """Fallback: encounter has no _patient_id; patient found via subject.reference."""
        from core.medical.narrative_prompts import assemble_clinical_context

        reg, patient_id, enc_id = _build_test_registry()
        # Remove the internal _patient_id field
        enc = reg.get_resource("Encounter", enc_id)
        enc_copy = {k: v for k, v in enc.items() if k != "_patient_id"}
        # Re-register without _patient_id
        reg._resources[f"Encounter/{enc_id}"] = enc_copy
        ctx = assemble_clinical_context(reg, enc_id)
        assert ctx["patient"]["id"] == patient_id


class TestDetermineDocTypes:
    """Tests for determine_doc_types()."""

    def test_inpatient_gets_discharge_summary_and_clinical_note(self):
        from core.medical.narrative_prompts import determine_doc_types

        context = {
            "encounter": {"class_code": "IMP"},
            "procedures": [],
            "imaging": [],
        }
        types = determine_doc_types(context, allowed_types=None)
        assert "discharge_summary" in types
        assert "clinical_note" in types

    def test_inpatient_with_procedures_gets_operative_and_pathology(self):
        from core.medical.narrative_prompts import determine_doc_types

        context = {
            "encounter": {"class_code": "IMP"},
            "procedures": [{"display": "Appendectomy"}],
            "imaging": [],
        }
        types = determine_doc_types(context, allowed_types=None)
        assert "operative_note" in types
        assert "pathology_report" in types

    def test_ambulatory_gets_clinical_note_only(self):
        from core.medical.narrative_prompts import determine_doc_types

        context = {
            "encounter": {"class_code": "AMB"},
            "procedures": [],
            "imaging": [],
        }
        types = determine_doc_types(context, allowed_types=None)
        assert "clinical_note" in types
        assert "discharge_summary" not in types

    def test_emergency_gets_clinical_note_only(self):
        from core.medical.narrative_prompts import determine_doc_types

        context = {
            "encounter": {"class_code": "EMER"},
            "procedures": [],
            "imaging": [],
        }
        types = determine_doc_types(context, allowed_types=None)
        assert "clinical_note" in types
        assert "discharge_summary" not in types

    def test_imaging_adds_radiology_report(self):
        from core.medical.narrative_prompts import determine_doc_types

        context = {
            "encounter": {"class_code": "AMB"},
            "procedures": [],
            "imaging": [{"id": "img-001"}],
        }
        types = determine_doc_types(context, allowed_types=None)
        assert "radiology_report" in types

    def test_allowed_types_filters_result(self):
        from core.medical.narrative_prompts import determine_doc_types

        context = {
            "encounter": {"class_code": "IMP"},
            "procedures": [{"display": "Surgery"}],
            "imaging": [{"id": "img-001"}],
        }
        types = determine_doc_types(context, allowed_types=["clinical_note", "radiology_report"])
        assert "clinical_note" in types
        assert "radiology_report" in types
        assert "discharge_summary" not in types
        assert "operative_note" not in types


class TestFormatUserPrompt:
    """Tests for format_user_prompt()."""

    def test_returns_non_empty_string_for_all_doc_types(self):
        from core.medical.narrative_prompts import ALL_DOC_TYPES, format_user_prompt

        reg, patient_id, enc_id = _build_test_registry()
        from core.medical.narrative_prompts import assemble_clinical_context

        context = assemble_clinical_context(reg, enc_id)
        for doc_type in ALL_DOC_TYPES:
            prompt = format_user_prompt(doc_type, context)
            assert isinstance(prompt, str), f"Expected str for {doc_type}"
            assert len(prompt) > 50, f"Prompt too short for {doc_type}: {prompt!r}"


# ===========================================================================
# Task 2 Tests: narrative_engine module
# ===========================================================================


class TestBuildDocumentReference:
    """Tests for build_document_reference()."""

    def test_basic_structure(self):
        from core.medical.narrative_engine import build_document_reference

        doc = build_document_reference(
            doc_id="doc-001",
            doc_type="discharge_summary",
            encounter_id="enc-001",
            patient_id="pat-001",
            narrative_text="Patient was admitted for chest pain.",
            authored_date="2024-01-14",
        )
        assert doc["resourceType"] == "DocumentReference"
        assert doc["id"] == "doc-001"
        assert doc["status"] == "current"

    def test_content_attachment_base64(self):
        from core.medical.narrative_engine import build_document_reference

        narrative = "Patient was admitted for acute MI and discharged in stable condition."
        doc = build_document_reference(
            doc_id="doc-002",
            doc_type="discharge_summary",
            encounter_id="enc-001",
            patient_id="pat-001",
            narrative_text=narrative,
            authored_date="2024-01-14",
        )
        content = doc["content"]
        assert isinstance(content, list)
        assert len(content) >= 1
        attachment = content[0]["attachment"]
        assert attachment["contentType"] == "text/plain"
        decoded = base64.b64decode(attachment["data"]).decode("utf-8")
        assert decoded == narrative

    def test_subject_and_context_references(self):
        from core.medical.narrative_engine import build_document_reference

        doc = build_document_reference(
            doc_id="doc-003",
            doc_type="clinical_note",
            encounter_id="enc-001",
            patient_id="pat-001",
            narrative_text="Clinical note content.",
            authored_date="2024-01-10",
        )
        assert doc["subject"]["reference"] == "Patient/pat-001"
        ctx = doc.get("context", {})
        encounter_refs = ctx.get("encounter", [])
        assert any("enc-001" in ref.get("reference", "") for ref in encounter_refs)

    def test_loinc_code_in_type(self):
        from core.medical.narrative_engine import build_document_reference
        from core.medical.narrative_prompts import DOC_TYPE_LOINC

        doc = build_document_reference(
            doc_id="doc-004",
            doc_type="radiology_report",
            encounter_id="enc-001",
            patient_id="pat-001",
            narrative_text="CT scan shows no acute findings.",
            authored_date="2024-01-10",
        )
        loinc_info = DOC_TYPE_LOINC["radiology_report"]
        coding = doc["type"]["coding"]
        assert any(c["code"] == loinc_info["code"] for c in coding)


class MockProvider:
    """Minimal mock LLM provider with chat_complete matching AlchemyProvider interface."""

    def __init__(self, response_text: str = "Mock narrative text generated by LLM."):
        self._response_text = response_text

    def chat_complete(self, messages: list[dict], **kwargs):
        text = self._response_text

        class _Msg:
            content = text

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class TestClinicalNarrativeEngine:
    """Tests for ClinicalNarrativeEngine."""

    def test_generate_for_encounter_returns_document_references(self):
        from core.medical.narrative_engine import ClinicalNarrativeEngine

        reg, patient_id, enc_id = _build_test_registry()
        provider = MockProvider("This is a discharge summary narrative.")
        engine = ClinicalNarrativeEngine(provider=provider)
        docs = engine.generate_for_encounter(reg, enc_id)
        assert len(docs) >= 1
        for doc in docs:
            assert doc["resourceType"] == "DocumentReference"

    def test_generate_for_encounter_registers_documents(self):
        from core.medical.narrative_engine import ClinicalNarrativeEngine

        reg, patient_id, enc_id = _build_test_registry()
        provider = MockProvider("Narrative for registration test.")
        engine = ClinicalNarrativeEngine(provider=provider)
        engine.generate_for_encounter(reg, enc_id, register=True)
        doc_ids = reg.get_ids("DocumentReference")
        assert len(doc_ids) >= 1

    def test_generate_for_encounter_respects_allowed_types(self):
        from core.medical.narrative_engine import ClinicalNarrativeEngine

        reg, patient_id, enc_id = _build_test_registry()
        provider = MockProvider("Clinical note only.")
        engine = ClinicalNarrativeEngine(provider=provider)
        docs = engine.generate_for_encounter(reg, enc_id, allowed_types=["clinical_note"])
        # Only clinical_note should have been generated
        for doc in docs:
            type_coding = doc["type"]["coding"]
            # clinical_note LOINC code is 11506-3
            codes = [c["code"] for c in type_coding]
            from core.medical.narrative_prompts import DOC_TYPE_LOINC

            clinical_code = DOC_TYPE_LOINC["clinical_note"]["code"]
            assert clinical_code in codes, f"Expected clinical_note LOINC {clinical_code}, got {codes}"

    def test_generate_for_all_encounters(self):
        from core.medical.narrative_engine import ClinicalNarrativeEngine

        reg, patient_id, enc_id = _build_test_registry()

        # Add a second encounter
        enc2_id = "enc-002"
        enc2 = {
            "resourceType": "Encounter",
            "id": enc2_id,
            "status": "finished",
            "class": {"code": "AMB", "display": "ambulatory"},
            "subject": {"reference": f"Patient/{patient_id}"},
            "period": {"start": "2024-02-01T09:00:00", "end": "2024-02-01T10:00:00"},
            "_patient_id": patient_id,
        }
        reg.register("Encounter", enc2_id, enc2)

        provider = MockProvider("Multi-encounter narrative.")
        engine = ClinicalNarrativeEngine(provider=provider)
        all_docs = engine.generate_for_all_encounters(reg)
        assert len(all_docs) >= 2

    def test_llm_error_produces_fallback_or_skips(self):
        """If LLM raises an exception, engine should not crash."""
        from core.medical.narrative_engine import ClinicalNarrativeEngine

        class FailingProvider:
            def chat_complete(self, *args, **kwargs):
                raise RuntimeError("LLM unavailable")

        reg, patient_id, enc_id = _build_test_registry()
        engine = ClinicalNarrativeEngine(provider=FailingProvider())
        # Should not raise; may return empty list or docs with error text
        try:
            docs = engine.generate_for_encounter(reg, enc_id)
            # Either empty or contains error fallbacks — just must not crash
            assert isinstance(docs, list)
        except Exception as e:
            pytest.fail(f"Engine raised unexpectedly: {e}")

    def test_document_reference_narrative_text_present(self):
        from core.medical.narrative_engine import ClinicalNarrativeEngine

        narrative = "The patient presented with acute chest pain radiating to the left arm."
        reg, patient_id, enc_id = _build_test_registry()
        provider = MockProvider(narrative)
        engine = ClinicalNarrativeEngine(provider=provider)
        docs = engine.generate_for_encounter(reg, enc_id, allowed_types=["discharge_summary"])
        assert len(docs) >= 1
        doc = docs[0]
        attachment = doc["content"][0]["attachment"]
        decoded = base64.b64decode(attachment["data"]).decode("utf-8")
        assert decoded == narrative


# ===========================================================================
# Task 3 Tests: validator updates
# ===========================================================================


class TestDocumentReferenceValidation:
    """Tests that DocumentReference validation is wired into the validator."""

    def test_valid_document_reference_passes(self):
        from core.medical.fhir.validator import validate_resource
        from core.medical.narrative_engine import build_document_reference

        doc = build_document_reference(
            doc_id="doc-val-001",
            doc_type="discharge_summary",
            encounter_id="enc-001",
            patient_id="pat-001",
            narrative_text="Discharge summary text.",
            authored_date="2024-01-14",
        )
        errors = validate_resource(doc)
        assert errors == [], f"Unexpected validation errors: {errors}"

    def test_invalid_document_reference_fails(self):
        from core.medical.fhir.validator import validate_resource

        # Missing required fields: type, subject, content
        doc = {
            "resourceType": "DocumentReference",
            "id": "doc-bad-001",
            "status": "current",
        }
        errors = validate_resource(doc)
        error_fields = [e["path"] for e in errors]
        assert (
            any("type" in f for f in error_fields)
            or any("subject" in f for f in error_fields)
            or any("content" in f for f in error_fields)
        )

    def test_document_reference_invalid_status_fails(self):
        from core.medical.fhir.validator import VALID_STATUSES, validate_resource

        assert "DocumentReference" in VALID_STATUSES
        doc = {
            "resourceType": "DocumentReference",
            "id": "doc-bad-002",
            "status": "bogus-status",
            "type": {"coding": [{"system": "http://loinc.org", "code": "18842-5", "display": "Discharge summary"}]},
            "subject": {"reference": "Patient/pat-001"},
            "content": [{"attachment": {"contentType": "text/plain", "data": "dGVzdA=="}}],
        }
        errors = validate_resource(doc)
        assert any("invalid_status" in e.get("error", "") for e in errors)


# ===========================================================================
# Task 4 Tests: MedicalEngine integration
# ===========================================================================


class TestMedicalEngineIntegration:
    def test_generate_with_narrative_provider(self):
        from core.medical.engine import MedicalEngine

        engine = MedicalEngine(seed=42)
        registry = engine.generate(
            resource_types=[
                "Patient",
                "Encounter",
                "Condition",
                "Observation",
                "MedicationRequest",
                "Procedure",
                "DocumentReference",
            ],
            patient_count=3,
            encounters_per_patient={"min": 1, "max": 2},
            clinical_density="moderate",
            narrative_provider=MockProvider(),
        )
        doc_ids = registry.get_ids("DocumentReference")
        assert len(doc_ids) > 0

    def test_generate_without_narrative_provider_skips(self):
        from core.medical.engine import MedicalEngine

        engine = MedicalEngine(seed=42)
        registry = engine.generate(
            resource_types=["Patient", "Encounter", "Condition", "DocumentReference"],
            patient_count=3,
            encounters_per_patient={"min": 1, "max": 2},
        )
        doc_ids = registry.get_ids("DocumentReference")
        assert len(doc_ids) == 0


# ===========================================================================
# Task 5 Tests: narrative API routes
# ===========================================================================


class TestNarrativeAPI:
    def test_fhir_generate_request_has_narrative_fields(self):
        from api.medical_routes import FHIRGenerateRequest

        req = FHIRGenerateRequest(
            include_narrative=True,
            narrative_doc_types=["clinical_note"],
            narrative_provider="ollama",
        )
        assert req.include_narrative is True
        assert req.narrative_doc_types == ["clinical_note"]
        assert req.narrative_provider == "ollama"

    def test_narrative_generate_request_model(self):
        from api.medical_routes import NarrativeGenerateRequest

        req = NarrativeGenerateRequest(
            bundle={"resourceType": "Bundle", "entry": []},
            doc_types=["discharge_summary"],
            provider="ollama",
        )
        assert req.provider == "ollama"
        assert req.doc_types == ["discharge_summary"]
