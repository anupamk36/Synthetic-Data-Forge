"""Tests for the ImagingEngine — DICOM metadata generation."""

import random

from core.medical.dicom.metadata import (
    generate_full_study,
    generate_instance_metadata,
    generate_series_metadata,
    generate_study_metadata,
    to_dicom_json,
)
from core.medical.dicom.uid_generator import generate_uid, get_sop_class_uid
from core.medical.imaging_engine import ImagingEngine


class TestDICOMUIDGeneration:
    def test_uid_format(self):
        uid = generate_uid(rng=random.Random(42))
        assert uid.startswith("1.2.826.0.1.3680043.8.498")
        assert len(uid) <= 64

    def test_uid_uniqueness(self):
        rng = random.Random(42)
        uids = [generate_uid(rng) for _ in range(100)]
        assert len(set(uids)) == 100

    def test_sop_class_uid(self):
        assert get_sop_class_uid("CT") == "1.2.840.10008.5.1.4.1.1.2"
        assert get_sop_class_uid("MR") == "1.2.840.10008.5.1.4.1.1.4"
        assert get_sop_class_uid("US") == "1.2.840.10008.5.1.4.1.1.6.1"


class TestStudyMetadata:
    def test_generate_study(self):
        study = generate_study_metadata("CT", rng=random.Random(42))
        assert study["modality"] == "CT"
        assert "study_instance_uid" in study
        assert "study_date" in study
        assert "accession_number" in study
        assert "patient_id" in study

    def test_study_with_patient_info(self):
        patient = {"id": "P001", "name": "DOE^JANE", "birth_date": "19850315", "sex": "F"}
        study = generate_study_metadata("MR", patient_info=patient, rng=random.Random(42))
        assert study["patient_id"] == "P001"
        assert study["patient_name"] == "DOE^JANE"
        assert study["patient_sex"] == "F"


class TestSeriesMetadata:
    def test_generate_series(self):
        study = generate_study_metadata("CT", rng=random.Random(42))
        series = generate_series_metadata(study, "CT", series_number=1, rng=random.Random(42))
        assert series["modality"] == "CT"
        assert "series_instance_uid" in series
        assert "acquisition_parameters" in series
        assert series["number_of_instances"] > 0

    def test_modality_specific_params(self):
        study = generate_study_metadata("CT", rng=random.Random(42))
        series = generate_series_metadata(study, "CT", rng=random.Random(42))
        params = series["acquisition_parameters"]
        assert "kvp" in params or "slice_thickness" in params


class TestInstanceMetadata:
    def test_generate_instance(self):
        study = generate_study_metadata("CT", rng=random.Random(42))
        series = generate_series_metadata(study, "CT", rng=random.Random(42))
        instance = generate_instance_metadata(series, instance_number=1, rng=random.Random(42))
        assert "sop_instance_uid" in instance
        assert instance["instance_number"] == 1
        assert instance["rows"] > 0
        assert instance["columns"] > 0

    def test_window_values(self):
        study = generate_study_metadata("CT", rng=random.Random(42))
        series = generate_series_metadata(study, "CT", rng=random.Random(42))
        instance = generate_instance_metadata(series, 1, rng=random.Random(42))
        assert "window_center" in instance
        assert "window_width" in instance


class TestFullStudyGeneration:
    def test_generate_full_ct_study(self):
        result = generate_full_study("CT", rng=random.Random(42))
        assert "study" in result
        assert "series" in result
        assert "instances" in result
        assert result["total_series"] >= 1
        assert result["total_instances"] > 0

    def test_generate_full_mr_study(self):
        result = generate_full_study("MR", rng=random.Random(42))
        assert result["study"]["modality"] == "MR"
        assert result["total_series"] >= 1

    def test_generate_without_instances(self):
        result = generate_full_study("US", include_instances=False, rng=random.Random(42))
        assert result["instances"] == {}
        assert result["total_instances"] == 0


class TestDICOMJSON:
    def test_dicom_json_format(self):
        result = generate_full_study("CT", rng=random.Random(42))
        dj = to_dicom_json(result["study"], result["series"], result["instances"])
        assert "0020000D" in dj  # Study Instance UID
        assert "00080060" in dj  # Modality
        assert "00100020" in dj  # Patient ID
        assert "series" in dj
        assert len(dj["series"]) == result["total_series"]

    def test_dicom_json_has_instances(self):
        result = generate_full_study("CT", rng=random.Random(42))
        dj = to_dicom_json(result["study"], result["series"], result["instances"])
        series_with_instances = [s for s in dj["series"] if len(s.get("instances", [])) > 0]
        assert len(series_with_instances) > 0


class TestImagingEngine:
    def test_generate_basic(self):
        engine = ImagingEngine(seed=42)
        result = engine.generate(modalities=["CT"], num_studies=5)
        assert result["stats"]["num_studies"] == 5
        assert result["stats"]["total_series"] > 0
        assert len(result["studies"]) == 5
        assert len(result["fhir_resources"]) == 5

    def test_multiple_modalities(self):
        engine = ImagingEngine(seed=42)
        result = engine.generate(modalities=["CT", "MR", "US"], num_studies=15)
        assert result["stats"]["num_studies"] == 15
        used_modalities = set(result["stats"]["modalities"])
        assert len(used_modalities) >= 2

    def test_body_part_filter(self):
        engine = ImagingEngine(seed=42)
        result = engine.generate(modalities=["CT"], body_parts=["CHEST"], num_studies=10)
        for study in result["studies"]:
            assert study["study"]["body_part_examined"] == "CHEST"

    def test_fhir_imaging_study_resources(self):
        engine = ImagingEngine(seed=42)
        result = engine.generate(modalities=["CT"], num_studies=5)
        for fhir_res in result["fhir_resources"]:
            assert fhir_res["resourceType"] == "ImagingStudy"
            assert fhir_res["status"] == "available"
            assert fhir_res["numberOfSeries"] > 0
            assert "subject" in fhir_res

    def test_output_dicom_json(self):
        engine = ImagingEngine(seed=42)
        result = engine.generate(modalities=["CT"], num_studies=3)
        output = engine.build_output(result, output_format="dicom_json")
        assert output["format"] == "dicom_json"
        assert len(output["data"]) == 3

    def test_output_csv(self):
        engine = ImagingEngine(seed=42)
        result = engine.generate(modalities=["CT"], num_studies=5)
        output = engine.build_output(result, output_format="csv")
        assert output["format"] == "csv"
        assert "studies" in output["data"]
        assert "series" in output["data"]
        assert "instances" in output["data"]
        assert len(output["data"]["studies"]) == 5

    def test_output_fhir(self):
        engine = ImagingEngine(seed=42)
        result = engine.generate(modalities=["MR"], num_studies=3)
        output = engine.build_output(result, output_format="fhir")
        assert output["format"] == "fhir"
        assert output["data"]["resourceType"] == "Bundle"

    def test_trial_integration(self):
        from core.medical.trial_engine import TrialEngine

        trial_engine = TrialEngine(seed=42)
        trial_registry = trial_engine.generate(
            profile_id="oncology_phase2",
            num_sites=2,
            subjects_per_arm=10,
        )

        imaging_engine = ImagingEngine(seed=42)
        result = imaging_engine.generate(
            modalities=["CT"],
            num_studies=10,
            trial_registry=trial_registry,
        )

        # Imaging studies should reference trial patients
        for fhir_res in result["fhir_resources"]:
            subject_ref = fhir_res.get("subject", {}).get("reference", "")
            assert "Patient/" in subject_ref

    def test_seed_reproducibility(self):
        engine1 = ImagingEngine(seed=99)
        engine2 = ImagingEngine(seed=99)
        r1 = engine1.generate(modalities=["CT"], num_studies=5)
        r2 = engine2.generate(modalities=["CT"], num_studies=5)
        uid1 = r1["studies"][0]["study"]["study_instance_uid"]
        uid2 = r2["studies"][0]["study"]["study_instance_uid"]
        assert uid1 == uid2

    def test_no_instance_metadata(self):
        engine = ImagingEngine(seed=42)
        result = engine.generate(modalities=["DX"], num_studies=5, include_instance_metadata=False)
        assert result["stats"]["total_instances"] == 0
