"""DICOM metadata generation per modality — study, series, instance levels."""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

from core.medical.dicom.uid_generator import (
    generate_accession_number,
    generate_instance_uid,
    generate_series_uid,
    generate_study_uid,
    get_sop_class_uid,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_template(modality: str) -> dict:
    filename = f"{modality.lower()}_template.json"
    path = TEMPLATES_DIR / filename
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _pick(options: dict, rng: random.Random) -> float | str | int:
    """Pick a value from a parameter spec (values list, min/max range, or typical)."""
    if "values" in options:
        return rng.choice(options["values"])
    if "min" in options and "max" in options:
        if isinstance(options["min"], float) or isinstance(options["max"], float):
            return round(rng.uniform(options["min"], options["max"]), 2)
        return rng.randint(options["min"], options["max"])
    if "typical" in options:
        return options["typical"]
    return 0


def generate_study_metadata(
    modality: str,
    patient_info: dict | None = None,
    study_date: date | None = None,
    referring_physician: str | None = None,
    institution: str | None = None,
    body_part: str | None = None,
    rng: random.Random | None = None,
) -> dict:
    """Generate DICOM study-level metadata."""
    rng = rng or random.Random()
    template = _load_template(modality)
    study_date = study_date or date.today() - timedelta(days=rng.randint(0, 365))
    study_time = f"{rng.randint(6, 20):02d}{rng.randint(0, 59):02d}{rng.randint(0, 59):02d}"

    patient = patient_info or {}
    patient_id = patient.get("id", f"PAT{rng.randint(10000, 99999)}")
    patient_name = patient.get("name", "DOE^JOHN")
    patient_dob = patient.get("birth_date", "19700101")
    patient_sex = patient.get("sex", rng.choice(["M", "F"]))

    study = {
        "study_instance_uid": generate_study_uid(rng),
        "study_date": study_date.strftime("%Y%m%d"),
        "study_time": study_time,
        "accession_number": generate_accession_number(rng),
        "modality": modality,
        "referring_physician_name": referring_physician
        or f"DR^{rng.choice(['SMITH', 'JONES', 'PATEL', 'CHEN', 'GARCIA'])}",
        "institution_name": institution
        or rng.choice(["University Hospital", "Regional Medical Center", "General Hospital"]),
        "study_description": f"{template.get('modality_display', modality)} {body_part or 'Examination'}",
        "patient_id": patient_id,
        "patient_name": patient_name,
        "patient_birth_date": patient_dob,
        "patient_sex": patient_sex,
        "body_part_examined": body_part or "",
    }
    return study


def generate_series_metadata(
    study: dict,
    modality: str,
    series_number: int = 1,
    series_description: str | None = None,
    rng: random.Random | None = None,
) -> dict:
    """Generate DICOM series-level metadata with modality-specific parameters."""
    rng = rng or random.Random()
    template = _load_template(modality)

    if not series_description:
        series_types = template.get("series_types", [{"description": "Default"}])
        weights = [s.get("weight", 1) for s in series_types]
        series_description = rng.choices(series_types, weights=weights, k=1)[0]["description"]

    params = template.get("parameters", {})
    acq_params = {}
    for key, spec in params.items():
        acq_params[key] = _pick(spec, rng)

    contrast_info = None
    contrast_spec = template.get("contrast")
    if contrast_spec and rng.random() < contrast_spec.get("probability", 0):
        contrast_info = {
            "agent": rng.choice(contrast_spec.get("agents", ["Contrast"])),
            "volume_ml": _pick(contrast_spec.get("volume_ml", {"typical": 100}), rng),
            "rate_ml_s": _pick(contrast_spec.get("rate_ml_s", {"typical": 3.0}), rng)
            if "rate_ml_s" in contrast_spec
            else None,
        }

    inst_spec = template.get("instances_per_series", {"min": 10, "max": 100})
    num_instances = _pick(inst_spec, rng)

    series = {
        "series_instance_uid": generate_series_uid(rng),
        "series_number": series_number,
        "series_description": series_description,
        "modality": modality,
        "sop_class_uid": template.get("sop_class_uid", get_sop_class_uid(modality)),
        "body_part_examined": study.get("body_part_examined", ""),
        "study_instance_uid": study["study_instance_uid"],
        "acquisition_parameters": acq_params,
        "contrast": contrast_info,
        "number_of_instances": int(num_instances),
    }
    return series


def generate_instance_metadata(
    series: dict,
    instance_number: int,
    rng: random.Random | None = None,
) -> dict:
    """Generate DICOM instance-level metadata."""
    rng = rng or random.Random()
    modality = series["modality"]
    template = _load_template(modality)
    img_dims = template.get("image_dimensions", {})

    rows = _pick(img_dims.get("rows", {"values": [512]}), rng)
    cols = _pick(img_dims.get("columns", {"values": [512]}), rng)
    acq = series.get("acquisition_parameters", {})
    pixel_spacing = acq.get("pixel_spacing", 1.0)
    slice_thickness = acq.get("slice_thickness", 5.0)

    window_presets = template.get("window_presets", [{"center": 40, "width": 400}])
    window = rng.choice(window_presets)

    instance = {
        "sop_instance_uid": generate_instance_uid(rng),
        "sop_class_uid": series["sop_class_uid"],
        "instance_number": instance_number,
        "series_instance_uid": series["series_instance_uid"],
        "rows": int(rows),
        "columns": int(cols),
        "bits_allocated": img_dims.get("bits_allocated", 16),
        "bits_stored": img_dims.get("bits_stored", 12),
        "pixel_spacing": [pixel_spacing, pixel_spacing] if isinstance(pixel_spacing, int | float) else pixel_spacing,
        "slice_thickness": slice_thickness,
        "window_center": window.get("center", 40),
        "window_width": window.get("width", 400),
    }
    return instance


def to_dicom_json(study: dict, series_list: list[dict], instances_by_series: dict[str, list[dict]]) -> dict:
    """Convert metadata to DICOM JSON model (PS3.18 Annex F format)."""
    result = {
        "00100020": {"vr": "LO", "Value": [study["patient_id"]]},
        "00100010": {"vr": "PN", "Value": [{"Alphabetic": study["patient_name"]}]},
        "00100030": {"vr": "DA", "Value": [study["patient_birth_date"]]},
        "00100040": {"vr": "CS", "Value": [study["patient_sex"]]},
        "0020000D": {"vr": "UI", "Value": [study["study_instance_uid"]]},
        "00080020": {"vr": "DA", "Value": [study["study_date"]]},
        "00080030": {"vr": "TM", "Value": [study["study_time"]]},
        "00080050": {"vr": "SH", "Value": [study["accession_number"]]},
        "00080060": {"vr": "CS", "Value": [study["modality"]]},
        "00080080": {"vr": "LO", "Value": [study["institution_name"]]},
        "00080090": {"vr": "PN", "Value": [{"Alphabetic": study["referring_physician_name"]}]},
        "00081030": {"vr": "LO", "Value": [study["study_description"]]},
        "00180015": {"vr": "CS", "Value": [study["body_part_examined"]]},
        "series": [],
    }

    for s in series_list:
        series_json = {
            "0020000E": {"vr": "UI", "Value": [s["series_instance_uid"]]},
            "00200011": {"vr": "IS", "Value": [s["series_number"]]},
            "0008103E": {"vr": "LO", "Value": [s["series_description"]]},
            "00080060": {"vr": "CS", "Value": [s["modality"]]},
            "instances": [],
        }
        for inst in instances_by_series.get(s["series_instance_uid"], []):
            inst_json = {
                "00080018": {"vr": "UI", "Value": [inst["sop_instance_uid"]]},
                "00200013": {"vr": "IS", "Value": [inst["instance_number"]]},
                "00280010": {"vr": "US", "Value": [inst["rows"]]},
                "00280011": {"vr": "US", "Value": [inst["columns"]]},
                "00281050": {"vr": "DS", "Value": [inst["window_center"]]},
                "00281051": {"vr": "DS", "Value": [inst["window_width"]]},
            }
            series_json["instances"].append(inst_json)
        result["series"].append(series_json)

    return result


def generate_full_study(
    modality: str,
    patient_info: dict | None = None,
    body_part: str | None = None,
    num_series: int | None = None,
    include_instances: bool = True,
    rng: random.Random | None = None,
) -> dict:
    """Generate a complete DICOM study with series and optional instances."""
    rng = rng or random.Random()
    template = _load_template(modality)

    study = generate_study_metadata(modality, patient_info=patient_info, body_part=body_part, rng=rng)

    if num_series is None:
        series_types = template.get("series_types", [{"description": "Default"}])
        num_series = min(rng.randint(2, 4), len(series_types))

    series_list = []
    instances_by_series: dict[str, list[dict]] = {}
    total_instances = 0

    selected_types = template.get("series_types", [{"description": "Series"}])
    if len(selected_types) >= num_series:
        selected_types = rng.sample(selected_types, num_series)

    for i, stype in enumerate(selected_types):
        series = generate_series_metadata(
            study,
            modality,
            series_number=i + 1,
            series_description=stype.get("description"),
            rng=rng,
        )
        series_list.append(series)

        if include_instances:
            n_inst = series["number_of_instances"]
            instances = []
            for j in range(1, n_inst + 1):
                inst = generate_instance_metadata(series, j, rng=rng)
                instances.append(inst)
            instances_by_series[series["series_instance_uid"]] = instances
            total_instances += n_inst

    return {
        "study": study,
        "series": series_list,
        "instances": instances_by_series if include_instances else {},
        "total_series": len(series_list),
        "total_instances": total_instances,
    }
