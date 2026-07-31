"""Convert generated FHIR resources to HL7v2 pipe-delimited messages."""

from __future__ import annotations

import random
from datetime import datetime, timezone

from core.medical.fhir.references import ReferenceRegistry

_FIELD_SEP = "|"
_COMPONENT_SEP = "^"
_ESCAPE = "\\"
_REPEAT_SEP = "~"
_SUBCOMPONENT_SEP = "&"


def _hl7_timestamp(iso_date: str | None) -> str:
    if not iso_date:
        return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    clean = iso_date.replace("-", "").replace(":", "").replace("T", "").split("+")[0].split("Z")[0]
    return clean[:14]


def _msh_segment(message_type: str, trigger: str, control_id: str) -> str:
    fields = [
        "MSH",
        "^~\\&",
        "FORGE",
        "SYNTHETIC_DATA_FORGE",
        "RECEIVING_APP",
        "RECEIVING_FACILITY",
        _hl7_timestamp(None),
        "",
        f"{message_type}{_COMPONENT_SEP}{trigger}",
        control_id,
        "P",
        "2.5.1",
    ]
    return _FIELD_SEP.join(fields)


def _pid_segment(patient: dict) -> str:
    names = patient.get("name", [{}])
    name = names[0] if names else {}
    family = name.get("family", "")
    given = name.get("given", [""])[0] if name.get("given") else ""

    gender = (patient.get("gender", "U") or "U")[0].upper()
    birth_date = _hl7_timestamp(patient.get("birthDate", ""))[:8]

    addr = patient.get("address", [{}])
    address = addr[0] if addr else {}
    street = (address.get("line") or [""])[0]
    city = address.get("city", "")
    state = address.get("state", "")
    zip_code = address.get("postalCode", "")

    identifiers = patient.get("identifier", [{}])
    mrn = identifiers[0].get("value", "") if identifiers else ""

    fields = [
        "PID",
        "1",
        "",
        mrn,
        "",
        f"{family}{_COMPONENT_SEP}{given}",
        "",
        birth_date,
        gender,
        "",
        "",
        f"{street}{_COMPONENT_SEP}{_COMPONENT_SEP}{city}{_COMPONENT_SEP}{state}{_COMPONENT_SEP}{zip_code}",
    ]
    return _FIELD_SEP.join(fields)


def _pv1_segment(encounter: dict) -> str:
    enc_class = encounter.get("class", {}).get("code", "O")
    class_map = {"AMB": "O", "IMP": "I", "EMER": "E"}
    patient_class = class_map.get(enc_class, "O")

    period = encounter.get("period", {})
    admit_time = _hl7_timestamp(period.get("start"))
    discharge_time = _hl7_timestamp(period.get("end")) if period.get("end") else ""

    participants = encounter.get("participant", [])
    attending = ""
    if participants:
        ind = participants[0].get("individual", {})
        attending = ind.get("display", "").replace(" ", _COMPONENT_SEP)

    fields = [
        "PV1",
        "1",
        patient_class,
        "",
        "",
        "",
        "",
        attending,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        admit_time,
        discharge_time,
    ]
    return _FIELD_SEP.join(fields)


def _obr_segment(observation: dict, set_id: int = 1) -> str:
    code = observation.get("code", {}).get("coding", [{}])[0]
    loinc_code = code.get("code", "")
    loinc_display = code.get("display", "")
    obs_time = _hl7_timestamp(observation.get("effectiveDateTime"))

    fields = [
        "OBR",
        str(set_id),
        "",
        "",
        f"{loinc_code}{_COMPONENT_SEP}{loinc_display}{_COMPONENT_SEP}LN",
        "",
        obs_time,
    ]
    return _FIELD_SEP.join(fields)


def _obx_segment(observation: dict, set_id: int = 1) -> str:
    code = observation.get("code", {}).get("coding", [{}])[0]
    loinc_code = code.get("code", "")
    loinc_display = code.get("display", "")

    vq = observation.get("valueQuantity")
    if vq:
        value_type = "NM"
        value = str(vq.get("value", ""))
        units = vq.get("unit", "")
    else:
        value_type = "ST"
        value = observation.get("valueString", "")
        units = ""

    ref_range = ""
    ranges = observation.get("referenceRange", [])
    if ranges:
        rr = ranges[0]
        low = rr.get("low", {}).get("value", "")
        high = rr.get("high", {}).get("value", "")
        ref_range = f"{low}-{high}"

    interp = ""
    interpretations = observation.get("interpretation", [])
    if interpretations:
        interp_code = interpretations[0].get("coding", [{}])[0].get("code", "")
        interp = interp_code

    status = "F"

    fields = [
        "OBX",
        str(set_id),
        value_type,
        f"{loinc_code}{_COMPONENT_SEP}{loinc_display}{_COMPONENT_SEP}LN",
        "",
        value,
        units,
        ref_range,
        interp,
        "",
        "",
        status,
    ]
    return _FIELD_SEP.join(fields)


def fhir_to_adt_a01(patient: dict, encounter: dict) -> str:
    """Convert Patient + Encounter to HL7v2 ADT^A01 (Admission) message."""
    control_id = str(random.randint(100000, 999999))
    segments = [
        _msh_segment("ADT", "A01", control_id),
        _pid_segment(patient),
        _pv1_segment(encounter),
    ]
    return "\r".join(segments)


def fhir_to_adt_a03(patient: dict, encounter: dict) -> str:
    """Convert Patient + Encounter to HL7v2 ADT^A03 (Discharge) message."""
    control_id = str(random.randint(100000, 999999))
    segments = [
        _msh_segment("ADT", "A03", control_id),
        _pid_segment(patient),
        _pv1_segment(encounter),
    ]
    return "\r".join(segments)


def fhir_to_oru_r01(patient: dict, encounter: dict, observations: list[dict]) -> str:
    """Convert Patient + Observations to HL7v2 ORU^R01 (Lab Result) message."""
    control_id = str(random.randint(100000, 999999))
    segments = [
        _msh_segment("ORU", "R01", control_id),
        _pid_segment(patient),
        _pv1_segment(encounter),
    ]

    for i, obs in enumerate(observations, 1):
        segments.append(_obr_segment(obs, i))
        segments.append(_obx_segment(obs, i))

    return "\r".join(segments)


def convert_registry_to_hl7v2(
    registry: ReferenceRegistry,
    message_types: list[str] | None = None,
) -> list[str]:
    """Convert all applicable resources in a registry to HL7v2 messages.

    Args:
        registry: The generated resources.
        message_types: Filter to specific types ["ADT_A01", "ADT_A03", "ORU_R01"].

    Returns:
        List of HL7v2 message strings.
    """
    if message_types is None:
        message_types = ["ADT_A01", "ORU_R01"]

    messages = []

    encounters = registry.resources_by_type("Encounter")
    for enc in encounters:
        patient_ref = enc.get("subject", {}).get("reference", "")
        patient_id = patient_ref.replace("Patient/", "")
        patient = registry.get_resource("Patient", patient_id)
        if not patient:
            continue

        if "ADT_A01" in message_types:
            messages.append(fhir_to_adt_a01(patient, enc))

        if "ADT_A03" in message_types and enc.get("status") == "finished":
            messages.append(fhir_to_adt_a03(patient, enc))

        if "ORU_R01" in message_types:
            enc_id = enc.get("id", "")
            obs_list = [
                obs
                for obs in registry.resources_by_type("Observation")
                if obs.get("encounter", {}).get("reference") == f"Encounter/{enc_id}"
            ]
            if obs_list:
                messages.append(fhir_to_oru_r01(patient, enc, obs_list))

    return messages
