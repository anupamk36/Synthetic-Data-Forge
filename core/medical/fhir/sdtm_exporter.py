"""CDISC SDTM domain export from generated FHIR resources."""

from __future__ import annotations

from core.medical.fhir.references import ReferenceRegistry


def export_dm(registry: ReferenceRegistry, study_id: str) -> list[dict]:
    """Export Demographics (DM) domain."""
    rows = []
    study = registry.get_resource("ResearchStudy", study_id)
    nct = ""
    if study:
        ids = study.get("identifier", [])
        nct = ids[0].get("value", "") if ids else ""

    for subj in registry.resources_by_type("ResearchSubject"):
        patient_ref = subj.get("individual", {}).get("reference", "")
        patient_id = patient_ref.replace("Patient/", "")
        patient = registry.get_resource("Patient", patient_id)
        if not patient:
            continue

        period = subj.get("period", {})

        rows.append(
            {
                "STUDYID": nct,
                "DOMAIN": "DM",
                "USUBJID": f"{nct}-{subj.get('identifier', [{}])[0].get('value', subj['id'])}",
                "SUBJID": subj.get("identifier", [{}])[0].get("value", subj["id"]),
                "SITEID": "",
                "AGE": patient.get("_age", ""),
                "AGEU": "YEARS",
                "SEX": {"male": "M", "female": "F"}.get(patient.get("gender", ""), "U"),
                "RACE": "",
                "ARM": subj.get("assignedArm", ""),
                "ARMCD": subj.get("assignedArm", ""),
                "RFSTDTC": period.get("start", ""),
                "RFENDTC": period.get("end", ""),
            }
        )
    return rows


def export_sv(registry: ReferenceRegistry) -> list[dict]:
    """Export Subject Visits (SV) domain."""
    rows = []
    for enc in registry.resources_by_type("Encounter"):
        visit_def = enc.get("_visit_def", {})
        period = enc.get("period", {})
        subj_id = enc.get("_subject_id", "")

        subject = registry.get_resource("ResearchSubject", subj_id) if subj_id else None
        usubjid = ""
        if subject:
            ids = subject.get("identifier", [{}])
            usubjid = ids[0].get("value", "") if ids else ""

        rows.append(
            {
                "DOMAIN": "SV",
                "USUBJID": usubjid,
                "VISITNUM": visit_def.get("visit_num", ""),
                "VISIT": visit_def.get("visit_name", ""),
                "SVSTDTC": period.get("start", ""),
                "SVENDTC": period.get("end", ""),
            }
        )
    return rows


def export_ae(registry: ReferenceRegistry) -> list[dict]:
    """Export Adverse Events (AE) domain."""
    rows = []
    for cond in registry.resources_by_type("Condition"):
        ae_data = cond.get("_ae_data")
        if not ae_data:
            continue

        rows.append(
            {
                "DOMAIN": "AE",
                "AETERM": ae_data.get("pt_name", ""),
                "AEDECOD": ae_data.get("pt_name", ""),
                "AEBODSYS": ae_data.get("soc_name", ""),
                "AESEV": ae_data.get("severity", "").upper(),
                "AESER": "Y" if ae_data.get("serious") else "N",
                "AEREL": ae_data.get("causality", "").upper(),
                "AEOUT": ae_data.get("outcome", "").upper(),
                "AESTDTC": ae_data.get("onset_date", ""),
                "AEENDTC": ae_data.get("end_date", ""),
            }
        )
    return rows


def export_lb(registry: ReferenceRegistry) -> list[dict]:
    """Export Laboratory Results (LB) domain."""
    rows = []
    for obs in registry.resources_by_type("Observation"):
        category = obs.get("category", [{}])
        cat_code = category[0].get("coding", [{}])[0].get("code", "") if category else ""
        if cat_code != "laboratory":
            continue

        code = obs.get("code", {}).get("coding", [{}])[0]
        vq = obs.get("valueQuantity", {})
        ref_ranges = obs.get("referenceRange", [{}])
        ref_range = ref_ranges[0] if ref_ranges else {}

        rows.append(
            {
                "DOMAIN": "LB",
                "LBTESTCD": code.get("code", ""),
                "LBTEST": code.get("display", ""),
                "LBORRES": str(vq.get("value", "")),
                "LBORRESU": vq.get("unit", ""),
                "LBORNRLO": str(ref_range.get("low", {}).get("value", "")),
                "LBORNRHI": str(ref_range.get("high", {}).get("value", "")),
                "LBDTC": obs.get("effectiveDateTime", ""),
            }
        )
    return rows


def export_vs(registry: ReferenceRegistry) -> list[dict]:
    """Export Vital Signs (VS) domain."""
    rows = []
    for obs in registry.resources_by_type("Observation"):
        category = obs.get("category", [{}])
        cat_code = category[0].get("coding", [{}])[0].get("code", "") if category else ""
        if cat_code != "vital-signs":
            continue

        code = obs.get("code", {}).get("coding", [{}])[0]
        vq = obs.get("valueQuantity", {})

        rows.append(
            {
                "DOMAIN": "VS",
                "VSTESTCD": code.get("code", ""),
                "VSTEST": code.get("display", ""),
                "VSORRES": str(vq.get("value", "")),
                "VSORRESU": vq.get("unit", ""),
                "VSDTC": obs.get("effectiveDateTime", ""),
            }
        )
    return rows


def export_tu(registry: ReferenceRegistry) -> list[dict]:
    """Export Tumor Identification (TU) domain — oncology only."""
    return []


def export_tr(registry: ReferenceRegistry) -> list[dict]:
    """Export Tumor Results (TR) domain — from RECIST observations."""
    rows = []
    for obs in registry.resources_by_type("Observation"):
        code = obs.get("code", {}).get("coding", [{}])[0]
        if code.get("code") != "96902-1":
            continue

        vq = obs.get("valueQuantity", {})
        rows.append(
            {
                "DOMAIN": "TR",
                "TRTEST": "SUMDIAM",
                "TRORRES": str(vq.get("value", "")),
                "TRORRESU": vq.get("unit", "mm"),
                "TRDTC": obs.get("effectiveDateTime", ""),
            }
        )
    return rows


def export_rs(registry: ReferenceRegistry) -> list[dict]:
    """Export Disease Response (RS) domain — RECIST responses."""
    rows = []
    for obs in registry.resources_by_type("Observation"):
        code = obs.get("code", {}).get("coding", [{}])[0]
        if code.get("code") != "21976-6":
            continue

        rows.append(
            {
                "DOMAIN": "RS",
                "RSTEST": "OVRLRESP",
                "RSORRES": obs.get("valueString", ""),
                "RSEVAL": "INVESTIGATOR",
                "RSDTC": obs.get("effectiveDateTime", ""),
            }
        )
    return rows


def export_ex(registry: ReferenceRegistry) -> list[dict]:
    """Export Exposure (EX) domain — study drug administration."""
    rows = []
    for medreq in registry.resources_by_type("MedicationRequest"):
        code = medreq.get("medicationCodeableConcept", {}).get("coding", [{}])[0]
        rows.append(
            {
                "DOMAIN": "EX",
                "EXTRT": code.get("display", ""),
                "EXDOSE": "",
                "EXDOSU": "",
                "EXROUTE": "",
                "EXSTDTC": medreq.get("authoredOn", ""),
                "EXENDTC": medreq.get("authoredOn", ""),
            }
        )
    return rows


def export_all(registry: ReferenceRegistry, study_id: str, therapeutic_area: str = "oncology") -> dict[str, list[dict]]:
    """Export all applicable SDTM domains."""
    domains = {
        "DM": export_dm(registry, study_id),
        "SV": export_sv(registry),
        "AE": export_ae(registry),
        "LB": export_lb(registry),
        "VS": export_vs(registry),
        "EX": export_ex(registry),
    }

    if therapeutic_area == "oncology":
        domains["TR"] = export_tr(registry)
        domains["RS"] = export_rs(registry)

    # Filter out empty domains
    return {k: v for k, v in domains.items() if v}
