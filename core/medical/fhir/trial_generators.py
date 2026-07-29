"""Clinical trial FHIR resource generators."""

from __future__ import annotations

from datetime import date, timedelta

from core.medical.fhir.generators import (
    FHIRGeneratorContext,
    _make_codeable,
    _meta,
)
from core.medical.terminologies import loinc
from core.medical.terminologies.meddra import ae_for_profile, random_severity
from core.medical.terminologies.tnm import (
    assign_recist_trajectory,
    classify_recist_response,
    generate_tumor_measurements,
)


def generate_research_study(ctx: FHIRGeneratorContext, profile: dict) -> dict:
    """Generate a ResearchStudy resource from a trial profile."""
    study_id = ctx.uid()
    sponsor_ref = ctx.random_ref("Organization")
    pi_ref = ctx.random_ref("Practitioner")

    phase_map = {
        "Phase 1": "phase-1",
        "Phase 2": "phase-2",
        "Phase 3": "phase-3",
        "Phase 4": "phase-4",
    }
    phase_code = phase_map.get(profile["phase"], "phase-2")

    arms = []
    for arm in profile.get("arms", []):
        arms.append(
            {
                "name": arm["name"],
                "type": _make_codeable(
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/research-study-arm-type",
                        "code": "experimental" if arm["code"] != "PBO" else "placebo-comparator",
                        "display": arm["name"],
                    }
                ),
                "description": arm.get("description", ""),
            }
        )

    site_ids = ctx.registry.get_ids("Organization")
    site_refs = [ctx.ref("Organization", sid) for sid in site_ids]

    study = {
        "resourceType": "ResearchStudy",
        "id": study_id,
        "meta": _meta(),
        "identifier": [{"system": "http://clinicaltrials.gov", "value": f"NCT{ctx.rng.randint(10000000, 99999999)}"}],
        "title": f"{profile['display_name']} - {profile.get('primary_endpoint', 'Safety and Efficacy')}",
        "status": "active",
        "phase": _make_codeable(
            {
                "system": "http://terminology.hl7.org/CodeSystem/research-study-phase",
                "code": phase_code,
                "display": profile["phase"],
            }
        ),
        "condition": [
            _make_codeable(
                {
                    "system": "http://snomed.info/sct",
                    "code": "363346000",
                    "display": profile.get("therapeutic_area", "oncology").title(),
                }
            )
        ],
        "arm": arms,
        "description": profile.get("description", ""),
    }
    if sponsor_ref:
        study["sponsor"] = sponsor_ref
    if pi_ref:
        study["principalInvestigator"] = pi_ref
    if site_refs:
        study["site"] = site_refs

    ctx.registry.register("ResearchStudy", study_id, study)
    return study


def generate_research_subjects(
    ctx: FHIRGeneratorContext,
    profile: dict,
    patient_ids: list[str],
    study_id: str,
    subjects_per_arm: int,
) -> list[dict]:
    """Enroll patients as research subjects with randomization."""
    arms = profile.get("arms", [{"code": "TRT", "name": "Treatment"}])
    ratio_str = profile.get("randomization_ratio", "1:1")
    ratios = [int(r) for r in ratio_str.split(":")]

    subjects = []
    arm_assignments = []
    for i, arm in enumerate(arms):
        count = subjects_per_arm * ratios[i] // ratios[0] if i > 0 else subjects_per_arm
        arm_assignments.extend([arm] * count)
    ctx.rng.shuffle(arm_assignments)

    study_start = date.today() - timedelta(days=ctx.rng.randint(60, 365))

    for i, pid in enumerate(patient_ids):
        if i >= len(arm_assignments):
            break
        arm = arm_assignments[i]
        subj_id = ctx.uid()
        consent_date = study_start + timedelta(days=ctx.rng.randint(0, 90))

        subject = {
            "resourceType": "ResearchSubject",
            "id": subj_id,
            "meta": _meta(),
            "identifier": [{"system": "http://trial.org/subjects", "value": f"SUBJ-{i+1:04d}"}],
            "status": "on-study",
            "study": ctx.ref("ResearchStudy", study_id),
            "individual": ctx.ref("Patient", pid),
            "assignedArm": arm["code"],
            "period": {"start": consent_date.isoformat()},
        }
        subject["_arm"] = arm
        subject["_consent_date"] = consent_date
        subject["_subject_num"] = i + 1
        ctx.registry.register("ResearchSubject", subj_id, subject)
        subjects.append(subject)

    return subjects


def generate_trial_visits(
    ctx: FHIRGeneratorContext,
    subjects: list[dict],
    profile: dict,
    dropout_rate: float = 0.15,
) -> dict[str, list[dict]]:
    """Generate encounters (visits) per subject following the trial visit schedule.

    Returns: {subject_id: [encounter_dicts]}
    """
    visit_schedule = profile.get("visit_schedule", [])
    visits_by_subject: dict[str, list[dict]] = {}

    for subject in subjects:
        subj_id = subject["id"]
        pid = subject["individual"]["reference"].replace("Patient/", "")
        consent_date = subject["_consent_date"]
        dropped = False

        subject_visits = []
        for visit_def in visit_schedule:
            if dropped:
                break

            # Dropout check (increases toward end of study)
            visit_week = visit_def.get("week", 0)
            total_weeks = profile.get("duration_weeks", 48)
            progress = max(0, visit_week / total_weeks) if total_weeks > 0 else 0
            if visit_week > 0 and ctx.rng.random() < dropout_rate * progress:
                dropped = True
                subject["status"] = "withdrawn"
                break

            window = visit_def.get("window_days", 0)
            actual_offset = visit_def["week"] * 7 + ctx.rng.randint(-window, window)
            visit_date = consent_date + timedelta(days=max(0, actual_offset))

            enc_id = ctx.uid()
            enc = {
                "resourceType": "Encounter",
                "id": enc_id,
                "meta": _meta(),
                "status": "finished",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "AMB",
                    "display": "ambulatory",
                },
                "type": [
                    _make_codeable(
                        {"system": "http://snomed.info/sct", "code": "185349003", "display": visit_def["visit_name"]}
                    )
                ],
                "subject": ctx.ref("Patient", pid),
                "period": {"start": visit_date.isoformat(), "end": visit_date.isoformat()},
            }
            enc["_visit_def"] = visit_def
            enc["_visit_date"] = visit_date
            enc["_subject_id"] = subj_id
            enc["_patient_id"] = pid
            enc["_arm"] = subject["_arm"]["code"]
            ctx.registry.register("Encounter", enc_id, enc)
            subject_visits.append(enc)

        visits_by_subject[subj_id] = subject_visits

    return visits_by_subject


def generate_trial_vitals(
    ctx: FHIRGeneratorContext,
    visits_by_subject: dict[str, list[dict]],
) -> list[dict]:
    """Generate vital sign observations at each visit."""
    resources = []
    vital_codes = loinc.vital_sign_codes()
    loinc_uri = loinc.system_uri()

    for _subj_id, visits in visits_by_subject.items():
        for enc in visits:
            visit_def = enc.get("_visit_def", {})
            if "vitals" not in visit_def.get("assessments", []):
                continue

            pid = enc["_patient_id"]
            enc_id = enc["id"]
            visit_date = (
                enc["_visit_date"].isoformat() if hasattr(enc["_visit_date"], "isoformat") else str(enc["_visit_date"])
            )

            for code_entry in vital_codes[:5]:
                obs_id = ctx.uid()
                ref_range = code_entry.get("reference_range") or {}
                low_val = ref_range.get("low", 60)
                high_val = ref_range.get("high", 120)
                value = round(ctx.rng.uniform(low_val, high_val), 1)

                obs = {
                    "resourceType": "Observation",
                    "id": obs_id,
                    "meta": _meta(),
                    "status": "final",
                    "category": [
                        {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                    "code": "vital-signs",
                                    "display": "Vital Signs",
                                }
                            ]
                        }
                    ],
                    "code": _make_codeable(
                        {"system": loinc_uri, "code": code_entry["code"], "display": code_entry["display"]}
                    ),
                    "subject": ctx.ref("Patient", pid),
                    "encounter": ctx.ref("Encounter", enc_id),
                    "effectiveDateTime": visit_date,
                    "valueQuantity": {
                        "value": value,
                        "unit": ref_range.get("unit", ""),
                        "system": "http://unitsofmeasure.org",
                    },
                }
                ctx.registry.register("Observation", obs_id, obs)
                resources.append(obs)

    return resources


def generate_trial_labs(
    ctx: FHIRGeneratorContext,
    visits_by_subject: dict[str, list[dict]],
    profile: dict,
) -> list[dict]:
    """Generate lab observations at each visit based on profile lab panel."""
    resources = []
    lab_codes = loinc.lab_codes()
    loinc_uri = loinc.system_uri()
    panel_size = min(10, len(lab_codes))

    for _subj_id, visits in visits_by_subject.items():
        selected_labs = ctx.rng.sample(lab_codes, panel_size)

        for enc in visits:
            visit_def = enc.get("_visit_def", {})
            if "labs" not in visit_def.get("assessments", []):
                continue

            pid = enc["_patient_id"]
            enc_id = enc["id"]
            visit_date = (
                enc["_visit_date"].isoformat() if hasattr(enc["_visit_date"], "isoformat") else str(enc["_visit_date"])
            )

            for code_entry in selected_labs:
                obs_id = ctx.uid()
                ref_range = code_entry.get("reference_range") or {}
                low_val = ref_range.get("low", 0)
                high_val = ref_range.get("high", 100)
                value = round(ctx.rng.uniform(low_val * 0.8, high_val * 1.2), 1)

                obs = {
                    "resourceType": "Observation",
                    "id": obs_id,
                    "meta": _meta(),
                    "status": "final",
                    "category": [
                        {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                    "code": "laboratory",
                                    "display": "Laboratory",
                                }
                            ]
                        }
                    ],
                    "code": _make_codeable(
                        {"system": loinc_uri, "code": code_entry["code"], "display": code_entry["display"]}
                    ),
                    "subject": ctx.ref("Patient", pid),
                    "encounter": ctx.ref("Encounter", enc_id),
                    "effectiveDateTime": visit_date,
                    "valueQuantity": {
                        "value": value,
                        "unit": ref_range.get("unit", ""),
                        "system": "http://unitsofmeasure.org",
                    },
                }
                if ref_range:
                    obs["referenceRange"] = [
                        {
                            "low": {"value": low_val, "unit": ref_range.get("unit", "")},
                            "high": {"value": high_val, "unit": ref_range.get("unit", "")},
                        }
                    ]
                ctx.registry.register("Observation", obs_id, obs)
                resources.append(obs)

    return resources


def generate_adverse_events(
    ctx: FHIRGeneratorContext,
    visits_by_subject: dict[str, list[dict]],
    profile: dict,
) -> list[dict]:
    """Generate adverse events with arm-based frequency differences."""
    resources = []
    ae_profile = profile.get("ae_profile", {})
    common_aes = ae_profile.get("common_aes", ["fatigue", "nausea", "headache"])
    treatment_multiplier = ae_profile.get("treatment_multiplier", 1.5)
    base_ae_prob = 0.15

    for _subj_id, visits in visits_by_subject.items():
        for enc in visits:
            visit_def = enc.get("_visit_def", {})
            if "ae_check" not in visit_def.get("assessments", []):
                continue

            arm = enc.get("_arm", "TRT")
            prob = base_ae_prob * (treatment_multiplier if arm not in ("PBO", "placebo") else 1.0)

            if ctx.rng.random() > prob:
                continue

            ae_term = ae_for_profile(common_aes, rng=ctx.rng)
            severity = random_severity(ae_term, rng=ctx.rng)
            pid = enc["_patient_id"]
            enc_id = enc["id"]
            visit_date = enc["_visit_date"]

            duration_days = ctx.rng.randint(1, 30)
            end_date = visit_date + timedelta(days=duration_days)
            serious = severity == "severe" and ctx.rng.random() < 0.3

            cond_id = ctx.uid()
            cond = {
                "resourceType": "Condition",
                "id": cond_id,
                "meta": _meta(),
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active",
                            "display": "Active",
                        }
                    ]
                },
                "verificationStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                            "code": "confirmed",
                            "display": "Confirmed",
                        }
                    ]
                },
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                "code": "problem-list-item",
                                "display": "Problem List Item",
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "https://www.meddra.org",
                            "code": ae_term.get("pt_code", ""),
                            "display": ae_term.get("pt_name", ""),
                        }
                    ],
                    "text": ae_term.get("pt_name", ""),
                },
                "subject": ctx.ref("Patient", pid),
                "encounter": ctx.ref("Encounter", enc_id),
                "onsetDateTime": visit_date.isoformat(),
                "abatementDateTime": end_date.isoformat(),
                "severity": _make_codeable(
                    {
                        "system": "http://snomed.info/sct",
                        "code": "255604002"
                        if severity == "mild"
                        else "6736007"
                        if severity == "moderate"
                        else "24484000",
                        "display": severity.title(),
                    }
                ),
            }
            cond["_ae_data"] = {
                "pt_name": ae_term.get("pt_name", ""),
                "pt_code": ae_term.get("pt_code", ""),
                "soc_name": ae_term.get("soc_name", ""),
                "severity": severity,
                "serious": serious,
                "causality": "related"
                if ctx.rng.random() < ae_term.get("treatment_related_pct", 0.5)
                else "not related",
                "outcome": "recovered" if ctx.rng.random() < 0.8 else "recovering",
                "onset_date": visit_date.isoformat(),
                "end_date": end_date.isoformat(),
            }
            ctx.registry.register("Condition", cond_id, cond)
            resources.append(cond)

    return resources


def generate_disease_assessments(
    ctx: FHIRGeneratorContext,
    subjects: list[dict],
    visits_by_subject: dict[str, list[dict]],
    profile: dict,
) -> list[dict]:
    """Generate disease-specific assessments (RECIST for oncology, DAS28 for RA, etc.)."""
    therapeutic_area = profile.get("therapeutic_area", "oncology")

    if therapeutic_area == "oncology":
        return _generate_oncology_assessments(ctx, subjects, visits_by_subject, profile)
    elif therapeutic_area == "immunology":
        return _generate_ra_assessments(ctx, subjects, visits_by_subject, profile)
    elif therapeutic_area == "neuroscience":
        return _generate_neuro_assessments(ctx, subjects, visits_by_subject, profile)
    elif therapeutic_area == "ophthalmology":
        return _generate_amd_assessments(ctx, subjects, visits_by_subject, profile)
    return []


def _generate_oncology_assessments(ctx, subjects, visits_by_subject, profile) -> list[dict]:
    """Generate TNM staging and RECIST tumor response."""
    resources = []
    cancer_types = profile.get("cancer_types", ["lung"])

    for subject in subjects:
        subj_id = subject["id"]
        pid = subject["individual"]["reference"].replace("Patient/", "")
        arm = subject["_arm"]["code"]
        ctx.rng.choice(cancer_types)

        trajectory = assign_recist_trajectory(arm, rng=ctx.rng)
        imaging_visits = [
            v for v in visits_by_subject.get(subj_id, []) if "imaging" in v.get("_visit_def", {}).get("assessments", [])
        ]

        measurements = generate_tumor_measurements(trajectory, len(imaging_visits), rng=ctx.rng)

        for i, enc in enumerate(imaging_visits):
            if i >= len(measurements):
                break
            meas = measurements[i]
            visit_date = (
                enc["_visit_date"].isoformat() if hasattr(enc["_visit_date"], "isoformat") else str(enc["_visit_date"])
            )
            enc_id = enc["id"]

            obs_id = ctx.uid()
            obs = {
                "resourceType": "Observation",
                "id": obs_id,
                "meta": _meta(),
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "imaging",
                                "display": "Imaging",
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "https://loinc.org",
                            "code": "96902-1",
                            "display": "Sum of diameters of target lesions",
                        }
                    ],
                    "text": "RECIST Target Lesion Sum",
                },
                "subject": ctx.ref("Patient", pid),
                "encounter": ctx.ref("Encounter", enc_id),
                "effectiveDateTime": visit_date,
                "valueQuantity": {
                    "value": meas["sum_mm"],
                    "unit": "mm",
                    "system": "http://unitsofmeasure.org",
                    "code": "mm",
                },
            }
            ctx.registry.register("Observation", obs_id, obs)
            resources.append(obs)

            if i > 0:
                response = classify_recist_response(meas["pct_change"])
                resp_id = ctx.uid()
                resp_obs = {
                    "resourceType": "Observation",
                    "id": resp_id,
                    "meta": _meta(),
                    "status": "final",
                    "code": {
                        "coding": [{"system": "https://loinc.org", "code": "21976-6", "display": "RECIST response"}],
                        "text": "RECIST 1.1 Response",
                    },
                    "subject": ctx.ref("Patient", pid),
                    "encounter": ctx.ref("Encounter", enc_id),
                    "effectiveDateTime": visit_date,
                    "valueString": response,
                }
                ctx.registry.register("Observation", resp_id, resp_obs)
                resources.append(resp_obs)

    return resources


def _generate_ra_assessments(ctx, subjects, visits_by_subject, profile) -> list[dict]:
    """Generate DAS28 and ACR response for RA trials."""
    resources = []
    for subject in subjects:
        subj_id = subject["id"]
        pid = subject["individual"]["reference"].replace("Patient/", "")
        arm = subject["_arm"]["code"]

        baseline_das28 = ctx.rng.uniform(4.5, 7.0)
        improvement_rate = 0.03 if arm in ("PBO", "placebo") else 0.08

        for enc in visits_by_subject.get(subj_id, []):
            visit_def = enc.get("_visit_def", {})
            visit_week = visit_def.get("week", 0)
            visit_date = (
                enc["_visit_date"].isoformat() if hasattr(enc["_visit_date"], "isoformat") else str(enc["_visit_date"])
            )

            das28 = max(1.0, baseline_das28 - (improvement_rate * visit_week) + ctx.rng.uniform(-0.3, 0.3))
            obs_id = ctx.uid()
            obs = {
                "resourceType": "Observation",
                "id": obs_id,
                "meta": _meta(),
                "status": "final",
                "code": {
                    "coding": [{"system": "https://loinc.org", "code": "77597-3", "display": "DAS28-CRP score"}],
                    "text": "DAS28-CRP",
                },
                "subject": ctx.ref("Patient", pid),
                "encounter": ctx.ref("Encounter", enc["id"]),
                "effectiveDateTime": visit_date,
                "valueQuantity": {"value": round(das28, 2), "unit": "{score}", "system": "http://unitsofmeasure.org"},
            }
            ctx.registry.register("Observation", obs_id, obs)
            resources.append(obs)

    return resources


def _generate_neuro_assessments(ctx, subjects, visits_by_subject, profile) -> list[dict]:
    """Generate ADAS-Cog and MMSE for Alzheimer's trials."""
    resources = []
    for subject in subjects:
        subj_id = subject["id"]
        pid = subject["individual"]["reference"].replace("Patient/", "")
        arm = subject["_arm"]["code"]

        baseline_adas = ctx.rng.uniform(20, 40)
        decline_rate = 0.15 if arm in ("PBO", "placebo") else 0.08

        for enc in visits_by_subject.get(subj_id, []):
            visit_def = enc.get("_visit_def", {})
            visit_week = visit_def.get("week", 0)
            visit_date = (
                enc["_visit_date"].isoformat() if hasattr(enc["_visit_date"], "isoformat") else str(enc["_visit_date"])
            )

            adas = baseline_adas + (decline_rate * visit_week) + ctx.rng.uniform(-2, 2)
            obs_id = ctx.uid()
            obs = {
                "resourceType": "Observation",
                "id": obs_id,
                "meta": _meta(),
                "status": "final",
                "code": {
                    "coding": [{"system": "https://loinc.org", "code": "58151-2", "display": "ADAS-Cog score"}],
                    "text": "ADAS-Cog 13",
                },
                "subject": ctx.ref("Patient", pid),
                "encounter": ctx.ref("Encounter", enc["id"]),
                "effectiveDateTime": visit_date,
                "valueQuantity": {"value": round(adas, 1), "unit": "{score}", "system": "http://unitsofmeasure.org"},
            }
            ctx.registry.register("Observation", obs_id, obs)
            resources.append(obs)

    return resources


def _generate_amd_assessments(ctx, subjects, visits_by_subject, profile) -> list[dict]:
    """Generate BCVA and OCT measurements for AMD trials."""
    resources = []
    for subject in subjects:
        subj_id = subject["id"]
        pid = subject["individual"]["reference"].replace("Patient/", "")
        arm = subject["_arm"]["code"]

        baseline_bcva = ctx.rng.randint(45, 65)
        improvement_rate = 0.05 if arm in ("PBO", "placebo", "PRN") else 0.15

        for enc in visits_by_subject.get(subj_id, []):
            visit_date = (
                enc["_visit_date"].isoformat() if hasattr(enc["_visit_date"], "isoformat") else str(enc["_visit_date"])
            )
            visit_week = enc.get("_visit_def", {}).get("week", 0)

            bcva = baseline_bcva + (improvement_rate * visit_week) + ctx.rng.uniform(-3, 3)
            obs_id = ctx.uid()
            obs = {
                "resourceType": "Observation",
                "id": obs_id,
                "meta": _meta(),
                "status": "final",
                "code": {
                    "coding": [
                        {"system": "https://loinc.org", "code": "79880-1", "display": "Best corrected visual acuity"}
                    ],
                    "text": "BCVA (ETDRS letters)",
                },
                "subject": ctx.ref("Patient", pid),
                "encounter": ctx.ref("Encounter", enc["id"]),
                "effectiveDateTime": visit_date,
                "valueQuantity": {"value": round(bcva, 0), "unit": "letters", "system": "http://unitsofmeasure.org"},
            }
            ctx.registry.register("Observation", obs_id, obs)
            resources.append(obs)

    return resources


def generate_study_drug_exposure(
    ctx: FHIRGeneratorContext,
    visits_by_subject: dict[str, list[dict]],
    profile: dict,
) -> list[dict]:
    """Generate MedicationRequest resources for study drug administration."""
    resources = []
    arms = {arm["code"]: arm for arm in profile.get("arms", [])}

    for _subj_id, visits in visits_by_subject.items():
        for enc in visits:
            visit_def = enc.get("_visit_def", {})
            if "drug_admin" not in visit_def.get("assessments", []):
                continue

            arm_code = enc.get("_arm", "TRT")
            arm = arms.get(arm_code, {})
            drug_name = arm.get("description", "Study Drug")
            pid = enc["_patient_id"]
            visit_date = (
                enc["_visit_date"].isoformat() if hasattr(enc["_visit_date"], "isoformat") else str(enc["_visit_date"])
            )

            med_id = ctx.uid()
            medreq = {
                "resourceType": "MedicationRequest",
                "id": med_id,
                "meta": _meta(),
                "status": "completed",
                "intent": "order",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": str(ctx.rng.randint(100000, 999999)),
                            "display": drug_name,
                        }
                    ],
                    "text": drug_name,
                },
                "subject": ctx.ref("Patient", pid),
                "encounter": ctx.ref("Encounter", enc["id"]),
                "authoredOn": visit_date,
            }
            ctx.registry.register("MedicationRequest", med_id, medreq)
            resources.append(medreq)

    return resources
