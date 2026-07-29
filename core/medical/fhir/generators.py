"""Per-resource-type FHIR generators with clinical coherence."""

from __future__ import annotations

import random
import uuid
from datetime import date, datetime, timedelta, timezone

from faker import Faker

from core.medical.fhir.references import ReferenceRegistry
from core.medical.terminologies import icd10, loinc, rxnorm, snomed
from core.medical.terminologies.loader import _load_codeset

_ENCOUNTER_CLASSES = [
    {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB", "display": "ambulatory"},
    {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "IMP", "display": "inpatient encounter"},
    {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "EMER", "display": "emergency"},
]

_ENCOUNTER_TYPES = [
    {"code": "185349003", "display": "Encounter for check up", "system": "http://snomed.info/sct"},
    {"code": "270427003", "display": "Patient-initiated encounter", "system": "http://snomed.info/sct"},
    {"code": "390906007", "display": "Follow-up encounter", "system": "http://snomed.info/sct"},
]

_MARITAL_STATUSES = [
    {"code": "M", "display": "Married"},
    {"code": "S", "display": "Never Married"},
    {"code": "D", "display": "Divorced"},
    {"code": "W", "display": "Widowed"},
]

_ORG_TYPES = [
    {
        "code": "prov",
        "display": "Healthcare Provider",
        "system": "http://terminology.hl7.org/CodeSystem/organization-type",
    },
    {
        "code": "dept",
        "display": "Hospital Department",
        "system": "http://terminology.hl7.org/CodeSystem/organization-type",
    },
    {
        "code": "laboratory",
        "display": "Laboratory",
        "system": "http://terminology.hl7.org/CodeSystem/organization-type",
    },
]

_SPECIALTIES = [
    "General Practice",
    "Internal Medicine",
    "Cardiology",
    "Oncology",
    "Pulmonology",
    "Neurology",
    "Orthopedics",
    "Gastroenterology",
    "Endocrinology",
    "Nephrology",
    "Radiology",
    "Pathology",
    "Emergency Medicine",
    "Surgery",
]

_ALLERGY_MANIFESTATIONS = [
    {"code": "39579001", "display": "Anaphylaxis", "system": "http://snomed.info/sct"},
    {"code": "247472004", "display": "Urticaria", "system": "http://snomed.info/sct"},
    {"code": "271807003", "display": "Rash", "system": "http://snomed.info/sct"},
    {"code": "267036007", "display": "Dyspnea", "system": "http://snomed.info/sct"},
    {"code": "422587007", "display": "Nausea", "system": "http://snomed.info/sct"},
    {"code": "418290006", "display": "Itching", "system": "http://snomed.info/sct"},
]

_SEVERITY_CODES = {
    "mild": {"code": "255604002", "display": "Mild", "system": "http://snomed.info/sct"},
    "moderate": {"code": "6736007", "display": "Moderate", "system": "http://snomed.info/sct"},
    "severe": {"code": "24484000", "display": "Severe", "system": "http://snomed.info/sct"},
}

_CLINICAL_STATUS_ACTIVE = {
    "coding": [
        {"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active", "display": "Active"}
    ]
}
_CLINICAL_STATUS_RESOLVED = {
    "coding": [
        {
            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
            "code": "resolved",
            "display": "Resolved",
        }
    ]
}
_VERIFICATION_CONFIRMED = {
    "coding": [
        {
            "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
            "code": "confirmed",
            "display": "Confirmed",
        }
    ]
}
_ALLERGY_ACTIVE = {
    "coding": [
        {
            "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
            "code": "active",
            "display": "Active",
        }
    ]
}
_ALLERGY_VERIFIED = {
    "coding": [
        {
            "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification",
            "code": "confirmed",
            "display": "Confirmed",
        }
    ]
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _meta() -> dict:
    return {"lastUpdated": _now_iso()}


def _make_codeable(code_entry: dict) -> dict:
    return {
        "coding": [{"system": code_entry["system"], "code": code_entry["code"], "display": code_entry["display"]}],
        "text": code_entry["display"],
    }


class FHIRGeneratorContext:
    """Shared state for a generation run."""

    def __init__(self, seed: int | None = None, terminology_focus: str | None = None):
        self.registry = ReferenceRegistry()
        self.fake = Faker()
        self.rng = random.Random(seed)
        self.terminology_focus = terminology_focus
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)

    def uid(self) -> str:
        """Generate a deterministic UUID from the seeded RNG."""
        return str(uuid.UUID(int=self.rng.getrandbits(128), version=4))

    def ref(self, resource_type: str, resource_id: str, display: str | None = None) -> dict:
        return self.registry.make_reference(resource_type, resource_id, display)

    def random_ref(self, resource_type: str) -> dict | None:
        ids = self.registry.get_ids(resource_type)
        if not ids:
            return None
        rid = self.rng.choice(ids)
        resource = self.registry.get_resource(resource_type, rid)
        display = None
        if resource:
            if resource_type == "Patient":
                names = resource.get("name", [])
                if names:
                    display = names[0].get("text")
            elif resource_type == "Practitioner":
                names = resource.get("name", [])
                if names:
                    display = names[0].get("text")
            elif resource_type == "Organization":
                display = resource.get("name")
        return {"reference": f"{resource_type}/{rid}", "display": display}


def generate_organizations(ctx: FHIRGeneratorContext, count: int) -> list[dict]:
    resources = []
    for _i in range(count):
        org_id = ctx.uid()
        org_type = ctx.rng.choice(_ORG_TYPES)
        org_suffix = ctx.rng.choice(
            ["General Hospital", "Medical Center", "Regional Health System", "University Hospital", "Community Clinic"]
        )
        org = {
            "resourceType": "Organization",
            "id": org_id,
            "meta": _meta(),
            "identifier": [
                {"system": "http://hl7.org/fhir/sid/us-npi", "value": str(ctx.rng.randint(1000000000, 9999999999))}
            ],
            "name": f"{ctx.fake.city()} {org_suffix}",
            "type": [_make_codeable(org_type)],
            "telecom": [{"system": "phone", "value": ctx.fake.phone_number(), "use": "work"}],
            "address": [
                {
                    "line": [ctx.fake.street_address()],
                    "city": ctx.fake.city(),
                    "state": ctx.fake.state_abbr(),
                    "postalCode": ctx.fake.zipcode(),
                    "country": "US",
                }
            ],
        }
        ctx.registry.register("Organization", org_id, org)
        resources.append(org)
    return resources


def generate_practitioners(ctx: FHIRGeneratorContext, count: int) -> list[dict]:
    resources = []
    for _i in range(count):
        prac_id = ctx.uid()
        gender = ctx.rng.choice(["male", "female"])
        first = ctx.fake.first_name_male() if gender == "male" else ctx.fake.first_name_female()
        last = ctx.fake.last_name()
        specialty = ctx.rng.choice(_SPECIALTIES)

        org_ref = ctx.random_ref("Organization")

        prac = {
            "resourceType": "Practitioner",
            "id": prac_id,
            "meta": _meta(),
            "identifier": [
                {"system": "http://hl7.org/fhir/sid/us-npi", "value": str(ctx.rng.randint(1000000000, 9999999999))}
            ],
            "name": [{"family": last, "given": [first], "text": f"Dr. {first} {last}"}],
            "telecom": [{"system": "email", "value": f"{first.lower()}.{last.lower()}@hospital.org", "use": "work"}],
            "gender": gender,
            "qualification": [
                {
                    "code": {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v2-0360",
                                "code": "MD",
                                "display": "Doctor of Medicine",
                            }
                        ],
                        "text": specialty,
                    },
                    "issuer": org_ref if org_ref else {"reference": "Organization/unknown"},
                }
            ],
        }
        ctx.registry.register("Practitioner", prac_id, prac)
        resources.append(prac)
    return resources


def generate_patients(ctx: FHIRGeneratorContext, count: int) -> list[dict]:
    resources = []
    for _i in range(count):
        patient_id = ctx.uid()
        gender = ctx.rng.choice(["male", "female"])
        age = ctx.rng.randint(1, 95)
        birth_year = date.today().year - age
        birth_date = date(birth_year, ctx.rng.randint(1, 12), ctx.rng.randint(1, 28))

        first = ctx.fake.first_name_male() if gender == "male" else ctx.fake.first_name_female()
        last = ctx.fake.last_name()

        marital = ctx.rng.choice(_MARITAL_STATUSES) if age >= 18 else None
        org_ref = ctx.random_ref("Organization")

        patient = {
            "resourceType": "Patient",
            "id": patient_id,
            "meta": _meta(),
            "identifier": [{"system": "http://hospital.org/mrn", "value": f"MRN-{ctx.rng.randint(100000, 999999)}"}],
            "name": [{"family": last, "given": [first], "text": f"{first} {last}"}],
            "gender": gender,
            "birthDate": birth_date.isoformat(),
            "address": [
                {
                    "line": [ctx.fake.street_address()],
                    "city": ctx.fake.city(),
                    "state": ctx.fake.state_abbr(),
                    "postalCode": ctx.fake.zipcode(),
                    "country": "US",
                }
            ],
            "telecom": [
                {"system": "phone", "value": ctx.fake.phone_number(), "use": "home"},
                {"system": "email", "value": ctx.fake.email()},
            ],
        }
        if marital:
            patient["maritalStatus"] = {
                "coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus", **marital}],
                "text": marital["display"],
            }
        if org_ref:
            patient["managingOrganization"] = org_ref
        if age > 80 and ctx.rng.random() < 0.05:
            patient["deceasedBoolean"] = True

        # Store age for clinical coherence in later resources
        patient["_age"] = age
        ctx.registry.register("Patient", patient_id, patient)
        resources.append(patient)
    return resources


def generate_encounters(
    ctx: FHIRGeneratorContext,
    patient_ids: list[str],
    min_per_patient: int = 1,
    max_per_patient: int = 5,
) -> list[dict]:
    resources = []
    for pid in patient_ids:
        n_encounters = ctx.rng.randint(min_per_patient, max_per_patient)
        base_date = date.today() - timedelta(days=ctx.rng.randint(30, 730))

        for j in range(n_encounters):
            enc_id = ctx.uid()
            enc_class = ctx.rng.choice(_ENCOUNTER_CLASSES)
            enc_type = ctx.rng.choice(_ENCOUNTER_TYPES)
            start = base_date + timedelta(days=j * ctx.rng.randint(7, 90))
            duration_hours = ctx.rng.randint(1, 72) if enc_class["code"] == "IMP" else ctx.rng.randint(1, 4)
            end = datetime.combine(start, datetime.min.time()) + timedelta(hours=duration_hours)

            practitioner_ref = ctx.random_ref("Practitioner")
            org_ref = ctx.random_ref("Organization")

            enc = {
                "resourceType": "Encounter",
                "id": enc_id,
                "meta": _meta(),
                "status": "finished",
                "class": enc_class,
                "type": [_make_codeable(enc_type)],
                "subject": ctx.ref("Patient", pid),
                "period": {"start": start.isoformat(), "end": end.isoformat()},
            }
            if practitioner_ref:
                enc["participant"] = [{"individual": practitioner_ref}]
            if org_ref:
                enc["serviceProvider"] = org_ref

            # Store patient ref for later use
            enc["_patient_id"] = pid
            enc["_start_date"] = start.isoformat()
            ctx.registry.register("Encounter", enc_id, enc)
            resources.append(enc)
    return resources


def generate_conditions(ctx: FHIRGeneratorContext, encounter_ids: list[str], density: str = "moderate") -> list[dict]:
    resources = []
    counts = {"low": (0, 1), "moderate": (1, 2), "high": (2, 4)}
    lo, hi = counts.get(density, (1, 2))

    for enc_id in encounter_ids:
        enc = ctx.registry.get_resource("Encounter", enc_id)
        if not enc:
            continue
        pid = enc.get("_patient_id", "")
        patient = ctx.registry.get_resource("Patient", pid)
        age = patient.get("_age", 50) if patient else 50
        gender = patient.get("gender", "male") if patient else None
        enc_date = enc.get("_start_date", date.today().isoformat())

        n = ctx.rng.randint(lo, hi)
        for _ in range(n):
            dx = icd10.random_diagnosis(
                category=ctx.terminology_focus,
                age=age,
                gender=gender,
                rng=ctx.rng,
            )
            cond_id = ctx.uid()
            severity_key = ctx.rng.choice(["mild", "moderate", "severe"])

            cond = {
                "resourceType": "Condition",
                "id": cond_id,
                "meta": _meta(),
                "clinicalStatus": _CLINICAL_STATUS_ACTIVE,
                "verificationStatus": _VERIFICATION_CONFIRMED,
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                "code": "encounter-diagnosis",
                                "display": "Encounter Diagnosis",
                            }
                        ]
                    }
                ],
                "code": _make_codeable({"system": icd10.system_uri(), **dx}),
                "subject": ctx.ref("Patient", pid),
                "encounter": ctx.ref("Encounter", enc_id),
                "onsetDateTime": enc_date,
                "recordedDate": enc_date,
                "severity": _make_codeable(_SEVERITY_CODES[severity_key]),
            }
            # Store diagnosis for medication/procedure coherence
            cond["_icd10_category"] = dx.get("category")
            ctx.registry.register("Condition", cond_id, cond)
            resources.append(cond)
    return resources


def generate_observations(ctx: FHIRGeneratorContext, encounter_ids: list[str], density: str = "moderate") -> list[dict]:
    resources = []
    counts = {"low": (1, 2), "moderate": (2, 5), "high": (5, 10)}
    lo, hi = counts.get(density, (2, 5))

    lab_codes = loinc.lab_codes()
    vital_codes = loinc.vital_sign_codes()
    loinc_uri = loinc.system_uri()

    for enc_id in encounter_ids:
        enc = ctx.registry.get_resource("Encounter", enc_id)
        if not enc:
            continue
        pid = enc.get("_patient_id", "")
        enc_date = enc.get("_start_date", date.today().isoformat())
        practitioner_ref = ctx.random_ref("Practitioner")

        n_labs = ctx.rng.randint(lo, hi)
        n_vitals = ctx.rng.randint(1, 3)

        for code_entry in ctx.rng.sample(vital_codes, min(n_vitals, len(vital_codes))):
            obs = _build_observation(ctx, code_entry, loinc_uri, pid, enc_id, enc_date, practitioner_ref, "vital-signs")
            resources.append(obs)

        for code_entry in ctx.rng.sample(lab_codes, min(n_labs, len(lab_codes))):
            obs = _build_observation(ctx, code_entry, loinc_uri, pid, enc_id, enc_date, practitioner_ref, "laboratory")
            resources.append(obs)

    return resources


def _build_observation(ctx, code_entry, system_uri, pid, enc_id, enc_date, practitioner_ref, category_code) -> dict:
    obs_id = ctx.uid()
    ref_range = code_entry.get("reference_range")

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
                        "code": category_code,
                        "display": category_code.replace("-", " ").title(),
                    }
                ]
            }
        ],
        "code": _make_codeable({"system": system_uri, "code": code_entry["code"], "display": code_entry["display"]}),
        "subject": ctx.ref("Patient", pid),
        "encounter": ctx.ref("Encounter", enc_id),
        "effectiveDateTime": enc_date,
    }

    if ref_range and "unit" in ref_range:
        low_val = ref_range.get("low", 0)
        high_val = ref_range.get("high", 100)
        # Generate value slightly outside range sometimes for realism
        if ctx.rng.random() < 0.15:
            value = round(ctx.rng.uniform(low_val * 0.7, high_val * 1.3), 1)
        else:
            value = round(ctx.rng.uniform(low_val, high_val), 1)

        obs["valueQuantity"] = {
            "value": value,
            "unit": ref_range["unit"],
            "system": "http://unitsofmeasure.org",
            "code": ref_range.get("code", ref_range["unit"]),
        }
        obs["referenceRange"] = [
            {
                "low": {"value": low_val, "unit": ref_range["unit"], "system": "http://unitsofmeasure.org"},
                "high": {"value": high_val, "unit": ref_range["unit"], "system": "http://unitsofmeasure.org"},
            }
        ]

        if value < low_val:
            obs["interpretation"] = [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            "code": "L",
                            "display": "Low",
                        }
                    ]
                }
            ]
        elif value > high_val:
            obs["interpretation"] = [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            "code": "H",
                            "display": "High",
                        }
                    ]
                }
            ]
        else:
            obs["interpretation"] = [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            "code": "N",
                            "display": "Normal",
                        }
                    ]
                }
            ]
    else:
        obs["valueString"] = ctx.fake.sentence(nb_words=5)

    if practitioner_ref:
        obs["performer"] = [practitioner_ref]

    ctx.registry.register("Observation", obs_id, obs)
    return obs


def generate_medication_requests(
    ctx: FHIRGeneratorContext, encounter_ids: list[str], density: str = "moderate"
) -> list[dict]:
    resources = []
    counts = {"low": (0, 1), "moderate": (1, 2), "high": (2, 3)}
    lo, hi = counts.get(density, (1, 2))

    for enc_id in encounter_ids:
        enc = ctx.registry.get_resource("Encounter", enc_id)
        if not enc:
            continue
        pid = enc.get("_patient_id", "")
        enc_date = enc.get("_start_date", date.today().isoformat())
        practitioner_ref = ctx.random_ref("Practitioner")

        n = ctx.rng.randint(lo, hi)
        for _ in range(n):
            med = rxnorm.random_medication(category=ctx.terminology_focus, rng=ctx.rng)
            med_id = ctx.uid()
            route = med.get("route", "oral")
            form = med.get("form", "tablet")

            medreq = {
                "resourceType": "MedicationRequest",
                "id": med_id,
                "meta": _meta(),
                "status": ctx.rng.choice(["active", "completed"]),
                "intent": "order",
                "medicationCodeableConcept": _make_codeable(
                    {"system": rxnorm.system_uri(), "code": med["code"], "display": med["display"]}
                ),
                "subject": ctx.ref("Patient", pid),
                "encounter": ctx.ref("Encounter", enc_id),
                "authoredOn": enc_date,
                "dosageInstruction": [
                    {
                        "text": f"Take 1 {form} by {route} daily",
                        "timing": {"repeat": {"frequency": 1, "period": 1, "periodUnit": "d"}},
                        "route": {
                            "coding": [{"system": "http://snomed.info/sct", "code": "26643006", "display": route}]
                        },
                    }
                ],
            }
            if practitioner_ref:
                medreq["requester"] = practitioner_ref

            ctx.registry.register("MedicationRequest", med_id, medreq)
            resources.append(medreq)
    return resources


def generate_procedures(ctx: FHIRGeneratorContext, encounter_ids: list[str], density: str = "moderate") -> list[dict]:
    resources = []
    counts = {"low": (0, 0), "moderate": (0, 1), "high": (1, 2)}
    lo, hi = counts.get(density, (0, 1))

    for enc_id in encounter_ids:
        enc = ctx.registry.get_resource("Encounter", enc_id)
        if not enc:
            continue
        pid = enc.get("_patient_id", "")
        enc_date = enc.get("_start_date", date.today().isoformat())
        practitioner_ref = ctx.random_ref("Practitioner")

        n = ctx.rng.randint(lo, hi)
        for _ in range(n):
            proc_code = snomed.random_procedure(rng=ctx.rng)
            proc_id = ctx.uid()
            body_site = snomed.random_body_structure(rng=ctx.rng)

            proc = {
                "resourceType": "Procedure",
                "id": proc_id,
                "meta": _meta(),
                "status": "completed",
                "code": _make_codeable(
                    {"system": snomed.system_uri(), "code": proc_code["code"], "display": proc_code["display"]}
                ),
                "subject": ctx.ref("Patient", pid),
                "encounter": ctx.ref("Encounter", enc_id),
                "performedDateTime": enc_date,
                "bodySite": [
                    _make_codeable(
                        {"system": snomed.system_uri(), "code": body_site["code"], "display": body_site["display"]}
                    )
                ],
                "outcome": _make_codeable(
                    {"system": "http://snomed.info/sct", "code": "385669000", "display": "Successful"}
                ),
            }
            if practitioner_ref:
                proc["performer"] = [{"actor": practitioner_ref}]

            ctx.registry.register("Procedure", proc_id, proc)
            resources.append(proc)
    return resources


def generate_diagnostic_reports(ctx: FHIRGeneratorContext, encounter_ids: list[str]) -> list[dict]:
    resources = []
    for enc_id in encounter_ids:
        enc = ctx.registry.get_resource("Encounter", enc_id)
        if not enc:
            continue
        pid = enc.get("_patient_id", "")
        enc_date = enc.get("_start_date", date.today().isoformat())
        practitioner_ref = ctx.random_ref("Practitioner")

        # Collect observations for this encounter
        obs_ids = [
            oid
            for oid in ctx.registry.get_ids("Observation")
            if (obs := ctx.registry.get_resource("Observation", oid))
            and obs.get("encounter", {}).get("reference") == f"Encounter/{enc_id}"
            and obs.get("category", [{}])[0].get("coding", [{}])[0].get("code") == "laboratory"
        ]

        if not obs_ids:
            continue

        report_id = ctx.uid()
        report = {
            "resourceType": "DiagnosticReport",
            "id": report_id,
            "meta": _meta(),
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "LAB",
                            "display": "Laboratory",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [{"system": "https://loinc.org", "code": "11502-2", "display": "Laboratory report"}],
                "text": "Laboratory Report",
            },
            "subject": ctx.ref("Patient", pid),
            "encounter": ctx.ref("Encounter", enc_id),
            "effectiveDateTime": enc_date,
            "issued": _now_iso(),
            "result": [ctx.ref("Observation", oid) for oid in obs_ids],
            "conclusion": f"Laboratory results within expected parameters. {len(obs_ids)} tests performed.",
        }
        if practitioner_ref:
            report["performer"] = [practitioner_ref]

        ctx.registry.register("DiagnosticReport", report_id, report)
        resources.append(report)
    return resources


def generate_allergy_intolerances(ctx: FHIRGeneratorContext, patient_ids: list[str]) -> list[dict]:
    resources = []
    for pid in patient_ids:
        if ctx.rng.random() > 0.3:
            continue

        n = ctx.rng.randint(1, 3)
        for _ in range(n):
            substance = snomed.random_substance(rng=ctx.rng)
            allergy_id = ctx.uid()
            severity = ctx.rng.choice(["mild", "moderate", "severe"])
            manifestation = ctx.rng.choice(_ALLERGY_MANIFESTATIONS)

            allergy = {
                "resourceType": "AllergyIntolerance",
                "id": allergy_id,
                "meta": _meta(),
                "clinicalStatus": _ALLERGY_ACTIVE,
                "verificationStatus": _ALLERGY_VERIFIED,
                "type": "allergy",
                "category": [ctx.rng.choice(["medication", "food", "environment"])],
                "criticality": ctx.rng.choice(["low", "high", "unable-to-assess"]),
                "code": _make_codeable(
                    {"system": snomed.system_uri(), "code": substance["code"], "display": substance["display"]}
                ),
                "patient": ctx.ref("Patient", pid),
                "onsetDateTime": (date.today() - timedelta(days=ctx.rng.randint(365, 3650))).isoformat(),
                "reaction": [
                    {
                        "substance": _make_codeable(
                            {"system": snomed.system_uri(), "code": substance["code"], "display": substance["display"]}
                        ),
                        "manifestation": [_make_codeable(manifestation)],
                        "severity": severity,
                    }
                ],
            }
            ctx.registry.register("AllergyIntolerance", allergy_id, allergy)
            resources.append(allergy)
    return resources


def generate_imaging_studies(
    ctx: FHIRGeneratorContext, encounter_ids: list[str], density: str = "moderate"
) -> list[dict]:
    resources = []
    probability = {"low": 0.1, "moderate": 0.2, "high": 0.4}.get(density, 0.2)

    try:
        dicom_data = _load_codeset("dicom_modalities.json")
        modalities = dicom_data.get("modalities", [])
        body_parts = dicom_data.get("body_parts", [])
    except Exception:
        return resources

    if not modalities or not body_parts:
        return resources

    for enc_id in encounter_ids:
        if ctx.rng.random() > probability:
            continue

        enc = ctx.registry.get_resource("Encounter", enc_id)
        if not enc:
            continue
        pid = enc.get("_patient_id", "")
        enc_date = enc.get("_start_date", date.today().isoformat())
        practitioner_ref = ctx.random_ref("Practitioner")

        body_part = ctx.rng.choice(body_parts)
        compatible_modalities = [m for m in modalities if m["code"] in body_part.get("modalities", [])]
        if not compatible_modalities:
            compatible_modalities = modalities
        modality = ctx.rng.choice(compatible_modalities)

        n_series = ctx.rng.randint(1, 4)
        n_instances = sum(ctx.rng.randint(10, 60) for _ in range(n_series))

        study_id = ctx.uid()
        series = []
        for s in range(n_series):
            series.append(
                {
                    "uid": f"2.25.{ctx.rng.randint(10**10, 10**15)}",
                    "number": s + 1,
                    "modality": {
                        "system": "http://dicom.nema.org/resources/ontology/DCM",
                        "code": modality["code"],
                        "display": modality["display"],
                    },
                    "bodySite": {
                        "system": "http://snomed.info/sct",
                        "code": body_part.get("snomed_code", body_part["code"]),
                        "display": body_part["display"],
                    },
                    "numberOfInstances": ctx.rng.randint(10, 60),
                    "description": f"Series {s + 1} - {modality['display']} {body_part['display']}",
                }
            )

        study = {
            "resourceType": "ImagingStudy",
            "id": study_id,
            "meta": _meta(),
            "status": "available",
            "subject": ctx.ref("Patient", pid),
            "encounter": ctx.ref("Encounter", enc_id),
            "started": enc_date + "T08:00:00Z" if "T" not in enc_date else enc_date,
            "numberOfSeries": n_series,
            "numberOfInstances": n_instances,
            "modality": [
                {
                    "system": "http://dicom.nema.org/resources/ontology/DCM",
                    "code": modality["code"],
                    "display": modality["display"],
                }
            ],
            "series": series,
            "description": f"{modality['display']} of {body_part['display']}",
        }
        if practitioner_ref:
            study["referrer"] = practitioner_ref

        ctx.registry.register("ImagingStudy", study_id, study)
        resources.append(study)
    return resources
