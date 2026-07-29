"""
FHIR R4 Pydantic resource models for synthetic clinical data generation.

Defines structurally valid FHIR R4 models focused on the fields actually used
in synthetic data generation, not the full spec surface area.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ──────────────────────────────────────────────────────────────────────────────
# Shared building blocks (FHIR data types)
# ──────────────────────────────────────────────────────────────────────────────


class Coding(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    system: str
    code: str
    display: str


class CodeableConcept(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    coding: list[Coding]
    text: str | None = None


class Reference(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    reference: str  # e.g., "Patient/uuid-here"
    display: str | None = None


class Period(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    start: str  # ISO datetime
    end: str | None = None


class Identifier(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    system: str
    value: str


class HumanName(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    family: str
    given: list[str]
    text: str | None = None


class Address(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    line: list[str] | None = None
    city: str | None = None
    state: str | None = None
    postalCode: str | None = None
    country: str | None = None


class ContactPoint(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    system: str  # "phone" | "email"
    value: str
    use: str | None = None  # "home" | "work" | "mobile"


class Quantity(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    value: float
    unit: str
    system: str = "http://unitsofmeasure.org"
    code: str | None = None


class ReferenceRange(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    low: Quantity | None = None
    high: Quantity | None = None
    text: str | None = None


class Dosage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    timing: dict | None = None
    route: CodeableConcept | None = None
    doseAndRate: list[dict] | None = None


class Narrative(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: str = "generated"
    div: str  # XHTML content


class ImagingSeries(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    uid: str
    number: int
    modality: Coding
    bodySite: Coding | None = None
    numberOfInstances: int
    description: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# FHIR R4 Resource models
# ──────────────────────────────────────────────────────────────────────────────


class Organization(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resourceType: Literal["Organization"] = "Organization"
    id: str
    meta: dict  # {"lastUpdated": "..."}
    text: Narrative | None = None
    identifier: list[Identifier]
    name: str
    type: list[CodeableConcept] | None = None
    telecom: list[ContactPoint] | None = None
    address: list[Address] | None = None


class Qualification(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    code: CodeableConcept
    issuer: Reference


class Practitioner(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resourceType: Literal["Practitioner"] = "Practitioner"
    id: str
    meta: dict
    text: Narrative | None = None
    identifier: list[Identifier]
    name: list[HumanName]
    telecom: list[ContactPoint] | None = None
    gender: str | None = None
    qualification: list[Qualification] | None = None


class Patient(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resourceType: Literal["Patient"] = "Patient"
    id: str
    meta: dict
    text: Narrative | None = None
    identifier: list[Identifier]
    name: list[HumanName]
    gender: str | None = None
    birthDate: str | None = None
    address: list[Address] | None = None
    telecom: list[ContactPoint] | None = None
    maritalStatus: CodeableConcept | None = None
    managingOrganization: Reference | None = None
    deceasedBoolean: bool | None = None


class EncounterParticipant(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    individual: Reference


class Encounter(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resourceType: Literal["Encounter"] = "Encounter"
    id: str
    meta: dict
    text: Narrative | None = None
    status: str
    class_: Coding = Field(alias="class")
    type: list[CodeableConcept] | None = None
    subject: Reference
    participant: list[EncounterParticipant] | None = None
    period: Period | None = None
    serviceProvider: Reference | None = None
    reasonCode: list[CodeableConcept] | None = None


class Condition(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resourceType: Literal["Condition"] = "Condition"
    id: str
    meta: dict
    text: Narrative | None = None
    clinicalStatus: CodeableConcept
    verificationStatus: CodeableConcept
    category: list[CodeableConcept] | None = None
    code: CodeableConcept  # ICD-10
    subject: Reference
    encounter: Reference | None = None
    onsetDateTime: str | None = None
    recordedDate: str | None = None
    severity: CodeableConcept | None = None


class Observation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resourceType: Literal["Observation"] = "Observation"
    id: str
    meta: dict
    text: Narrative | None = None
    status: str
    category: list[CodeableConcept] | None = None
    code: CodeableConcept  # LOINC
    subject: Reference
    encounter: Reference | None = None
    effectiveDateTime: str | None = None
    valueQuantity: Quantity | None = None
    valueString: str | None = None
    interpretation: list[CodeableConcept] | None = None
    referenceRange: list[ReferenceRange] | None = None
    performer: list[Reference] | None = None


class MedicationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resourceType: Literal["MedicationRequest"] = "MedicationRequest"
    id: str
    meta: dict
    text: Narrative | None = None
    status: str
    intent: str
    medicationCodeableConcept: CodeableConcept  # RxNorm
    subject: Reference
    encounter: Reference | None = None
    authoredOn: str | None = None
    requester: Reference | None = None
    dosageInstruction: list[Dosage] | None = None
    reasonCode: list[CodeableConcept] | None = None


class ProcedurePerformer(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    actor: Reference


class Procedure(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resourceType: Literal["Procedure"] = "Procedure"
    id: str
    meta: dict
    text: Narrative | None = None
    status: str
    code: CodeableConcept  # SNOMED
    subject: Reference
    encounter: Reference | None = None
    performedDateTime: str | None = None
    performedPeriod: Period | None = None
    performer: list[ProcedurePerformer] | None = None
    bodySite: list[CodeableConcept] | None = None
    outcome: CodeableConcept | None = None


class DiagnosticReport(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resourceType: Literal["DiagnosticReport"] = "DiagnosticReport"
    id: str
    meta: dict
    text: Narrative | None = None
    status: str
    category: list[CodeableConcept] | None = None
    code: CodeableConcept
    subject: Reference
    encounter: Reference | None = None
    effectiveDateTime: str | None = None
    issued: str | None = None
    performer: list[Reference] | None = None
    result: list[Reference] | None = None  # References to Observations
    conclusion: str | None = None


class AllergyReaction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    substance: CodeableConcept | None = None
    manifestation: list[CodeableConcept]
    severity: str | None = None  # "mild" | "moderate" | "severe"


class AllergyIntolerance(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resourceType: Literal["AllergyIntolerance"] = "AllergyIntolerance"
    id: str
    meta: dict
    text: Narrative | None = None
    clinicalStatus: CodeableConcept
    verificationStatus: CodeableConcept
    type_: str | None = Field(default=None, alias="type")
    category: list[str] | None = None  # "food" | "medication" | "environment" | "biologic"
    criticality: str | None = None  # "low" | "high" | "unable-to-assess"
    code: CodeableConcept
    patient: Reference
    encounter: Reference | None = None
    onsetDateTime: str | None = None
    reaction: list[AllergyReaction] | None = None


class ImagingStudy(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resourceType: Literal["ImagingStudy"] = "ImagingStudy"
    id: str
    meta: dict
    text: Narrative | None = None
    status: str
    subject: Reference
    encounter: Reference | None = None
    started: str | None = None
    numberOfSeries: int | None = None
    numberOfInstances: int | None = None
    modality: list[Coding] | None = None
    referrer: Reference | None = None
    series: list[ImagingSeries] | None = None
    description: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Clinical Trial Resources
# ──────────────────────────────────────────────────────────────────────────────


class ResearchStudyArm(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    type: CodeableConcept | None = None
    description: str | None = None


class ResearchStudy(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resourceType: Literal["ResearchStudy"] = "ResearchStudy"
    id: str
    meta: dict
    text: Narrative | None = None
    identifier: list[Identifier] | None = None
    title: str
    status: str  # "active" | "completed" | "closed-to-accrual"
    phase: CodeableConcept | None = None
    category: list[CodeableConcept] | None = None
    condition: list[CodeableConcept] | None = None
    sponsor: Reference | None = None
    principalInvestigator: Reference | None = None
    site: list[Reference] | None = None
    arm: list[ResearchStudyArm] | None = None
    description: str | None = None


class ResearchSubject(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resourceType: Literal["ResearchSubject"] = "ResearchSubject"
    id: str
    meta: dict
    text: Narrative | None = None
    identifier: list[Identifier] | None = None
    status: str  # "candidate" | "eligible" | "follow-up" | "on-study" | "withdrawn"
    study: Reference
    individual: Reference  # Patient
    assignedArm: str | None = None
    consent: Reference | None = None
    period: Period | None = None


class Specimen(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resourceType: Literal["Specimen"] = "Specimen"
    id: str
    meta: dict
    text: Narrative | None = None
    identifier: list[Identifier] | None = None
    status: str | None = None  # "available" | "unavailable"
    type: CodeableConcept | None = None
    subject: Reference
    receivedTime: str | None = None
    collection: dict | None = None  # {collectedDateTime, bodySite, method}


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────


def resource_to_dict(resource: BaseModel) -> dict:
    """Serialize a FHIR resource to a dict, excluding None values."""
    return resource.model_dump(by_alias=True, exclude_none=True)
