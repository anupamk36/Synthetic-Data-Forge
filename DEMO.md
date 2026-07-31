# Clinical Data Forge — Demo Guide

## Synthetic Data Platform for Life Sciences & Enterprise

> **Version:** 3.0.0 | **Stack:** Python 3.11, FastAPI, Next.js, Multi-Provider LLM (Claude / OpenAI / Gemini / Ollama), Polars, SciPy, SQLite
> **Ports:** Frontend :3000 | API :8100 | Ollama :11434

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Our Solution — Clinical Data Forge](#our-solution--clinical-data-forge)
3. [Architecture Overview](#architecture-overview)
4. [Live Demo Walkthrough](#live-demo-walkthrough)
   - [Demo 1: FHIR R4 Resource Generation](#demo-1-fhir-r4-resource-generation)
   - [Demo 2: Clinical Trial Simulation (SDTM)](#demo-2-clinical-trial-simulation-sdtm)
   - [Demo 3: Medical Imaging (DICOM)](#demo-3-medical-imaging-dicom)
   - [Demo 4: Single Table Generator](#demo-4-single-table-generator--patients-dataset)
   - [Demo 5: Multi-Table Relational Data (Hydra)](#demo-5-multi-table-relational-data-hydra)
   - [Demo 6: Privacy Compliance Audit](#demo-6-privacy-compliance-audit)
   - [Demo 7: Data Quality Dashboard](#demo-7-data-quality-dashboard)
   - [Demo 8: Schema Library & History](#demo-8-schema-library--history)
5. [REST API for Programmatic Access](#rest-api-for-programmatic-access)
6. [Deployment](#deployment)
7. [Impact & Use Cases](#impact--use-cases)

---

## The Problem

Organizations working with clinical and sensitive data face a critical dilemma:

| Challenge | Impact |
|-----------|--------|
| **Regulatory constraints** (GDPR, HIPAA, GxP) | Cannot share real patient data across teams |
| **Clinical data complexity** | FHIR, DICOM, HL7v2, SDTM standards are hard to generate correctly |
| **Multi-table complexity** | Parent-child relationships break when data is scrambled |
| **Privacy leaks** | Naive anonymization still leaks sensitive information |
| **No quality assurance** | No way to measure if synthetic data actually resembles real data |

---

## Our Solution — Clinical Data Forge

A medical-first synthetic data platform that generates compliant, realistic datasets with FHIR R4, DICOM, HL7v2, SDTM support — plus general-purpose tabular data generation with privacy guarantees.

### Key Differentiators

| Feature | What It Does |
|---------|--------------|
| **FHIR R4 Bundles** | Generate 11 resource types with cross-resource referential integrity and 6 terminology systems |
| **Clinical Trials** | Phase I-III simulation with CDISC SDTM export, randomization, dropout, adverse events |
| **DICOM Metadata** | Study/Series/Instance hierarchy for CT, MR, US, DX, MG, PT with modality-specific templates |
| **HL7v2 Conversion** | Convert FHIR bundles to ADT/ORU messages for legacy integration |
| **Privacy Compliance** | DCR + k-anonymity + l-diversity + epsilon estimation with downloadable compliance reports |
| **Multi-Provider LLM** | Claude, OpenAI, Gemini, or Ollama for semantic data generation and validation |
| **Gaussian Copula** | Preserves cross-column correlations using mathematically rigorous copula-based generation |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│         Next.js Frontend (:3000)                │
│   Landing → Clinical Data → Generic → QA       │
└──────────────────────┬──────────────────────────┘
                       │ HTTP /api/v1/*
                       ▼
┌─────────────────────────────────────────────────┐
│            FastAPI Backend (:8100)               │
├───────────┬──────────────┬──────────────────────┤
│ Medical   │ Generation   │ Privacy / Quality    │
│ ├─ FHIR   │ ├─ Single    │ ├─ DCR              │
│ ├─ Trials  │ ├─ Relational│ ├─ k-Anonymity      │
│ ├─ Imaging │ └─ LLM Logic │ └─ Compliance       │
│ └─ HL7v2   │              │                      │
└───────────┴──────────────┴──────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
     Ollama (:11434)    Core Engines (Python)
```

---

## Live Demo Walkthrough

### Demo 1: FHIR R4 Resource Generation

**Story:** *"A development team needs realistic FHIR R4 bundles to test their EHR integration — with proper ICD-10 diagnoses, LOINC observations, and cross-resource references."*

**Steps:**

1. Navigate to **Clinical Data → FHIR Generator**
2. Select resource types: Patient, Encounter, Condition, Observation, MedicationRequest
3. Configure: 100 patients, 2-5 encounters each, high clinical density
4. Choose terminology focus: Oncology
5. Enable HL7v2 conversion
6. Click **Generate**

**What to highlight:**
- 11 FHIR resource types with dependency-aware generation order
- Age/gender-filtered diagnosis codes (ICD-10)
- Cross-resource references validated automatically
- Dual download: FHIR Bundle JSON + HL7v2 messages

---

### Demo 2: Clinical Trial Simulation (SDTM)

**Story:** *"A CRO needs synthetic Phase II oncology trial data with CDISC SDTM export for regulatory submission testing."*

**Steps:**

1. Navigate to **Clinical Data → Clinical Trials**
2. Select trial profile: Oncology Phase II
3. Configure: 5 sites, 50 subjects per arm, 15% dropout
4. Select output: SDTM + FHIR
5. Click **Generate**

**What to highlight:**
- SDTM domains: DM (Demographics), SV (Subject Visits), AE (Adverse Events), LB (Laboratory)
- Realistic dropout curves and adverse event timing
- FHIR + SDTM dual export in one generation

---

### Demo 3: Medical Imaging (DICOM)

**Steps:**

1. Navigate to **Clinical Data → Imaging Data**
2. Select modalities: CT, MR
3. Filter body parts: Head, Chest
4. Generate 50 studies with instance-level metadata
5. Download DICOM JSON

---

### Demo 4: Single Table Generator — Patients Dataset

**File:** `demo_data/patients.csv`

**Steps:**

1. Navigate to **Data Generation → Single Table**
2. Upload `demo_data/patients.csv`
3. Review auto-inferred schema
4. Set record count to 10,000
5. Click **Generate**

**What to highlight:**
- Smart column detection maps `diagnosis`, `email`, `age` to realistic Faker providers
- Gaussian copula preserves cross-column correlations from the sample data
- Optional LLM semantic validation catches logical inconsistencies

---

### Demo 5: Multi-Table Relational Data (Hydra)

**Files:** `demo_data/clinical_sites.csv`, `clinical_subjects.csv`, `clinical_visits.csv`

**Steps:**

1. Navigate to **Data Generation → Multi-Table**
2. Upload all 3 CSV files
3. Define FK relationships: subjects.site_id → sites.site_id, visits.subject_id → subjects.subject_id
4. Set record counts: Sites=20, Subjects=200, Visits=500
5. Click **Generate All Tables**

**What to highlight:**
- DAG-based topological sort ensures parents generated before children
- Every FK value guaranteed to exist in parent table
- Circular dependency detection

---

### Demo 6: Privacy Compliance Audit

**Files:** `demo_data/privacy_real_employees.csv`, `demo_data/privacy_synthetic_employees.csv`

**Steps:**

1. Navigate to **Analyze → Privacy Audit**
2. Upload real and synthetic datasets
3. Select quasi-identifier columns
4. Click **Run Full Audit**

**What to see:**
- **DCR Metrics** — Distance to Closest Record with risk badge
- **k-Anonymity** — Minimum group size for quasi-identifiers
- **l-Diversity** — Sensitive attribute diversity per group
- **Epsilon Estimate** — Empirical privacy loss measurement
- **Compliance Report** — Downloadable JSON report

---

### Demo 7: Data Quality Dashboard

**Steps:**

1. Navigate to **Analyze → Data Quality**
2. Upload original and synthetic datasets
3. View realism grade (A-F) with KS tests, chi-squared tests, and correlation preservation

---

### Demo 8: Schema Library & History

- **Schema Library** — Save, browse, and reuse schema definitions across teams
- **Generation History** — Full audit trail with schema, settings, provider, elapsed time

---

## REST API for Programmatic Access

```bash
# FHIR generation
curl -X POST http://localhost:8100/api/v1/medical/fhir/generate \
  -H "Content-Type: application/json" \
  -d '{"resource_types": ["Patient", "Encounter"], "patient_count": 100}'

# Clinical trial generation
curl -X POST http://localhost:8100/api/v1/medical/trials/generate \
  -d '{"profile": "oncology_phase2", "subjects_per_arm": 50}'

# Privacy compliance report
curl -X POST http://localhost:8100/api/v1/privacy/report \
  -d '{"real_data": [...], "synthetic_data": [...], "quasi_identifiers": ["age", "zip"]}'

# Single table generation
curl -X POST http://localhost:8100/api/v1/generate \
  -d '{"schema": {"name": "String", "age": "Int64"}, "count": 1000}'
```

---

## Deployment

```bash
docker compose up -d
```

| Service | Port | Purpose |
|---------|------|---------|
| **frontend** | 3000 | Next.js dashboard |
| **api** | 8100 | FastAPI REST API |
| **ollama** | 11434 | Local LLM |

---

## Impact & Use Cases

### Pharma & Life Sciences

| Use Case | How Clinical Data Forge Solves It |
|----------|----------------------------------|
| **EHR integration testing** | FHIR R4 bundles with proper terminology codes and references |
| **Regulatory submissions** | SDTM export from clinical trial simulation |
| **PACS testing** | DICOM metadata with realistic study hierarchies |
| **Privacy-safe data sharing** | Compliance reports with DCR + k-anonymity + epsilon |
| **Legacy system testing** | HL7v2 message generation from FHIR bundles |

---

> **Clinical Data Forge** — *Generate compliant clinical data. Guarantee privacy. Ship faster.*
