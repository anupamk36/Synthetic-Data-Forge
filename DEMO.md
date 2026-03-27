# ForgeFlow AI — Hackathon Demo Guide

## Synthetic Data Platform for Enterprise & Life Sciences

> **Version:** 2.0.0 | **Stack:** Python 3.13, Streamlit, FastAPI, Ollama LLM, Polars, SQLite  
> **Team:** Synthetic-Data-Forge | **Hackathon Presentation**

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Our Solution — ForgeFlow AI](#our-solution--forgeflow-ai)
3. [Architecture Overview](#architecture-overview)
4. [Live Demo Walkthrough](#live-demo-walkthrough)
   - [Demo 1: Single Table Generator](#demo-1-single-table-generator--patients-dataset)
   - [Demo 2: LLM-Powered Smart Generation](#demo-2-llm-powered-smart-generation)
   - [Demo 3: Multi-Table Relational Data (Hydra)](#demo-3-multi-table-relational-data-hydra)
   - [Demo 4: Time Travel Simulator](#demo-4-time-travel-simulator--pharma-orders)
   - [Demo 5: Privacy Audit](#demo-5-privacy-audit--employee-data)
   - [Demo 6: Data Quality Dashboard](#demo-6-data-quality-dashboard)
   - [Demo 7: Schema Library & History](#demo-7-schema-library--history)
5. [REST API for Programmatic Access](#rest-api-for-programmatic-access)
6. [Enterprise Features](#enterprise-features)
7. [Deployment](#deployment)
8. [Impact & Use Cases](#impact--use-cases)

---

## The Problem

Organizations working with sensitive data face a critical dilemma:

| Challenge | Impact |
|-----------|--------|
| **Regulatory constraints** (GDPR, HIPAA, GxP) | Cannot share real patient/employee data across teams |
| **Slow data provisioning** | Dev & QA teams wait weeks for anonymized test datasets |
| **Privacy leaks** | Naive anonymization still leaks sensitive information |
| **Multi-table complexity** | Parent-child relationships break when data is scrambled |
| **Temporal patterns** | Time-series data loses realistic trends and seasonality |
| **No quality assurance** | No way to measure if synthetic data actually resembles real data |

**Result:** Data scientists, QA engineers, and analytics teams are blocked — slowing down drug development, clinical trials, and business intelligence.

---

## Our Solution — ForgeFlow AI

**ForgeFlow AI** is a full-stack synthetic data platform that generates realistic, privacy-safe datasets in seconds — with an intuitive UI, a REST API, LLM-powered intelligence, and built-in privacy verification.

### Key Differentiators

| Feature | What It Does |
|---------|--------------|
| **Smart Column Detection** | Automatically maps column names like `email`, `phone`, `diagnosis` to realistic Faker providers |
| **LLM-Powered Generation** | Uses a local Ollama LLM (llama3.2:3b) to generate semantically coherent data with domain constraints |
| **Multi-Table Integrity** | Generates referentially-consistent parent → child → grandchild datasets using DAG-based topological sort |
| **Time Travel Simulation** | Models temporal patterns — trends, seasonality, and calendar spikes — for realistic time-series data |
| **Privacy Scorecard** | Computes Distance to Closest Record (DCR) metrics to mathematically prove synthetic data cannot be traced back |
| **Data Quality Engine** | Measures completeness, uniqueness, schema match, and distribution fidelity — with an overall quality score |
| **Schema Library** | Save, version, search, import/export reusable schemas across teams |
| **Full Audit Trail** | Every generation logged with schema, settings, engine, elapsed time, and status |
| **REST API** | 14 FastAPI endpoints for CI/CD pipelines, Jupyter notebooks, and automated workflows |
| **Pharma Safe Mode** | One toggle to disable SSN, credit card, and IBAN generation for regulated environments |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        ForgeFlow AI v2.0                        │
├──────────────────────────┬──────────────────────────────────────┤
│                          │                                      │
│   Streamlit UI (:8501)   │     FastAPI REST API (:8100)         │
│   ┌──────────────────┐   │     ┌────────────────────────────┐   │
│   │ 7 Interactive     │   │     │ /api/v1/generate           │   │
│   │ Pages:            │   │     │ /api/v1/generate/async     │   │
│   │ • Single Table    │   │     │ /api/v1/jobs/{id}          │   │
│   │ • Multi-Table     │   │     │ /api/v1/privacy/audit      │   │
│   │ • Time Travel     │   │     │ /api/v1/schemas (CRUD)     │   │
│   │ • Privacy Audit   │   │     │ /api/v1/history            │   │
│   │ • Quality         │   │     └────────────────────────────┘   │
│   │ • Schema Library  │   │                                      │
│   │ • History         │   │                                      │
│   └──────────────────┘   │                                      │
│                          │                                      │
├──────────────────────────┴──────────────────────────────────────┤
│                     Core Engines                                │
│  ┌──────────┐ ┌──────────────┐ ┌────────────┐ ┌─────────────┐  │
│  │  Forge   │ │  Relational  │ │   Time     │ │  Privacy    │  │
│  │  Engine  │ │  Engine      │ │   Travel   │ │  Scorecard  │  │
│  │ (Faker + │ │  (DAG Sort)  │ │  (Trends)  │ │  (DCR)      │  │
│  │  LLM)    │ │              │ │            │ │             │  │
│  └──────────┘ └──────────────┘ └────────────┘ └─────────────┘  │
│  ┌──────────┐ ┌──────────────┐ ┌────────────┐                  │
│  │ Quality  │ │   Audit      │ │  Schema    │                  │
│  │ Engine   │ │   Trail      │ │  Registry  │                  │
│  └──────────┘ └──────────────┘ └────────────┘                  │
├─────────────────────────────────────────────────────────────────┤
│                     Data Sinks                                  │
│          ┌────────────┐      ┌────────────────┐                 │
│          │ Local FS   │      │  Amazon S3     │                 │
│          │ (Parquet/  │      │  (Direct       │                 │
│          │  CSV/JSON) │      │   Upload)      │                 │
│          └────────────┘      └────────────────┘                 │
├─────────────────────────────────────────────────────────────────┤
│               Ollama LLM (llama3.2:3b, local GPU)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Live Demo Walkthrough

> All demos use pre-built datasets in the `demo_data/` folder. Follow each step exactly.

---

### Demo 1: Single Table Generator — Patients Dataset

**Story:** *"A clinical data team needs 10,000 patient records for testing a new analytics dashboard, but cannot use real patient data."*

**File:** `demo_data/patients.csv`

**What the demo data looks like:**

| patient_id | first_name | last_name | age | sex | diagnosis | blood_type | region | admission_date | billing_amount |
|---|---|---|---|---|---|---|---|---|---|
| P-1001 | Emily | Rodriguez | 34 | F | Hypertension | A+ | North America | 2024-03-15 | 4,250.00 |
| P-1002 | James | Chen | 58 | M | Type 2 Diabetes | O+ | Asia Pacific | 2024-04-01 | 8,730.50 |
| P-1003 | Sophia | Müller | 27 | F | Asthma | B- | Europe | 2024-02-20 | 2,100.75 |
| *... 15 rows total* | | | | | | | | | |

**Steps:**

1. Open **ForgeFlow AI** → Navigate to **"📊 Single Table"** in the sidebar
2. Click **"Upload a file"** → Select `demo_data/patients.csv`
3. The schema is auto-inferred:
   - `patient_id` → String, `first_name` → String, `age` → Int64, `diagnosis` → String, `billing_amount` → Float64, etc.
4. Set **Record Count** to `10,000`
5. Choose **Output Format:** Parquet
6. *(Optional)* Enable **Hive Partitioning** → Partition by `region`
7. Click **Generate**

**What to highlight:**
- **Smart column detection** — `first_name`, `last_name`, `age`, `diagnosis` are auto-mapped to realistic medical-context Faker providers
- **Real-time progress bar** showing records/sec throughput
- **Post-generation analytics:** column profiler shows uniqueness, null %, and sample values per column
- **4 download formats:** CSV, JSON, Parquet, and ZIP (all formats bundled)
- **Hive partitioning** creates folder structure like `region=North America/part_0.parquet`

**Talking point:** *"In under 10 seconds, we generated 10,000 realistic patient records that are structurally identical to the original — but contain zero real patient data."*

---

### Demo 2: LLM-Powered Smart Generation

**Story:** *"The team needs data where business rules are enforced — billing amounts should correlate with diagnosis severity, and ages should be realistic for each condition."*

**Steps:**

1. On the same **Single Table** page, toggle on **"🤖 Use LLM (Smart Mode)"**
2. Write **field descriptions** (semantic hints for the LLM):
   - `diagnosis` → *"Common chronic diseases: hypertension, diabetes, COPD, asthma, CKD"*
   - `billing_amount` → *"Between 1000 and 20000, higher for severe diagnoses"*
   - `age` → *"Between 18 and 85, older for cardiac and renal conditions"*
3. Set **Record Count** to `50` (LLM generation is slower but higher quality)
4. Click **Generate**

**What to highlight:**
- The LLM (llama3.2:3b running **locally on-device** — no data leaves the machine) interprets natural language constraints
- Generated data shows semantic coherence: elderly patients with COPD, younger patients with asthma
- Batched generation: 10 records per LLM call with automatic retry and lenient JSON parsing
- **Full privacy**: the LLM runs on Ollama locally — no OpenAI, no cloud API calls, no data exfiltration

**Talking point:** *"Unlike random Faker data, LLM mode understands medical context. An 85-year-old has COPD, not acne. And it all runs locally — no data ever leaves the laptop."*

---

### Demo 3: Multi-Table Relational Data (Hydra)

**Story:** *"A clinical trial database has 3 related tables — Sites, Subjects, and Visits. We need synthetic data where every Subject belongs to a real Site, and every Visit belongs to a real Subject."*

**Files:**
- `demo_data/clinical_sites.csv` (8 records — parent)
- `demo_data/clinical_subjects.csv` (15 records — child)
- `demo_data/clinical_visits.csv` (15 records — grandchild)

**The relationship chain:**

```
clinical_sites (8 rows)          clinical_subjects (15 rows)         clinical_visits (15 rows)
┌─────────────────────┐          ┌───────────────────────┐           ┌────────────────────────┐
│ site_id (PK)        │─────────▶│ site_id (FK)          │           │ visit_id (PK)          │
│ site_name           │          │ subject_id (PK)       │──────────▶│ subject_id (FK)        │
│ country             │          │ enrollment_date       │           │ visit_date             │
│ city                │          │ age, sex              │           │ systolic_bp, heart_rate│
│ principal_inv.      │          │ weight_kg, height_cm  │           │ adverse_event          │
│ phase, status       │          │ arm, consent_signed   │           │ lab_glucose_mg_dl      │
└─────────────────────┘          └───────────────────────┘           └────────────────────────┘
```

**Sample data — clinical_sites.csv:**

| site_id | site_name | country | city | principal_investigator | phase | status |
|---|---|---|---|---|---|---|
| SITE-001 | Johns Hopkins Clinical Center | USA | Baltimore | Dr. Sarah Mitchell | Phase III | Active |
| SITE-002 | Charité Research Hospital | Germany | Berlin | Dr. Klaus Weber | Phase III | Active |
| SITE-003 | Tokyo University Hospital | Japan | Tokyo | Dr. Yuki Sato | Phase II | Active |

**Steps:**

1. Navigate to **"🔗 Multi-Table (Hydra)"**
2. Upload all 3 CSV files from `demo_data/`
3. Define **two foreign key relationships:**
   - `clinical_subjects.site_id` → `clinical_sites.site_id`
   - `clinical_visits.subject_id` → `clinical_subjects.subject_id`
4. Click **"Show Relationship Diagram"** — a Mermaid ER diagram renders showing the 3-table chain
5. Set record counts: Sites = 20, Subjects = 200, Visits = 500
6. Click **Generate All Tables**

**What to highlight:**
- **DAG-based topological sort** — Kahn's algorithm ensures Sites are generated first, then Subjects (with valid `site_id` FK values from Sites), then Visits (with valid `subject_id` FK values from Subjects)
- **Circular dependency detection** — if you try to create a cycle, it's caught immediately
- **Mermaid diagram** — visual confirmation of the relationship map before generation
- **Referential integrity guaranteed** — every `site_id` in Subjects exists in Sites; every `subject_id` in Visits exists in Subjects

**Talking point:** *"This is the 'Hydra' feature — generates an entire relational database in one click, with guaranteed foreign key integrity. No broken joins, no orphan records."*

---

### Demo 4: Time Travel Simulator — Pharma Orders

**Story:** *"The supply chain team needs 12 months of pharma order data with realistic seasonal trends — Q4 holiday spike, steady quarterly growth."*

**File:** `demo_data/pharma_orders.csv`

**Sample data:**

| order_id | order_date | product_category | product_name | quantity | unit_price | customer_region | channel |
|---|---|---|---|---|---|---|---|
| ORD-10001 | 2024-01-05 | Oncology | Avastin 400mg | 12 | 4,850.00 | North America | Hospital |
| ORD-10004 | 2024-01-25 | Diagnostics | cobas 6800 | 3 | 85,000.00 | North America | Direct |
| ORD-10010 | 2024-03-15 | Neuroscience | Evrysdi 60mg | 18 | 2,300.00 | North America | Pharmacy |

**Steps:**

1. Navigate to **"⏰ Time Travel"**
2. Upload `demo_data/pharma_orders.csv`
3. Configure the schema (auto-inferred)
4. Set parameters:
   - **Date range:** 2024-01-01 → 2024-12-31
   - **Frequency:** Monthly
   - **Base records/period:** 50
   - **Trend:** +5% (5% month-over-month growth)
5. Add **volume spikes:**
   - November 15, 2024 → Multiplier: **3×** (holiday surge)
   - December 15, 2024 → Multiplier: **2×** (year-end)
6. Click **"Preview Volume"** — see the bar chart showing ramping volume with spike in Nov/Dec
7. Click **Generate**

**What to highlight:**
- **Compound growth formula:** `base × (1 + 0.05)^month` — month 12 has ~80% more records than month 1
- **Calendar spike injection** — November and December show the expected 3×/2× volume multipliers
- **Auto-partitioning** — output is split by `_period` column for easy Spark/BigQuery ingestion
- **Volume preview** — leadership can see the expected data shape before generation starts

**Talking point:** *"This feature is designed for testing ETL pipelines, capacity planning, and seasonal business intelligence. It generates realistic load patterns that mirror real-world ordering behavior."*

---

### Demo 5: Privacy Audit — Employee Data

**Story:** *"Before sharing synthetic employee data with an external vendor, compliance needs mathematical proof that no real employee can be re-identified."*

**Files:**
- `demo_data/privacy_real_employees.csv` (20 real employees)
- `demo_data/privacy_synthetic_employees.csv` (20 synthetic employees)

**Sample comparison:**

| | Real (E-1001) | Synthetic (E-5001) |
|---|---|---|
| **Department** | Engineering | Engineering |
| **Job Title** | Senior Developer | Senior Developer |
| **Salary** | $125,000 | $128,500 |
| **Years XP** | 8 | 9 |
| **City** | San Francisco | San Francisco |

*Similar structure — but the synthetic record has slightly different values. Can we quantify the difference?*

**Steps:**

1. Navigate to **"🛡️ Privacy Audit"**
2. Upload `demo_data/privacy_real_employees.csv` in the **"Real Data"** panel
3. Upload `demo_data/privacy_synthetic_employees.csv` in the **"Synthetic Data"** panel
4. Click **"Run Privacy Audit"**

**What to see:**

- **Risk Badge:** 🟢 Low / 🟡 Medium / 🔴 High — based on DCR thresholds
- **Metric Cards:**
  - **Min DCR** — closest distance between any real-synthetic pair (should be > 0.01)
  - **Mean DCR** — average distance across all pairs
  - **Median DCR** — typical distance
  - **% Exact Matches** — should be 0% for safe data
- **Histogram** — distribution of distances (should be bell-shaped, shifted right)

**Risk thresholds (configurable):**
- 🔴 **High Risk:** >5% exact matches OR min DCR < 0.005
- 🟡 **Medium Risk:** >1% exact matches OR min DCR < 0.02
- 🟢 **Low Risk:** Everything else

**Talking point:** *"This is our compliance killer feature. Before any synthetic dataset leaves the organization, we compute the mathematical distance from every synthetic record to every real record. If any synthetic record is too close to a real one — we flag it. This is the same DCR methodology used in published privacy research."*

---

### Demo 6: Data Quality Dashboard

**Story:** *"After generating synthetic data, the data science team wants to verify it statistically resembles the original dataset."*

**Steps:**

1. Navigate to **"📈 Data Quality"**
2. Upload the original `demo_data/patients.csv` as the **reference dataset**
3. Upload (or use) the synthetic patients generated in Demo 1
4. View the **Quality Report:**

**Metrics displayed:**

| Metric | What It Measures | Target |
|---|---|---|
| **Completeness** | % of non-null values across all columns | > 95% |
| **Uniqueness** | Average unique values per column | Varies by column |
| **Schema Match** | % of columns matching expected data types | 100% |
| **Distribution Fidelity** | How closely synthetic data distributions mirror the original | > 70% |
| **Overall Score** | Weighted composite (25% completeness, 25% schema, 30% distribution, 20% uniqueness) | > 80% |

**What to highlight:**
- **Per-column detail table** — see completeness, unique count, and dtype for every column
- **Distribution comparison charts** — overlapping histograms for numeric columns (age, billing_amount)
- **Warnings** — automatically flagged if completeness < 95%, distribution fidelity < 70%, or columns have only 1 unique value

**Talking point:** *"This closes the loop. We don't just generate data and hope for the best — we measure synthetic data quality with the same rigor as a statistical test. If the distributions diverge, we know immediately."*

---

### Demo 7: Schema Library & History

**Story:** *"Teams across the organization need to reuse the same schemas. And compliance needs a full audit trail of every dataset ever generated."*

#### Schema Library (📚)

1. Navigate to **"📚 Schema Library"**
2. **Save a schema:** After generating data on the Single Table page, come here to save the schema with a name (e.g., "Clinical Patient v2"), description, and tags
3. **Browse saved schemas** — search by name or tag
4. **Import/Export** — download as JSON for version control, or import from a colleague's export
5. **Use in Generator** — one-click loads a saved schema into the Single Table page

#### Generation History (📜)

1. Navigate to **"📜 History"**
2. See every generation run logged:
   - **Summary metrics:** Total runs, total records, success rate, error count
   - **Per-run details:** Schema, engine (Faker or LLM), elapsed time, record count, output path
   - **Filter** by feature type (Single / Multi-Table / Time Travel)
   - **Error tracking** — failed runs show full error messages

**Talking point:** *"Full traceability. Every dataset generated is logged with the exact schema, settings, and engine used. For GxP compliance, you can answer: 'Who generated what, when, with which configuration?'"*

---

## REST API for Programmatic Access

ForgeFlow AI includes a full **FastAPI REST API** running on port 8100 for CI/CD pipelines, Jupyter notebooks, and automated testing workflows.

### Key Endpoints

```bash
# Health check
curl http://localhost:8100/health

# Generate 1,000 records (synchronous)
curl -X POST http://localhost:8100/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "schema": {"patient_id": "String", "age": "Int64", "diagnosis": "String"},
    "count": 1000,
    "seed": 42
  }'

# Async generation (for large datasets)
curl -X POST http://localhost:8100/api/v1/generate/async \
  -d '{"schema": {"id": "Int64", "name": "String"}, "count": 1000000}'
# Returns: {"job_id": "abc123", "status": "running"}

# Poll job status
curl http://localhost:8100/api/v1/jobs/abc123

# Run privacy audit via API
curl -X POST http://localhost:8100/api/v1/privacy/audit \
  -H "Content-Type: application/json" \
  -d '{"real_data": [...], "synthetic_data": [...]}'

# Schema CRUD
curl http://localhost:8100/api/v1/schemas          # List all
curl -X POST http://localhost:8100/api/v1/schemas   # Create
curl -X PUT http://localhost:8100/api/v1/schemas/id  # Update
curl -X DELETE http://localhost:8100/api/v1/schemas/id # Delete

# Generation history
curl http://localhost:8100/api/v1/history?feature=single&limit=50
```

### Use Case Examples

| Scenario | API Call |
|----------|----------|
| **CI pipeline** needs fresh test data before each test run | `POST /api/v1/generate` with `seed=42` for deterministic output |
| **Jupyter notebook** needs synthetic data for model training | `POST /api/v1/generate` returns data as JSON directly into pandas |
| **Automated privacy check** after every data export | `POST /api/v1/privacy/audit` in a post-export hook |
| **Schema versioning** across teams | `GET/POST /api/v1/schemas` for centralized schema management |

---

## Enterprise Features

### Security & Compliance

| Feature | Detail |
|---------|--------|
| **Pharma Safe Mode** | Disables SSN, credit card, IBAN generation — one env toggle |
| **Local LLM** | Ollama runs on-device — no data sent to external APIs. Full air-gap compatible |
| **Input Validation** | All schemas, uploads, and parameters validated; control characters stripped |
| **Non-root Docker** | Application runs as unprivileged `forge` user in container |
| **Audit Trail** | SQLite-backed logging of every generation with full provenance |
| **CORS Configuration** | Configurable cross-origin settings for API access |

### Scalability

| Feature | Detail |
|---------|--------|
| **Batch Processing** | 500-record chunks with progress callbacks — handles millions |
| **Async API Jobs** | Background generation with polling for large datasets |
| **Hive Partitioning** | Output as partitioned Parquet for direct use in Spark, BigQuery, Athena |
| **S3 Direct Upload** | Stream to S3 without intermediate disk files — supports IAM roles and explicit credentials |
| **Seed Reproducibility** | `seed=42` produces identical output every time — deterministic CI/CD |

### Observability

| Feature | Detail |
|---------|--------|
| **Structured Logging** | JSON-formatted logs for production (ELK/Splunk-ready) |
| **System Status** | Sidebar shows Ollama availability, loaded models, session stats |
| **Generation Metrics** | Elapsed time, records/sec throughput, per-column profiling |
| **Quality Scoring** | Automated quality checks with configurable warning thresholds |

---

## Deployment

### Local Development (Fastest)

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama (macOS — uses Metal GPU)
brew install ollama && ollama serve
ollama pull llama3.2:3b

# Launch the UI
streamlit run app/main.py --server.port 8501

# Launch the API (optional)
uvicorn api.server:app --host 0.0.0.0 --port 8100
```

### Docker Compose (Production)

```bash
docker compose up -d
```

Launches 3 services:

| Service | Port | Purpose |
|---------|------|---------|
| **app** | 8501 | Streamlit UI |
| **api** | 8100 | FastAPI REST API |
| **ollama** | 11434 | Local LLM (auto-pulls llama3.2:3b) |

Resource limits: App & API at 2GB RAM / 2 CPUs. Ollama at 16GB RAM / 10 CPUs.

---

## Impact & Use Cases

### Pharma & Life Sciences

| Use Case | How ForgeFlow Solves It |
|----------|------------------------|
| **Clinical trial test data** | Multi-Table Hydra generates Sites → Subjects → Visits with FK integrity |
| **Privacy-safe data sharing** | Privacy Audit proves DCR compliance before any data leaves the org |
| **Regulatory submissions** | Audit trail provides GxP-compatible provenance for every generated dataset |
| **Drug supply forecasting** | Time Travel generates 12+ months of order data with realistic trends + spikes |

### Engineering & DevOps

| Use Case | How ForgeFlow Solves It |
|----------|------------------------|
| **CI/CD test fixtures** | REST API with `seed` parameter generates deterministic datasets per build |
| **Database load testing** | Generate millions of records with Hive partitioning → direct S3 upload |
| **Schema evolution testing** | Schema Library stores versioned schemas for regression testing |
| **Dashboard prototyping** | Generate realistic data in seconds for Tableau/PowerBI demos |

### Data Science & Analytics

| Use Case | How ForgeFlow Solves It |
|----------|------------------------|
| **Model training data** | LLM mode generates semantically coherent, constraint-respecting datasets |
| **Data quality benchmarking** | Quality Dashboard measures synthetic-vs-real distribution fidelity |
| **Cross-team data catalogs** | Schema Library + History provides discoverability and governance |

---

## Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **UI** | Streamlit 1.36 | Rapid prototyping, interactive widgets, beautiful dashboards |
| **API** | FastAPI 0.111 + Uvicorn | Async-ready, auto-docs (OpenAPI/Swagger), high performance |
| **Data** | Polars 1.1 | 10× faster than pandas, zero-copy, native Parquet support |
| **Synthetic Data** | Faker 26.0 | 200+ realistic providers (names, addresses, medical, financial) |
| **LLM** | Ollama + llama3.2:3b | Local, GPU-accelerated, air-gap compatible — no cloud dependency |
| **Privacy** | SciPy + NumPy | DCR computation via optimized Euclidean distance matrices |
| **Storage** | SQLite + S3 (boto3) | Zero-config audit trail + enterprise cloud storage |
| **Container** | Docker Compose | One-command deployment with health checks and resource limits |

---

## Quick Reference — Demo Files Cheat Sheet

| Demo | Navigate To | Upload | What To Show |
|------|------------|--------|--------------|
| **Single Table** | 📊 Single Table | `patients.csv` | Schema inference → generate 10K → column profiler → download |
| **LLM Smart Mode** | 📊 Single Table | `patients.csv` | Toggle LLM → add field descriptions → generate 50 → show semantic coherence |
| **Multi-Table** | 🔗 Multi-Table | `clinical_sites.csv` + `clinical_subjects.csv` + `clinical_visits.csv` | Define 2 FKs → Mermaid diagram → generate → verify FK integrity |
| **Time Travel** | ⏰ Time Travel | `pharma_orders.csv` | Monthly, +5% trend, Nov 3× spike → preview volume chart → generate |
| **Privacy Audit** | 🛡️ Privacy Audit | `privacy_real_employees.csv` + `privacy_synthetic_employees.csv` | Run audit → show DCR metrics → risk badge → histogram |
| **Quality** | 📈 Data Quality | Original `patients.csv` + synthetic from Demo 1 | Quality score → distribution charts → per-column details |
| **Schema Library** | 📚 Schema Library | *(use saved schemas)* | Save → browse → search → export JSON → import |
| **History** | 📜 History | *(automatic)* | Filter by feature → view run details → compliance audit log |

---

> **ForgeFlow AI** — *Generate realistic data. Guarantee privacy. Ship faster.*
