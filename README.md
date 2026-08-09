# Synthetic Data Forge

**Generate realistic, privacy-safe synthetic data — from a single CSV to full FHIR R4 clinical trial datasets.**

[![CI](https://github.com/anupamk36/Synthetic-Data-Forge/actions/workflows/ci.yml/badge.svg)](https://github.com/anupamk36/Synthetic-Data-Forge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

Synthetic Data Forge is an open-source platform for generating synthetic datasets that preserve the statistical shape and correlations of real data — without exposing any of it. It covers general-purpose tabular data as well as deep healthcare/life-sciences support (FHIR R4, DICOM, HL7v2, CDISC SDTM), plus built-in privacy and quality scoring so you can prove the output is safe to share.

---

## Why

Teams that need realistic test data are usually stuck choosing between two bad options: use real production data (and inherit its compliance risk), or hand-roll fixtures that don't reflect real-world distributions, correlations, or edge cases. Synthetic Data Forge generates data that statistically resembles your source data — including cross-column correlations — while measuring and reporting how private the result actually is.

## Features

| Area | What it does |
|---|---|
| **Single-table generation** | Upload a CSV, get an auto-inferred schema, generate any number of rows via Faker + a Gaussian copula that preserves cross-column correlations |
| **Multi-table relational data (Hydra)** | Define foreign-key relationships across tables; a DAG-based engine generates parents before children and guarantees every FK resolves |
| **LLM semantic validation** | Optional pass through Claude, OpenAI, Gemini, or a local Ollama model to catch logically inconsistent rows (e.g. age vs. job seniority) |
| **Privacy Scorecard** | Distance-to-Closest-Record (DCR), k-anonymity, l-diversity, and empirical epsilon estimation, with a downloadable compliance report |
| **Data Quality Grading** | KS tests, chi-squared tests, and correlation-preservation checks against a real sample, rolled up into an A–F realism grade |
| **FHIR R4 generation** | 11 resource types (Patient, Encounter, Condition, Observation, MedicationRequest, …) with cross-resource referential integrity and real terminology (ICD-10, LOINC, SNOMED, RxNorm) |
| **Clinical trial simulation** | Phase I–III trial data with CDISC SDTM export, randomization, dropout curves, and adverse events |
| **DICOM metadata** | Study/Series/Instance hierarchies for CT, MR, US, DX, MG, PT with modality-specific templates |
| **HL7v2 conversion** | Convert generated FHIR bundles into ADT/ORU messages for legacy system testing |
| **AI Test Intelligence** | LLM-driven generation of categorized edge-case test data (boundary, invalid, security, unicode, nulls) with coverage scoring |
| **Zero-copy cloud sinks** | Push generated data straight to S3-compatible storage without an intermediate disk write |
| **Audit trail** | Every generation run, schema save, or privacy audit is logged to SQLite — who generated what, when, with which settings |

## Architecture

```
┌─────────────────────────────────────────────┐
│        Next.js Frontend (:3000)              │
│   Generate → Analyze → Manage → Medical      │
└───────────────────┬───────────────────────────┘
                     │ HTTP /api/v1/*
                     ▼
┌─────────────────────────────────────────────┐
│           FastAPI Backend (:8100)             │
├───────────┬──────────────┬────────────────────┤
│ Medical   │ Generation   │ Privacy / Quality   │
│ ├─ FHIR   │ ├─ Single    │ ├─ DCR              │
│ ├─ Trials │ ├─ Relational│ ├─ k-Anonymity      │
│ ├─ Imaging│ └─ LLM Logic │ └─ Compliance       │
│ └─ HL7v2  │              │                     │
└───────────┴──────────────┴────────────────────┘
                     │
            ┌────────┴────────┐
            ▼                 ▼
   LLM Provider(s)      Core Engines (Python)
   (Ollama / Claude /   (Polars, SciPy, Faker)
    OpenAI / Gemini)
```

**Backend:** Python, FastAPI, Polars, SciPy, Faker
**Frontend:** Next.js, TypeScript, Tailwind, shadcn/ui
**LLM providers:** Ollama (local, default), Anthropic Claude, OpenAI, Google Gemini — pick any at runtime

## Quickstart

### Docker (recommended)

```bash
git clone https://github.com/anupamk36/Synthetic-Data-Forge.git
cd Synthetic-Data-Forge
cp .env.example .env   # add API keys if you want cloud LLM providers
docker compose up -d
```

| Service | URL | Purpose |
|---|---|---|
| Frontend | http://localhost:3000 | Web dashboard |
| API | http://localhost:8100 | REST API |
| Ollama | http://localhost:11434 | Local LLM (no API key needed) |

### Manual setup

```bash
# Backend
pip install -r requirements.txt
uvicorn api.server:app --host 0.0.0.0 --port 8100

# Frontend (separate terminal)
cd frontend
npm install
npm run dev

# Local LLM (optional, separate terminal)
ollama serve
ollama pull llama3.2:3b
```

## API examples

```bash
# Single-table generation
curl -X POST http://localhost:8100/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"schema": {"name": "String", "age": "Int64"}, "count": 1000}'

# FHIR R4 bundle generation
curl -X POST http://localhost:8100/api/v1/medical/fhir/generate \
  -H "Content-Type: application/json" \
  -d '{"resource_types": ["Patient", "Encounter"], "patient_count": 100}'

# Clinical trial simulation (SDTM export)
curl -X POST http://localhost:8100/api/v1/medical/trials/generate \
  -H "Content-Type: application/json" \
  -d '{"profile": "oncology_phase2", "subjects_per_arm": 50}'

# Privacy compliance report
curl -X POST http://localhost:8100/api/v1/privacy/report \
  -H "Content-Type: application/json" \
  -d '{"real_data": [...], "synthetic_data": [...], "quasi_identifiers": ["age", "zip"]}'
```

See [DEMO.md](DEMO.md) for a full walkthrough of every feature, including UI screenshots and demo datasets in [demo_data/](demo_data/).

## Configuration

All configuration is via environment variables — see [.env.example](.env.example) for the full list. Key ones:

| Variable | Purpose |
|---|---|
| `FORGE_OLLAMA_URL` | Local Ollama endpoint (default: `http://ollama:11434`) |
| `FORGE_ANTHROPIC_API_KEY` / `FORGE_OPENAI_API_KEY` / `FORGE_GEMINI_API_KEY` | Enable a cloud LLM provider |
| `FORGE_PHARMA_SAFE_MODE` | Disables regulated-PII generators (SSN, credit card, etc.) |
| `FORGE_OUTPUT_ROOT` | Directory generated files are written to |
| `FORGE_DCR_*` | Thresholds for the privacy scorecard's risk classification |

## Project structure

```
core/               # Generation, privacy, quality, relational, LLM engines
  medical/           # FHIR, DICOM, HL7v2, SDTM clinical modules
api/                # FastAPI routes
frontend/           # Next.js web app
tests/              # pytest suite
demo_data/          # Sample CSVs for trying every feature
```

## Development

```bash
pip install -r requirements-dev.txt

ruff check .              # lint
ruff format --check .     # format check
pytest --cov=core --cov=app --cov-fail-under=70   # tests
bandit -r core/ app/ -ll  # security scan
```

CI runs lint, tests, security scan, and a Docker build on every push — see [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Contributing

Contributions are welcome. Please open an issue to discuss significant changes before submitting a PR. Make sure `ruff check`, `ruff format --check`, and `pytest` all pass before opening one.

## License

[MIT](LICENSE)
