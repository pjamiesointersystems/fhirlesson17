# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lesson 17 of the Vanderbilt FHIR Training course. Demonstrates using local LLMs (via LM Studio) to summarize FHIR patient data. The main application is a Python TUI that retrieves a patient's complete FHIR bundle, chunks it, and progressively summarizes it using a local LLM.

## Architecture

All application code lives in `FHIRAI/`:

- **`fhirapp.py`** — Main entry point. A Textual TUI app (`FHIRApp`) that:
  1. Fetches a patient's `$everything` bundle from a local FHIR server
  2. Chunks the JSON text into 1500-token segments (using tiktoken)
  3. Sends each chunk to a local LLM (via `lmstudio` SDK) for summarization
  4. Merges batch summaries in two rounds (intermediate → final) using a hierarchical merge strategy
  5. Displays progress and results in a Textual UI with Markdown widgets

- **`getSearchPatients.py`** — FHIR client module. Contains functions for querying an InterSystems IRIS for Health FHIR R4 server:
  - `get_patient_from_server()` — fetch single Patient by ID
  - `get_patients_from_server()` — fetch all Patients
  - `search_patients_get_ids()` — search Patients and extract IDs via FHIRPath
  - `get_observations_for_patient()` — fetch Observations for a patient
  - `get_everything_for_patient()` — fetch Patient/$everything bundle (used by fhirapp.py)
  - Uses `fhirpathpy` for FHIRPath evaluation and `fhir.resources` for FHIR model parsing

- **`llamachat.py`** — Minimal LM Studio test script

- **`100Set/`** — 115 Synthea-generated FHIR Bundle JSON files (transaction bundles) for sample patient data

## FHIR Server Configuration

The app connects to a local InterSystems IRIS for Health FHIR R4 endpoint:
- Base URL: `http://127.0.0.1:8080/csp/healthshare/demo/fhir/r4`
- Auth: Basic (`_System` / `ISCDEMO`)

## LLM Configuration

Uses LM Studio running locally. The model is configured in `fhirapp.py` constructor (currently `llama-3.2-3b-instruct`). LM Studio must be running with the model loaded before launching the app.

## Running

```bash
cd FHIRAI
# Activate the virtual environment
source .venv/bin/activate

# Run the main TUI app (defaults to patient FHIR ID "2")
python fhirapp.py

# Run the FHIR client standalone test
python getSearchPatients.py

# Test LM Studio connectivity
python llamachat.py
```

## Dependencies

Managed via `FHIRAI/requirements.txt`. Key packages:
- `fhir.resources==6.1.0` (requires pydantic <2.0)
- `fhirpathpy` — FHIRPath expression evaluation
- `textual==0.44.1` — TUI framework
- `lmstudio` — LM Studio Python SDK
- `tiktoken` — Token-aware text chunking
- `requests` — HTTP client for FHIR server
- `sentence-transformers`, `intersystems-irispython` — additional dependencies

Python 3.11 virtual environment at `FHIRAI/.venv/`.
