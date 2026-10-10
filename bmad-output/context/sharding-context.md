# Sharding Context — Lesson 17 Restructuring

## Track & Sizing
- **Track:** Quick Flow (tech-spec only, no separate PRD/architecture)
- **Sizing rule:** Each story = one dev-day (~2-8h). Split if larger. NO story points.
- **Source document:** `bmad-output/tech-spec.md`
- **Epic map:** `bmad-output/epics.md`
- **Decision log:** `bmad-output/decision-log.md`
- **Story template:** Use the BMAD story context object format (see below)

## Critical Decision: No Textual TUI
**Textual has been dropped.** Exercise 3 is a CLI pipeline, not a TUI app. The instructor demos via Angular (out of scope for stories). Remove all Textual references from dependencies, stories, and acceptance criteria. See decision-log.md entry "Drop Textual TUI" (2026-10-08).

## Project Context
- **What:** Restructuring Lesson 17 ("Using LLMs with FHIR") of the Vanderbilt FHIR Training Course
- **Who:** Healthcare informatics students with FHIR knowledge, no prior AI/ML background
- **Series:** Lesson 17 is #1 in a 4-part AI+FHIR series (followed by Agentic AI Parts One/Two and RAG Lab)
- **Non-Goals:** Agentic AI, MCP, RAG, vector stores, terminology normalization (all covered in companion lessons)

## File Structure
All code lives in `FHIRAI/` subdirectory. Key paths:
- `FHIRAI/requirements.txt` — dependency pins
- `FHIRAI/config.py` — NEW: central configuration loader
- `FHIRAI/.env.example` — NEW: configuration template
- `FHIRAI/fhir_client.py` — NEW: refactored from getSearchPatients.py
- `FHIRAI/exercise1_llm_chat.py` — NEW: Exercise 1
- `FHIRAI/exercise2_fhir_explore.py` — NEW: Exercise 2
- `FHIRAI/exercise3_summarizer.py` — NEW: Exercise 3 (CLI pipeline, NOT TUI)
- `FHIRAI/exercise4_prompt_compare.py` — NEW: Exercise 4
- `FHIRAI/getSearchPatients.py` — EXISTING: to be refactored into fhir_client.py
- `FHIRAI/fhirapp.py` — EXISTING: concepts absorbed into exercise3
- `FHIRAI/llamachat.py` — EXISTING: replaced by exercise1
- `FHIRAI/100Set/` — EXISTING: Synthea bundles, keep as-is
- `FHIRAI/archive/` — NEW: holds archived undocumented files
- `FHIRAI/Readme.md` — EXISTING: to be rewritten

## Dependency Table (post-Textual-removal)
| Package | Version | Notes |
|---------|---------|-------|
| `fhir.resources` | `>=7.0.0` | R4B sub-package; `model_validate()` |
| `fhirpathpy` | `>=2.2.0` | Stable |
| `ollama` | `>=0.4.0` | Primary LLM |
| `lmstudio` | `>=1.5.0` | Secondary LLM |
| `requests` | `>=2.28.0,<3.0.0` | HTTP client |
| `tiktoken` | `>=0.5.0` | Token counting |
| `python-dotenv` | `>=1.0.0` | .env loading |

**Removed:** `textual` (dropped), `sentence-transformers`, `einops`, `intersystems-irispython` (moved to RAG Lab)

## Epic Summary
- **Epic 1 (Infrastructure):** Stories 1.1-1.3. Sequential. Must complete before Epic 2.
- **Epic 2 (Exercises):** Stories 2.1-2.4. Depends on Epic 1. Internal partial parallelism.
- **Epic 3 (Content & Cleanup):** Stories 3.1-3.5. Slides/safety can parallelize with Epic 2; Readme waits for Epic 2; cleanup runs last.

## Story Compilation Rules
1. Every story is a self-contained context object (~8K tokens)
2. Acceptance Criteria: numbered, testable, 3-7 items. LOCKED.
3. Every Task maps to an AC via `(AC: #N)`
4. Dev Notes cite sources: `[Source: tech-spec.md#FR-XXX]` or `[Inference]`
5. Testing: strategy only, never run tests
6. Owned File/Module Scope: explicit paths, disjoint across parallel stories
7. Dev Agent Record: leave EMPTY
8. Status: `backlog` initially
