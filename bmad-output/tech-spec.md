# Technical Specification: Lesson 17 — Using LLMs with FHIR (Restructuring)

**Date:** 2026-10-07
**Author:** Patrick W. Jamieson, M.D. (assisted by BMAD planning)
**Version:** 1.0
**Track:** Quick Flow (1-15 stories)
**Status:** Draft

> **Quick Flow track** — this document replaces a separate PRD and architecture file for
> small-scope work. If scope grows beyond ~15 stories, migrate to the BMad Method track
> (bmad-prd + bmad-architecture) before continuing.

---

## Related Documents

- SPEC kernel: `bmad-output/SPEC.md`
- Decision log: `bmad-output/decision-log.md`
- Research report: `bmad-output/research-report.md`
- Existing codebase assessment: `bmad-output/research-scratch/existing-assessment.md`
- Companion lessons: `Using_Agentic_AI_with_FHIR_Part_One.pdf`, `Using_Agentic_AI_with_FHIR_Part_Two.pdf`, `Lab-_LLMs_and_FHIR_Application_Development.pptx`

---

## Problem & Solution

### Problem Statement

Lesson 17 is the entry point for a four-part AI+FHIR series, but it currently fails to teach the foundational AI/LLM concepts that the downstream agentic AI lectures (Parts One and Two) and RAG lab assume students already know. The existing code (May 2025) has outdated dependencies that prevent installation on modern Python environments, code quality issues (duplicate functions, bugs), and no structured progression from concepts to hands-on practice. Students arrive at the agentic AI and RAG materials without understanding tokens, embeddings, context windows, prompt engineering, or why FHIR's structured data matters for AI consumption.

### Proposed Solution

Restructure Lesson 17 as a three-phase foundational lesson:

1. **Phase 1 — AI Foundations** (slides/lecture): Core LLM concepts taught through healthcare analogies, model selection guidance, local vs. cloud deployment trade-offs, and a brief safety/governance framing.

2. **Phase 2 — Hands-On Exercises** (code): Progressive exercises from LLM "hello world" through FHIR bundle retrieval to a complete patient summarization pipeline with prompt engineering comparison.

3. **Phase 3 — Bridge to Series** (slides/discussion): How the concepts connect to agentic AI (Parts One/Two) and RAG (Lab), what's coming next, and safety/governance discussion.

Update all dependencies, fix existing bugs, externalize configuration, and reorganize files into a clean structure.

### Goals

- Provide the conceptual foundation that makes the rest of the AI+FHIR series coherent
- Give students their first working AI+FHIR application (patient summarizer)
- Update all code to current library versions (Pydantic v2, Textual 8.x, fhir.resources 7.x)
- Teach prompt engineering as a practical skill, not just theory

---

## Scope

### In Scope

- New/revised slide content for AI foundations (Phase 1) and series bridge (Phase 3)
- Updated Readme.md with lesson structure, learning objectives, prerequisites, and setup instructions
- Dependency updates (requirements.txt) and all associated code migrations
- Four progressive exercises with clean, documented code
- Configuration externalization (.env or config.toml)
- File cleanup: remove/archive undocumented files, fix bugs in retained files

### Out of Scope

- Agentic AI patterns, MCP servers/clients, tool-use (covered in Parts One/Two)
- RAG pipelines, vector stores, HNSW indexes, embeddings for retrieval (covered in RAG Lab)
- Terminology normalization and code resolution (covered in Parts One/Two)
- Graph RAG (covered in RAG Lab)
- Fine-tuning or training custom models
- Slide visual design (content only; visual formatting is instructor's preference)

---

## Requirements

### Functional Requirements

#### FR-001: AI Foundations Slide Content [MUST]

Create slide/lecture content covering core LLM concepts with healthcare analogies. Content must be suitable for students with FHIR knowledge but no prior AI/ML background.

**Topics to cover:**
- What is an LLM? (architecture at a conceptual level — no math)
- Tokens and tokenization (analogy: pharmacist reading prescriptions — each word/subword is a unit)
- Context windows (analogy: clinician working memory — limited chart info at a time)
- Embeddings (brief conceptual introduction — "meaning as numbers"; detailed coverage in RAG Lab)
- Temperature and sampling (analogy: confident specialist vs. creative brainstormer)
- Prompts and system roles (analogy: clinical protocols guiding behavior)
- Hallucination (analogy: clinical confabulation — confident but wrong)
- Local vs. cloud models (privacy, latency, capability trade-offs; HIPAA implications)
- Model selection basics (size vs. capability, when local is sufficient vs. when cloud is needed)

**Acceptance Criteria:**
- Content covers all listed topics with at least one healthcare analogy per concept
- Content explicitly does NOT cover agentic AI, MCP, or RAG in depth (references forward to Parts One/Two and Lab)
- Content is reviewable as a markdown document or slide outline that the instructor can format

---

#### FR-002: Exercise 1 — Local LLM Setup and First Chat [MUST]

A minimal script that connects to a local LLM and sends a healthcare-related prompt. Serves as the "hello world" that confirms the student's environment works.

**Implementation:**
- Primary: Ollama with `ollama` Python package (simpler install, no GUI required)
- Secondary: LM Studio with `lmstudio` SDK (existing approach, kept as alternative)
- Script sends a simple clinical prompt (e.g., "Explain what an A1c test measures") and prints the response
- Includes error handling for "LLM not running" and "model not loaded" cases
- Configuration (model name, endpoint) read from `.env` or `config.toml`

**Acceptance Criteria:**
- Script runs successfully with Ollama and prints LLM response
- Script includes commented alternative for LM Studio
- Clear error message if LLM is not running
- Model name is configurable, not hardcoded

---

#### FR-003: Exercise 2 — FHIR Bundle Retrieval and Exploration [MUST]

A script that retrieves a patient's complete FHIR record and explores its structure, demonstrating why raw FHIR JSON needs processing before LLM consumption.

**Implementation:**
- Refactored `getSearchPatients.py` module (renamed to `fhir_client.py`) with:
  - Configuration externalized (FHIR server URL, credentials from `.env`)
  - Duplicate `get_patient_from_server()` removed
  - `parse_obj()` replaced with `model_validate()` throughout
  - Imports updated for fhir.resources 7.x R4B sub-package
- Script retrieves `Patient/$everything` for a configurable patient ID
- Prints: total bundle size in characters, resource count by type, token count estimate
- Demonstrates the "too big for context window" problem concretely with numbers
- Shows a naive "send raw JSON to LLM" attempt and its poor result, motivating the need for processing

**Acceptance Criteria:**
- Script retrieves bundle from IRIS FHIR server successfully
- Output shows concrete numbers (e.g., "Bundle: 45,000 characters, ~12,000 tokens, 47 resources")
- Pydantic v2 `model_validate()` used throughout (no `parse_obj()`)
- fhir.resources 7.x R4B imports compile and work correctly
- No duplicate function definitions

---

#### FR-004: Exercise 3 — Patient Summarization Pipeline [MUST]

The core exercise: a complete pipeline from FHIR bundle to clinical narrative summary via local LLM. This is the enhanced version of the existing `fhirapp.py`.

**Implementation:**
- Textual TUI app (updated for Textual 8.x API) that:
  1. Fetches `Patient/$everything` via `fhir_client.py`
  2. Groups resources by type (Patient, Condition, Observation, MedicationRequest, etc.)
  3. Extracts clinically relevant fields per resource type (structured feature extraction, as previewed in Part One slides)
  4. Produces a compact intermediate representation (not raw JSON chunks)
  5. Sends structured data to local LLM with a clinical summarization prompt
  6. Displays the progressive output in the TUI
- Dynamic chunk handling (no fixed 10-chunk limit)
- Tokenization: use tiktoken but explicitly note in comments and slides that it's an approximation for local models; explain why
- System prompt uses structured output format (Patient Summary with sections: Demographics, Active Conditions, Medications, Recent Encounters, Recent Observations, Allergies, Notable Findings)

**Acceptance Criteria:**
- TUI app launches and displays progressive summarization
- Chunk count is dynamic based on actual content
- Resource-type grouping produces better summaries than raw JSON chunking
- Works with Textual 8.x (updated API calls)
- Tokenization approximation is documented in code comments

---

#### FR-005: Exercise 4 — Prompt Engineering Comparison [MUST]

A script that demonstrates how different prompting strategies produce different summarization results, teaching prompt engineering as a practical skill.

**Implementation:**
- Takes the same FHIR patient data and runs it through 3+ prompt variations:
  1. **Minimal prompt**: "Summarize this patient data" (baseline)
  2. **Structured system prompt**: Clinical role, output format specification, "do not invent" instruction
  3. **Few-shot prompt**: Includes an example input/output pair before the actual data
- Side-by-side or sequential output comparison
- Discussion prompts in comments: "Notice how prompt #2 produces sections while #1 is a wall of text"
- Optional: temperature comparison (0.1 vs. 0.7 vs. 1.0) on the same prompt

**Acceptance Criteria:**
- At least 3 distinct prompt strategies are demonstrated
- Output differences are visible and meaningful
- Code includes inline comments explaining why each strategy produces different results
- Students can modify prompts and re-run to experiment

---

#### FR-006: Updated Readme.md [MUST]

Complete rewrite of the lesson Readme to serve as the lesson guide.

**Structure:**
1. Lesson overview and position in the AI+FHIR series
2. Learning objectives (mapped to SPEC capabilities)
3. Prerequisites (prior FHIR lessons, Python, local FHIR server running)
4. Environment setup instructions (Ollama install, model download, Python venv, dependencies)
5. Phase 1 summary (reference to slides)
6. Exercise walkthroughs (Exercises 1-4 with expected outputs)
7. Phase 3 summary (what comes next in the series)
8. Safety and governance discussion points
9. Further reading / resources

**Acceptance Criteria:**
- All exercises are documented with setup steps and expected output
- Series context is clearly explained (what this lesson covers vs. what's in Parts One/Two and Lab)
- Setup instructions work on macOS (primary) and include notes for Windows/Linux

---

#### FR-007: Configuration Externalization [MUST]

Move all hardcoded values to a central configuration.

**Implementation:**
- `.env` file (gitignored) with:
  - `FHIR_BASE_URL=http://127.0.0.1:8080/csp/healthshare/demo/fhir/r4`
  - `FHIR_USERNAME=_System`
  - `FHIR_PASSWORD=ISCDEMO`
  - `LLM_MODEL=llama3.2:3b`
  - `LLM_PROVIDER=ollama` (or `lmstudio`)
- `.env.example` file (committed) with placeholder values
- `config.py` module that loads from `.env` with sensible defaults
- All exercise scripts import from `config.py`

**Acceptance Criteria:**
- No hardcoded URLs, credentials, or model names in exercise scripts
- `.env.example` is committed; `.env` is in `.gitignore`
- Scripts work with defaults if `.env` is missing (with warning)

---

#### FR-008: Dependency Update [MUST]

Update `requirements.txt` to current, compatible versions.

**Target versions:**
| Package | Old | New | Migration Notes |
|---------|-----|-----|----------------|
| `fhir.resources` | `==6.1.0` | `>=7.0.0` | R4B sub-package imports; `model_validate()` |
| `pydantic` | `<2.0` | (remove pin) | Pulled in by fhir.resources 7.x |
| `textual` | `==0.44.1` | `>=8.0.0` | API changes in compose/widget methods |
| `fhirpathpy` | unpinned | `>=2.2.0` | No changes needed |
| `lmstudio` | unpinned | `>=1.5.0` | No changes needed |
| `requests` | `>=2.20.0,<3.0.0` | `>=2.28.0,<3.0.0` | No changes needed |
| `tiktoken` | unpinned | keep unpinned | No changes needed |
| `ollama` | (new) | `>=0.4.0` | New primary LLM interface |
| `python-dotenv` | (new) | `>=1.0.0` | For .env loading |

**Remove from lesson scope** (move to RAG Lab dependencies):
- `sentence-transformers` (used in RAG, not in this lesson)
- `einops` (dependency of sentence-transformers)
- `intersystems-irispython` (used in RAG Lab's vector store exercises)

**Acceptance Criteria:**
- `pip install -r requirements.txt` succeeds on Python 3.11+ without version conflicts
- No `pydantic<2.0` constraint
- All exercise scripts import successfully with new versions

---

#### FR-009: File Cleanup and Reorganization [SHOULD]

Reorganize FHIRAI/ directory for clarity.

**File disposition:**

| Current File | Action | Rationale |
|-------------|--------|-----------|
| `fhirapp.py` | **Rewrite** → Exercise 3 TUI | Core exercise, needs Textual 8.x + structured extraction |
| `getSearchPatients.py` | **Refactor** → `fhir_client.py` | Fix duplicate function, update imports, externalize config |
| `llamachat.py` | **Replace** → Exercise 1 | Replace with Ollama-primary version |
| `requirements.txt` | **Update** | See FR-008 |
| `Readme.md` | **Rewrite** | See FR-006 |
| `FHIRSummaryApp_by_resource_type.py` | **Archive** → `archive/` | Undocumented; concepts absorbed into Exercise 3 |
| `FHIRSummaryRevised.py` | **Archive** → `archive/` | Undocumented; concepts absorbed into Exercise 3 |
| `OpenAIFHIRSummary.py` | **Archive** → `archive/` | Cloud LLM variant; could be revisited but not in scope |
| `fhirvectorflattened.py` | **Move** → RAG Lab | This is RAG Lab content, not Lesson 17 |
| `simplevector.py` | **Move** → RAG Lab | This is RAG Lab content |
| `testrag.py` | **Move** → RAG Lab | This is RAG Lab content |
| `fhir_rag_chat.py` | **Move** → RAG Lab | This is RAG Lab content |
| `testdriver.py` | **Delete** | Has bugs, duplicates testrag.py functionality |
| `testpydriver.py` | **Delete** | IRIS connection test, not educational |
| `100Set/` | **Keep** | Synthea bundles used by exercises |
| `.vscode/` | **Keep** | Editor settings |

**New files to create:**

| File | Purpose |
|------|---------|
| `config.py` | Central configuration loader |
| `.env.example` | Configuration template |
| `exercise1_llm_chat.py` | Exercise 1: LLM hello world |
| `exercise2_fhir_explore.py` | Exercise 2: FHIR bundle exploration |
| `exercise3_summarizer.py` | Exercise 3: Patient summarization TUI |
| `exercise4_prompt_compare.py` | Exercise 4: Prompt engineering comparison |
| `fhir_client.py` | Refactored FHIR client module |

**Acceptance Criteria:**
- FHIRAI/ directory has clear, numbered exercise files
- No undocumented files in the main directory
- Archive directory preserves original work
- RAG-related files identified for transfer to Lab materials

---

#### FR-010: Safety and Governance Discussion Content [SHOULD]

Create discussion content (slides or markdown) covering AI safety in healthcare as a conceptual frame for the series.

**Topics:**
- Hallucination risks in clinical contexts (with examples)
- Why local models matter for PHI (HIPAA considerations)
- Human-in-the-loop: AI as decision support, not decision maker
- The "golden rule": never trust LLM output without validation
- Brief mention of regulatory landscape (ONC HTI-1, FDA AI devices) — awareness level, not deep dive
- Forward reference to safety patterns in Part One (least privilege, structured validation, audit logging)

**Acceptance Criteria:**
- Content is reviewable as markdown or slide outline
- Each topic connects to a concrete FHIR+AI scenario
- Content frames safety as a series-long theme, not just a Lesson 17 topic

---

#### FR-011: Series Bridge Content [COULD]

Create a brief "what's next" section connecting Lesson 17 to the rest of the series.

**Content:**
- How the patient summarization pipeline becomes a subagent in Part One's architecture
- How structured feature extraction feeds into the RAG Lab's vector pipeline
- How prompt engineering applies to agent prompts in Parts One/Two
- Preview of MCP as the standardized way to connect LLMs to FHIR (without teaching it here)

**Acceptance Criteria:**
- Students understand where each Lesson 17 concept reappears in the series
- No content that duplicates what Parts One/Two or Lab teach

---

### Non-Functional Requirements

#### Performance

- Exercise scripts should return results within 60 seconds on a machine running a 3B-parameter local model (the summarization TUI may take longer due to multiple LLM calls, but individual calls should complete in <30 seconds)
- FHIR server queries should complete in <5 seconds on local IRIS

#### Security

- No real PHI in any exercise code or sample data (Synthea only)
- `.env` with credentials must be gitignored
- Slide content must explicitly address PHI risks of cloud LLM usage

#### Other

- All code must run on Python 3.11+ (course standard)
- Primary development/testing on macOS; should work on Windows/Linux with documented adjustments
- Code should follow the existing course style (functional, well-commented, educational clarity over production patterns)

---

## Technical Approach

### Technology Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| FHIR Server | InterSystems IRIS for Health | Course infrastructure, FHIR R4 |
| FHIR Client | `requests` + `fhir.resources` 7.x + `fhirpathpy` | Pydantic v2, R4B sub-package |
| LLM (primary) | Ollama + `ollama` Python package | Simpler setup than LM Studio |
| LLM (secondary) | LM Studio + `lmstudio` SDK | Kept as alternative for GUI preference |
| TUI Framework | Textual 8.x | For Exercise 3 summarization app |
| Tokenization | `tiktoken` | Approximation for token counting |
| Configuration | `python-dotenv` | .env-based config |
| Sample Data | Synthea FHIR R4 bundles | Existing 100Set/ directory |

### Architecture Overview

The lesson's code follows a simple pipeline architecture. There is no persistent data model or API surface — these are educational scripts.

```
[FHIR Server (IRIS)]
        |
        | REST / $everything
        v
[fhir_client.py] -- shared FHIR access module
        |
        v
[Exercise Scripts 1-4] -- each builds on concepts from previous
        |
        | Python SDK calls
        v
[Local LLM (Ollama/LM Studio)]
        |
        v
[Terminal Output / Textual TUI]
```

Configuration flows from `.env` → `config.py` → all scripts.

### Key Components

#### fhir_client.py (refactored from getSearchPatients.py)

**Purpose:** Shared FHIR REST client used by all exercises.

**Responsibilities:**
- Authenticate and query the IRIS FHIR R4 server
- Retrieve Patient, Observation, and $everything bundles
- Parse responses into fhir.resources objects (Pydantic v2)
- Extract resources via FHIRPath

**Key changes from existing code:**
- Remove duplicate `get_patient_from_server()` (lines 69-90 of current file)
- Replace all `Patient.parse_obj()` → `Patient.model_validate()`
- Update imports: `from fhir.resources.R4B.patient import Patient` (or equivalent for 7.x R4B)
- Read FHIR server URL and credentials from `config.py`
- Add `get_resource_counts(bundle_dict)` helper that returns a dict of resource_type → count

---

#### config.py (new)

**Purpose:** Central configuration for all exercise scripts.

**Responsibilities:**
- Load `.env` file via `python-dotenv`
- Expose typed config values: `FHIR_BASE_URL`, `FHIR_AUTH`, `LLM_MODEL`, `LLM_PROVIDER`
- Provide an `get_llm_client()` factory that returns either Ollama or LM Studio client based on `LLM_PROVIDER`

---

#### exercise3_summarizer.py (rewrite of fhirapp.py)

**Purpose:** The main TUI application — patient summarization pipeline.

**Responsibilities:**
- Fetch patient bundle via fhir_client
- Group resources by type (not raw JSON chunking)
- Extract structured clinical features per resource type
- Build compact intermediate representation
- Send to LLM with structured summarization prompt
- Display progressive results in Textual TUI

**Key changes from existing fhirapp.py:**
- Textual 8.x API (verify `compose()`, `ProgressBar`, `Markdown` widget APIs)
- Dynamic chunk count (remove hardcoded `[:10]` and fixed 10 widgets)
- Resource-type-aware extraction instead of blind JSON chunking
- LLM provider abstraction via config.py (Ollama primary, LM Studio secondary)
- Structured output prompt format (matches the format introduced in Part One slides)

---

### Data Model

No persistent data model. All data flows through in-memory Python objects:
- FHIR Bundle → dict (from `response.json()`)
- Resource lists → `fhir.resources` Pydantic models
- LLM input → formatted strings
- LLM output → string responses displayed in TUI

### API Design

No API surface. These are standalone scripts and a TUI application.

### Error Handling Strategy

- **FHIR server unreachable**: Clear error message with setup instructions reference
- **LLM not running**: Detect connection failure, print "Please start Ollama/LM Studio" with instructions
- **Model not loaded**: Catch model-not-found errors, suggest `ollama pull <model>`
- **Empty patient bundle**: Graceful message rather than crash
- **LLM timeout/failure**: Per-call try/except with fallback message in TUI

---

## Story List

| # | Epic | Story Title | Notes |
|---|------|-------------|-------|
| 1 | Infrastructure | Update dependencies and verify compatibility | requirements.txt, test imports |
| 2 | Infrastructure | Create config module and .env template | config.py, .env.example, .gitignore update |
| 3 | Infrastructure | Refactor FHIR client module | getSearchPatients.py → fhir_client.py, fix bugs, Pydantic v2 |
| 4 | Exercises | Build Exercise 1 — LLM hello world | Ollama + LM Studio, error handling |
| 5 | Exercises | Build Exercise 2 — FHIR bundle exploration | Retrieve, count, demonstrate size problem |
| 6 | Exercises | Build Exercise 3 — Patient summarization TUI | Textual 8.x rewrite with structured extraction |
| 7 | Exercises | Build Exercise 4 — Prompt engineering comparison | 3+ prompt strategies, side-by-side output |
| 8 | Content | Write AI foundations slide/lecture content | Healthcare analogies, concept explanations |
| 9 | Content | Write safety and governance discussion content | Hallucination, PHI, regulatory frame |
| 10 | Content | Rewrite Readme.md as lesson guide | Full lesson walkthrough, setup, series context |
| 11 | Cleanup | Archive/remove undocumented files, organize directory | File disposition per FR-009 |
| 12 | Content | Write series bridge content (what's next) | Connect to Parts One/Two and RAG Lab |

**Total stories:** 12 (Quick Flow ceiling: 15)

---

## Testing Strategy

### Unit Testing Focus

- `fhir_client.py`: Verify that each function correctly parses sample FHIR JSON (use a small Synthea bundle from 100Set/ as fixture). Key test: `model_validate()` works with fhir.resources 7.x for Patient, Observation, Bundle.
- `config.py`: Verify .env loading, defaults when .env missing, LLM provider factory.

### Integration / End-to-End Scenarios

- **Full pipeline test**: Run Exercise 3 against live IRIS FHIR server with a known patient ID, verify that a summary is produced without errors.
- **Dependency smoke test**: `pip install -r requirements.txt` in a clean venv, then `python -c "import fhir.resources; import textual; import ollama; import fhirpathpy"` succeeds.
- **Each exercise runs**: Execute each exercise script and verify non-error output. This is manual testing appropriate for educational code.

### Performance / Load Considerations

Not applicable for educational scripts. The only performance concern is LLM response time, which depends on the student's hardware and model choice. Document minimum hardware recommendations (8GB RAM for 3B models, 16GB for 7B).

### Security Testing Notes

- Verify `.env` is in `.gitignore`
- Verify no credentials appear in committed code
- Verify sample data is Synthea (synthetic), not real PHI

---

## Dependencies

### External Dependencies

| Dependency | Version / Constraint | Purpose | Risk |
|------------|---------------------|---------|------|
| `fhir.resources` | `>=7.0.0` | FHIR R4B model parsing | R4 → R4B import path changes; must verify all resource types used |
| `textual` | `>=8.0.0` | TUI framework for Exercise 3 | Major version jump from 0.44; API changes in widget composition |
| `ollama` | `>=0.4.0` | Primary LLM interface | Requires Ollama app installed separately |
| `lmstudio` | `>=1.5.0` | Secondary LLM interface | Requires LM Studio app installed separately |
| `fhirpathpy` | `>=2.2.0` | FHIRPath expression evaluation | Stable, no changes expected |
| `requests` | `>=2.28.0,<3.0.0` | HTTP client for FHIR server | Stable |
| `tiktoken` | `>=0.5.0` | Token counting approximation | Stable |
| `python-dotenv` | `>=1.0.0` | .env file loading | Stable, minimal |

### Internal / Shared Dependencies

- InterSystems IRIS for Health running locally with FHIR R4 endpoint and Synthea data loaded (course setup from prior lessons)
- Ollama installed with at least one model pulled (e.g., `ollama pull llama3.2:3b`)
- The `100Set/` Synthea bundle files (already present in repository)

---

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Textual 0.44 → 8.x has breaking API changes in widget composition | Exercise 3 TUI may need significant rework | High | Research Textual 8.x API before starting Exercise 3; prototype compose() method first |
| fhir.resources 7.x R4B sub-package imports differ from R4 | Import errors across all files | High | Test R4B imports early in Story 1; may need `from fhir.resources.R4B.patient import Patient` |
| Students' machines may not support Ollama (older hardware, ARM vs x86) | Exercise 1-4 blocked | Medium | Keep LM Studio as fallback; document minimum hardware; provide pre-recorded output screenshots |
| Tiktoken approximation may confuse students who notice token count discrepancies | Pedagogical confusion | Low | Explicitly address in slides and code comments; frame as a teaching moment about tokenizer differences |

---

## Assumptions & Constraints

### Assumptions

1. Students have completed prior FHIR lessons and have a working IRIS for Health instance with Synthea data loaded
2. Students have Python 3.11+ installed with ability to create virtual environments
3. Students can install Ollama (or already have LM Studio) on their machines
4. The InterSystems IRIS FHIR endpoint is accessible at the configured URL during class
5. A 3B-parameter model (e.g., Llama 3.2 3B) is sufficient for demonstrating summarization concepts

### Constraints

1. Must use InterSystems IRIS for Health as the FHIR server (course infrastructure)
2. Must use Python (course language)
3. Must not duplicate content from Parts One/Two or RAG Lab
4. Must work offline (local LLM, local FHIR server) — no cloud API dependencies for core exercises
5. All sample data must be synthetic (Synthea)

---

## Success Criteria

How we know this work is complete:

- [ ] All four exercises run successfully against IRIS FHIR server with updated dependencies
- [ ] `pip install -r requirements.txt` succeeds in a clean Python 3.11+ venv
- [ ] No `parse_obj()`, `pydantic<2.0`, or `textual==0.44.1` references remain in code
- [ ] Readme.md documents the complete lesson flow with setup instructions
- [ ] AI foundations content covers all concepts listed in FR-001
- [ ] Exercise 4 demonstrates at least 3 prompt strategies with visibly different outputs
- [ ] Configuration is externalized (no hardcoded URLs, credentials, or model names in exercise scripts)
- [ ] Safety/governance content provides conceptual frame for series
- [ ] All MUST functional requirements implemented and accepted
- [ ] All stories reach `done` status

---

## Decisions Log Summary

| Decision | Rationale | Date |
|----------|-----------|------|
| Ollama as primary LLM, LM Studio as secondary | Ollama has simpler install (CLI, no GUI), wider model support, better for classroom deployment. LM Studio kept for students who prefer GUI. | 2026-10-07 |
| Resource-type grouping instead of raw JSON chunking | Produces better summaries, aligns with Part One's structured feature extraction approach, and is more pedagogically current than blind text chunking | 2026-10-07 |
| Move RAG/vector files to Lab materials | SPEC explicitly scopes RAG to the Lab. Keeping these files in Lesson 17 creates confusion about what this lesson teaches. | 2026-10-07 |
| Keep tiktoken for token counting with documented approximation caveat | Replacing with model-native tokenizers adds complexity without pedagogical benefit; the approximation itself is a teaching moment | 2026-10-07 |
| .env configuration over config.toml | Simpler for students, well-established pattern, python-dotenv is lightweight | 2026-10-07 |

---

## Next Steps

This tech spec is the Quick Flow planning artifact. Proceed to story creation:

1. Use **bmad-epics-and-stories** to expand the Story List into full story files under
   `bmad-output/stories/`.
2. Story file naming: `{epic}.{story}.{slug}.story.md`
   (e.g., `1.1.update-dependencies.story.md`)
3. Once stories reach `ready-for-dev` status, hand off to your dev tool / plugin.

If scope has grown beyond 15 stories, switch to the BMad Method track before creating
stories: run **bmad-prd** to capture full requirements, then **bmad-architecture** to
design the system, then return to story planning.

---

*Technical Specification — Quick Flow Track — BMAD Method by the BMAD Code Organization*
