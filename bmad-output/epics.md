# Epics — Lesson 17: Using LLMs with FHIR (Restructuring)

> The epic MAP. A thin index, not a context object. Each epic lists its goal, the
> requirements it covers (cited to tech-spec.md), its ordered stories, and cross-epic
> dependencies. Story detail lives in the individual {epic}.{story}.{slug}.story.md files.
>
> Track: Quick Flow
> Sources: tech-spec.md

---

## Epic 1: Infrastructure

**Goal:** Update the foundational plumbing — dependencies, configuration, and shared FHIR client — so all exercise scripts build on a modern, working base.

**In scope (cited):**
- FR-008 — Dependency update (fhir.resources 7.x, Textual 8.x, Ollama, python-dotenv) [Source: tech-spec.md#FR-008]
- FR-007 — Configuration externalization (.env + config.py) [Source: tech-spec.md#FR-007]
- FR-003 (partial) — FHIR client refactor (getSearchPatients.py → fhir_client.py) [Source: tech-spec.md#FR-003]

**Architecture touchpoints:** requirements.txt, config.py, fhir_client.py, .env.example, .gitignore [Source: tech-spec.md#technical-approach]

**Out of scope:** Exercise scripts, slide content, Readme (those are Epic 2 and 3)

**Stories (ordered):**

| ID | Slug | Intent | Status |
|------|------|--------|--------|
| 1.1 | update-dependencies | Update requirements.txt to current versions and verify imports | ready-for-dev |
| 1.2 | create-config-module | Create config.py, .env.example, and .gitignore update | ready-for-dev |
| 1.3 | refactor-fhir-client | Refactor getSearchPatients.py → fhir_client.py with Pydantic v2 | ready-for-dev |

**Cross-epic dependencies:**
- Blocked by: none — this is the foundation
- Blocks: Epic 2 — exercises depend on updated deps, config module, and fhir_client.py

---

## Epic 2: Exercises

**Goal:** Build the four progressive hands-on exercises that give students their first working AI+FHIR experience.

**In scope (cited):**
- FR-002 — Exercise 1: Local LLM setup and first chat [Source: tech-spec.md#FR-002]
- FR-003 (remainder) — Exercise 2: FHIR bundle retrieval and exploration [Source: tech-spec.md#FR-003]
- FR-004 — Exercise 3: Patient summarization TUI [Source: tech-spec.md#FR-004]
- FR-005 — Exercise 4: Prompt engineering comparison [Source: tech-spec.md#FR-005]

**Architecture touchpoints:** exercise1_llm_chat.py, exercise2_fhir_explore.py, exercise3_summarizer.py, exercise4_prompt_compare.py [Source: tech-spec.md#technical-approach]

**Out of scope:** Slide content, Readme (Epic 3); infrastructure plumbing (Epic 1)

**Stories (ordered):**

| ID | Slug | Intent | Status |
|------|------|--------|--------|
| 2.1 | exercise1-llm-chat | Build Exercise 1 — LLM hello world with Ollama | ready-for-dev |
| 2.2 | exercise2-fhir-explore | Build Exercise 2 — FHIR bundle exploration | ready-for-dev |
| 2.3 | exercise3-summarizer-cli | Build Exercise 3 — Patient summarization CLI pipeline | ready-for-dev |
| 2.4 | exercise4-prompt-compare | Build Exercise 4 — Prompt engineering comparison | ready-for-dev |

**Cross-epic dependencies:**
- Blocked by: Epic 1 — all exercises import config.py and fhir_client.py
- Blocks: Epic 3 (partially) — Readme documents exercises, so exercises should exist first

---

## Epic 3: Content & Cleanup

**Goal:** Create the lecture/slide content, rewrite the Readme as a lesson guide, clean up the file structure, and write the series bridge — completing the lesson as a deliverable.

**In scope (cited):**
- FR-001 — AI foundations slide/lecture content [Source: tech-spec.md#FR-001]
- FR-010 — Safety and governance discussion content [Source: tech-spec.md#FR-010]
- FR-006 — Updated Readme.md [Source: tech-spec.md#FR-006]
- FR-009 — File cleanup and reorganization [Source: tech-spec.md#FR-009]
- FR-011 — Series bridge content [Source: tech-spec.md#FR-011]

**Architecture touchpoints:** FHIRAI/Readme.md, slide content files, archive/ directory [Source: tech-spec.md#FR-009]

**Out of scope:** Exercise code (Epic 2); infrastructure (Epic 1)

**Stories (ordered):**

| ID | Slug | Intent | Status |
|------|------|--------|--------|
| 3.1 | ai-foundations-slides | Write AI foundations slide/lecture content with healthcare analogies | ready-for-dev |
| 3.2 | safety-governance | Write safety and governance discussion content | ready-for-dev |
| 3.3 | readme-rewrite | Rewrite Readme.md as complete lesson guide | ready-for-dev |
| 3.4 | file-cleanup | Archive/remove undocumented files, organize directory | ready-for-dev |
| 3.5 | series-bridge | Write series bridge content connecting to Parts One/Two and RAG Lab | ready-for-dev |

**Cross-epic dependencies:**
- Blocked by: Epic 1 (partially — Readme references updated file names); Epic 2 (partially — Readme documents exercises)
- Blocks: none — this is the final epic

**Parallelization notes:** Stories 3.1 (slides) and 3.2 (safety) can run in parallel with Epic 2 exercises since they produce different file types. Story 3.3 (Readme) should wait for Epic 2 completion. Story 3.4 (cleanup) should run last. Story 3.5 (bridge) can parallelize with 3.1/3.2.

---

## Delivery Tracking (count-based)

No story points, velocity, or burndown. Track by COUNT only:

- Total stories: 12
- Done: 0
- Remaining: 12
- Completion rate: 0/12

## Notes

**Sequencing rationale:**
- Epic 1 is the critical path — nothing else can start until deps, config, and FHIR client are in place
- Within Epic 1: stories are strictly sequential (1.1 → 1.2 → 1.3) because each depends on the prior
- Within Epic 2: stories 2.1-2.2 can partially parallelize (different files, both import config); 2.3 depends on 2.2 (uses fhir_client); 2.4 can parallelize with 2.3 (different file, shares FHIR data pattern)
- Within Epic 3: 3.1, 3.2, 3.5 can parallelize (different content files); 3.3 should follow Epic 2; 3.4 runs last
- Epic 3 stories 3.1/3.2/3.5 can begin as soon as Epic 1 completes (they don't depend on exercise code)

**Risk epics:** Epic 2 carries the highest risk due to Textual 8.x migration (Story 2.3) and fhir.resources 7.x R4B imports (validated in Epic 1 but first heavily used in 2.2).
