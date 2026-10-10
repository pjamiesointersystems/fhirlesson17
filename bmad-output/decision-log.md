# Decision Log

## Research: Restructuring Lesson 17 (AI and FHIR) -- 2026-10-07
- Mode: Create
- Types: Domain, Technical, Educational
- Key finding: The AI+FHIR ecosystem has matured from experimental to production (224+ repos, 6+ MCP servers, Epic/Cumulus deployments); the lesson needs fundamental restructuring — not a refresh — into a three-phase structure (foundations, hands-on exercises, advanced topics) with updated dependencies and new content for MCP, agentic AI, RAG, and safety.
- Report: bmad-output/research-report.md
- Next skill: bmad-spec or bmad-product-brief

## SPEC created — 2026-10-07
- Source: Research report + full series analysis (Part One, Part Two, RAG Lab PDFs/PPTX)
- Key scope decision: Lesson 17 is the foundational on-ramp only — agentic AI, MCP, RAG, terminology normalization, and vector stores are explicitly out of scope (covered by companion lessons). The SPEC positions Lesson 17 to teach AI/LLM fundamentals and first hands-on FHIR+LLM experience that downstream lessons assume.

## Tech Spec created — 2026-10-07
- Artifact: bmad-output/tech-spec.md
- Scope: 12 stories across 3 epics (Infrastructure, Exercises, Content)
- Key decisions:
  - **Ollama as primary LLM** (simpler install, CLI-based, LM Studio kept as secondary)
  - **Resource-type grouping** replaces raw JSON chunking for summarization (aligns with Part One's structured feature extraction)
  - **RAG/vector files moved to Lab** (simplevector.py, fhirvectorflattened.py, testrag.py, fhir_rag_chat.py belong in RAG Lab scope)
  - **tiktoken kept** with documented approximation caveat (teaching moment about tokenizer differences)
  - **.env configuration** over config.toml (simpler for students)
- Next skill: bmad-epics-and-stories (to compile stories from the story list)

## Drop Textual TUI; Exercise 3 becomes CLI pipeline — 2026-10-08
- **Decision:** Remove Textual framework entirely. Exercise 3 is a CLI pipeline (fetch → group → extract → LLM → print). Instructor will demo artifacts via Angular front end separately; Angular is out of scope for Lesson 17 stories.
- **Rationale:** Students will not build many artifacts themselves — the instructor demos with Angular. A CLI pipeline is simpler, eliminates the highest-risk dependency migration (Textual 0.44→8.x), and keeps the lesson focused on AI+FHIR concepts rather than UI framework specifics.
- **Impact:** FR-004 rewritten as CLI; Textual removed from FR-008 dependency table; Story 2.3 simplified; Textual API migration risk eliminated.

## Epics & Stories created — 2026-10-08
- Artifact: bmad-output/epics.md + 12 story files in bmad-output/stories/
- Track: Quick Flow (12 stories, 3 epics)
- Epic 1 (Infrastructure): 3 stories — deps, config, FHIR client
- Epic 2 (Exercises): 4 stories — Exercises 1-4
- Epic 3 (Content & Cleanup): 5 stories — slides, safety, readme, cleanup, bridge
