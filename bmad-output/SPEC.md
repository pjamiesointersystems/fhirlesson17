# SPEC — Lesson 17: Using LLMs with FHIR (Foundational Lesson Restructuring)

> **This is the kernel.** Five fields. Keep it lean.
> Downstream skills (PRD, tech-spec, architecture) expand each field.
> Do not add sections. If something doesn't fit, put it in the PRD.

**Created:** 2026-10-07
**Source:** BMAD research report + analysis of full lesson series (Part One, Part Two, RAG Lab)
**Track:** quick-flow
**Status:** confirmed

**Series context:** This lesson is the first in a four-part AI+FHIR series:
1. **Lesson 17 — Using LLMs with FHIR** (this spec) — foundational concepts and first hands-on
2. **Using Agentic AI with FHIR Part One** — agent architecture, MCP, tool-use patterns, terminology normalization
3. **Using Agentic AI with FHIR Part Two** — FHIR as knowledge layer, subagents, summarizer pipeline, safety patterns
4. **Lab: LLMs and FHIR Application Development** — RAG, vector stores, HNSW indexes, hybrid search, Graph RAG

---

## Problem

Healthcare informatics students in the Vanderbilt FHIR Training Course encounter AI and FHIR integration for the first time in Lesson 17 ("Using LLMs with FHIR"), which then feeds into subsequent lessons on agentic AI (Parts One and Two) and a hands-on RAG lab. The current Lesson 17 (May 2025) has outdated dependencies (Pydantic v1, Textual 0.44, fhir.resources 6.x), code quality issues (duplicate functions, bugs), and most critically fails to teach the foundational AI/LLM concepts that the downstream lessons assume students already know. Students arriving at the agentic AI lectures and the RAG lab have no structured introduction to how LLMs work, what tokens and embeddings are, how context windows affect clinical data processing, or why FHIR's structured data is uniquely well-suited for AI consumption. Without this foundation, the subsequent lessons lose pedagogical coherence.

---

## Capabilities

- Students can explain core LLM concepts (tokens, context windows, embeddings, temperature, prompts) using healthcare analogies that connect to their FHIR knowledge
- Students can set up and interact with a local LLM (Ollama as primary, LM Studio as secondary) from Python, understanding the privacy implications of local vs. cloud model deployment
- Students can retrieve a FHIR patient bundle ($everything) and transform it into LLM-ready input, understanding why raw FHIR JSON must be processed before feeding to an LLM
- Students can build a working patient summarization pipeline that demonstrates the end-to-end flow from FHIR server to structured clinical narrative
- Students can apply basic prompt engineering techniques (system roles, structured output instructions, few-shot examples) and observe their effect on clinical summarization quality
- Students can articulate the safety and governance considerations for using AI with healthcare data, providing the conceptual frame for safety patterns covered in the agentic AI lessons
- The lesson's exercises produce working code and conceptual understanding that directly feed into the agentic AI lectures (Part One/Two) and the RAG lab

---

## Constraints

- Must run against InterSystems IRIS for Health as the FHIR R4 server (course infrastructure)
- Must use Python as the primary language (consistent with all other lessons)
- Must work with updated dependencies: fhir.resources 7.x (Pydantic v2), current Textual, current LM Studio/Ollama SDKs
- Must use Synthea-generated patient bundles (no real PHI) — the existing 100Set or equivalent
- Must serve as the first lesson in the AI+FHIR series, assuming no prior AI/ML knowledge beyond general familiarity with LLMs
- Must not duplicate content covered in depth by the agentic AI lectures (Part One/Two) or the RAG lab

---

## Non-Goals

- Teaching agentic AI patterns, tool-use, or multi-step reasoning (covered in "Using Agentic AI with FHIR" Parts One and Two)
- Building or demonstrating MCP servers or clients (covered in Parts One and Two)
- Implementing RAG pipelines, vector stores, or HNSW indexes (covered in the RAG Lab)
- Terminology normalization or code resolution via MCP (covered in Parts One and Two)
- Graph RAG or advanced retrieval strategies (covered in the RAG Lab)
- Fine-tuning models or training custom healthcare LLMs

---

## Success Metrics

- Students can correctly define and explain at least 5 core AI/LLM concepts (tokens, embeddings, context windows, temperature, hallucination) using healthcare analogies in a post-lesson assessment
- All exercise code runs successfully against the course FHIR server with updated dependencies (zero Pydantic v1 or outdated library errors)
- Students produce a working patient summary from a FHIR bundle using a local LLM within the lesson timeframe
- Students entering the agentic AI lectures and RAG lab report that Lesson 17 adequately prepared them (measured by pre-lesson readiness survey or instructor observation)
- The lesson's code examples demonstrate at least two distinct prompt engineering techniques with observable differences in output quality

---

<!-- SPEC END — anything else belongs in the PRD, tech-spec, or architecture doc. -->
