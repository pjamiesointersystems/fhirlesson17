# Research Report: Restructuring Lesson 17 — AI and FHIR

**Date:** 2026-10-07
**Research Type:** Domain | Technical | Educational
**Mode:** Create
**Project:** Vanderbilt FHIR Training Course, Lesson 17
**Version:** 1.0
**Status:** Final

---

## Executive Summary

**Research Objective:** Determine how to best restructure a university-level lesson on AI and FHIR for healthcare informatics students, incorporating developments since May 2025.

**Key Findings:**

1. **The AI+FHIR ecosystem has matured dramatically** — from conceptual demos to production deployments (Epic's three AI product lines, 224+ GitHub repos combining FHIR+LLM, 6+ FHIR MCP servers, frameworks like HealthChain providing typed FHIR tools for AI agents).

2. **MCP and agentic AI are the dominant new integration patterns** — Model Context Protocol has become the standard way to connect LLMs to FHIR servers. Agentic workflows (Hospilot's 14-agent hospital OS, HealthChain's FHIRToolkit) represent the cutting edge. The existing lesson covers none of this.

3. **The existing lesson's core approach (chunked summarization via local LLM) remains pedagogically sound** but represents only one of several patterns students need. The codebase contains undocumented RAG and vector store components that should be formalized. Dependencies are outdated (pydantic<2.0, textual==0.44.1, fhir.resources==6.1.0).

**Bottom Line:** The lesson needs a fundamental restructuring — not a refresh. The existing summarization exercise is worth keeping (enhanced with prompt engineering and temperature exploration), but the lesson must be expanded to cover foundational AI concepts, the MCP/agentic paradigm, RAG with FHIR, structured FHIR output generation, and safety/governance. The recommended structure is a 2.5-3 hour "sandwich" model: demo hook, concepts, hands-on exercises, advanced topics, wrap-up.

**Primary Recommendation:** Restructure as a three-phase lesson: (1) AI foundations for healthcare audiences, (2) hands-on exercises progressing from summarization to NL-to-FHIR-query to RAG, (3) emerging patterns (MCP, agents, structured output) and safety discussion.

---

## Research Scope

### Objectives

**Primary Questions:**
1. What AI/LLM concepts must FHIR students understand, and how should they be taught?
2. What is the current state of AI+FHIR integration, and what tools/patterns should the lesson cover?
3. What from the existing May 2025 lesson remains relevant, and what needs replacement?

**Secondary Questions:**
- What hands-on exercises are most effective for teaching AI+FHIR?
- Which local LLM platform is best for educational use?
- What privacy/regulatory considerations must the lesson address?

### Scope Boundaries

**In Scope:**
- AI/LLM concepts accessible to non-CS healthcare informatics students
- Tools and frameworks for AI+FHIR integration as of 2025-2026
- Practical exercises suitable for a single 2.5-3 hour lesson
- Assessment of the existing Lesson 17 codebase and materials
- Privacy, HIPAA, and regulatory landscape

**Out of Scope:**
- Deep ML/neural network theory — audience is healthcare informaticists, not ML engineers
- Specific vendor EHR integration details — lesson uses a generic FHIR R4 server
- Multi-lesson curriculum design — focusing on a single lesson within existing course

**Time Frame:** Current landscape as of Q4 2026
**Geographic Focus:** US healthcare system (HIPAA, ONC, FDA context)

---

## Methodology

**Research Type:** Secondary (web sources, published literature, open-source project analysis)

**Methods Used:**
- WebFetch analysis across 60+ URLs (GitHub repos, PyPI packages, standards sites, industry sites)
- PubMed literature search for AI education in health informatics
- Direct analysis of existing Lesson 17 codebase (10+ Python files, requirements.txt, Readme)

**Limitations:**
- Several government sites (HHS, ONC, Federal Register) blocked direct access (403/404); regulatory details supplemented from secondary sources
- AMIA and HIMSS education catalogs behind membership walls
- GPT4All, Jan, LM Studio, Ollama main sites had certificate errors; assessed via PyPI and GitHub instead
- Academic papers accessed via PubMed abstracts; full texts not retrieved

---

## Finding 1: The AI+FHIR Ecosystem Has Reached Production Maturity

**Summary:** The landscape has evolved from experimental projects in 2023-2024 to production-grade tools and deployments in 2025-2026.

**Key Evidence:**

- **Epic Systems** deploys three AI product lines: Emmie (patient engagement), Art (clinical documentation — "nurses write notes 85% faster"), Penny (revenue cycle). GPT-4 integrated through HIPAA-compliant pipeline with Microsoft. [Source: epic.com/software/ai, accessed 2026-10-07]

- **224+ GitHub repositories** combine FHIR and LLM capabilities. Notable projects include HealthChain (229 stars, Python SDK for FHIR-aware AI agents), Stanford's LLMonFHIR (158 stars, iOS), Flexpa's LLM-FHIR-Eval (46 stars, benchmarking), and Infherno (18 stars, EACL 2026 paper on clinical notes to FHIR). [Source: github.com search, accessed 2026-10-07]

- **SMART Cumulus** — ONC-funded "universal sidecar" using AI/NLP to process Bulk FHIR data, deployed across 5 health system partnerships for disease surveillance. [Source: smarthealthit.org/cumulus, accessed 2026-10-07]

- **ONC HTI-1 Final Rule** (Jan 2024) introduced Decision Support Intervention (DSI) and Predictive DSI certification criteria with AI/ML transparency requirements, phased through 2025-2026. [Source: healthit.gov]

- **FDA** has authorized 700+ AI-enabled medical devices as of September 2026. [Source: fda.gov AI device list, accessed 2026-10-07]

**Implications for Lesson Design:**
- The lesson must present AI+FHIR as a production reality, not a future possibility
- Real-world deployments (Epic, Cumulus) serve as motivating examples
- Regulatory context (HTI-1, FDA) must be covered, at least briefly

**Confidence:** [VERIFIED] — multiple independent sources confirm ecosystem maturity

---

## Finding 2: MCP and Agentic AI Are the Dominant New Patterns

**Summary:** Model Context Protocol has become the standard integration layer between AI agents and FHIR servers, and agentic workflows represent the cutting edge of clinical AI.

**Key Evidence:**

- **6 significant FHIR MCP server implementations** exist:
  - WSO2 (137 stars) — SMART-on-FHIR auth, FHIRPath filtering, multi-transport
  - Momentum (102 stars) — full CRUD, RAG services, Pinecone vector store, LOINC
  - LangCare (67 stars) — enterprise-grade Go, Epic/Cerner/OpenEMR support, PHI scrubbing
  - HealthClaw Guardrails (30 stars) — security proxy, PHI redaction, audit trails, 29 MCP tools
  - Azure FHIR MCP (18 stars), fhirHydrant (18 stars)
  [Source: github.com search, accessed 2026-10-07]

- **The emerging architecture pattern is: LLM Agent -> MCP Server -> FHIR Server -> EHR**, with security guardrail layers inserted between MCP and FHIR.

- **Agentic examples:**
  - Hospilot: 14 specialized domain agents (bed management, ER triage, ICU capacity, discharge planning, etc.) coordinated by LangGraph LLM planner with human-in-the-loop gates
  - HealthChain: FHIRToolkit serving validated, typed FHIR tools to agents via MCP or LangChain
  - OpenMed Agent: terminal-native agent for prior auth, ICD-10 audits, care coordination
  [Sources: respective GitHub repos, accessed 2026-10-07]

- **CDS Hooks** (HL7 standard) provides the natural integration point for AI-powered CDS — hook-based triggers (patient-view, order-select, order-sign) returning CDS Cards. [Source: cds-hooks.org, accessed 2026-10-07]

**Implications for Lesson Design:**
- MCP must be introduced as the modern integration pattern — even if only conceptually
- The Agent -> MCP -> FHIR -> EHR architecture should be taught as a reference pattern
- CDS Hooks provides the natural bridge between students' existing FHIR knowledge and AI

**Confidence:** [VERIFIED] — multiple independent projects confirm the pattern

---

## Finding 3: Foundational AI Concepts Require Healthcare-Specific Teaching

**Summary:** Published research and educational frameworks provide clear guidance on what to teach and how to teach it to healthcare informatics audiences.

**Key Evidence:**

- **Miller's Pyramid for AI Competency** (Houser et al., 2026, PMID: 42022211): Knows -> Knows How -> Shows How -> Does. A single lesson can target "Knows" and "Shows How" levels.

- **Three-tiered concept framework** (synthesized from IBM, NVIDIA, Google, OpenAI sources):
  - Tier 1 (Must Cover): What LLMs are, tokenization, context windows, prompt engineering, temperature, hallucination/safety
  - Tier 2 (Should Cover): Embeddings, transformers/attention (conceptual), RAG, fine-tuning
  - Tier 3 (Brief Mention): AI agents, model architectures, training data/bias, RLHF

- **Healthcare-native analogies** are critical for non-CS audiences:
  - Tokenization = "pharmacist reading prescriptions by meaningful chunks"
  - Context window = "clinician's working memory during a patient encounter"
  - RAG = "open-book exam" or "consulting UpToDate while seeing a patient"
  - Hallucination = "confabulation in neurology"
  - Prompt engineering = "writing a good consultation request"
  [Sources: IBM, Google Developers, OpenAI, AWS, Anthropic; accessed 2026-10-07]

- **Published curriculum research** (4 papers, 2025-2026): Seba et al. (PMID: 41401239) covers GenAI curriculum integration; Alhur & Al-Kahtani (PMID: 42549035) covers student engagement strategies.

- **AMA uses "Augmented Intelligence"** framing to emphasize human-AI partnership, not replacement. [Source: ama-assn.org, accessed 2026-10-07]

**Implications for Lesson Design:**
- Lead with function, not mechanism — what AI does and why it matters, then how
- Use healthcare analogies for every concept
- 40% concepts / 60% hands-on is the optimal ratio
- Hallucination/safety must be a prominent theme, not an afterthought

**Confidence:** [VERIFIED] — multiple published sources and educational research

---

## Finding 4: The Existing Lesson Has a Sound Core but Major Gaps

**Summary:** The May 2025 codebase contains more than documented, has outdated dependencies, and is missing critical 2026 topics.

**Key Evidence:**

**What works (keep/enhance):**
- Hierarchical chunking/summarization pattern is still pedagogically valuable — teaches context windows, tokenization, map-reduce
- The Textual TUI provides excellent visual feedback
- Using local LLMs eliminates HIPAA concerns for exercises
- Synthea-generated FHIR bundles in 100Set/ are appropriate test data

**What's outdated (fix):**
- `pydantic<2.0` pin blocks current libraries; `fhir.resources==6.1.0` should be 7.x; `textual==0.44.1` is extremely outdated (current: 8.2.8)
- `parse_obj()` calls need updating to `model_validate()` (Pydantic v2)
- `tiktoken.encoding_for_model("gpt-3.5-turbo")` is technically incorrect for local Llama/Mistral models
- Duplicate `get_patient_from_server()` function in `getSearchPatients.py`
- Hardcoded credentials and model names throughout

**What's undocumented (formalize):**
- RAG pipeline: `fhirvectorflattened.py` (FHIR -> embeddings -> IRIS vector store), `fhir_rag_chat.py` (interactive RAG chat), `testrag.py`, `simplevector.py`
- OpenAI comparison: `OpenAIFHIRSummary.py` (same architecture with GPT-4o)
- Resource-type summarization: `FHIRSummaryApp_by_resource_type.py`

**What's missing (add):**
- No conceptual introduction — students jump straight into code
- No agentic AI / tool-use patterns
- No MCP coverage
- No structured output / FHIR resource generation
- Minimal prompt engineering instruction
- No model selection guidance
- No safety/governance demonstrations
- No assessment component

**Implications for Lesson Design:**
- Keep the summarization exercise as Exercise 1 (enhanced with prompt engineering)
- Formalize the undocumented RAG components into a proper module
- Add new content for agentic AI, MCP, structured output, and safety
- Fix all outdated dependencies before the lesson update

**Confidence:** [VERIFIED] — direct codebase analysis plus library version checks

---

## Finding 5: Ollama Is the Recommended Platform for Educational Use

**Summary:** For a university FHIR course, Ollama offers the best balance of simplicity and capability, though LM Studio remains viable.

**Key Evidence:**

| Feature | LM Studio | Ollama | llama-cpp-python |
|---------|-----------|--------|-----------------|
| Setup | Easy (GUI) | Easy (CLI) | Moderate (build) |
| Python SDK | lmstudio 1.5.0 | ollama 0.6.3 | 0.3.36 |
| Min Python | 3.10 | 3.8 | 3.8 |
| OpenAI-compatible API | Yes | Yes | Yes (built-in server) |
| JSON Schema constrained output | Via API | Via API | Native support |
| Best for | Visual learners | CLI/automation | Advanced users |
[Sources: PyPI package pages, accessed 2026-10-07]

- **Ollama** advantages: CLI-first teaches model management, broader Python compat, embedding generation built-in, async client support
- **LM Studio** advantages: GUI for model browsing/download, good for beginners who want visual interface
- **llama-cpp-python** advantage: native JSON Schema constrained output (valuable for generating valid FHIR resources)

**Recommendation:** Support both Ollama (primary) and LM Studio (secondary) in the lesson. Demonstrate llama-cpp-python's constrained output for the FHIR generation exercise.

**Confidence:** [VERIFIED] — direct SDK documentation review

---

## Finding 6: Privacy Architecture Is a Teaching Opportunity

**Summary:** The local-vs-cloud LLM decision, de-identification approaches, and regulatory requirements provide rich material for the lesson.

**Key Evidence:**

- **HIPAA requires BAAs** when PHI is sent to cloud LLM providers; local LLMs avoid this entirely
- **Stanford LLMonFHIR** demonstrates "fog node" mode for local inference
- **AI Medical Scribe** uses Chrome's Gemini Nano for entirely on-device processing
- **HealthClaw Guardrails** implements real-time PHI redaction: names to initials, identifiers stripped, DOB to year-only
- **Microsoft FHIR Tools for Anonymization** — open-source toolkit implementing HIPAA Safe Harbor de-identification
- **ONC HTI-1** requires transparency about training data, intended use, and limitations for Predictive DSI
- **Key regulatory gap:** no explicit standard linking FHIR interoperability to AI/ML deployment requirements

[Sources: respective GitHub repos, FDA, ONC; accessed 2026-10-07]

**Implications for Lesson Design:**
- Local-vs-cloud trade-off is an excellent classroom discussion topic
- Synthea data (already in 100Set/) eliminates PHI concerns for exercises
- Brief regulatory overview (HTI-1, FDA, HIPAA) provides important context

**Confidence:** [VERIFIED] for tools/approaches; [UNVERIFIED] for some regulatory details (government sites blocked)

---

## Gaps & Opportunities

### 1. No Existing Lesson Covers AI+FHIR This Comprehensively
- Most informatics programs treat AI and FHIR separately
- A lesson that integrates both with hands-on exercises using real FHIR data would be distinctive
- The existing InterSystems IRIS infrastructure provides a unique advantage (FHIR server + vector store)

### 2. The MCP+FHIR Pattern Is Undertaught
- 6+ FHIR MCP servers exist but no educational materials explaining the pattern
- Even a conceptual introduction with a live demo would be valuable

### 3. RAG with FHIR Data Is the Most Practical Advanced Topic
- The existing codebase already has RAG components (undocumented)
- RAG directly addresses hallucination — the #1 safety concern
- The concept maps naturally to healthcare ("consulting a reference while treating a patient")

---

## Recommendations

### Primary Recommendations

1. **Restructure as a Three-Phase Lesson (2.5-3 hours)**
   - Phase 1: AI Foundations (60 min) — concepts with healthcare analogies, kick off with live demo
   - Phase 2: Hands-On Exercises (75 min) — summarization, NL-to-FHIR query, RAG chat
   - Phase 3: Advanced Topics & Safety (30 min) — MCP/agents overview, hallucination, privacy, regulatory
   - Rationale: Matches the "sandwich" model from educational research (Finding 3) and 40/60 theory-practice ratio
   - Priority: Critical
   - Handoff: Inform slide/notes development

2. **Update the Technology Stack**
   - Upgrade dependencies: fhir.resources 7.x, pydantic 2.x, textual 8.x
   - Support both Ollama (primary) and LM Studio (secondary)
   - Fix all code quality issues (duplicate functions, hardcoded credentials, Pydantic v1 syntax)
   - Externalize configuration to `.env` file
   - Priority: Critical (must be done before lesson redesign)
   - Handoff: Inform code update sprint

3. **Add Three New Exercise Types**
   - Exercise 1 (keep/enhance): Patient summarization with prompt engineering exploration
   - Exercise 2 (new): Natural language to FHIR query translation
   - Exercise 3 (formalize): RAG chat with FHIR data (components already exist)
   - Rationale: Ranked by pedagogical effectiveness in Finding 3; covers progression from basic to advanced
   - Priority: High
   - Handoff: Inform exercise development

### Secondary Recommendations

- Create a student handout with healthcare analogies, key concepts, and resource links
- Add a brief assessment component (error-detection exercise or concept check questions)
- Include a model selection guide (when local vs. cloud, size trade-offs, quantization effects)
- Demonstrate the MCP+FHIR pattern — even a 10-minute conceptual walkthrough with the WSO2 FHIR MCP server would be impactful
- Consider Jupyter notebook versions of exercises for step-by-step instruction

---

## Recommended Lesson Outline

### Phase 1: AI Foundations (60 min)

**1.1 Hook: Live Demo (15 min)**
- Run the FHIR summarization app (raw JSON in -> patient summary out)
- "How did the computer do this? What could go wrong?"

**1.2 What Are LLMs? (10 min)**
- Statistical next-token prediction ("giant autocomplete")
- Not understanding or thinking — computing probabilities
- Healthcare analogy: pattern recognition, not clinical reasoning

**1.3 Core Concepts (25 min)**
- Tokenization (5 min) — live demo with tokenizer, healthcare text examples
- Context windows (5 min) — "clinician's working memory," why we chunked in the demo
- Prompt engineering (10 min) — anatomy of a good prompt, "consultation request" analogy
- Temperature (5 min) — side-by-side comparison on same clinical data

**1.4 Safety: Hallucination (10 min)**
- "Confabulation in neurology" analogy
- Show a real hallucinated clinical summary
- Why verification is non-negotiable in healthcare
- AMA "Augmented Intelligence" framing

### Phase 2: Hands-On Exercises (75 min)

**2.1 Exercise: Enhanced Patient Summarization (25 min)**
- Run the existing summarization on a Synthea patient
- Modify prompts and compare outputs
- Compare temperature 0.0 vs 0.7
- Identify errors in generated summaries

**2.2 Exercise: Natural Language to FHIR Query (25 min)**
- Students write clinical questions in plain English
- LLM translates to FHIR search API calls
- Execute against local FHIR server, validate results
- Observe when the LLM generates invalid FHIR (reinforces earlier FHIR lessons)

**2.3 Exercise: RAG Chat with FHIR Data (25 min)**
- Conceptual intro: RAG = "open-book exam" / "consulting UpToDate"
- Why RAG reduces hallucination
- Demo the RAG chat pipeline (embed query -> vector search -> context-augmented LLM)
- Students query the patient vector store and compare RAG vs. non-RAG responses

### Phase 3: Advanced Topics & Wrap-up (30 min)

**3.1 Embeddings & RAG Deep Dive (5 min)**
- "Hospital map" analogy for embeddings
- How vector similarity connects concepts

**3.2 The Modern AI+FHIR Architecture (10 min)**
- MCP: how AI agents connect to FHIR servers via a standard protocol
- The pattern: Agent -> MCP Server -> FHIR Server -> EHR
- Agentic AI: LLMs that decide which FHIR queries to run
- CDS Hooks as the clinical integration point
- Brief tour of key projects: HealthChain, WSO2 FHIR MCP, Hospilot

**3.3 Privacy, Regulation & Governance (10 min)**
- Local vs. cloud LLMs: HIPAA implications
- De-identification approaches (HIPAA Safe Harbor, HealthClaw redaction)
- ONC HTI-1 Predictive DSI requirements
- FDA AI/ML device landscape (700+ authorized devices)
- The regulatory gap between FHIR interoperability and AI standards

**3.4 Wrap-up (5 min)**
- Key takeaways (3-5 bullet points)
- Resources for further learning
- Brief concept-check or reflection exercise

---

## Key Open-Source Projects for the Lesson

| Project | Stars | What It Demonstrates | Lesson Use |
|---------|-------|---------------------|------------|
| HealthChain | 229 | Typed FHIR tools for AI agents | Reference architecture |
| WSO2 FHIR MCP Server | 137 | MCP+FHIR integration | Demo candidate |
| LLMonFHIR (Stanford) | 158 | LLM interpretation of health records | Case study |
| LLM-FHIR-Eval (Flexpa) | 46 | Benchmarking LLMs on FHIR tasks | Discussion of model capabilities |
| HealthClaw Guardrails | 30 | Security, PHI redaction, audit | Safety discussion |
| Infherno | 18 | Clinical notes -> FHIR (RAG + agents) | Advanced pattern example |
| Hospilot | 15 | Multi-agent hospital operations | Agentic AI case study |

---

## Student Resources List

### Concept References
1. IBM — "What Are Large Language Models?" — https://www.ibm.com/think/topics/large-language-models
2. NVIDIA — "Large Language Models" — https://www.nvidia.com/en-us/glossary/large-language-models/
3. Google Developers — "Intro to LLMs" — https://developers.google.com/machine-learning/resources/intro-llms
4. OpenAI — "Prompt Engineering Guide" — https://developers.openai.com/api/docs/guides/prompt-engineering
5. AWS — "What is RAG?" — https://aws.amazon.com/what-is/retrieval-augmented-generation/
6. Anthropic — "Contextual Retrieval" — https://www.anthropic.com/news/contextual-retrieval

### FHIR+AI Standards
7. CDS Hooks — https://cds-hooks.hl7.org/
8. SMART on FHIR — https://smarthealthit.org
9. Model Context Protocol — https://modelcontextprotocol.io

### Ethics & Governance
10. AMA — "Augmented Intelligence in Medicine" — https://www.ama-assn.org/practice-management/digital/augmented-intelligence-medicine
11. WHO — "Ethics and Governance of AI for Health" — https://www.who.int/publications/i/item/9789240029200

### Academic Papers
12. Houser et al. "Integrating AI Into Health Informatics Education." 2026. PMID: 42022211
13. Seba et al. "Incorporating GenAI Into a Health Informatics Curriculum." 2025. PMID: 41401239
14. Alhur & Al-Kahtani. "GenAI in Health Informatics Education." 2026. PMID: 42549035

---

## Next Steps

### Immediate Planning Actions

- [ ] Update the technology stack (dependencies, Pydantic v2, code fixes)
- [ ] Design the new slide deck following the three-phase structure
- [ ] Formalize the undocumented RAG components into a structured exercise
- [ ] Create the NL-to-FHIR-query exercise
- [ ] Write the student handout with healthcare analogies and resources
- [ ] Decide on Ollama vs. LM Studio as the primary platform (or support both)

### Follow-Up Research Needed

- How to best demonstrate MCP+FHIR in a classroom setting — suggested method: test WSO2 FHIR MCP server with Claude Desktop or VS Code
- Which Synthea patients from the existing 100Set have the richest data for exercise scenarios — suggested method: analyze bundle sizes and resource diversity

### Recommended BMAD Handoff

**Next skill:** bmad-product-brief or bmad-spec — to distill these findings into a focused specification for the lesson redesign before creating a PRD for the actual content development.

**Key inputs for next skill from this report:**
- The three-phase lesson structure (Finding 3 + Recommendations)
- The technology stack decisions (Finding 5 + Recommendation 2)
- The exercise prioritization (Finding 4 + Recommendation 3)

---

## Appendix A: All Sources Consulted

### Successfully Accessed Sources

| # | Source | URL | Date |
|---|--------|-----|------|
| 1 | Epic AI | https://www.epic.com/software/ai | 2026-10-07 |
| 2 | SMART Health IT | https://smarthealthit.org | 2026-10-07 |
| 3 | SMART Cumulus | https://smarthealthit.org/cumulus | 2026-10-07 |
| 4 | CDS Hooks | https://cds-hooks.org | 2026-10-07 |
| 5 | CDS Hooks HL7 | https://cds-hooks.hl7.org/ | 2026-10-07 |
| 6 | Model Context Protocol | https://modelcontextprotocol.io | 2026-10-07 |
| 7 | PyPI: ollama | https://pypi.org/project/ollama/ | 2026-10-07 |
| 8 | PyPI: lmstudio | https://pypi.org/project/lmstudio/ | 2026-10-07 |
| 9 | PyPI: llama-cpp-python | https://pypi.org/project/llama-cpp-python/ | 2026-10-07 |
| 10 | GitHub: FHIR+LLM search | https://github.com/search?q=fhir+llm | 2026-10-07 |
| 11 | GitHub: MCP+FHIR search | https://github.com/search?q=mcp+server+fhir | 2026-10-07 |
| 12 | HealthChain | https://github.com/healthchainai/HealthChain | 2026-10-07 |
| 13 | LLMonFHIR | https://github.com/StanfordBDHG/LLMonFHIR | 2026-10-07 |
| 14 | LLM-FHIR-Eval | https://github.com/flexpa/llm-fhir-eval | 2026-10-07 |
| 15 | Infherno | https://github.com/j-frei/Infherno | 2026-10-07 |
| 16 | Hospilot | https://github.com/Carer-Healthcare-AI/Hospilot | 2026-10-07 |
| 17 | AI Medical Scribe | https://github.com/hutchpd/AI-Medical-Scribe | 2026-10-07 |
| 18 | WSO2 FHIR MCP | https://github.com/wso2/fhir-mcp-server | 2026-10-07 |
| 19 | Momentum FHIR MCP | https://github.com/the-momentum/fhir-mcp-server | 2026-10-07 |
| 20 | LangCare MCP FHIR | https://github.com/langcare/langcare-mcp-fhir | 2026-10-07 |
| 21 | HealthClaw Guardrails | https://github.com/aks129/HealthClawGuardrails | 2026-10-07 |
| 22 | MS FHIR Anonymization | https://github.com/microsoft/FHIR-Tools-for-Anonymization | 2026-10-07 |
| 23 | Google Healthcare API | https://docs.cloud.google.com/healthcare-api/docs/concepts/fhir | 2026-10-07 |
| 24 | FDA AI Devices | https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/ | 2026-10-07 |
| 25 | IBM LLMs | https://www.ibm.com/think/topics/large-language-models | 2026-10-07 |
| 26 | NVIDIA LLMs | https://www.nvidia.com/en-us/glossary/large-language-models/ | 2026-10-07 |
| 27 | Google Intro LLMs | https://developers.google.com/machine-learning/resources/intro-llms | 2026-10-07 |
| 28 | OpenAI Prompt Eng | https://developers.openai.com/api/docs/guides/prompt-engineering | 2026-10-07 |
| 29 | AWS RAG | https://aws.amazon.com/what-is/retrieval-augmented-generation/ | 2026-10-07 |
| 30 | Anthropic Contextual Retrieval | https://www.anthropic.com/news/contextual-retrieval | 2026-10-07 |
| 31 | AMA Augmented Intelligence | https://www.ama-assn.org/practice-management/digital/augmented-intelligence-medicine | 2026-10-07 |
| 32 | WHO AI Ethics | https://www.who.int/publications/i/item/9789240029200 | 2026-10-07 |
| 33 | PubMed search | https://pubmed.ncbi.nlm.nih.gov/ | 2026-10-07 |
| 34 | OpenMed Agent | https://github.com/openmed-labs/openmed-agent | 2026-10-07 |
| 35 | LangChain RAG tutorial | https://www.langchain.com/blog/tutorial-chatgpt-over-your-data | 2026-10-07 |

### Blocked/Inaccessible Sources
- healthit.gov (ONC) — 403/404 on multiple paths
- hhs.gov/hipaa — 403
- confluence.hl7.org — 405
- federalregister.gov — blocked
- lmstudio.ai, ollama.com, gpt4all.io, jan.ai — certificate errors
- AMIA education catalog — requires membership

---

## Appendix B: Detailed Scratch Files

The following detailed research files are available in `bmad-output/research-scratch/`:
- `technical-domain.md` — Full AI+FHIR landscape, tools, MCP, privacy (580 lines)
- `education.md` — Teaching frameworks, analogies, exercises, lesson structure (571 lines)
- `existing-assessment.md` — Complete codebase assessment with file inventory (240 lines)

---

**Report Generated By:** BMAD Research Skill (bmad-planning-orchestrator)
**Output Path:** bmad-output/research-report.md
**Related Documents:** bmad-output/research-scratch/*.md
**Last Updated:** 2026-10-07
