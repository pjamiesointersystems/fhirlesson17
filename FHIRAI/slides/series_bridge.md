# From Foundations to Production: The AI+FHIR Series

**Lesson 17, Phase 3 — Vanderbilt FHIR Training Course**

> You've built the foundation. This section shows where each concept reappears
> and how it scales up in the lessons ahead.

---

## The Series at a Glance

| Lesson | What You Learn | What You Build |
|--------|---------------|----------------|
| **Lesson 17** (this lesson) | LLM concepts, prompt engineering, safety principles | Patient summarization CLI pipeline |
| **Part One** | Agent architecture, MCP, tool-use, structured validation | FHIR-connected AI agent with audit logging |
| **Part Two** | FHIR as knowledge layer, subagent coordination, human review | Multi-agent clinical workflow |
| **RAG Lab** | Embeddings, vector stores, retrieval-augmented generation | Retrieval pipeline grounded in patient data |

Each lesson builds directly on the one before it. The concepts and code patterns from Lesson 17 are not replaced — they are embedded into larger systems.

---

## 1. Your Summarization Pipeline Becomes a Subagent

**In Lesson 17 (Exercise 3),** you built a linear pipeline:

```
FHIR Server → fetch bundle → group by type → extract features → LLM → clinical summary
```

This is a single-purpose workflow: one patient, one summary, one output. You wrote every step explicitly in Python.

**In Parts One and Two,** this same pipeline becomes a **subagent** — a modular component within a multi-agent architecture. Instead of a standalone script, the summarizer is one agent among several (others might normalize terminology, generate differential diagnoses, or flag missing data). An orchestrating agent decides when to invoke the summarizer, what patient data to pass it, and how to combine its output with results from other agents.

**The key insight:** Lesson 17 teaches the "inner loop" — the core logic of fetching, extracting, and summarizing. Parts One and Two teach the "outer loop" — how agents coordinate, share context, and enforce safety constraints across a multi-step workflow. You need to understand the inner loop before you can reason about the outer loop.

> *Suggested visual:* Lesson 17's linear pipeline on the left; Part One's multi-agent architecture on the right, with the summarizer pipeline highlighted as one component within the larger system.

---

## 2. Your Structured Extraction Feeds the RAG Pipeline

**In Lesson 17 (Exercise 3),** you transformed a raw FHIR bundle (~100K tokens of JSON) into structured, clinically relevant text (~1K tokens). Resource-type grouping and field extraction produced clean data: patient demographics, active conditions, recent observations, current medications.

**In the RAG Lab,** this same structured extraction is the **input** to the retrieval pipeline. The Lab teaches how to:

- Convert structured clinical text into **embeddings** (numeric vectors that capture meaning)
- Store those embeddings in a **vector database** with efficient similarity indexes
- Retrieve the most relevant patient data for a given clinical question using **semantic search**

Raw FHIR JSON is poor input for embeddings — curly braces, field names, and URLs add noise without clinical meaning. The structured extraction you learned in Exercise 3 ensures that embeddings capture what matters: diagnoses, lab values, medications, and clinical events.

**The key insight:** Exercise 3's extraction is "RAG prep." You can't build an effective retrieval pipeline without first transforming FHIR data into meaningful text. Lesson 17 teaches that transformation; the RAG Lab teaches what comes after it.

> *Suggested visual:* Exercise 3's extraction output flowing into the RAG Lab's embedding → vector store → retrieval → generation pipeline.

---

## 3. Your Prompt Engineering Skills Apply to Agent Prompts

**In Lesson 17 (Exercise 4),** you experimented with prompting strategies:

- **System roles** that set the LLM's behavior ("You are a clinical AI assistant")
- **Structured output instructions** that control format ("Summarize into sections: Demographics, Conditions, Medications")
- **Few-shot examples** that guide style without explicit rules
- **Temperature** that controls creativity vs. determinism

**In Parts One and Two,** these same techniques apply to **agent prompts** — the instructions that control how AI agents behave within larger systems. Agent prompts are more complex (they include context from other agents' outputs, tool-use instructions, and safety constraints), but the core principles are identical:

- Clear role specification produces predictable behavior
- Structured output makes agent results parseable by downstream components
- Safety constraints ("do not invent") prevent hallucination in multi-step workflows

**The key insight:** Prompt engineering is not a one-lesson skill — it is the primary interface between human intent and AI behavior in every application you build. Exercise 4 teaches the fundamentals; Parts One and Two show how to apply them when the stakes and complexity increase.

---

## 4. From Manual FHIR Calls to MCP

**In Lesson 17,** you interact with the FHIR server manually. Your Python code explicitly constructs URLs, sends HTTP requests via `requests`, and parses JSON responses. You decide which endpoint to call, what parameters to include, and how to handle errors.

**In Parts One and Two,** this manual interaction is replaced by **MCP (Model Context Protocol)** — a standardized way for LLMs to connect to external systems like FHIR servers. With MCP:

- An **MCP server** exposes FHIR operations as **tools** (e.g., "fetch patient", "search observations by code", "get medication list")
- The **LLM decides** which tool to call based on the user's question or the current task
- The MCP server **executes the FHIR query** and returns structured results to the LLM
- The entire interaction follows a **standard protocol**, so the same LLM can connect to any MCP-compatible server

Think of it this way: in Lesson 17, you are the intermediary between the LLM and the FHIR server. With MCP, the LLM connects to the FHIR server directly (through the standardized protocol), and you design the tools and safety constraints.

**The key insight:** You need "manual mode" understanding before "autonomous mode" makes sense. When an MCP-connected agent makes an unexpected FHIR query, you need to understand FHIR endpoints, resource types, and query parameters to debug it. Lesson 17 gives you that understanding.

> *Suggested visual:* Two diagrams side by side — Lesson 17: "You → Python code → FHIR Server → You → LLM"; Parts One/Two: "User → LLM → MCP Server → FHIR Server → MCP Server → LLM → User".

---

## Bringing It Together

| Lesson 17 Concept | Where It Scales Up | Why the Foundation Matters |
|---|---|---|
| Summarization pipeline (Ex. 3) | Part One: subagent in multi-agent system | You must understand the component before orchestrating many |
| Resource-type grouping & extraction (Ex. 3) | RAG Lab: input to embedding and vector pipeline | Clean extraction produces meaningful embeddings |
| Prompt engineering (Ex. 4) | Parts One/Two: agent prompts with tool-use and safety | Same principles, higher complexity and stakes |
| Manual FHIR access (`requests`) | Parts One/Two: MCP-mediated tool calls | Manual understanding enables debugging autonomous systems |
| Safety principles (Phase 3) | Every lesson: least privilege, validation, audit logging | Safety is a design constraint, not an afterthought |

The pattern across the series is consistent: **learn the concept, build it manually, then see how it operates at scale in a production-like architecture.** Lesson 17 is where every concept starts.
