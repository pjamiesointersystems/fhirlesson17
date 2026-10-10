# Lesson 17: Using LLMs with FHIR

**Vanderbilt FHIR Training Course**
**Author:** Patrick W. Jamieson, M.D.

This lesson introduces Large Language Models (LLMs) in the context of healthcare data, using FHIR as the structured data layer. Students learn foundational AI concepts, build a working patient summarization pipeline, and explore prompt engineering — all using a local LLM to keep patient data private.

This lesson does **not** cover agentic AI, Model Context Protocol (MCP), or Retrieval-Augmented Generation (RAG). Those topics are taught in the downstream lessons listed below.

---

## Series Position

Lesson 17 is the **first lesson** in a four-part AI+FHIR series:

| # | Lesson | Focus |
|---|--------|-------|
| 1 | **Lesson 17** (this lesson) | AI foundations, hands-on exercises, safety framing |
| 2 | Agentic AI — Part One | Agent architecture, MCP, tool-use, structured validation, audit logging |
| 3 | Agentic AI — Part Two | FHIR as knowledge layer, subagents, human review checkpoints |
| 4 | RAG Lab | Vector stores, embeddings for retrieval, hybrid search, Graph RAG |

Lesson 17 establishes the vocabulary, mental models, and safety awareness that the subsequent lessons assume. After completing this lesson, students can explain what an LLM does, why raw FHIR bundles don't fit in a context window, and why validation is non-negotiable in clinical AI.

---

## Learning Objectives

After completing this lesson, students will be able to:

1. **Explain core LLM concepts** (tokens, embeddings, context windows, temperature, prompts, hallucination) using healthcare analogies
2. **Set up and interact with a local LLM** (Ollama or LM Studio) from Python
3. **Transform a FHIR patient bundle** into LLM-ready input through resource grouping and feature extraction
4. **Build a patient summarization pipeline** that fetches FHIR data, reduces it by ~99%, and produces a clinical narrative
5. **Apply prompt engineering techniques** (system roles, structured output, few-shot examples) and observe how they change LLM output
6. **Articulate AI safety and governance principles** including the "golden rule" of validation, human-in-the-loop, and PHI/HIPAA considerations

---

## Prerequisites

- **Prior FHIR lessons completed** — familiarity with Patient, Observation, Condition, MedicationRequest, and Bundle resources, as well as basic FHIRPath expressions
- **Python 3.11+** installed with virtual environment support (`venv`)
- **Local FHIR server** — InterSystems IRIS for Health running with Synthea patient data loaded (from prior lessons)
- **Basic command-line familiarity** (navigating directories, running Python scripts)

---

## Environment Setup

### 1. Install Ollama

Ollama runs LLMs locally on your machine. Install for your platform:

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download the installer from the Ollama website.

After installation, start the Ollama service:
```bash
ollama serve
```

> **LM Studio alternative:** If you prefer LM Studio, install it and start the local server with `lms start`. Set `LLM_PROVIDER=lmstudio` in your `.env` file.

### 2. Download a Model

Pull the default model used in the exercises:
```bash
ollama pull llama3.2:3b
```

This is a 3B-parameter model (~2 GB download) that runs on any modern laptop with 8 GB RAM. Students with more powerful hardware can try larger models (e.g., `llama3.1:8b`) by changing the `LLM_MODEL` setting in `.env`.

### 3. Create Python Virtual Environment

```bash
cd FHIRAI
python3 -m venv .venv
```

Activate the environment:

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```cmd
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment

Copy the example configuration file and edit it for your environment:
```bash
cp .env.example .env
```

Edit `.env` with your settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `FHIR_BASE_URL` | `http://127.0.0.1:8080/csp/healthshare/demo/fhir/r4` | Your FHIR R4 endpoint |
| `FHIR_USERNAME` | `_System` | FHIR server username |
| `FHIR_PASSWORD` | (none) | FHIR server password |
| `LLM_MODEL` | `llama3.2:3b` | Ollama model name |
| `LLM_PROVIDER` | `ollama` | `ollama` or `lmstudio` |

### 6. Verify Setup

Confirm your FHIR server is running and accessible:
```bash
curl http://127.0.0.1:8080/csp/healthshare/demo/fhir/r4/Patient/2
```

You should see a JSON Patient resource in the response.

---

## Phase 1: AI Foundations

Before the hands-on exercises, the instructor presents foundational AI concepts using healthcare analogies. Topics covered:

1. What is a Large Language Model?
2. Tokens and tokenization (pharmacist reading a prescription)
3. Context windows (clinician working memory)
4. Embeddings (clinical concept similarity)
5. Temperature and sampling (confident specialist vs. creative brainstormer)
6. Prompts and system roles (clinical protocols)
7. Hallucination (clinical confabulation)
8. Local vs. cloud models (local lab vs. reference lab)
9. Model selection basics (size vs. capability)

Full content: [`slides/ai_foundations.md`](slides/ai_foundations.md)

---

## Phase 2: Hands-On Exercises

### Exercise 1: LLM Hello World

**What it teaches:** Verifying your local LLM setup works and seeing your first LLM response from Python.

**Setup:** Ensure Ollama is running with the model loaded (`ollama serve` in a separate terminal).

**Run:**
```bash
python exercise1_llm_chat.py
```

**Expected output:** The LLM responds to a healthcare prompt explaining what an A1c test measures, in language suitable for a patient. You should see a multi-sentence response printed to the terminal.

**What to notice:** The system prompt sets the role ("clinical AI assistant"), and the user prompt asks a specific question. If you get a connection error, verify that Ollama is running and the model is loaded.

---

### Exercise 2: FHIR Bundle Exploration

**What it teaches:** How large a real FHIR patient bundle is, why raw JSON doesn't fit in a small model's context window, and what resource types are present.

**Setup:** Ensure your FHIR server is running with Synthea data loaded.

**Run:**
```bash
python exercise2_fhir_explore.py
```

To explore a different patient:
```bash
python exercise2_fhir_explore.py --patient 5
```

**Expected output:**
```
=== FHIR Bundle Statistics ===
Total characters:     310,482
Estimated tokens:     ~101,049
Total resources:      224

=== Resource Breakdown ===
ExplanationOfBenefit    72
Encounter               47
Observation             43
...

=== Context Window Analysis ===
This bundle is ~1,263% of an 8,000-token context window.
```

**What to notice:** A single patient's `$everything` bundle is far too large for a small model's context window. This motivates the resource grouping and feature extraction approach used in Exercise 3.

---

### Exercise 3: Patient Summarization Pipeline

**What it teaches:** The core technique of the lesson — transforming a large FHIR bundle into a concise clinical summary using a local LLM. This is a CLI pipeline (not a graphical interface).

**Setup:** Ensure both Ollama and the FHIR server are running.

**Run:**
```bash
python exercise3_summarizer.py
```

To summarize a different patient:
```bash
python exercise3_summarizer.py --patient 5
```

**Expected output:** The script runs through five phases:

1. **Fetch** — retrieves the Patient/$everything bundle from the FHIR server
2. **Group** — organizes resources by type (Condition, Observation, MedicationRequest, etc.)
3. **Extract** — pulls clinically relevant features from each resource type
4. **Summarize** — sends the extracted features to the LLM with a clinical system prompt
5. **Output** — prints the LLM's clinical narrative summary

You'll see progress messages for each phase, followed by the clinical summary. The script also reports token reduction (typically ~99% — from ~100K tokens to ~1K tokens).

**What to notice:** The "do not invent" instruction in the system prompt is a safety guardrail. The LLM should only report what's present in the extracted features. Compare the summary against the extracted features to verify accuracy.

**Discussion:** Could a clinician use this summary directly? What would need to change for clinical use? (Answer: human verification against source data — the "golden rule.")

---

### Exercise 4: Prompt Engineering Comparison

**What it teaches:** How different prompting strategies produce different LLM outputs from the same patient data, and how temperature affects output variability.

**Setup:** Same as Exercise 3 (Ollama + FHIR server running).

**Run:**
```bash
python exercise4_prompt_compare.py
```

**Expected output:** The script runs six LLM calls and displays results in labeled sections:

- **Strategy 1: Minimal Prompt** — generic "summarize this" with no structure or role
- **Strategy 2: Structured System Prompt** — clinical role, section format, "do not invent" constraint
- **Strategy 3: Few-Shot Prompt** — includes an example input/output pair to guide format
- **Temperature Comparison** — Strategy 2 run at temperatures 0.1, 0.7, and 1.0

Each section includes inline comments explaining what to notice.

**What to notice:**
- Strategy 1 typically produces an unstructured paragraph; Strategy 2 produces organized sections
- The few-shot example in Strategy 3 guides the format without explicit format instructions
- Temperature 0.1 produces focused, deterministic output; temperature 1.0 produces more variable output

**Discussion:** Which strategy would you choose for a clinical summarization tool? Why? Consider predictability, format consistency, and safety.

> **Experiment:** The prompt constants are defined at the top of the file. Try modifying them and re-running to see how your changes affect the output.

---

## Phase 3: Safety, Governance, and Series Bridge

### Safety and Governance Discussion

The lesson concludes with a discussion of AI safety principles that frame the entire four-part series:

- **Hallucination risks** — LLMs can invent medications, fabricate lab values, or conflate patients
- **The golden rule** — never trust LLM output without validating it against source FHIR data
- **Human-in-the-loop** — AI assists clinical decisions; clinicians make them
- **PHI and HIPAA** — why local models matter; the risk of sending patient data to cloud APIs without a Business Associate Agreement
- **Regulatory awareness** — ONC HTI-1 and FDA AI/ML as medical devices (awareness level)

These principles are not afterthoughts — they are design constraints that shape how production AI+FHIR systems are built. The subsequent lessons operationalize them with least-privilege access, structured validation, and audit logging.

Full content: [`slides/safety_governance.md`](slides/safety_governance.md)

### What Comes Next

- **Part One** teaches agentic AI architecture: agents that call FHIR APIs as tools, MCP for standardized LLM-to-server communication, structured output validation, and audit logging
- **Part Two** adds FHIR as a knowledge layer, subagent coordination, and human review checkpoints in multi-agent pipelines
- **RAG Lab** covers embedding patient data into vector stores, retrieval-augmented generation, and grounding LLM output with source documents

---

## Further Reading

- **Ollama** — local LLM runner, model library, and API reference
- **HL7 FHIR R4 Specification** — the data standard used throughout this series
- **InterSystems IRIS for Health** — the FHIR server platform used in the course
- **"Attention Is All You Need" (Vaswani et al., 2017)** — the original transformer architecture paper (optional, for deeper technical understanding)

---

## File Reference

| File | Purpose |
|------|---------|
| `exercise1_llm_chat.py` | Exercise 1: LLM connectivity hello world |
| `exercise2_fhir_explore.py` | Exercise 2: FHIR bundle size and token analysis |
| `exercise3_summarizer.py` | Exercise 3: Patient summarization CLI pipeline |
| `exercise4_prompt_compare.py` | Exercise 4: Prompt engineering comparison |
| `config.py` | Configuration loader (reads `.env`) |
| `fhir_client.py` | FHIR server client functions |
| `.env.example` | Configuration template |
| `slides/ai_foundations.md` | Phase 1 lecture content |
| `slides/safety_governance.md` | Phase 3 safety and governance content |
| `requirements.txt` | Python dependencies |
