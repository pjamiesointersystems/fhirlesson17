# Lesson 17 Existing Content Assessment

**Assessment Date:** October 7, 2026
**Original Lesson Created:** May 2025
**Assessor:** Automated analysis of codebase plus web research on current library/framework status

---

## 1. Inventory of Existing Files

### Core Files (referenced in Readme)
- `Readme.md` -- Lesson overview, learning objectives, getting-started instructions
- `fhirapp.py` -- Main Textual TUI app: fetches Patient/$everything, chunks JSON, hierarchical LLM summarization
- `getSearchPatients.py` -- FHIR REST client using requests, fhir.resources, and fhirpathpy
- `llamachat.py` -- Minimal LM Studio test script (lmstudio SDK "hello world")
- `requirements.txt` -- Dependency pins

### Additional Files (not documented in Readme)
- `FHIRSummaryApp_by_resource_type.py` -- Variation that summarizes by FHIR resource type, pulls from InterSystems IRIS vector table
- `FHIRSummaryRevised.py` -- Similar to above, uses LM Studio with Mistral, tighter prompt engineering with word/sentence limits
- `OpenAIFHIRSummary.py` -- Same architecture but uses OpenAI API (gpt-4o-mini for partials, gpt-4o for final)
- `fhirvectorflattened.py` -- Ingestion pipeline: fetches $everything, flattens FHIR resources to text, generates embeddings via SentenceTransformer (nomic-embed-text-v1.5), stores in IRIS with HNSW vector index
- `simplevector.py` -- Standalone in-memory vector search demo with hardcoded patient summaries and cosine similarity
- `testrag.py` -- Vector search utility against IRIS PatientVectors table
- `fhir_rag_chat.py` -- Interactive RAG chat TUI: embeds user query, retrieves from IRIS vector store, feeds context to LLM
- `testdriver.py` -- RAG query script (has a bug: references `self.conn` in a standalone function)
- `testpydriver.py` -- IRIS connection and row insert/retrieval test

### Key Observation
The Readme only describes the basic summarization flow (`fhirapp.py`). The more advanced RAG, vector store, and OpenAI integration files are undocumented and appear to be works-in-progress or instructor demo scripts. Several files have duplicated code and inconsistent naming.

---

## 2. Technology Stack Assessment

### 2.1 LM Studio Python SDK (`lmstudio`)

**Current Status:** Production-stable (v1.5.0 released August 2025; v1.6.0b1 pre-release October 2025). Supports Python 3.10-3.13.

**Assessment:**
- The SDK is still the official recommended approach for Python interaction with LM Studio.
- The existing code uses `lms.Client()` and `model.complete()` / `model.respond()` -- these remain valid API calls.
- The code hardcodes model names like `"llama-3.2-3b-instruct"` and `"mistral-7b-instruct-v0.3"`. These are fine as examples but the lesson should note that available models depend on what the user has downloaded in LM Studio. Students should be guided on model selection.
- **No changes needed** to the basic SDK usage pattern; it remains current.

### 2.2 Textual TUI Framework

**Current Status:** v8.2.8 (released June 30, 2026). Very actively maintained -- 37.4k GitHub stars, 13,100+ commits.

**Assessment:**
- Textual is thriving and remains an excellent choice for educational terminal demos.
- The `requirements.txt` pins `textual==0.44.1`, which is extremely outdated (the library is now at v8.x). This pin should be updated. There may be breaking API changes between 0.44 and 8.x that would require code adjustments.
- **Action needed:** Update the Textual version pin and verify existing TUI code against the current API. The composability model and widget names are likely similar but import paths or method signatures may have changed.

### 2.3 FHIR Python Libraries

#### `fhir.resources`
**Current Status:** Version 7.x+ is available, now powered by Pydantic v2, with FHIR R5 as the default and R4B as a sub-package.

**Assessment:**
- The `requirements.txt` pins `fhir.resources==6.1.0` with `pydantic<2.0`. This is significantly outdated.
- Version 7.0.0+ dropped standalone R4 support; R4B is available as a sub-package. Since the lesson uses R4 resources (`Patient`, `Observation`, `Bundle`), migration requires verifying that R4B sub-package imports work as replacements.
- The `pydantic<2.0` constraint is no longer necessary and actively harmful -- it prevents using current versions of many libraries.
- The code uses `Patient.parse_obj()` which is Pydantic v1 syntax. Pydantic v2 uses `model_validate()` instead (`parse_obj` may still work as a compatibility shim but should be updated).
- **Action needed:** Upgrade to fhir.resources 7.x, remove the pydantic<2.0 pin, update `parse_obj()` calls to `model_validate()`, and adjust imports for R4B sub-package.

#### `fhirpathpy`
**Current Status:** v2.2.4 (released August 2026). Production/Stable. Supports Python 3.10-3.14.

**Assessment:**
- Still actively maintained and a good choice.
- Current code usage (`from fhirpathpy import evaluate as fhirpath`) remains valid.
- **No changes needed** to the fhirpathpy usage.

### 2.4 Other Dependencies

| Dependency | Current Pin | Assessment |
|---|---|---|
| `requests` | `>=2.20.0,<3.0.0` | Fine, still current |
| `tiktoken` | unpinned | Still functional but see note below on tokenization |
| `sentence-transformers` | unpinned | Still current, actively maintained |
| `einops` | unpinned | Needed by some transformer models, fine |
| `intersystems-irispython` | unpinned | Vendor-specific, acceptable for InterSystems demos |

---

## 3. Architectural Assessment of the Core Approach

### 3.1 What Works Well

**The hierarchical chunking/summarization pattern (`fhirapp.py`) is pedagogically valuable:**
- It clearly demonstrates the fundamental problem of LLM context windows vs. real-world FHIR bundle sizes.
- The map-reduce summarization strategy (batch summaries -> partial summaries -> final summary) is a real pattern used in production systems.
- It teaches students about tokenization, chunking strategies, and the trade-offs of hierarchical summarization (information loss at each merge step).
- The Textual TUI provides immediate visual feedback with progress bars and live-updating summaries, making the process transparent.

**The FHIR data pipeline is sound:**
- Using `Patient/$everything` to retrieve a complete patient record is a realistic starting point.
- The `getSearchPatients.py` module demonstrates proper FHIR REST interactions with fhir.resources validation.
- FHIRPath evaluation for extracting resources from bundles is a useful technique.

**The multi-approach demonstration is valuable:**
- Showing both local (LM Studio) and hosted (OpenAI) approaches gives students perspective on deployment trade-offs.
- The progression from simple completion to RAG represents a natural learning arc.

### 3.2 What Is Outdated or Needs Improvement

**Tokenization approach:**
- Using `tiktoken.encoding_for_model("gpt-3.5-turbo")` for token counting when the actual model is a local Llama or Mistral model is technically incorrect. Different models use different tokenizers. GPT-3.5-turbo's cl100k_base encoding does not match Llama's or Mistral's tokenizer.
- For pedagogical purposes this is acceptable as an approximation, but the lesson should acknowledge the mismatch and explain why it matters (over- or under-estimating actual token usage).
- Better approach: use the model's own tokenizer, or explain the approximation explicitly.

**Hardcoded model names and connection strings:**
- FHIR server URLs, credentials, IRIS connection strings, and model names are scattered as hardcoded values across multiple files.
- Should use environment variables or a central config file.

**Code quality issues:**
- `getSearchPatients.py` defines `get_patient_from_server()` twice (lines 13-34 and 69-90).
- `testdriver.py` references `self.conn` in a standalone function (bug).
- Several files have commented-out code, unused imports, and inconsistent formatting.
- No error handling for LM Studio not running or model not loaded.

**The 10-chunk fixed limit in `fhirapp.py`:**
- The code hardcodes `chunks[:10]` and pre-creates 10 Markdown widgets. This is brittle -- patients with small records waste slots, patients with large records lose data.
- Should be dynamic based on actual content size.

### 3.3 Is Chunking/Summarization Still Pedagogically Relevant?

**Yes, but with important caveats:**

The chunking pattern remains relevant because:
- Many real-world FHIR bundles still exceed even 128K-200K context windows when serialized as JSON (especially for patients with long histories).
- The pattern teaches fundamental concepts about information processing pipelines.
- It illustrates the tension between context length, cost, latency, and accuracy.

However, the lesson should be updated to acknowledge:
- Modern models (GPT-4o, Claude, Gemini) have 128K-1M+ token context windows, which can fit most patient records without chunking.
- The field has largely moved toward "stuff it all in the context" for moderate-sized records, with chunking reserved for truly large datasets.
- The hierarchical merge strategy loses information at each step -- modern approaches often prefer a single pass with a large context window.
- The lesson's 1,500-token chunk size is very conservative by 2026 standards, even for small local models.

---

## 4. What Is Missing from the Current Lesson

### 4.1 RAG and Vector Stores (Partially Present but Undocumented)

The codebase actually contains RAG components (`fhirvectorflattened.py`, `testrag.py`, `fhir_rag_chat.py`, `simplevector.py`) but they are:
- Not referenced in the Readme
- Not structured as a learning progression
- Tightly coupled to InterSystems IRIS (which is appropriate for this course but limits portability)
- Missing conceptual explanation of why RAG matters for FHIR

**Recommendation:** These should be formalized into a proper lesson module with:
- Conceptual introduction to embeddings and vector similarity
- Why RAG is important for healthcare (grounding, reducing hallucination)
- Step-by-step walkthrough of the ingestion pipeline
- Interactive RAG chat demo

### 4.2 Agentic AI Patterns

**Completely absent.** This is a significant gap given the state of the field in 2026:
- Tool-use / function-calling where an LLM decides which FHIR queries to execute
- Multi-step reasoning (e.g., "Find this patient's latest A1c and compare to their medications")
- Agent loops that iteratively refine FHIR queries based on results
- Frameworks like LangGraph, CrewAI, or custom agent loops

### 4.3 MCP (Model Context Protocol)

**Completely absent.** MCP has become a standard approach for connecting LLMs to external data sources:
- A FHIR MCP server would allow any MCP-compatible client (Claude, etc.) to query FHIR resources
- This is highly relevant to the course's focus on interoperability
- Demonstrates a modern, standards-based approach to LLM-FHIR integration

### 4.4 Structured Output / FHIR Resource Generation

**Completely absent.** The current lesson only does FHIR-to-text (summarization). The reverse direction is equally important:
- Generating valid FHIR resources from natural language descriptions
- Using structured output modes (JSON mode, function calling) to ensure valid FHIR JSON
- Validating LLM-generated FHIR resources against profiles
- Clinical note -> FHIR Condition/Observation extraction

### 4.5 Prompt Engineering

**Minimal coverage.** The existing prompts are basic system-role instructions. Missing:
- Few-shot prompting with FHIR examples
- Chain-of-thought prompting for clinical reasoning
- Prompt templates and parameterization
- Discussion of prompt injection risks in healthcare
- Comparison of prompt strategies and their effects on output quality

### 4.6 Model Selection and Trade-offs

**Not discussed.** The code uses various models (Llama 3.2 3B, Mistral 7B, GPT-4o-mini, GPT-4o) but never explains:
- Why choose one model over another
- Size vs. capability trade-offs (3B vs 7B vs 70B vs hosted)
- Cost considerations for hosted APIs
- Quantization effects on clinical accuracy
- When local models are appropriate vs. when hosted models are needed
- Healthcare-specific model considerations (clinical vs. general models)

### 4.7 Safety and Governance

**Briefly mentioned in Readme but no code demonstrations:**
- Hallucination detection and mitigation
- Human-in-the-loop validation
- Audit logging of LLM interactions
- PHI/PII considerations when using hosted LLMs
- Regulatory considerations (FDA, 21st Century Cures Act implications)

---

## 5. Summary of Recommended Actions

### Must Fix (Blocking Issues)
1. **Update `requirements.txt`**: Remove `pydantic<2.0` pin, update `fhir.resources` to 7.x, update `textual` version pin.
2. **Fix Pydantic v1 syntax**: Replace `parse_obj()` with `model_validate()` across all files.
3. **Fix duplicate function definition** in `getSearchPatients.py`.
4. **Fix bug in `testdriver.py`** (`self.conn` reference in standalone function).

### Should Update (Important Improvements)
5. **Document the additional files** in the Readme or restructure as a progressive lesson.
6. **Externalize configuration** (FHIR server URLs, credentials, model names) to `.env` or config file.
7. **Add error handling** for LM Studio not running, model not loaded, FHIR server unreachable.
8. **Update tokenization discussion** to acknowledge the tiktoken/local-model mismatch.
9. **Make chunk count dynamic** rather than hardcoded to 10.

### Should Add (New Content for 2026 Relevance)
10. **Agentic AI with FHIR**: Tool-use patterns where LLM decides which FHIR queries to run.
11. **MCP server for FHIR**: Demonstrate Model Context Protocol integration.
12. **Structured output**: FHIR resource generation from natural language.
13. **Prompt engineering module**: Systematic coverage of prompting techniques for clinical data.
14. **Model selection guide**: When to use local vs. hosted, small vs. large models.
15. **Safety module**: Hallucination detection, PHI considerations, audit logging.

### Consider (Nice to Have)
16. **Replace the simple vector demo** (`simplevector.py`) with a more portable solution (e.g., ChromaDB or an in-process SQLite-based vector store) as an alternative to the IRIS-dependent version.
17. **Add notebook versions** of key demos for easier step-by-step instruction.
18. **Add unit tests** for the FHIR data processing functions.
