# AI Foundations for FHIR Practitioners

**Lesson 17, Phase 1 — Vanderbilt FHIR Training Course**

> This content assumes FHIR knowledge (resources, bundles, FHIRPath, FHIR servers)
> but no prior AI/ML background. Each concept is grounded in a healthcare analogy.

---

## 1. What Is a Large Language Model?

A Large Language Model (LLM) is a software system trained on vast amounts of text to predict what comes next in a sequence. Given a prompt, it generates a response word by word, selecting each word based on patterns learned during training.

**Healthcare analogy:** Think of an LLM as a clinician who has read every textbook, every journal article, and every clinical note ever written — but has never seen an actual patient. When you present a case, the clinician draws on this enormous body of reading to compose a response. The knowledge is broad but indirect: learned from text, not from lived clinical experience. This is both the LLM's strength (vast knowledge) and its limitation (no grounding in the specific patient in front of you).

**Key points for FHIR practitioners:**
- LLMs do not "understand" FHIR resources the way a FHIR server does — they process text
- An LLM has no database, no schema validation, no referential integrity — only patterns in language
- When we combine LLMs with FHIR, we provide the structured data that the LLM lacks

> *Suggested visual:* Diagram showing training data (books, journals, notes) flowing into a model, then a prompt going in and a response coming out.

---

## 2. Tokens and Tokenization

Before an LLM can process text, it must break the input into **tokens** — the fundamental units the model reads. A token is typically a word or subword fragment. The sentence "The patient's hemoglobin A1c was 7.2%" might tokenize as: `["The", " patient", "'s", " hem", "oglobin", " A", "1", "c", " was", " 7", ".", "2", "%"]`.

**Healthcare analogy — pharmacist reading a prescription:** A pharmacist doesn't read a prescription as a single block of text. They parse it into discrete units: drug name, dosage, route, frequency. Each unit is processed separately. Similarly, an LLM parses input into tokens and processes each one in sequence. Token boundaries don't always match word boundaries — "unresponsive" might become `["un", "responsive"]`, just as a pharmacist might parse "acetaminophen/hydrocodone" into its two components.

**Why this matters for FHIR:**
- A FHIR Patient/$everything bundle can be tens of thousands of tokens when serialized as JSON
- Token count determines whether your data fits in the LLM's processing window
- Raw FHIR JSON is token-inefficient: curly braces, quotes, field names, and URLs consume tokens without adding clinical meaning
- Exercise 2 demonstrates this concretely with real token counts

> *Note:* Different models use different tokenizers. The `tiktoken` library (used in our exercises) provides an approximation. Actual token counts may vary ~20% across models — this is itself a teaching moment about the diversity of LLM implementations.

---

## 3. Context Windows

The **context window** is the maximum number of tokens an LLM can process in a single interaction — both the input (your prompt + data) and the output (the LLM's response) must fit within this limit.

| Model Type | Typical Context Window |
|------------|----------------------|
| Small local models (3B params) | 4K–8K tokens |
| Medium local models (7B params) | 8K–32K tokens |
| Large cloud models (GPT-4, Claude) | 128K–1M+ tokens |

**Healthcare analogy — clinician working memory:** A physician reviewing a complex patient chart cannot hold the entire record in mind simultaneously. If a patient has 20 years of encounters, hundreds of lab results, and dozens of conditions, the clinician must focus on a subset — recent visits, active problems, current medications. The context window is the LLM's equivalent of this cognitive limit. Exceeding it means data is simply ignored or lost.

**Why this matters for FHIR:**
- A typical Synthea patient bundle (~100K tokens as raw JSON) far exceeds a small model's 8K window
- This is why Exercise 3 groups resources by type and extracts clinical features — reducing 100K tokens to ~1K tokens (99% reduction)
- Larger context windows don't eliminate the need for processing; they shift the trade-off toward cost, latency, and attention quality

> *Suggested visual:* A "sliding window" over a long patient timeline, showing what fits inside vs. what's cut off.

---

## 4. Embeddings

An **embedding** is a list of numbers (a vector) that represents the meaning of a word, sentence, or document. Words with similar meanings have similar vectors — "hypertension" and "high blood pressure" are close together in embedding space, while "hypertension" and "appendectomy" are far apart.

**Healthcare analogy — clinical concept similarity:** Clinicians intuitively know that "MI" and "myocardial infarction" and "heart attack" refer to the same concept, even though the words are different. Embeddings give computers this same ability: they map text to numeric representations where semantic similarity is preserved. This is how systems can find "related" patient records or match a natural-language question to relevant clinical data.

**Key point:** Embeddings are the foundation of Retrieval-Augmented Generation (RAG), where a system finds relevant documents by comparing embedding similarity before sending them to an LLM. We introduce the concept here; the **RAG Lab** covers embeddings, vector stores, and similarity search in depth.

> *Forward reference:* The RAG Lab builds on this concept with hands-on exercises: embedding patient summaries, storing vectors in IRIS, and building a retrieval pipeline.

---

## 5. Temperature and Sampling

When an LLM generates a response, it doesn't just pick the single most likely next word. Instead, it assigns probabilities to all possible next words and **samples** from this distribution. The **temperature** parameter controls how this sampling works:

| Temperature | Behavior | Clinical Analogy |
|------------|----------|-----------------|
| Low (0.1) | Almost always picks the highest-probability word. Deterministic, focused, repetitive. | **Confident specialist** following established guidelines — predictable, reliable, by-the-book. |
| Medium (0.7) | Balanced. Mostly picks likely words but occasionally explores alternatives. | **Experienced clinician** — generally follows evidence but exercises clinical judgment. |
| High (1.0+) | Samples more uniformly. Creative, diverse, sometimes surprising or incoherent. | **Creative brainstormer** generating differential diagnoses — casts a wide net, not all ideas are good. |

**For clinical summarization,** lower temperature (0.1–0.3) is typically safer — you want the LLM to faithfully report what's in the data, not get creative. For generating hypotheses or exploring possibilities, higher temperature may be useful.

Exercise 4 demonstrates this directly: the same patient data summarized at temperatures 0.1, 0.7, and 1.0 produces visibly different outputs.

---

## 6. Prompts and System Roles

A **prompt** is the instruction you give the LLM. Most modern LLMs support two types of prompts:

- **System prompt:** Sets the LLM's role, behavior rules, and output format. Persists across the conversation. Think of it as the LLM's "standing orders."
- **User prompt:** The specific question or data for this interaction.

**Healthcare analogy — clinical protocols:** A system prompt is like a hospital policy or clinical protocol: "Always verify allergies before prescribing," "Use structured SOAP format for notes," "Flag critical values immediately." These protocols shape every clinician-patient interaction without being restated each time. The user prompt is the specific clinical question: "What are this patient's active conditions?"

**Prompt engineering** is the practice of crafting prompts to get better LLM outputs. Key techniques:

1. **Role specification:** "You are a clinical AI assistant" (not just "You are helpful")
2. **Output format:** "Summarize into sections: Demographics, Conditions, Medications..."
3. **Safety constraints:** "Do not invent information not present in the data"
4. **Few-shot examples:** Providing an example input/output pair before the real data

Exercise 4 compares these techniques side-by-side, showing how each affects output quality.

> *Forward reference:* Parts One and Two of the agentic AI lessons show how prompt engineering applies to agent prompts — system prompts that control multi-step reasoning and tool-use behavior.

---

## 7. Hallucination

**Hallucination** occurs when an LLM generates information that is plausible-sounding but factually incorrect or entirely fabricated. The LLM doesn't "know" it's wrong — it produces text that fits the patterns it learned, regardless of truth.

**Healthcare analogy — clinical confabulation:** In neurology, confabulation is when a patient with memory impairment fills gaps with fabricated memories, presented with full confidence. The patient isn't lying — they genuinely believe the fabricated memory. LLM hallucination works similarly: the model fills gaps in its knowledge with plausible-sounding but invented information, presented with the same confident tone as accurate information.

**Why hallucination is critical in healthcare AI:**
- An LLM might invent a medication the patient never took
- It might fabricate a lab result or diagnosis not present in the data
- It might cite a clinical guideline that doesn't exist
- The confident tone makes hallucinations hard to detect without verification

**Mitigation strategies** (introduced here, expanded in the series):
- **Grounding:** Provide the actual data (FHIR resources) and instruct the LLM to summarize only what's present
- **"Do not invent" instructions:** Explicit system prompt constraints (used in Exercise 3)
- **Human-in-the-loop:** AI assists but a clinician validates
- **RAG:** Retrieve relevant source documents before generation (covered in RAG Lab)

> *Forward reference:* Part One introduces structured validation patterns; Part Two covers safety patterns including least privilege, audit logging, and output verification.

---

## 8. Local vs. Cloud Models

LLMs can run locally on your machine or be accessed via cloud APIs. Each approach has distinct trade-offs:

| Factor | Local Models | Cloud Models |
|--------|-------------|-------------|
| **Privacy** | Data never leaves your machine | Data sent to third-party servers |
| **HIPAA** | No BAA needed — PHI stays local | Requires BAA with provider |
| **Capability** | Limited by your hardware (3B–13B params typical) | Access to largest models (100B+ params) |
| **Cost** | Free after hardware investment | Pay per token |
| **Latency** | Depends on hardware; no network round-trip | Network latency + processing queue |
| **Availability** | Always available (your machine) | Depends on service uptime |

**Healthcare analogy — local lab vs. reference lab:** A hospital's local lab provides fast turnaround and keeps specimens on-site (privacy). But it can only run common tests. A reference lab (Quest, LabCorp) offers specialized, high-complexity tests but requires shipping specimens off-site, takes longer, and costs more. The choice depends on what you need: routine CBC → local lab. Rare genetic panel → reference lab.

**For this course:** We use **Ollama** to run models locally (with LM Studio as an alternative). This ensures:
- No PHI leaves the student's machine (even though we use synthetic Synthea data)
- No API keys, billing, or network dependencies for exercises
- Students understand the privacy-first approach to healthcare AI

> *Forward reference:* Parts One and Two discuss MCP (Model Context Protocol) as a standardized way for LLMs to connect to FHIR servers and other data sources, whether the LLM is local or cloud-hosted.

---

## 9. Model Selection Basics

LLMs come in many sizes, measured roughly by parameter count:

| Size | Examples | Strengths | Limitations |
|------|----------|-----------|------------|
| **Small (1B–3B)** | Llama 3.2 1B/3B, Phi-3 Mini | Fast, runs on laptops, 8GB RAM | Limited reasoning, shorter outputs |
| **Medium (7B–13B)** | Llama 3.1 8B, Mistral 7B | Good balance of quality and speed, 16GB RAM | May struggle with complex clinical reasoning |
| **Large (30B–70B)** | Llama 3.1 70B, Mixtral 8x7B | Strong reasoning, nuanced outputs | Requires 32–64GB+ RAM or GPU |
| **Cloud-scale (100B+)** | GPT-4, Claude, Gemini | State-of-the-art capability | Cloud-only, cost per token, privacy concerns |

**How to choose:**
- **Simple extraction and summarization** (our exercises): A 3B model is sufficient. Speed matters more than depth.
- **Clinical reasoning and differential diagnosis:** Needs a larger model (13B+) or cloud API.
- **Production deployment with PHI:** Local models with appropriate hardware, or cloud with BAA.

**For this course:** We default to `llama3.2:3b` via Ollama — small enough to run on any modern laptop, capable enough for the summarization and prompt engineering exercises. Students with more powerful hardware can experiment with larger models by changing the `LLM_MODEL` setting in their `.env` file.

---

## Summary: Connecting to the Series

These nine concepts form the vocabulary and mental models for the entire AI+FHIR series:

| Concept | Where It Deepens |
|---------|-----------------|
| Tokens & context windows | Exercise 2 (concrete numbers), Exercise 3 (structured extraction) |
| Prompt engineering | Exercise 4 (strategy comparison), Parts One/Two (agent prompts) |
| Hallucination & safety | Exercise 3 ("do not invent"), Parts One/Two (safety patterns) |
| Embeddings | RAG Lab (vector stores, similarity search) |
| Local vs. cloud | Parts One/Two (MCP, deployment patterns) |
| Temperature & sampling | Exercise 4 (temperature comparison) |

With these foundations in place, you're ready for the hands-on exercises — and for the agentic AI and RAG lessons that follow.
