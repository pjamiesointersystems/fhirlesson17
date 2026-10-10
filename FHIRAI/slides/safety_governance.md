# AI Safety and Governance in Healthcare

**Lesson 17, Phase 3 — Vanderbilt FHIR Training Course**

> Safety is not a single lesson topic — it is a theme that runs through the entire
> AI+FHIR series. This content introduces the core principles; subsequent lessons
> operationalize them with concrete patterns and code.

---

## 1. Hallucination Risks in Clinical Contexts

**Hallucination** is when an LLM generates information that sounds authoritative but is fabricated. The model is not "lying" — it is producing text that fits learned patterns, regardless of whether the content is true.

### Why This Is Dangerous in Healthcare

In general-purpose applications, a hallucinated fact is an inconvenience. In healthcare, it can cause harm.

**Scenario 1 — Invented medication:** You run Exercise 3's summarizer on a patient's FHIR bundle. The summary states "Current Medications: Metformin 500mg twice daily, Lisinopril 10mg daily." But when you check the actual MedicationRequest resources, there is no Metformin — the LLM inferred it from the patient's diabetes diagnosis and fabricated the prescription. If a clinician acts on this summary without checking, a medication reconciliation error results.

**Scenario 2 — Fabricated lab value:** The LLM summary reports "Most recent A1c: 8.2%." But the patient's Observation resources contain no A1c result. The LLM filled a gap it expected to see with a plausible-sounding number. A clinician relying on this value could make inappropriate treatment decisions.

**Scenario 3 — Patient conflation:** In batch processing, an LLM processing multiple patients' data may blend facts from Patient A into Patient B's summary — particularly if context windows are reused without clearing.

### Mitigation

- Use **grounding**: provide the actual FHIR data and instruct the LLM to summarize only what's present
- Include **"do not invent" instructions** in system prompts (as demonstrated in Exercise 3)
- Always **verify LLM output against source data** before clinical use

> *Suggested visual:* Side-by-side comparison — LLM summary vs. actual FHIR resources, with fabricated items highlighted in red.

---

## 2. PHI and HIPAA Considerations

### The Core Risk

When you send patient data to a cloud LLM API (OpenAI, Anthropic, Google), the data leaves your control and traverses third-party infrastructure. Under HIPAA, this constitutes disclosure of Protected Health Information (PHI) to a business associate.

**Concrete scenario:** A well-meaning developer copies a patient's FHIR Patient/$everything bundle — 300,000 characters of clinical data including names, dates of birth, diagnoses, and medications — and pastes it into ChatGPT to generate a summary. This is a HIPAA violation. The data is now on OpenAI's servers, potentially logged, and outside the covered entity's control.

### Why Local Models Matter

Local models (Ollama, LM Studio) process data entirely on-premises:

| Aspect | Local Model | Cloud API |
|--------|------------|-----------|
| Data stays on your machine | Yes | No |
| Business Associate Agreement needed | No | Yes |
| Risk of external data exposure | None | Present |
| Suitable for real PHI | Yes (with proper controls) | Only with BAA in place |

**This is why our exercises use Ollama** — even though we use synthetic Synthea data, we practice the pattern that would be required with real PHI.

### When Cloud APIs Are Appropriate

Cloud LLMs are not off-limits in healthcare. They can be used when:
- A BAA is executed with the API provider (e.g., Azure OpenAI Service offers BAA-backed endpoints)
- Data is de-identified per HIPAA Safe Harbor or Expert Determination standards
- The API provides HIPAA-compliant data handling guarantees

The key principle: **know where your data goes and what protections are in place.**

> *Suggested visual:* Data flow diagram — FHIR server → local LLM (stays on-premises) vs. FHIR server → cloud API (crosses network boundary, BAA checkpoint required).

---

## 3. Human-in-the-Loop

### AI as Decision Support, Not Decision Maker

LLMs should **assist** clinical decision-making, not replace it. The pattern is:

1. LLM processes data and generates a suggestion or summary
2. Clinician reviews the output against source data
3. Clinician makes the clinical decision

**Scenario:** Exercise 3 generates a structured patient summary from FHIR data. A clinician reviewing this summary should treat it as a draft — a starting point that saves time reading raw records, not a final clinical document. The clinician verifies key facts (active conditions, current medications, recent critical values) against the FHIR source before using the summary to inform care.

### Why This Matters

- LLMs cannot assess clinical context that isn't in the data (patient's demeanor, family dynamics, socioeconomic factors)
- LLMs cannot take responsibility for clinical outcomes
- Regulatory frameworks (see Section 5) increasingly require human oversight of AI-assisted decisions

**The analogy:** A resident presents a case summary to an attending. The attending doesn't sign off without reviewing the chart. LLM output deserves the same scrutiny.

---

## 4. The Golden Rule: Never Trust LLM Output Without Validation

> **If you remember one thing from this lesson about AI safety, remember this:
> never use LLM output in a clinical context without verifying it against the source data.**

This is the single most important safety principle for the AI+FHIR series.

### Validation Strategies

**Cross-reference against source FHIR data:**
The LLM summary says "Active Conditions: Type 2 Diabetes, Hypertension, Obesity." Check the actual Condition resources in the FHIR bundle — are all three present with `clinicalStatus: active`? Are any conditions missing from the summary?

**Structured output validation:**
If the LLM is asked to produce JSON (e.g., a structured clinical summary), validate the output against a schema. Does it contain the required fields? Are values in expected formats? This can be automated — the LLM generates, the system validates.

**Human spot-checking:**
For critical outputs, a clinician reviews key facts. This doesn't require reading the entire output — focused verification of medications, allergies, and active diagnoses catches the most dangerous errors.

**Reproducibility testing:**
Run the same input through the LLM twice. Significantly different outputs (especially factual differences, not just phrasing) suggest hallucination or instability. Lower temperature (see Exercise 4) increases reproducibility.

### Connecting to the Exercises

- Exercise 3 uses a "do not invent" instruction — this is a prompt-level mitigation
- Exercise 4 shows how temperature affects output stability — lower temperature = more reproducible = easier to validate
- The extracted features from Exercise 3 are themselves a form of validation: the LLM only sees structured data that was explicitly extracted from FHIR resources

---

## 5. Regulatory Awareness

> **Note:** This section provides awareness-level coverage. Students should know these
> frameworks exist and understand why they matter. Deep compliance analysis is beyond
> Lesson 17's scope.

### ONC Health IT Certification (HTI-1)

The Office of the National Coordinator for Health IT (ONC) finalized the HTI-1 rule, which includes provisions for:

- **Decision support interventions**: Requirements for transparency about how AI-based clinical decision support tools work and what data they use
- **Predictive models**: Disclosure requirements for algorithms used in certified health IT, including source data, performance metrics, and known limitations

**Why it matters for FHIR practitioners:** If you build an AI+FHIR application that is used as part of certified health IT, these transparency and disclosure requirements may apply. Understanding this landscape helps you build applications that can meet regulatory expectations.

### FDA and AI/ML as Medical Devices

The FDA regulates AI/ML-based Software as a Medical Device (SaMD):

- If an AI tool is intended for **diagnosis, treatment, or prevention of disease**, it may be regulated as a medical device
- The FDA has approved hundreds of AI/ML-enabled medical devices and maintains a public list
- A patient summarization tool used for convenience (our exercises) is different from a diagnostic tool used for clinical decisions — but the line can be blurry

**Why it matters for FHIR practitioners:** Not every AI+FHIR application requires FDA clearance, but understanding when the threshold is crossed helps you design systems responsibly. If your application moves from "informational summary" to "clinical recommendation," regulatory considerations change.

### The Practical Takeaway

When building AI+FHIR applications:
1. **Know your intent** — is this informational or clinical?
2. **Know your data flow** — does PHI leave your infrastructure?
3. **Know your audience** — will a clinician make decisions based on this output?
4. **Know your obligations** — which regulatory frameworks apply?

These questions don't need legal answers at this stage. They need engineering awareness — the same kind of awareness that leads you to validate FHIR resources against profiles or enforce authentication on API endpoints.

---

## Safety as a Series Theme

The principles introduced here are operationalized throughout the AI+FHIR series:

| Principle | Where It Deepens |
|-----------|-----------------|
| Hallucination mitigation | **Part One**: structured validation patterns for agent outputs |
| PHI protection | **Part One**: least-privilege FHIR access scoping for agents |
| Human-in-the-loop | **Part Two**: human review checkpoints in multi-agent pipelines |
| Output validation | **Part One**: JSON schema validation of agent tool calls |
| Audit and accountability | **Part One**: audit logging of all agent actions and FHIR queries |
| Grounding with source data | **RAG Lab**: retrieval-augmented generation anchors LLM output to source documents |

These are not afterthoughts — they are design constraints that shape how production AI+FHIR systems are built. The exercises in Lesson 17 introduce the techniques (grounding, "do not invent" prompts, temperature control); the subsequent lessons show how to embed them in larger architectures.

> *Key message to students:* Safety in healthcare AI is not about avoiding AI — it's about
> using it with the same rigor you apply to any clinical tool. Validate, verify, and keep
> the human in the loop.
