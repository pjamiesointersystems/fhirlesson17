"""
Generate speaker notes for the Lesson 17 slide deck.
Timed for a 50-minute recorded presentation (Camtasia capture).
Tone: conversational, engaging, instructor-to-student.
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

doc = Document()

style = doc.styles["Normal"]
font = style.font
font.name = "Calibri"
font.size = Pt(11)

# ── Title ───────────────────────────────────────────────────────────────
doc.add_paragraph()
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("Speaker Notes\nUsing LLMs with FHIR — Lesson 17")
run.bold = True
run.font.size = Pt(24)
run.font.color.rgb = RGBColor(0, 51, 102)

doc.add_paragraph()
meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.add_run("Vanderbilt FHIR Training Course\n").font.size = Pt(13)
meta.add_run("Patrick W. Jamieson, M.D.\n").font.size = Pt(13)
meta.add_run("Total runtime target: 50 minutes\n").font.size = Pt(12)
run = meta.add_run("Recording format: Camtasia screen capture for asynchronous learners")
run.font.size = Pt(12)
run.italic = True

doc.add_paragraph()

# Timing note
box = doc.add_paragraph()
run = box.add_run("Recording Tips: ")
run.bold = True
run.font.size = Pt(11)
run.font.color.rgb = RGBColor(0, 102, 153)
box.add_run(
    "Speak at a measured pace — async learners can speed up playback but can't slow down "
    "a rushed delivery. Pause briefly between sections to create natural edit points for Camtasia. "
    "When running live demos, narrate what you're doing ('I'm switching to the terminal now...') "
    "since screen transitions are harder to follow in a recording. Keep energy up — you're the "
    "only voice they'll hear for 50 minutes."
)

doc.add_page_break()

# ── Timing Overview ─────────────────────────────────────────────────────
doc.add_heading("Timing Overview", level=1)

timing = [
    ("Slides 1–4", "Opening & Context", "3 min"),
    ("Slides 5–14", "Phase 1: AI Foundations", "15 min"),
    ("Slides 15–21", "Phase 2: Exercises (with live demos)", "20 min"),
    ("Slides 22–31", "Phase 3: Safety, Governance & Bridge", "10 min"),
    ("Slides 32–34", "Takeaways, Thank You & Close", "2 min"),
]
table = doc.add_table(rows=len(timing) + 1, cols=3)
table.style = "Light Grid Accent 1"
for i, h in enumerate(["Slides", "Section", "Time"]):
    table.rows[0].cells[i].text = h
    for p in table.rows[0].cells[i].paragraphs:
        for r in p.runs:
            r.bold = True
for i, (slides, section, time) in enumerate(timing, 1):
    table.rows[i].cells[0].text = slides
    table.rows[i].cells[1].text = section
    table.rows[i].cells[2].text = time

doc.add_paragraph()
p = doc.add_paragraph()
run = p.add_run("Note: ")
run.bold = True
p.add_run(
    "The exercise demos in Phase 2 are the heart of the lesson and get the most time. "
    "If running long, compress the Phase 1 foundations by spending less on embeddings and "
    "model selection (concepts students will revisit in later lessons)."
)

doc.add_page_break()

# ── Helper ──────────────────────────────────────────────────────────────
def slide_notes(slide_num, title, duration, notes_text, transitions=None, demo=None):
    """Write notes for one slide."""
    h = doc.add_heading(f"Slide {slide_num}: {title}", level=2)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0, 51, 102)

    p = doc.add_paragraph()
    run = p.add_run(f"Duration: {duration}")
    run.bold = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(120, 120, 120)

    doc.add_paragraph()

    for para in notes_text.strip().split("\n\n"):
        doc.add_paragraph(para.strip())

    if demo:
        doc.add_paragraph()
        p = doc.add_paragraph()
        run = p.add_run("LIVE DEMO: ")
        run.bold = True
        run.font.color.rgb = RGBColor(40, 120, 60)
        p.add_run(demo)

    if transitions:
        doc.add_paragraph()
        p = doc.add_paragraph()
        run = p.add_run("Transition: ")
        run.bold = True
        run.font.color.rgb = RGBColor(0, 102, 153)
        run.italic = True
        run2 = p.add_run(transitions)
        run2.italic = True

    # Light separator
    doc.add_paragraph("─" * 70)


# ═══════════════════════════════════════════════════════════════════════
# OPENING (Slides 1–4) — ~3 minutes
# ═══════════════════════════════════════════════════════════════════════

doc.add_heading("Opening & Context", level=1)

slide_notes(1, "Title Slide", "30 sec", """
Welcome to Lesson 17 — Using Large Language Models with FHIR. I'm Dr. Jamieson, and this is the lesson where your FHIR knowledge meets AI.

If you've followed the course so far, you know FHIR inside and out — Patient resources, Observations, Bundles, FHIRPath. Today we're going to take all of that structured clinical data and put it to work with something completely new: a large language model running right here on your machine.

No cloud services, no API keys, no sending patient data anywhere. Everything stays local.""",
    transitions="This lesson is actually the start of something bigger — let me show you where it fits.")

slide_notes(2, "The AI+FHIR Series", "45 sec", """
This is lesson one of four. I want you to see this as a journey, not a standalone topic.

Today we build the foundation — you'll understand what an LLM is, you'll build a working summarization pipeline, and you'll learn why safety matters before you write a single line of agent code.

In Part One, we take that pipeline and turn it into an agent that can call FHIR APIs on its own through something called the Model Context Protocol. In Part Two, we orchestrate multiple agents working together. And in the RAG Lab, we build retrieval systems that ground the LLM's output in actual patient data.

Everything we cover today comes back. Nothing is throwaway.""",
    transitions="Let me show you exactly what you'll be able to do by the end of this session.")

slide_notes(3, "Learning Objectives", "45 sec", """
Six objectives — and they're practical, not theoretical.

By the time we're done, you'll be able to explain these AI concepts to a colleague using healthcare analogies they already understand. You'll have Ollama running locally and you'll have called it from Python. You'll have taken a real FHIR patient bundle — hundreds of thousands of characters — and reduced it to about a thousand tokens of clinically relevant features. You'll have built a complete summarization pipeline. You'll have compared different prompting strategies side by side. And you'll be able to articulate why you can't just trust what an LLM tells you.

That last one — validation — is the single most important idea in this entire series.""",
    transitions="Here's how we'll get there — three phases.")

slide_notes(4, "Lesson Structure: Three Phases", "45 sec", """
Three phases, each building on the last.

Phase 1 is the conceptual foundation — nine AI concepts, each connected to a healthcare analogy you already understand. This is the vocabulary you need before touching code.

Phase 2 is where it gets hands-on. Four exercises, each one building on the previous. We start with a simple hello-world to verify your setup, explore how large a FHIR bundle actually is, build a real summarization pipeline, and then compare different prompt engineering strategies.

Phase 3 is where we step back and talk about safety, governance, and how everything connects to the rest of the series. This is the part that separates responsible healthcare AI from reckless experimentation.

Let's dive into Phase 1.""")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: AI Foundations (Slides 5–14) — ~15 minutes
# ═══════════════════════════════════════════════════════════════════════

doc.add_page_break()
doc.add_heading("Phase 1: AI Foundations", level=1)

slide_notes(5, "Phase 1 Divider", "10 sec", """
Phase 1 — AI Foundations. Nine concepts, each grounded in healthcare. No prior AI knowledge required.""",
    transitions="Let's start with the most basic question.")

slide_notes(6, "What Is a Large Language Model?", "2 min", """
So what actually is a large language model?

At its core, it's a software system that was trained on an enormous amount of text — books, articles, websites, clinical notes — and it learned patterns in that text. When you give it a prompt, it generates a response one word at a time, picking each word based on the patterns it learned.

Here's the analogy I want you to hold onto: imagine a clinician who has read every medical textbook ever published, every journal article, every clinical note in every EHR — but has never actually seen a patient. When you present a case, they can compose a very articulate response drawn from all that reading. But the knowledge is indirect. It came from text, not from examining a patient.

That's the LLM's superpower and its fundamental limitation. Vast knowledge, no grounding.

Now, here's what matters for us as FHIR practitioners. The LLM does not understand FHIR resources the way a FHIR server does. A FHIR server has a schema, it validates data, it enforces referential integrity. The LLM has none of that — it only has patterns in language. When we combine LLMs with FHIR, we are providing the structured data that the LLM lacks on its own.""",
    transitions="But before the LLM can process any text, it has to break it into pieces. That's tokenization.")

slide_notes(7, "Tokens and Tokenization", "1.5 min", """
Before an LLM can read anything, it chops the input into tokens — small pieces the model can process. A token is usually a word or a piece of a word.

Look at this example on screen: "The patient's hemoglobin A1c was 7.2 percent." To us, that's a straightforward sentence. But the tokenizer breaks it into 13 separate tokens. Notice that "hemoglobin" gets split into "hem" and "oglobin" — the model doesn't see whole words the way we do.

The healthcare analogy: think of a pharmacist reading a prescription. They don't read it as a block of text. They parse it into units — drug name, dosage, route, frequency. Each unit is processed separately. Same idea.

Now, why does this matter for FHIR? Because a Patient/$everything bundle — all the data for one patient — can be tens of thousands of tokens when serialized as JSON. And all those curly braces, field names, and URL references? They consume tokens without adding any clinical meaning. That's token-inefficient. Exercise 2 is going to make this very concrete with real numbers.""",
    transitions="And those tokens have to fit somewhere — that's the context window.")

slide_notes(8, "Context Windows", "2 min", """
The context window is the maximum number of tokens the LLM can handle in one interaction. Both your input — the prompt plus the data — and the output have to fit inside this window.

Look at the sizes on screen. A small local model like the one we're using has roughly an 8,000-token window. A large cloud model like GPT-4 or Claude can handle over a million tokens. But bigger windows come with trade-offs: cost, latency, and the fact that models pay less attention to information buried in the middle of very long inputs.

The analogy: a physician reviewing a complex patient chart can't hold the entire record in their head. If there are 20 years of encounters and hundreds of lab results, they focus on what's relevant — recent visits, active problems, current medications. The context window is the LLM's version of that cognitive limit.

Now look at the right side of this slide. A typical Synthea patient bundle is about 101,000 tokens. That's over 1,200 percent of an 8,000-token window. The data is 12 times too large. You simply cannot paste raw FHIR JSON into a small model.

That's the entire motivation for Exercise 3 — we're going to extract just the clinically relevant features and get that 101,000 tokens down to about 1,000. A 99 percent reduction.""",
    transitions="Before we move on, let me briefly introduce embeddings.")

slide_notes(9, "Embeddings — Meaning as Numbers", "1 min", """
An embedding converts text into a list of numbers — a vector — that captures its meaning. The key insight is that similar concepts end up with similar vectors.

On screen you can see: "hypertension" and "high blood pressure" are close together in embedding space, while "hypertension" and "appendectomy" are far apart. Just like how you as a clinician know that "MI," "myocardial infarction," and "heart attack" all mean the same thing — embeddings give computers that same ability.

I'm keeping this brief intentionally. Embeddings are the foundation of Retrieval-Augmented Generation — RAG — which is covered in depth in the RAG Lab, the fourth lesson in our series. For today, just understand the concept: meaning as numbers, similar meaning means similar numbers.""",
    transitions="Now let's talk about a parameter you'll directly experiment with — temperature.")

slide_notes(10, "Temperature and Sampling", "1.5 min", """
When the LLM generates text, it doesn't just pick the single most likely next word. It assigns probabilities to all possible next words and samples from that distribution. Temperature controls how that sampling works.

Low temperature — 0.1 — is like a confident specialist following established guidelines. Nearly deterministic. Picks the highest-probability word almost every time. Very focused, very predictable.

Medium — 0.7 — is your experienced clinician. Generally follows the evidence but occasionally exercises judgment. This is the default for most applications.

High temperature — 1.0 and above — is your creative brainstormer throwing out differential diagnoses. Casts a wide net, generates diverse ideas, but not all of them are good.

For clinical summarization — what we're doing in our exercises — lower temperature is safer. You want the LLM to report what's in the data faithfully, not get creative with it. Exercise 4 will let you see this difference with your own eyes.""",
    transitions="Temperature controls how the model responds. But prompts control what it responds to.")

slide_notes(11, "Prompts and System Roles", "1.5 min", """
Modern LLMs work with two types of prompts. The system prompt is like standing orders — it sets the LLM's role, its behavior rules, its output format. It persists across the conversation. The user prompt is the specific question or data for this interaction.

The healthcare analogy is clinical protocols. A system prompt is like a hospital policy: "Always verify allergies before prescribing," or "Use structured SOAP format for notes." These shape every interaction without being restated each time. The user prompt is the specific clinical question: "What are this patient's active conditions?"

Prompt engineering — the practice of crafting these prompts to get better output — is a real skill. In Exercise 4, you'll compare four techniques: role specification, output format instructions, safety constraints, and few-shot examples where you show the LLM an example before giving it real data.

The prompts are defined as constants at the top of the exercise file. I want you to read them, modify them, re-run the script, and see what changes.""",
    transitions="And now the concept that connects directly to our safety discussion — hallucination.")

slide_notes(12, "Hallucination", "1.5 min", """
Hallucination is when the LLM generates information that sounds authoritative, reads perfectly, and is completely fabricated. The model isn't lying — it's producing text that fits patterns it learned, regardless of whether the content is true.

The healthcare analogy is clinical confabulation — a neurological phenomenon where a patient with memory impairment fills gaps with fabricated memories, presented with full confidence. They genuinely believe what they're saying. Same thing here.

Why is this dangerous in healthcare AI? Look at these examples on the left side. The LLM could invent a medication the patient never took — it sees diabetes and confidently adds Metformin even though no MedicationRequest exists. It could fabricate a lab value — "A1c: 8.2 percent" — that sounds perfectly plausible but corresponds to no actual Observation. It could even conflate two patients' data in batch processing.

On the right side — mitigation strategies. Grounding the LLM with actual FHIR data. Including "do not invent" instructions in your prompts. Keeping a human in the loop. And eventually, using RAG to anchor output in source documents.

We're going to use that "do not invent" instruction in Exercise 3. You'll see it in action.""",
    transitions="Now — where should that LLM actually run?")

slide_notes(13, "Local vs. Cloud Models", "1 min", """
Two options: run the model locally on your machine, or call a cloud API.

Local models — like what we're using with Ollama — keep everything on your machine. Data never leaves. No Business Associate Agreement needed. Free after the initial hardware investment. The trade-off is capability: you're limited to smaller models, typically 3 to 13 billion parameters.

Cloud models — GPT-4, Claude, Gemini — offer state-of-the-art capability. But your data crosses network boundaries and lands on third-party servers. For patient data, that means you need a BAA in place or you're looking at a HIPAA violation.

The analogy: local lab versus reference lab. Your hospital's local lab gives you fast results and the specimen never leaves the building. A reference lab can run exotic tests, but you have to ship the specimen out. Choose based on what you need.

For this course, we use Ollama. Even though our data is synthetic Synthea data, we practice the pattern you'd use with real PHI.""",
    transitions="One more concept — how do you choose which model to use?")

slide_notes(14, "Model Selection Basics", "1 min", """
Models come in a range of sizes, measured roughly by parameter count.

Small models — 1 to 3 billion parameters — run on any modern laptop with 8 gigs of RAM. Fast, great for extraction and summarization, which is exactly what our exercises do. Limited reasoning ability for complex tasks.

Medium models — 7 to 13 billion — offer a nice balance. Need 16 gigs of RAM or more.

Large models need serious hardware or a GPU. Cloud-scale models are cloud-only.

For this course, we default to llama3.2:3b — small enough for any laptop, capable enough for our exercises. If you have a more powerful machine, you can try larger models by changing the LLM_MODEL setting in your .env file.

That wraps up our nine foundational concepts. Let's put them to work.""")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Exercises (Slides 15–21) — ~20 minutes
# ═══════════════════════════════════════════════════════════════════════

doc.add_page_break()
doc.add_heading("Phase 2: Hands-On Exercises", level=1)

slide_notes(15, "Phase 2 Divider", "10 sec", """
Phase 2 — Hands-On Exercises. Four exercises that build from a simple hello-world all the way to a complete FHIR patient summarization pipeline.""",
    transitions="Let's start with the simplest possible test — can you talk to an LLM from Python?")

slide_notes(16, "Exercise 1: LLM Hello World", "3 min", """
Exercise 1 is your connectivity smoke test. We're verifying that Ollama is running, the model is loaded, and Python can talk to it.

The script sends two things to the LLM: a system prompt that says "You are a clinical AI assistant," and a user prompt that asks it to explain what an A1c test measures in language suitable for a patient.

Let me run this live so you can see what happens.""",
    demo=(
        "Switch to terminal. Run: python exercise1_llm_chat.py\n"
        "Narrate: 'I'm running the script now. It's sending the prompt to our local llama3.2 model... "
        "and here's the response. Notice it explains A1c in plain language — that's because the system "
        "prompt told it to act as a clinical AI assistant communicating with patients.'\n\n"
        "Point out: 'If you see a ConnectionError, that means Ollama isn't running. Start it with "
        "ollama serve in another terminal window. The first call after loading the model is always "
        "slower — it's loading the model into memory.'"
    ),
    transitions="Good — the plumbing works. Now let's look at actual FHIR data and understand the size problem.")

slide_notes(17, "Exercise 2: FHIR Bundle Exploration", "4 min", """
This exercise answers the question: how large is a real patient's data, and can it fit in a context window?

We're calling Patient/$everything for patient 2 on our FHIR server — the same operation you've used in prior lessons. But now, instead of looking at the clinical content, we're looking at the size.

Let me run it.""",
    demo=(
        "Switch to terminal. Run: python exercise2_fhir_explore.py\n"
        "Narrate the output as it appears:\n"
        "'310,000 characters. About 101,000 tokens. 224 resources in this bundle.\n"
        "And look at that context window analysis — this bundle is 1,263 percent of an 8,000-token "
        "window. It's 12 times too large.\n\n"
        "Look at the resource breakdown — ExplanationOfBenefit, Encounter, Observation... most of these "
        "ExplanationOfBenefit resources are insurance claims data. Clinically, how relevant is that for "
        "a patient summary? Probably not very. But they're consuming a huge number of tokens.\n\n"
        "So we have a problem. We can't paste this into an LLM. What are our options?'\n\n"
        "Pause and engage: 'Option 1: use a bigger model with a bigger window — expensive, cloud, "
        "privacy concerns. Option 2: extract and reduce — keep only what matters clinically. That's "
        "what we're about to do. Option 3: use retrieval to find relevant parts first — that's the "
        "RAG Lab.'"
    ),
    transitions="Let's solve this problem. Exercise 3 is the core of the entire lesson.")

slide_notes(18, "Exercise 3: Patient Summarization Pipeline", "5 min", """
This is the exercise everything has been building toward. We're going to take that 101,000-token bundle and turn it into a clinical narrative summary.

The pipeline has five phases — you can see them on screen. Fetch the bundle. Group resources by type — instead of a flat list of 224 resources, we organize them into Conditions, Observations, MedicationRequests, and so on. Extract clinically relevant features from each group — not the full JSON, just the fields that matter: condition names with onset dates, lab values with units, medication names with dosages. Send those extracted features — now about 1,000 tokens — to the LLM with a clinical system prompt. And print the result.

Let me run it.""",
    demo=(
        "Switch to terminal. Run: python exercise3_summarizer.py\n"
        "Narrate each phase as it prints:\n"
        "'Phase 1 — fetching the bundle from our FHIR server... got it.\n"
        "Phase 2 — grouping by resource type... look at that breakdown.\n"
        "Phase 3 — extracting features... here's what the LLM will actually see. Patient demographics, "
        "active conditions, recent observations with values and units, medications with dosages.\n"
        "Notice the token count — we went from about 101,000 to about 1,000 tokens. That's a 99 percent "
        "reduction.\n"
        "Phase 4 — sending to the LLM now...\n"
        "And Phase 5 — here's our clinical summary.'\n\n"
        "Read through the summary briefly. Point out how it's organized into sections. Then say: "
        "'Now — is this summary perfect? Does it match the extracted features exactly? Let's look...'"
    ),
    transitions="Before we move to the next exercise, there's a critical safety detail in this pipeline.")

slide_notes(19, "Exercise 3: The 'Do Not Invent' Instruction", "3 min", """
Look at the system prompt from our summarizer on screen. The last line says: "Do not invent or assume information not present in the data."

Why is that there? Without it, the LLM might see a diabetes diagnosis and helpfully add "Current Medications: Metformin 500mg twice daily" — even if no Metformin MedicationRequest exists in the FHIR bundle. It's filling in what it expects to see, based on its training data. That's a hallucination, and in a clinical context, it's dangerous.

Here's the discussion question I want you to think about: Could a clinician use the summary we just generated directly in patient care?

The answer is no. Not without checking it against the source data. The LLM might have gotten it right — probably did, mostly — but "probably mostly right" isn't good enough for clinical decisions. That brings us to the golden rule, which we'll cover in Phase 3.

This "do not invent" instruction is a prompt-level guardrail. It helps, but it's not foolproof. Real safety requires validation, which is why the rest of the series teaches structured output validation, audit logging, and human review checkpoints.""",
    transitions="Now let's see what happens when we change the prompt. Same data, different instructions, very different results.")

slide_notes(20, "Exercise 4: Prompt Engineering Comparison", "3 min", """
Exercise 4 takes the same patient data and runs it through three different prompting strategies.

Strategy 1 is the naive approach: "You are a helpful assistant. Summarize this data." No clinical role, no format, no safety instructions.

Strategy 2 is what we used in Exercise 3: clinical role, section format, "do not invent" constraint.

Strategy 3 is few-shot prompting: we give the LLM an example of what we want — a sample input and a sample output — before giving it the real data.

Let me run this so you can see the difference.""",
    demo=(
        "Switch to terminal. Run: python exercise4_prompt_compare.py\n"
        "Narrate: 'This makes six LLM calls — three strategies plus three temperature variants — "
        "so it'll take a minute or two.\n\n"
        "Here's Strategy 1... see how it's just a wall of text? No structure, no consistent format. "
        "It might be accurate, might not — hard to verify without structure.\n\n"
        "Now Strategy 2... look at the difference. Organized sections: Demographics, Active Conditions, "
        "Medications. Much easier to verify against the source data.\n\n"
        "Strategy 3 with the few-shot example... notice how it mimics the format of the example we gave it, "
        "even though we didn't explicitly say \"use this format.\" It learned from the example.\n\n"
        "Which would you want in a clinical tool? Strategy 2 — predictable, verifiable, safe.'"
    ),
    transitions="Now look at what temperature does to the same prompt.")

slide_notes(21, "Exercise 4: Temperature Comparison", "2 min", """
Same prompt — Strategy 2 — same data. Only the temperature changes.

At 0.1, the output is focused and deterministic. Run it twice, you'll get nearly the same thing. This is what you want for clinical reporting.

At 0.7, the default, you get some variation in phrasing but the facts stay consistent.

At 1.0, things get interesting. The model starts making more creative word choices. The phrasing changes significantly between runs. In clinical contexts, that variability is a risk — it makes the output harder to verify and more likely to include unexpected content.

The takeaway is straightforward: for anything clinical, use low temperature. Reproducibility means verifiability, and verifiability means safety.

I want to point out that the prompt constants — all six of them — are defined right at the top of the exercise file. I encourage you to open that file, change a prompt, and re-run the script. That's where the real learning happens — when you experiment on your own.""",
    transitions="We've built the skills. Now let's talk about using them responsibly.")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Safety & Bridge (Slides 22–31) — ~10 minutes
# ═══════════════════════════════════════════════════════════════════════

doc.add_page_break()
doc.add_heading("Phase 3: Safety, Governance & Series Bridge", level=1)

slide_notes(22, "Phase 3 Divider", "10 sec", """
Phase 3 — Safety, Governance, and the Series Bridge. This is where we step back from the code and talk about what responsible healthcare AI looks like.""",
    transitions="If you remember one thing from this entire lesson, remember this next slide.")

slide_notes(23, "The Golden Rule", "2 min", """
Here it is. The single most important principle in this series.

Never use LLM output in a clinical context without verifying it against the source data.

Say it again in your head. Write it down. This is the golden rule of healthcare AI.

Four validation strategies. First: cross-reference against the source FHIR data. The LLM says three active conditions? Check the actual Condition resources. Are all three there? Are any missing?

Second: structured output validation. If you ask the LLM to produce JSON, validate it against a schema. Does it have the required fields? Are values in the expected format? This can be automated.

Third: human spot-checking. A clinician reviews the key facts — medications, allergies, active diagnoses. You don't have to read every word. Focused verification of the most dangerous categories catches the most critical errors.

Fourth: reproducibility testing. Run the same input twice. If you get significantly different factual content — not just different phrasing — that's a red flag for hallucination.""",
    transitions="Let me show you exactly what hallucination looks like in clinical contexts.")

slide_notes(24, "Hallucination Risks in Clinical Contexts", "1.5 min", """
Three concrete scenarios.

First: invented medication. The LLM summary says "Metformin 500mg twice daily." But when you look at the actual MedicationRequest resources in the FHIR bundle, there is no Metformin. The LLM saw a diabetes diagnosis and inferred a prescription that doesn't exist. If a clinician acts on this during medication reconciliation, that's a real error.

Second: fabricated lab value. The summary reports "Most recent A1c: 8.2 percent." But there is no A1c Observation in the patient's data. The LLM filled a gap it expected to see with a number that sounds perfectly plausible. A treatment decision based on a non-existent lab value — that's harmful.

Third: patient conflation. In batch processing, the LLM blends data from Patient A into Patient B's summary. This happens when context windows are reused without being cleared between patients.

Every one of these can cause patient harm. And every one of them is preventable with validation.""",
    transitions="Now let's talk about where the data goes — because that matters just as much as what the AI does with it.")

slide_notes(25, "PHI and HIPAA Considerations", "1 min", """
Here's a scenario that has actually happened. A well-meaning developer copies a patient's entire FHIR bundle — names, dates of birth, diagnoses, medications, 300,000 characters of clinical data — and pastes it into ChatGPT to generate a summary.

That's a HIPAA violation. Full stop. That data is now on OpenAI's servers, potentially logged, and outside the covered entity's control.

This is why we use Ollama in our exercises. Data never leaves your machine. No Business Associate Agreement required. Even though we're using synthetic Synthea data today, we are practicing the pattern you would use with real patient data.

Cloud APIs are not off-limits — but they require a BAA with the provider. Azure OpenAI Service, for example, offers BAA-backed endpoints. The key principle: always know where your data goes and what protections are in place.""",
    transitions="And even with the right data protections, there's one more principle — keeping the human in the loop.")

slide_notes(26, "Human-in-the-Loop", "1 min", """
Three-step pattern. The LLM processes data and generates a summary or suggestion. The clinician reviews that output against the source data and their clinical judgment. The clinician makes the decision.

The LLM is decision support, not decision maker. Just like a resident who presents a case summary — the attending doesn't sign off without reviewing the chart themselves.

Why? Because LLMs can't assess context that isn't in the data. The patient's demeanor, family dynamics, socioeconomic factors — none of that is in a FHIR bundle. LLMs can't take responsibility for clinical outcomes. And regulatory frameworks are increasingly requiring human oversight of AI-assisted decisions.""",
    transitions="Speaking of regulatory frameworks — let me give you a quick awareness-level overview.")

slide_notes(27, "Regulatory Awareness", "1 min", """
Two frameworks to know about — at awareness level, not compliance depth.

ONC's HTI-1 rule includes transparency and disclosure requirements for AI-based clinical decision support in certified health IT. If you build an AI+FHIR application that's used in a certified system, these requirements may apply.

The FDA regulates AI and machine learning tools that are intended for diagnosis, treatment, or prevention of disease as medical devices. They've approved hundreds of AI-enabled medical devices. The line between "informational summary" and "clinical recommendation" can be blurry — and where you fall on that line changes your regulatory obligations.

Four practical questions to ask when building any AI+FHIR application: What's the intent — informational or clinical? Where does the data flow — does PHI leave your infrastructure? Who's the audience — will clinicians make decisions from this? And what are your obligations — which frameworks apply?

You don't need legal answers today. You need engineering awareness.""",
    transitions="Let me show you some real-world examples of where organizations are doing this today.")

slide_notes(28, "Real-World Use Cases", "45 sec", """
Quick tour of how LLMs are being used with healthcare data in the real world.

Patient record summarization — what we built in Exercise 3 — is being used by Mayo Clinic and others to assist diagnostic decision-making. Clinical note generation — Ambience, Abridge, Amazon HealthScribe are generating SOAP notes from conversation transcripts. Natural language to FHIR query — converting plain English like "Find all diabetics with A1c over 8" into actual FHIR search parameters. Structured data extraction — pulling medications, conditions, and labs from free-text notes and transforming them into FHIR resources. Clinical trial matching and predictive risk analysis round out the landscape.

These aren't research prototypes. These are production systems. And every one of them needs the safety principles we just discussed.""",
    transitions="Now let me show you where everything we learned today is headed.")

slide_notes(29, "Where These Concepts Scale Up", "1 min", """
Five connections between today and the rest of the series.

Your summarization pipeline from Exercise 3 becomes a subagent in Part One's multi-agent architecture. Your feature extraction feeds the RAG Lab's embedding and vector pipeline. Your prompt engineering skills apply directly to agent prompts in Parts One and Two. Your manual FHIR calls using the requests library get replaced by MCP — the Model Context Protocol — which standardizes how LLMs connect to FHIR servers. And the safety principles we just covered are operationalized as least-privilege access, structured validation, and audit logging.

The pattern is consistent: learn the concept, build it manually, then see how it operates at scale.""",
    transitions="Let me give you a quick preview of MCP specifically, since it's a big part of what comes next.")

slide_notes(30, "Preview: Model Context Protocol (MCP)", "1 min", """
In today's exercises, you are the intermediary. You wrote Python code that calls the FHIR server, parses the response, extracts features, and feeds them to the LLM. That's manual mode.

In Parts One and Two, the LLM connects to the FHIR server through MCP — the Model Context Protocol. An MCP server exposes FHIR operations as tools the LLM can call on its own. The LLM decides which tool to use based on the user's question.

Why do you need manual mode first? Because when that MCP-connected agent makes an unexpected FHIR query or returns surprising results, you need to understand FHIR endpoints, resource types, and query parameters well enough to debug it. Today gives you that foundation.""",
    transitions="One more slide on safety, then we'll wrap up.")

slide_notes(31, "Safety as a Series-Long Theme", "45 sec", """
Safety is not a checkbox on one lesson. It's a design constraint throughout.

Hallucination mitigation becomes structured validation patterns in Part One. PHI protection becomes least-privilege FHIR access scoping. Human-in-the-loop becomes formal review checkpoints in Part Two's multi-agent pipelines. Output validation becomes JSON schema validation of every agent tool call. Audit and accountability becomes comprehensive logging. And grounding with source data becomes full retrieval-augmented generation in the RAG Lab.

The message: safety in healthcare AI isn't about avoiding AI. It's about using it with the same rigor you apply to any clinical tool. Validate, verify, and keep the human in the loop.""",
    transitions="Let's bring it all together.")


# ═══════════════════════════════════════════════════════════════════════
# CLOSE (Slides 32–34) — ~2 minutes
# ═══════════════════════════════════════════════════════════════════════

doc.add_page_break()
doc.add_heading("Closing", level=1)

slide_notes(32, "Key Takeaways", "45 sec", """
Six takeaways.

LLMs are powerful healthcare tools — and they require responsible use. Both things are true.

Raw FHIR bundles don't fit in context windows. Structured extraction — what you built today — solves that problem.

Prompt engineering is a practical skill with direct safety implications. You saw this in Exercise 4.

The golden rule: never trust LLM output without validating it against the source data. If you remember nothing else, remember this.

Local models protect PHI. Cloud models require BAAs and careful data handling.

And everything you learned today — every concept, every technique — reappears in Parts One, Two, and the RAG Lab. This is your foundation.""")

slide_notes(33, "Thank You", "30 sec", """
Thank you for your time and engagement with this material. This is genuinely one of the most exciting intersections in health IT right now — the combination of structured clinical data standards like FHIR with the capabilities of large language models.

You've built real skills today. You can set up a local LLM, you can transform FHIR data into LLM-ready input, you can build a summarization pipeline, and you understand why safety isn't optional.""",
    transitions="If you're watching this recording and have questions, please post them on the training site.")

slide_notes(34, "Questions & Discussion", "30 sec", """
Next in the series is Agentic AI — Part One. We'll take the pipeline you built today and turn it into an autonomous agent that can query FHIR servers on its own through the Model Context Protocol.

If you're watching this asynchronously, feel free to post questions on the training site. I read every post and will respond.

Thank you, and I'll see you in Part One.""")


# ═══════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════
output_path = os.path.join(os.path.dirname(__file__),
                           "Lesson17_Speaker_Notes.docx")
doc.save(output_path)
print(f"Saved: {output_path}")
