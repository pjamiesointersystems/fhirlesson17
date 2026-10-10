"""
Generate the Lesson 17 slide deck: Using LLMs with FHIR.
Replaces the original 'Using LLMS with FHIR.pptx' with updated content
covering the restructured three-phase lesson with exercises.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# ── Color Palette ───────────────────────────────────────────────────────
DARK_BLUE = RGBColor(0, 51, 102)       # titles
MED_BLUE = RGBColor(0, 102, 153)       # subtitles, accents
LIGHT_BLUE = RGBColor(200, 220, 240)   # light backgrounds
WHITE = RGBColor(255, 255, 255)
BLACK = RGBColor(0, 0, 0)
DARK_GRAY = RGBColor(60, 60, 60)
MED_GRAY = RGBColor(120, 120, 120)
RED_ACCENT = RGBColor(180, 40, 40)     # safety/warning
GREEN_ACCENT = RGBColor(40, 120, 60)   # positive/local
GOLD = RGBColor(180, 140, 20)          # golden rule highlight
BG_DARK = RGBColor(0, 40, 80)          # dark slide backgrounds
PHASE_1_COLOR = RGBColor(0, 102, 153)
PHASE_2_COLOR = RGBColor(40, 120, 60)
PHASE_3_COLOR = RGBColor(180, 40, 40)


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


# ── Helper Functions ────────────────────────────────────────────────────

def add_background(slide, color):
    """Set solid background color for a slide."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, left, top, width, height, text, font_size=18,
                bold=False, italic=False, color=BLACK, alignment=PP_ALIGN.LEFT,
                font_name="Calibri"):
    """Add a text box to a slide."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.italic = italic
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_bullet_slide(slide, left, top, width, height, bullets, font_size=18,
                     color=BLACK, spacing=Pt(6), font_name="Calibri", bold_prefix=False):
    """Add a text box with bullet points."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_after = spacing
        p.font.name = font_name

        if bold_prefix and ": " in bullet:
            prefix, rest = bullet.split(": ", 1)
            run1 = p.add_run()
            run1.text = prefix + ": "
            run1.font.size = Pt(font_size)
            run1.font.bold = True
            run1.font.color.rgb = color
            run1.font.name = font_name
            run2 = p.add_run()
            run2.text = rest
            run2.font.size = Pt(font_size)
            run2.font.color.rgb = color
            run2.font.name = font_name
        else:
            p.text = bullet
            p.font.size = Pt(font_size)
            p.font.color.rgb = color

        p.level = 0
        # Add bullet character
        pPr = p._pPr
        if pPr is None:
            from pptx.oxml.ns import qn
            pPr = p._p.get_or_add_pPr()

    return txBox


def add_two_column_bullets(slide, left, top, width, height,
                           col1_title, col1_bullets, col2_title, col2_bullets,
                           font_size=16, title_color=DARK_BLUE):
    """Add two columns of content side by side."""
    col_width = width // 2 - Inches(0.25)

    # Column 1 title
    add_textbox(slide, left, top, col_width, Inches(0.5),
                col1_title, font_size=20, bold=True, color=title_color)
    add_bullet_slide(slide, left, top + Inches(0.5), col_width, height - Inches(0.5),
                     col1_bullets, font_size=font_size)

    # Column 2 title
    col2_left = left + width // 2 + Inches(0.25)
    add_textbox(slide, col2_left, top, col_width, Inches(0.5),
                col2_title, font_size=20, bold=True, color=title_color)
    add_bullet_slide(slide, col2_left, top + Inches(0.5), col_width, height - Inches(0.5),
                     col2_bullets, font_size=font_size)


def title_slide(title_text, subtitle_text, bg_color=BG_DARK, title_color=WHITE,
                sub_color=LIGHT_BLUE):
    """Create a title/section slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    add_background(slide, bg_color)
    add_textbox(slide, Inches(1), Inches(2), Inches(11), Inches(1.5),
                title_text, font_size=40, bold=True, color=title_color,
                alignment=PP_ALIGN.CENTER)
    add_textbox(slide, Inches(1), Inches(3.7), Inches(11), Inches(1),
                subtitle_text, font_size=22, color=sub_color,
                alignment=PP_ALIGN.CENTER)
    return slide


def content_slide(title_text, title_color=DARK_BLUE):
    """Create a content slide with a title and return it for adding content."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    add_background(slide, WHITE)
    # Title bar
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(1.1))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(240, 244, 248)
    shape.line.fill.background()
    add_textbox(slide, Inches(0.7), Inches(0.15), Inches(11.5), Inches(0.8),
                title_text, font_size=32, bold=True, color=title_color)
    return slide


def phase_divider(phase_num, phase_title, description, color):
    """Create a phase divider slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    add_background(slide, color)
    add_textbox(slide, Inches(1), Inches(1.5), Inches(11), Inches(1),
                f"PHASE {phase_num}", font_size=24, bold=True, color=WHITE,
                alignment=PP_ALIGN.CENTER)
    add_textbox(slide, Inches(1), Inches(2.5), Inches(11), Inches(1.5),
                phase_title, font_size=44, bold=True, color=WHITE,
                alignment=PP_ALIGN.CENTER)
    add_textbox(slide, Inches(2), Inches(4.5), Inches(9), Inches(1.5),
                description, font_size=20, color=RGBColor(220, 230, 240),
                alignment=PP_ALIGN.CENTER)
    return slide


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title Slide
# ═══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, BG_DARK)
add_textbox(slide, Inches(1), Inches(1.2), Inches(11), Inches(1.8),
            "Using Large Language Models\nwith FHIR",
            font_size=44, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)
add_textbox(slide, Inches(1), Inches(3.3), Inches(11), Inches(0.6),
            "Concepts, Exercises & Safety Foundations",
            font_size=24, color=LIGHT_BLUE, alignment=PP_ALIGN.CENTER)
add_textbox(slide, Inches(1), Inches(4.5), Inches(11), Inches(1.5),
            "Patrick W. Jamieson, M.D.\n"
            "Vanderbilt FHIR Training Course — Lesson 17\n"
            "Lesson 1 of 4 in the AI+FHIR Series",
            font_size=18, color=RGBColor(180, 200, 220), alignment=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 2 — Series Context
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("The AI+FHIR Series")
add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.5), Inches(0.6),
            "Lesson 17 is the first lesson in a four-part series that takes you from "
            "AI foundations to production-ready healthcare AI applications.",
            font_size=18, color=DARK_GRAY)

rows = [
    ("Lesson 17\n(This Lesson)", "AI foundations, hands-on exercises,\nsafety framing",
     "Patient summarization\nCLI pipeline"),
    ("Agentic AI\nPart One", "Agent architecture, MCP, tool-use,\nstructured validation",
     "FHIR-connected AI agent\nwith audit logging"),
    ("Agentic AI\nPart Two", "FHIR as knowledge layer, subagents,\nhuman review",
     "Multi-agent clinical\nworkflow"),
    ("RAG Lab", "Embeddings, vector stores,\nretrieval-augmented generation",
     "Retrieval pipeline grounded\nin patient data"),
]
y = Inches(2.2)
for i, (lesson, focus, build) in enumerate(rows):
    color = [PHASE_1_COLOR, DARK_BLUE, DARK_BLUE, DARK_BLUE][i]
    bg = [RGBColor(220, 240, 250), RGBColor(240, 244, 248),
          RGBColor(240, 244, 248), RGBColor(240, 244, 248)][i]
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(0.7), y, Inches(11.9), Inches(1.05))
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg
    shape.line.color.rgb = color
    shape.line.width = Pt(1.5)

    add_textbox(slide, Inches(0.9), y + Inches(0.08), Inches(2.5), Inches(0.9),
                lesson, font_size=15, bold=True, color=color)
    add_textbox(slide, Inches(3.8), y + Inches(0.08), Inches(4.5), Inches(0.9),
                focus, font_size=14, color=DARK_GRAY)
    add_textbox(slide, Inches(8.8), y + Inches(0.08), Inches(3.5), Inches(0.9),
                build, font_size=14, color=MED_GRAY, italic=True)
    y += Inches(1.2)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 3 — Learning Objectives
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Learning Objectives")
objectives = [
    "Explain core LLM concepts (tokens, embeddings, context windows, temperature, prompts, hallucination) using healthcare analogies",
    "Set up and interact with a local LLM (Ollama) from Python",
    "Transform a FHIR patient bundle into LLM-ready input through resource grouping and feature extraction",
    "Build a patient summarization pipeline that reduces ~100K tokens to ~1K tokens",
    "Apply prompt engineering techniques and observe how they change LLM output",
    "Articulate AI safety and governance principles for healthcare AI",
]
add_bullet_slide(slide, Inches(0.9), Inches(1.4), Inches(11), Inches(5.5),
                 [f"{i+1}.  {obj}" for i, obj in enumerate(objectives)],
                 font_size=20, color=DARK_GRAY, spacing=Pt(12))


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 4 — Lesson Structure (3 Phases)
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Lesson Structure: Three Phases")
phases = [
    (PHASE_1_COLOR, "Phase 1: AI Foundations",
     "9 core concepts with healthcare analogies\n"
     "What is an LLM? Tokens, context windows, embeddings,\n"
     "temperature, prompts, hallucination, local vs. cloud, model selection"),
    (PHASE_2_COLOR, "Phase 2: Hands-On Exercises",
     "Exercise 1: LLM Hello World\n"
     "Exercise 2: FHIR Bundle Exploration\n"
     "Exercise 3: Patient Summarization Pipeline\n"
     "Exercise 4: Prompt Engineering Comparison"),
    (PHASE_3_COLOR, "Phase 3: Safety & Series Bridge",
     "The Golden Rule of validation\n"
     "Hallucination risks, PHI/HIPAA, human-in-the-loop\n"
     "Regulatory awareness (ONC HTI-1, FDA)\n"
     "How Lesson 17 connects to Parts One/Two and RAG Lab"),
]
y = Inches(1.5)
for color, title, desc in phases:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(0.7), y, Inches(11.9), Inches(1.7))
    shape.fill.solid()
    shape.fill.fore_color.rgb = WHITE
    shape.line.color.rgb = color
    shape.line.width = Pt(2.5)
    # Color bar on left
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                  Inches(0.7), y, Inches(0.15), Inches(1.7))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    add_textbox(slide, Inches(1.1), y + Inches(0.1), Inches(4), Inches(0.5),
                title, font_size=22, bold=True, color=color)
    add_textbox(slide, Inches(5.5), y + Inches(0.1), Inches(7), Inches(1.5),
                desc, font_size=16, color=DARK_GRAY)
    y += Inches(1.9)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 DIVIDER
# ═══════════════════════════════════════════════════════════════════════
phase_divider(1, "AI Foundations",
              "9 core concepts, each grounded in a healthcare analogy\n"
              "No prior AI/ML knowledge required",
              PHASE_1_COLOR)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 5 — What Is an LLM?
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("1. What Is a Large Language Model?")
add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.5), Inches(1.2),
            "A software system trained on vast amounts of text to predict what comes next "
            "in a sequence. Given a prompt, it generates a response word by word, selecting "
            "each word based on patterns learned during training.",
            font_size=18, color=DARK_GRAY)

# Analogy box
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(2.8), Inches(11.9), Inches(2.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(240, 248, 255)
shape.line.color.rgb = MED_BLUE
add_textbox(slide, Inches(1.0), Inches(2.9), Inches(1.8), Inches(0.5),
            "Healthcare\nAnalogy", font_size=16, bold=True, color=MED_BLUE)
add_textbox(slide, Inches(3.0), Inches(2.9), Inches(9.3), Inches(1.8),
            "Think of an LLM as a clinician who has read every textbook, every journal "
            "article, and every clinical note ever written — but has never seen an actual "
            "patient. The knowledge is vast but indirect: learned from text, not from "
            "lived clinical experience.",
            font_size=17, color=DARK_GRAY)

add_bullet_slide(slide, Inches(0.9), Inches(5.2), Inches(11), Inches(2),
                 ["LLMs process text patterns — they don't \"understand\" FHIR resources like a FHIR server does",
                  "An LLM has no database, no schema validation — only patterns in language",
                  "When we combine LLMs with FHIR, we provide the structured data the LLM lacks"],
                 font_size=16, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 6 — Tokens and Tokenization
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("2. Tokens and Tokenization")
add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.5), Inches(0.8),
            "Before an LLM can process text, it breaks the input into tokens — the "
            "fundamental units the model reads. A token is typically a word or subword fragment.",
            font_size=18, color=DARK_GRAY)

# Example box
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(2.3), Inches(11.9), Inches(1.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(245, 245, 245)
shape.line.color.rgb = MED_GRAY
add_textbox(slide, Inches(1.0), Inches(2.4), Inches(11.3), Inches(0.8),
            '"The patient\'s hemoglobin A1c was 7.2%"\n'
            '   ["The", " patient", "\'s", " hem", "oglobin", " A", "1", "c", " was", " 7", ".", "2", "%"]    (13 tokens)',
            font_size=15, color=DARK_GRAY, font_name="Consolas")

# Analogy
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(3.6), Inches(5.5), Inches(2.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(240, 248, 255)
shape.line.color.rgb = MED_BLUE
add_textbox(slide, Inches(0.9), Inches(3.7), Inches(5.1), Inches(0.4),
            "Pharmacist Reading a Prescription", font_size=16, bold=True, color=MED_BLUE)
add_textbox(slide, Inches(0.9), Inches(4.1), Inches(5.1), Inches(1.4),
            "A pharmacist parses a prescription into discrete units: "
            "drug name, dosage, route, frequency. Token boundaries "
            "don't always match word boundaries.",
            font_size=15, color=DARK_GRAY)

# FHIR implications
add_textbox(slide, Inches(6.5), Inches(3.6), Inches(6), Inches(0.4),
            "Why This Matters for FHIR:", font_size=16, bold=True, color=DARK_BLUE)
add_bullet_slide(slide, Inches(6.5), Inches(4.1), Inches(6), Inches(2.5),
                 ["A Patient/$everything bundle = tens of thousands of tokens",
                  "Raw FHIR JSON is token-inefficient (braces, quotes, URLs)",
                  "Token count determines if data fits in the LLM's window",
                  "Exercise 2 demonstrates this with real numbers"],
                 font_size=15, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 7 — Context Windows
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("3. Context Windows")
add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.5), Inches(0.7),
            "The context window is the maximum number of tokens an LLM can process "
            "in a single interaction — both input and output must fit.",
            font_size=18, color=DARK_GRAY)

# Table-like comparison
models = [
    ("Small local (3B)", "4K–8K tokens", RGBColor(220, 240, 220)),
    ("Medium local (7B)", "8K–32K tokens", RGBColor(220, 235, 250)),
    ("Large cloud (GPT-4, Claude)", "128K–1M+ tokens", RGBColor(240, 230, 220)),
]
y = Inches(2.3)
for name, window, bg in models:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(0.7), y, Inches(5.5), Inches(0.55))
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg
    shape.line.fill.background()
    add_textbox(slide, Inches(0.9), y + Inches(0.05), Inches(3), Inches(0.45),
                name, font_size=15, bold=True, color=DARK_GRAY)
    add_textbox(slide, Inches(4.0), y + Inches(0.05), Inches(2), Inches(0.45),
                window, font_size=15, color=DARK_GRAY)
    y += Inches(0.65)

# Analogy
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(4.5), Inches(5.5), Inches(1.8))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(240, 248, 255)
shape.line.color.rgb = MED_BLUE
add_textbox(slide, Inches(0.9), Inches(4.6), Inches(5.1), Inches(0.4),
            "Clinician Working Memory", font_size=16, bold=True, color=MED_BLUE)
add_textbox(slide, Inches(0.9), Inches(5.0), Inches(5.1), Inches(1.2),
            "A physician reviewing a complex chart can't hold the entire record "
            "in mind at once. They focus on a subset: recent visits, active problems, "
            "current medications. The context window is the LLM's equivalent of this limit.",
            font_size=15, color=DARK_GRAY)

# The problem
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(6.5), Inches(2.3), Inches(6.1), Inches(4.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(255, 245, 240)
shape.line.color.rgb = RED_ACCENT
add_textbox(slide, Inches(6.7), Inches(2.4), Inches(5.7), Inches(0.4),
            "The Problem (Exercise 2 reveals this):", font_size=16, bold=True, color=RED_ACCENT)
add_textbox(slide, Inches(6.7), Inches(2.9), Inches(5.7), Inches(3.2),
            "A typical Synthea patient bundle:\n"
            "   ~310,000 characters\n"
            "   ~101,000 tokens\n"
            "   ~1,263% of an 8K window\n\n"
            "The data is 12x too large for a small\n"
            "model's context window.\n\n"
            "This is why Exercise 3 extracts features —\n"
            "reducing 101K tokens to ~1K tokens (99% reduction).",
            font_size=15, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 8 — Embeddings (Brief)
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("4. Embeddings — Meaning as Numbers")
add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.5), Inches(0.8),
            "An embedding is a list of numbers (a vector) that represents the meaning "
            "of a word, sentence, or document. Words with similar meanings have similar vectors.",
            font_size=18, color=DARK_GRAY)

# Similarity example
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(2.4), Inches(11.9), Inches(1.5))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(245, 245, 245)
shape.line.color.rgb = MED_GRAY
add_textbox(slide, Inches(1.0), Inches(2.5), Inches(11.3), Inches(1.3),
            '"hypertension"  ~  "high blood pressure"     (close in embedding space)\n'
            '"hypertension"  ≠  "appendectomy"              (far apart in embedding space)\n\n'
            'Like how clinicians know "MI" = "myocardial infarction" = "heart attack"\n'
            '— embeddings give computers this same ability.',
            font_size=17, color=DARK_GRAY)

shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(4.3), Inches(11.9), Inches(1.2))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(255, 250, 230)
shape.line.color.rgb = GOLD
add_textbox(slide, Inches(1.0), Inches(4.4), Inches(11.3), Inches(1.0),
            "We introduce the concept here; the RAG Lab covers embeddings, vector stores, "
            "and similarity search in depth. Embeddings are the foundation of "
            "Retrieval-Augmented Generation (RAG) — finding relevant documents by "
            "comparing vector similarity before sending them to an LLM.",
            font_size=16, italic=True, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 9 — Temperature and Sampling
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("5. Temperature and Sampling")
add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.5), Inches(0.7),
            "Temperature controls how the LLM chooses its next word. Low = focused and "
            "predictable. High = creative and variable.",
            font_size=18, color=DARK_GRAY)

temps = [
    ("Low (0.1)", "Confident Specialist",
     "Almost always picks the most likely word. Deterministic, focused. "
     "Like a specialist following established guidelines.",
     GREEN_ACCENT, RGBColor(235, 250, 235)),
    ("Medium (0.7)", "Experienced Clinician",
     "Balanced — mostly follows evidence but exercises clinical judgment. "
     "Default for most tasks.",
     MED_BLUE, RGBColor(235, 245, 255)),
    ("High (1.0+)", "Creative Brainstormer",
     "Samples more broadly. Creative, diverse, sometimes incoherent. "
     "Like generating a differential diagnosis — casts a wide net.",
     RED_ACCENT, RGBColor(255, 240, 240)),
]
y = Inches(2.3)
for temp, analogy, desc, color, bg in temps:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(0.7), y, Inches(11.9), Inches(1.4))
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg
    shape.line.color.rgb = color
    shape.line.width = Pt(1.5)
    add_textbox(slide, Inches(1.0), y + Inches(0.1), Inches(2.2), Inches(0.5),
                temp, font_size=18, bold=True, color=color)
    add_textbox(slide, Inches(1.0), y + Inches(0.6), Inches(2.2), Inches(0.5),
                analogy, font_size=14, italic=True, color=color)
    add_textbox(slide, Inches(3.5), y + Inches(0.2), Inches(8.8), Inches(1.0),
                desc, font_size=16, color=DARK_GRAY)
    y += Inches(1.55)

add_textbox(slide, Inches(0.7), Inches(6.6), Inches(11.5), Inches(0.5),
            "For clinical summarization, lower temperature (0.1–0.3) is typically safer. "
            "Exercise 4 demonstrates this side-by-side.",
            font_size=16, bold=True, color=DARK_BLUE)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 10 — Prompts and System Roles
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("6. Prompts and System Roles")

add_two_column_bullets(
    slide, Inches(0.7), Inches(1.4), Inches(12), Inches(2.5),
    "System Prompt (Standing Orders)",
    ["Sets the LLM's role and behavior rules",
     "Persists across the conversation",
     "\"You are a clinical AI assistant\"",
     "\"Do not invent information\""],
    "User Prompt (Clinical Question)",
    ["The specific question or data for this interaction",
     "\"Summarize this patient's conditions\"",
     "Changes with each request",
     "Contains the actual FHIR data"],
    font_size=16
)

shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(4.2), Inches(11.9), Inches(1.5))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(240, 248, 255)
shape.line.color.rgb = MED_BLUE
add_textbox(slide, Inches(1.0), Inches(4.3), Inches(11.3), Inches(0.4),
            "Healthcare Analogy — Clinical Protocols", font_size=16, bold=True, color=MED_BLUE)
add_textbox(slide, Inches(1.0), Inches(4.7), Inches(11.3), Inches(0.9),
            "A system prompt is like a hospital policy: \"Always verify allergies before prescribing.\" "
            "It shapes every interaction without being restated. The user prompt is the specific "
            "clinical question: \"What are this patient's active conditions?\"",
            font_size=16, color=DARK_GRAY)

add_textbox(slide, Inches(0.7), Inches(6.0), Inches(11.5), Inches(0.8),
            "Prompt Engineering Techniques:  1) Role specification  2) Output format  "
            "3) Safety constraints  4) Few-shot examples\n"
            "Exercise 4 compares these side-by-side.",
            font_size=16, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 11 — Hallucination
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("7. Hallucination")
add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.5), Inches(0.7),
            "Hallucination: when an LLM generates plausible-sounding but factually incorrect "
            "or entirely fabricated information — with full confidence.",
            font_size=18, color=DARK_GRAY)

shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(2.3), Inches(11.9), Inches(1.3))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(240, 248, 255)
shape.line.color.rgb = MED_BLUE
add_textbox(slide, Inches(1.0), Inches(2.4), Inches(11.3), Inches(0.4),
            "Healthcare Analogy — Clinical Confabulation", font_size=16, bold=True, color=MED_BLUE)
add_textbox(slide, Inches(1.0), Inches(2.8), Inches(11.3), Inches(0.7),
            "In neurology, confabulation is when a patient with memory impairment fills gaps "
            "with fabricated memories, presented with full confidence. LLM hallucination works the same way.",
            font_size=16, color=DARK_GRAY)

# Danger examples
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(3.9), Inches(5.8), Inches(2.8))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(255, 240, 240)
shape.line.color.rgb = RED_ACCENT
add_textbox(slide, Inches(0.9), Inches(4.0), Inches(5.4), Inches(0.4),
            "Danger in Healthcare AI:", font_size=16, bold=True, color=RED_ACCENT)
add_bullet_slide(slide, Inches(0.9), Inches(4.4), Inches(5.4), Inches(2.2),
                 ["Invents a medication the patient never took",
                  "Fabricates a lab value not in the FHIR record",
                  "Conflates two patients' data in batch processing",
                  "Cites a clinical guideline that doesn't exist"],
                 font_size=15, color=DARK_GRAY)

# Mitigations
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(6.8), Inches(3.9), Inches(5.8), Inches(2.8))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(235, 250, 235)
shape.line.color.rgb = GREEN_ACCENT
add_textbox(slide, Inches(7.0), Inches(4.0), Inches(5.4), Inches(0.4),
            "Mitigation Strategies:", font_size=16, bold=True, color=GREEN_ACCENT)
add_bullet_slide(slide, Inches(7.0), Inches(4.4), Inches(5.4), Inches(2.2),
                 ["Grounding: provide actual FHIR data",
                  "\"Do not invent\" system prompt instructions",
                  "Human-in-the-loop validation",
                  "RAG: retrieve source docs first (RAG Lab)"],
                 font_size=15, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 12 — Local vs. Cloud Models
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("8. Local vs. Cloud Models")

add_two_column_bullets(
    slide, Inches(0.7), Inches(1.4), Inches(12), Inches(3.5),
    "Local Models (Ollama, LM Studio)",
    ["Data never leaves your machine",
     "No BAA needed — PHI stays local",
     "Limited by your hardware (3B–13B typical)",
     "Free after hardware investment",
     "Always available (your machine)"],
    "Cloud Models (GPT-4, Claude, Gemini)",
    ["Data sent to third-party servers",
     "Requires BAA for PHI",
     "Access to largest models (100B+ params)",
     "Pay per token",
     "Depends on service uptime"],
    font_size=16,
    title_color=DARK_BLUE
)

shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(5.2), Inches(11.9), Inches(1.5))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(240, 248, 255)
shape.line.color.rgb = MED_BLUE
add_textbox(slide, Inches(1.0), Inches(5.3), Inches(11.3), Inches(0.4),
            "Healthcare Analogy — Local Lab vs. Reference Lab", font_size=16, bold=True, color=MED_BLUE)
add_textbox(slide, Inches(1.0), Inches(5.7), Inches(11.3), Inches(0.9),
            "Local lab: fast turnaround, specimens stay on-site (privacy), but limited test menu. "
            "Reference lab: specialized tests, but specimens shipped off-site, slower, more expensive.\n"
            "Our exercises use Ollama (local) — even with synthetic data, we practice the privacy-first pattern.",
            font_size=16, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 13 — Model Selection
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("9. Model Selection Basics")

sizes = [
    ("Small (1B–3B)", "Llama 3.2 1B/3B, Phi-3 Mini", "Fast, runs on laptops, 8GB RAM",
     "Limited reasoning, shorter outputs", RGBColor(220, 240, 220)),
    ("Medium (7B–13B)", "Llama 3.1 8B, Mistral 7B", "Good balance of quality and speed",
     "May struggle with complex reasoning", RGBColor(220, 235, 250)),
    ("Large (30B–70B)", "Llama 3.1 70B, Mixtral 8x7B", "Strong reasoning, nuanced",
     "Requires 32–64GB+ RAM or GPU", RGBColor(250, 240, 220)),
    ("Cloud-scale (100B+)", "GPT-4, Claude, Gemini", "State-of-the-art capability",
     "Cloud-only, cost, privacy concerns", RGBColor(255, 235, 235)),
]
y = Inches(1.5)
for size, examples, strengths, limits, bg in sizes:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(0.7), y, Inches(11.9), Inches(1.0))
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg
    shape.line.fill.background()
    add_textbox(slide, Inches(0.9), y + Inches(0.08), Inches(2.5), Inches(0.8),
                size, font_size=15, bold=True, color=DARK_BLUE)
    add_textbox(slide, Inches(3.5), y + Inches(0.05), Inches(2.8), Inches(0.4),
                examples, font_size=13, italic=True, color=MED_GRAY)
    add_textbox(slide, Inches(3.5), y + Inches(0.45), Inches(4.5), Inches(0.4),
                strengths, font_size=13, color=GREEN_ACCENT)
    add_textbox(slide, Inches(8.2), y + Inches(0.45), Inches(4.2), Inches(0.4),
                limits, font_size=13, color=RED_ACCENT)
    y += Inches(1.15)

shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(6.1), Inches(11.9), Inches(1.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(220, 240, 250)
shape.line.color.rgb = PHASE_1_COLOR
add_textbox(slide, Inches(1.0), Inches(6.2), Inches(11.3), Inches(0.8),
            "For this course: We use llama3.2:3b via Ollama — small enough for any modern laptop, "
            "capable enough for summarization and prompt engineering. "
            "Change LLM_MODEL in .env to try larger models.",
            font_size=16, bold=True, color=DARK_BLUE)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 DIVIDER
# ═══════════════════════════════════════════════════════════════════════
phase_divider(2, "Hands-On Exercises",
              "4 exercises that progressively build from LLM connectivity\n"
              "to a complete FHIR patient summarization pipeline",
              PHASE_2_COLOR)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 14 — Exercise 1: LLM Hello World
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Exercise 1: LLM Hello World")
add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.5), Inches(0.5),
            "Verify your local LLM setup and see your first AI response from Python.",
            font_size=20, italic=True, color=MED_BLUE)

# Code example
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(2.1), Inches(6.5), Inches(3.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(40, 44, 52)
shape.line.fill.background()
add_textbox(slide, Inches(1.0), Inches(2.2), Inches(6), Inches(2.8),
            '$ python exercise1_llm_chat.py\n\n'
            '=== Exercise 1: LLM Hello World ===\n\n'
            'Sending prompt to llama3.2:3b...\n\n'
            'Response:\n'
            'An A1c test, also called a hemoglobin\n'
            'A1c or HbA1c test, measures your\n'
            'average blood sugar level over the\n'
            'past 2-3 months...',
            font_size=14, color=RGBColor(200, 220, 200), font_name="Consolas")

# What to notice
add_textbox(slide, Inches(7.5), Inches(2.1), Inches(5.3), Inches(0.4),
            "What to Notice:", font_size=18, bold=True, color=DARK_BLUE)
add_bullet_slide(slide, Inches(7.5), Inches(2.6), Inches(5.3), Inches(3.5),
                 ["System prompt sets the role:\n  \"clinical AI assistant\"",
                  "User prompt asks a specific question:\n  \"Explain what an A1c test measures\"",
                  "The LLM generates natural language\n  from a structured prompt",
                  "If you get a connection error, Ollama\n  isn't running — start it first"],
                 font_size=15, color=DARK_GRAY, spacing=Pt(10))

add_textbox(slide, Inches(0.7), Inches(5.5), Inches(11.5), Inches(0.8),
            "Setup: Ensure Ollama is running (ollama serve) with the model loaded.\n"
            "Run:  python exercise1_llm_chat.py",
            font_size=16, color=DARK_GRAY, font_name="Calibri")


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 15 — Exercise 2: FHIR Bundle Exploration
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Exercise 2: FHIR Bundle Exploration")
add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.5), Inches(0.5),
            "How large is a real FHIR patient bundle? Can it fit in a context window?",
            font_size=20, italic=True, color=MED_BLUE)

# Output
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(2.1), Inches(6.5), Inches(4.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(40, 44, 52)
shape.line.fill.background()
add_textbox(slide, Inches(1.0), Inches(2.2), Inches(6), Inches(3.8),
            '$ python exercise2_fhir_explore.py\n\n'
            '=== FHIR Bundle Statistics ===\n'
            'Total characters:     310,482\n'
            'Estimated tokens:     ~101,049\n'
            'Total resources:      224\n\n'
            '=== Resource Breakdown ===\n'
            'ExplanationOfBenefit    72\n'
            'Encounter               47\n'
            'Observation             43\n'
            'Claim                   25\n'
            'Condition               14\n'
            '...\n\n'
            '=== Context Window Analysis ===\n'
            'This bundle is ~1,263% of an\n'
            '8,000-token context window.',
            font_size=13, color=RGBColor(200, 220, 200), font_name="Consolas")

# Key insight
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(7.5), Inches(2.1), Inches(5.3), Inches(2.2))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(255, 245, 240)
shape.line.color.rgb = RED_ACCENT
add_textbox(slide, Inches(7.7), Inches(2.2), Inches(4.9), Inches(0.4),
            "The Key Insight:", font_size=18, bold=True, color=RED_ACCENT)
add_textbox(slide, Inches(7.7), Inches(2.7), Inches(4.9), Inches(1.5),
            "One patient's data is 12x too large for a "
            "small model's context window.\n\n"
            "We can't just paste FHIR JSON into an LLM.\n"
            "We need a smarter approach...",
            font_size=16, color=DARK_GRAY)

# Discussion
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(7.5), Inches(4.6), Inches(5.3), Inches(1.5))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(240, 248, 255)
shape.line.color.rgb = MED_BLUE
add_textbox(slide, Inches(7.7), Inches(4.7), Inches(4.9), Inches(0.4),
            "Discussion: What Are Our Options?",
            font_size=15, bold=True, color=MED_BLUE)
add_bullet_slide(slide, Inches(7.7), Inches(5.1), Inches(4.9), Inches(1.2),
                 ["1. Bigger model (expensive, cloud, privacy)",
                  "2. Extract & reduce (Exercise 3)",
                  "3. Retrieve relevant parts (RAG Lab)"],
                 font_size=14, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 16 — Exercise 3: The Pipeline (Overview)
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Exercise 3: Patient Summarization Pipeline")
add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.5), Inches(0.5),
            "The core exercise — transform a FHIR bundle into a clinical narrative using a local LLM.",
            font_size=20, italic=True, color=MED_BLUE)

# Pipeline phases
phases = [
    ("1. FETCH", "Retrieve Patient/$everything\nbundle from FHIR server",
     DARK_BLUE, RGBColor(220, 235, 255)),
    ("2. GROUP", "Organize resources by type:\nCondition, Observation,\nMedicationRequest, etc.",
     MED_BLUE, RGBColor(230, 245, 255)),
    ("3. EXTRACT", "Pull clinically relevant\nfeatures from each type:\ndemographics, conditions,\nlabs, medications",
     PHASE_1_COLOR, RGBColor(235, 248, 245)),
    ("4. SUMMARIZE", "Send extracted features\n(~1K tokens) to LLM with\nclinical system prompt",
     PHASE_2_COLOR, RGBColor(235, 250, 235)),
    ("5. OUTPUT", "Print clinical narrative\nsummary and token\nreduction statistics",
     RGBColor(100, 60, 140), RGBColor(240, 235, 250)),
]
x = Inches(0.4)
for name, desc, color, bg in phases:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    x, Inches(2.3), Inches(2.35), Inches(3.0))
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg
    shape.line.color.rgb = color
    shape.line.width = Pt(2)
    add_textbox(slide, x + Inches(0.15), Inches(2.4), Inches(2.05), Inches(0.5),
                name, font_size=16, bold=True, color=color, alignment=PP_ALIGN.CENTER)
    add_textbox(slide, x + Inches(0.15), Inches(3.0), Inches(2.05), Inches(2.0),
                desc, font_size=13, color=DARK_GRAY, alignment=PP_ALIGN.CENTER)
    # Arrow between phases (except last)
    if name != "5. OUTPUT":
        add_textbox(slide, x + Inches(2.35), Inches(3.3), Inches(0.25), Inches(0.5),
                    ">", font_size=24, bold=True, color=MED_GRAY,
                    alignment=PP_ALIGN.CENTER)
    x += Inches(2.55)

# Token reduction callout
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(5.7), Inches(11.9), Inches(1.2))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(235, 250, 235)
shape.line.color.rgb = GREEN_ACCENT
shape.line.width = Pt(2)
add_textbox(slide, Inches(1.0), Inches(5.8), Inches(11.3), Inches(1.0),
            "Result: ~101,000 tokens  -->  ~1,040 tokens  (99% reduction)\n"
            "The LLM receives only clinically relevant features, not raw JSON. "
            "Run: python exercise3_summarizer.py",
            font_size=18, bold=True, color=DARK_BLUE)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 17 — Exercise 3: Safety Detail
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Exercise 3: The \"Do Not Invent\" Instruction")

# System prompt box
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(1.4), Inches(11.9), Inches(2.5))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(40, 44, 52)
shape.line.fill.background()
add_textbox(slide, Inches(1.0), Inches(1.5), Inches(11.3), Inches(2.3),
            'System Prompt (from exercise3_summarizer.py):\n\n'
            '"You are a clinical AI assistant reviewing FHIR patient data.\n'
            ' Provide a concise clinical summary organized by:\n'
            ' Demographics, Active Conditions, Current Medications,\n'
            ' Recent Observations, Recent Encounters, Allergies, and\n'
            ' Notable Findings.\n\n'
            ' Do not invent or assume information not present in the data."',
            font_size=15, color=RGBColor(200, 220, 200), font_name="Consolas")

# Discussion
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(4.2), Inches(5.8), Inches(2.8))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(255, 245, 240)
shape.line.color.rgb = RED_ACCENT
add_textbox(slide, Inches(0.9), Inches(4.3), Inches(5.4), Inches(0.4),
            "Why Is This Instruction Critical?", font_size=17, bold=True, color=RED_ACCENT)
add_bullet_slide(slide, Inches(0.9), Inches(4.8), Inches(5.4), Inches(2.0),
                 ["Without it, the LLM may fabricate data",
                  "Example: invents Metformin for a diabetic\n  patient who was never prescribed it",
                  "Clinician acting on hallucinated data\n  → medication reconciliation error",
                  "This is a prompt-level safety guardrail"],
                 font_size=15, color=DARK_GRAY)

shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(6.8), Inches(4.2), Inches(5.8), Inches(2.8))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(240, 248, 255)
shape.line.color.rgb = MED_BLUE
add_textbox(slide, Inches(7.0), Inches(4.3), Inches(5.4), Inches(0.4),
            "Discussion Question:", font_size=17, bold=True, color=MED_BLUE)
add_textbox(slide, Inches(7.0), Inches(4.8), Inches(5.4), Inches(2.0),
            "\"Could a clinician use this summary directly "
            "in patient care?\"\n\n"
            "Answer: No — it requires human verification "
            "against source data.\n\n"
            "This is the \"golden rule\" we'll discuss "
            "in Phase 3.",
            font_size=16, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 18 — Exercise 4: Prompt Engineering
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Exercise 4: Prompt Engineering Comparison")
add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.5), Inches(0.5),
            "Same patient data, different prompts — see how prompt design changes everything.",
            font_size=20, italic=True, color=MED_BLUE)

strategies = [
    ("Strategy 1: Minimal Prompt", "\"You are a helpful assistant.\"\n\"Summarize this patient data.\"",
     "Unstructured paragraph, no consistent format,\nmay hallucinate", RGBColor(255, 240, 240), RED_ACCENT),
    ("Strategy 2: Structured Prompt", "Clinical role + section format +\n\"do not invent\" constraint",
     "Organized sections matching requested format,\nsafer, more predictable", RGBColor(235, 250, 235), GREEN_ACCENT),
    ("Strategy 3: Few-Shot", "Includes an example input/output pair\nbefore the real data",
     "Mimics the example format and style without\nexplicit format instructions", RGBColor(235, 245, 255), MED_BLUE),
]
y = Inches(2.1)
for name, prompt, result, bg, color in strategies:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(0.7), y, Inches(11.9), Inches(1.3))
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg
    shape.line.color.rgb = color
    shape.line.width = Pt(1.5)
    add_textbox(slide, Inches(0.9), y + Inches(0.05), Inches(3.3), Inches(0.4),
                name, font_size=16, bold=True, color=color)
    add_textbox(slide, Inches(0.9), y + Inches(0.45), Inches(4.5), Inches(0.8),
                prompt, font_size=13, color=DARK_GRAY, italic=True)
    add_textbox(slide, Inches(6.8), y + Inches(0.15), Inches(5.5), Inches(1.0),
                result, font_size=14, color=DARK_GRAY)
    y += Inches(1.45)

add_textbox(slide, Inches(0.7), Inches(6.3), Inches(11.5), Inches(0.8),
            "Also includes temperature comparison (0.1, 0.7, 1.0) on Strategy 2. "
            "Prompt constants are at the top of the file — students can edit and re-run.\n"
            "Run:  python exercise4_prompt_compare.py",
            font_size=16, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 19 — Exercise 4: Temperature Comparison
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Exercise 4: Temperature Comparison")
add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.5), Inches(0.5),
            "Same prompt (Strategy 2), same data — only temperature changes.",
            font_size=20, italic=True, color=MED_BLUE)

temps = [
    ("Temperature 0.1", "Focused, deterministic", "Nearly identical output each run. "
     "Picks the highest-probability words consistently. Best for clinical summarization "
     "where accuracy and reproducibility matter.", GREEN_ACCENT, RGBColor(235, 250, 235)),
    ("Temperature 0.7", "Balanced (default)", "Reasonable variety in phrasing while "
     "maintaining accuracy. Good general-purpose setting. Some creative choices in wording.",
     MED_BLUE, RGBColor(235, 245, 255)),
    ("Temperature 1.0", "Creative, variable", "Output varies significantly across runs. "
     "May introduce unexpected phrasings or tangents. Higher hallucination risk. "
     "Useful for brainstorming, not clinical reporting.", RED_ACCENT, RGBColor(255, 240, 240)),
]
y = Inches(2.1)
for name, style, desc, color, bg in temps:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(0.7), y, Inches(11.9), Inches(1.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg
    shape.line.color.rgb = color
    shape.line.width = Pt(2)
    add_textbox(slide, Inches(1.0), y + Inches(0.1), Inches(2.5), Inches(0.5),
                name, font_size=18, bold=True, color=color)
    add_textbox(slide, Inches(1.0), y + Inches(0.6), Inches(2.5), Inches(0.5),
                style, font_size=14, italic=True, color=color)
    add_textbox(slide, Inches(4.0), y + Inches(0.2), Inches(8.3), Inches(1.1),
                desc, font_size=16, color=DARK_GRAY)
    y += Inches(1.65)

add_textbox(slide, Inches(0.7), Inches(7.0), Inches(11.5), Inches(0.4),
            "Takeaway: For clinical applications, use low temperature. "
            "Reproducibility = verifiability = safety.",
            font_size=17, bold=True, color=DARK_BLUE)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 DIVIDER
# ═══════════════════════════════════════════════════════════════════════
phase_divider(3, "Safety, Governance &\nSeries Bridge",
              "AI safety is not a single topic — it is a design constraint\n"
              "that runs through the entire AI+FHIR series",
              PHASE_3_COLOR)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 20 — The Golden Rule
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("The Golden Rule")

shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(1.5), Inches(1.8), Inches(10.3), Inches(2.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(255, 248, 220)
shape.line.color.rgb = GOLD
shape.line.width = Pt(3)
add_textbox(slide, Inches(2.0), Inches(2.0), Inches(9.3), Inches(1.6),
            "Never use LLM output in a clinical context\n"
            "without verifying it against the source data.",
            font_size=28, bold=True, color=RGBColor(140, 100, 0),
            alignment=PP_ALIGN.CENTER)

add_textbox(slide, Inches(0.7), Inches(4.2), Inches(11.5), Inches(0.5),
            "Validation Strategies:", font_size=20, bold=True, color=DARK_BLUE)

strategies = [
    "Cross-reference against source FHIR data: Does the summary match the actual resources?",
    "Structured output validation: Does the LLM output conform to the expected schema?",
    "Human spot-checking: Clinician verifies medications, allergies, active diagnoses",
    "Reproducibility testing: Run the same input twice — significant factual differences suggest hallucination",
]
add_bullet_slide(slide, Inches(0.9), Inches(4.8), Inches(11), Inches(2.5),
                 strategies, font_size=17, color=DARK_GRAY, spacing=Pt(10),
                 bold_prefix=True)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 21 — Hallucination Risks in Clinical Contexts
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Hallucination Risks in Clinical Contexts")

scenarios = [
    ("Invented Medication", "LLM summary says \"Metformin 500mg twice daily\" — "
     "but there is no Metformin MedicationRequest in the FHIR bundle. The LLM inferred it "
     "from the diabetes diagnosis.", RED_ACCENT),
    ("Fabricated Lab Value", "Summary reports \"Most recent A1c: 8.2%\" — but no A1c "
     "Observation exists. The LLM filled a gap with a plausible-sounding number.",
     RED_ACCENT),
    ("Patient Conflation", "In batch processing, the LLM blends Patient A's medications "
     "into Patient B's summary — especially if context windows are reused without clearing.",
     RED_ACCENT),
]
y = Inches(1.5)
for title, desc, color in scenarios:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(0.7), y, Inches(11.9), Inches(1.6))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(255, 245, 243)
    shape.line.color.rgb = color
    shape.line.width = Pt(1.5)
    add_textbox(slide, Inches(1.0), y + Inches(0.1), Inches(3.5), Inches(0.4),
                title, font_size=18, bold=True, color=color)
    add_textbox(slide, Inches(1.0), y + Inches(0.55), Inches(11.3), Inches(1.0),
                desc, font_size=16, color=DARK_GRAY)
    y += Inches(1.75)

add_textbox(slide, Inches(0.7), Inches(6.8), Inches(11.5), Inches(0.5),
            "Each of these can cause patient harm if a clinician acts on the LLM output "
            "without verifying against the source FHIR data.",
            font_size=16, bold=True, color=DARK_BLUE)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 22 — PHI and HIPAA
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("PHI and HIPAA Considerations")

shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(1.5), Inches(11.9), Inches(1.6))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(255, 240, 240)
shape.line.color.rgb = RED_ACCENT
shape.line.width = Pt(2)
add_textbox(slide, Inches(1.0), Inches(1.6), Inches(11.3), Inches(1.4),
            "Scenario: A developer copies a patient's FHIR Patient/$everything bundle — "
            "300,000 characters of clinical data including names, dates of birth, diagnoses, "
            "and medications — and pastes it into ChatGPT to generate a summary.\n\n"
            "This is a HIPAA violation. The data is now on external servers, "
            "potentially logged, and outside the covered entity's control.",
            font_size=17, color=DARK_GRAY)

add_two_column_bullets(
    slide, Inches(0.7), Inches(3.5), Inches(12), Inches(2.5),
    "Local Model (Ollama)",
    ["Data stays on your machine",
     "No BAA needed",
     "No risk of external exposure",
     "This is why our exercises use Ollama"],
    "Cloud API (with BAA)",
    ["BAA required with API provider",
     "Azure OpenAI offers BAA-backed endpoints",
     "De-identified data may be acceptable",
     "Know where your data goes"],
    font_size=16,
    title_color=DARK_BLUE
)

add_textbox(slide, Inches(0.7), Inches(6.5), Inches(11.5), Inches(0.6),
            "Key principle: Know where your data goes and what protections are in place.",
            font_size=18, bold=True, color=DARK_BLUE)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 23 — Human-in-the-Loop
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Human-in-the-Loop")
add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.5), Inches(0.6),
            "AI as Decision Support, Not Decision Maker",
            font_size=22, bold=True, color=DARK_BLUE)

# Three step process
steps = [
    ("1", "LLM Processes", "LLM processes data and\ngenerates a summary\nor suggestion",
     MED_BLUE, RGBColor(235, 245, 255)),
    ("2", "Clinician Reviews", "Clinician reviews output\nagainst source FHIR data\nand clinical context",
     GOLD, RGBColor(255, 248, 230)),
    ("3", "Clinician Decides", "Clinician makes the\nclinical decision based\non verified information",
     GREEN_ACCENT, RGBColor(235, 250, 235)),
]
x = Inches(0.7)
for num, title, desc, color, bg in steps:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    x, Inches(2.3), Inches(3.8), Inches(2.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg
    shape.line.color.rgb = color
    shape.line.width = Pt(2)
    add_textbox(slide, x + Inches(0.2), Inches(2.4), Inches(0.5), Inches(0.5),
                num, font_size=28, bold=True, color=color)
    add_textbox(slide, x + Inches(0.8), Inches(2.4), Inches(2.8), Inches(0.5),
                title, font_size=18, bold=True, color=color)
    add_textbox(slide, x + Inches(0.3), Inches(3.0), Inches(3.2), Inches(1.5),
                desc, font_size=15, color=DARK_GRAY)
    if num != "3":
        add_textbox(slide, x + Inches(3.8), Inches(3.1), Inches(0.4), Inches(0.5),
                    ">", font_size=28, bold=True, color=MED_GRAY)
    x += Inches(4.2)

shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(5.2), Inches(11.9), Inches(1.8))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(240, 248, 255)
shape.line.color.rgb = MED_BLUE
add_textbox(slide, Inches(1.0), Inches(5.3), Inches(11.3), Inches(0.4),
            "The Analogy:", font_size=16, bold=True, color=MED_BLUE)
add_textbox(slide, Inches(1.0), Inches(5.7), Inches(11.3), Inches(1.2),
            "A resident presents a case summary to an attending. The attending doesn't sign off "
            "without reviewing the chart. LLM output deserves the same scrutiny.\n\n"
            "LLMs cannot assess clinical context not in the data, cannot take responsibility for outcomes, "
            "and regulatory frameworks increasingly require human oversight.",
            font_size=16, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 24 — Regulatory Awareness
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Regulatory Awareness")
add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.5), Inches(0.4),
            "Awareness level — know these frameworks exist and why they matter.",
            font_size=18, italic=True, color=MED_GRAY)

# ONC
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(2.0), Inches(5.8), Inches(3.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(240, 244, 248)
shape.line.color.rgb = DARK_BLUE
add_textbox(slide, Inches(0.9), Inches(2.1), Inches(5.4), Inches(0.5),
            "ONC Health IT Certification (HTI-1)", font_size=17, bold=True, color=DARK_BLUE)
add_bullet_slide(slide, Inches(0.9), Inches(2.6), Inches(5.4), Inches(2.2),
                 ["Transparency requirements for AI-based\n  clinical decision support tools",
                  "Disclosure requirements for predictive\n  models in certified health IT",
                  "If your AI+FHIR application is part of\n  certified health IT, these rules apply"],
                 font_size=15, color=DARK_GRAY)

# FDA
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(6.8), Inches(2.0), Inches(5.8), Inches(3.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(240, 244, 248)
shape.line.color.rgb = DARK_BLUE
add_textbox(slide, Inches(7.0), Inches(2.1), Inches(5.4), Inches(0.5),
            "FDA — AI/ML as Medical Devices", font_size=17, bold=True, color=DARK_BLUE)
add_bullet_slide(slide, Inches(7.0), Inches(2.6), Inches(5.4), Inches(2.2),
                 ["AI tools for diagnosis/treatment may be\n  regulated as medical devices",
                  "FDA has approved hundreds of AI/ML-\n  enabled medical devices",
                  "\"Patient summary\" vs. \"clinical\n  recommendation\" = different regulatory bar"],
                 font_size=15, color=DARK_GRAY)

# Practical takeaway
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(5.3), Inches(11.9), Inches(1.8))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(255, 248, 220)
shape.line.color.rgb = GOLD
add_textbox(slide, Inches(1.0), Inches(5.4), Inches(11.3), Inches(0.4),
            "When Building AI+FHIR Applications, Ask:", font_size=17, bold=True, color=RGBColor(140, 100, 0))
add_bullet_slide(slide, Inches(1.0), Inches(5.8), Inches(11.3), Inches(1.2),
                 ["1. Know your intent — informational or clinical?",
                  "2. Know your data flow — does PHI leave your infrastructure?",
                  "3. Know your audience — will clinicians make decisions from this?",
                  "4. Know your obligations — which regulatory frameworks apply?"],
                 font_size=15, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 25 — Real-World Use Cases
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Real-World Use Cases: LLMs + FHIR")

cases = [
    ("Patient Record\nSummarization", "Mayo Clinic and others use LLMs to\nsummarize records for diagnostic support",
     "(Our Exercise 3!)"),
    ("Clinical Note\nGeneration", "Ambience, Abridge, Amazon HealthScribe\ngenerate SOAP notes from transcripts",
     ""),
    ("Natural Language\nto FHIR Query", "\"Find all diabetics with A1c > 8\" translated\ninto FHIR search parameters",
     ""),
    ("Structured Data\nExtraction", "LLMs extract medications, conditions, labs\nfrom free-text notes into FHIR resources",
     ""),
    ("Clinical Trial\nMatching", "Match patient FHIR profiles to trial\neligibility criteria automatically",
     ""),
    ("Predictive Risk\nAnalysis", "Combine clinical notes + structured data\nfor early detection (sepsis, readmission)",
     ""),
]
y = Inches(1.5)
for i, (title, desc, note) in enumerate(cases):
    col = i % 2
    row = i // 2
    x = Inches(0.7) + col * Inches(6.3)
    yy = Inches(1.5) + row * Inches(1.7)
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    x, yy, Inches(6.0), Inches(1.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(240, 244, 248)
    shape.line.color.rgb = MED_BLUE if not note else GREEN_ACCENT
    shape.line.width = Pt(2) if note else Pt(1)
    add_textbox(slide, x + Inches(0.2), yy + Inches(0.1), Inches(2.5), Inches(1.0),
                title, font_size=15, bold=True, color=DARK_BLUE)
    add_textbox(slide, x + Inches(2.7), yy + Inches(0.1), Inches(3.1), Inches(1.0),
                desc, font_size=13, color=DARK_GRAY)
    if note:
        add_textbox(slide, x + Inches(2.7), yy + Inches(1.0), Inches(3.1), Inches(0.4),
                    note, font_size=13, bold=True, color=GREEN_ACCENT)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 26 — Series Bridge: Where Concepts Scale Up
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Where These Concepts Scale Up")

connections = [
    ("Summarization Pipeline\n(Exercise 3)", "Becomes a subagent in\nPart One's multi-agent architecture",
     PHASE_2_COLOR),
    ("Feature Extraction\n(Exercise 3)", "Input to RAG Lab's embedding\nand vector store pipeline",
     RGBColor(100, 60, 140)),
    ("Prompt Engineering\n(Exercise 4)", "Applies to agent prompts\nin Parts One and Two",
     MED_BLUE),
    ("Manual FHIR Access\n(requests library)", "Replaced by MCP — standardized\nLLM-to-FHIR protocol",
     DARK_BLUE),
    ("Safety Principles\n(Phase 3)", "Operationalized as least-privilege,\nvalidation, audit logging",
     RED_ACCENT),
]
y = Inches(1.5)
for lesson17, future, color in connections:
    # Left box (Lesson 17)
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(0.7), y, Inches(4.0), Inches(0.9))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(240, 244, 248)
    shape.line.color.rgb = color
    add_textbox(slide, Inches(0.9), y + Inches(0.05), Inches(3.6), Inches(0.8),
                lesson17, font_size=14, bold=True, color=color)
    # Arrow
    add_textbox(slide, Inches(4.8), y + Inches(0.1), Inches(0.8), Inches(0.7),
                "-->", font_size=20, bold=True, color=MED_GRAY,
                alignment=PP_ALIGN.CENTER)
    # Right box (Future)
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(5.8), y, Inches(6.8), Inches(0.9))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(245, 250, 255)
    shape.line.color.rgb = color
    add_textbox(slide, Inches(6.0), y + Inches(0.05), Inches(6.4), Inches(0.8),
                future, font_size=14, color=DARK_GRAY)
    y += Inches(1.05)

add_textbox(slide, Inches(0.7), Inches(6.8), Inches(11.5), Inches(0.5),
            "Pattern: Learn the concept → build it manually → see it operate at scale "
            "in a production-like architecture.",
            font_size=17, bold=True, color=DARK_BLUE)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 27 — MCP Preview
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Preview: Model Context Protocol (MCP)")
add_textbox(slide, Inches(0.7), Inches(1.3), Inches(11.5), Inches(0.5),
            "From manual FHIR calls to standardized LLM-to-server communication.",
            font_size=20, italic=True, color=MED_BLUE)

# Today
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.7), Inches(2.1), Inches(5.8), Inches(3.5))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(240, 244, 248)
shape.line.color.rgb = MED_BLUE
add_textbox(slide, Inches(0.9), Inches(2.2), Inches(5.4), Inches(0.5),
            "Lesson 17 — Manual Mode", font_size=18, bold=True, color=MED_BLUE)
add_textbox(slide, Inches(0.9), Inches(2.8), Inches(5.4), Inches(2.5),
            "You write the FHIR query code yourself:\n\n"
            "  You --> Python code --> FHIR Server\n"
            "  You --> extracted data --> LLM\n"
            "  You --> read response --> make decisions\n\n"
            "You are the intermediary between\n"
            "the LLM and the FHIR server.",
            font_size=16, color=DARK_GRAY)

# Future
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(6.8), Inches(2.1), Inches(5.8), Inches(3.5))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(235, 250, 235)
shape.line.color.rgb = GREEN_ACCENT
add_textbox(slide, Inches(7.0), Inches(2.2), Inches(5.4), Inches(0.5),
            "Parts One/Two — MCP Mode", font_size=18, bold=True, color=GREEN_ACCENT)
add_textbox(slide, Inches(7.0), Inches(2.8), Inches(5.4), Inches(2.5),
            "The LLM connects to FHIR via MCP:\n\n"
            "  User --> LLM --> MCP Server --> FHIR\n"
            "  FHIR --> MCP Server --> LLM --> User\n\n"
            "MCP server exposes FHIR operations\n"
            "as tools the LLM can invoke.\n"
            "The LLM decides which tool to call.",
            font_size=16, color=DARK_GRAY)

add_textbox(slide, Inches(0.7), Inches(5.9), Inches(11.5), Inches(1.0),
            "You need \"manual mode\" understanding before \"autonomous mode\" makes sense.\n"
            "When an MCP-connected agent makes an unexpected FHIR query, you need to understand "
            "FHIR endpoints and resource types to debug it.",
            font_size=16, italic=True, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 28 — Safety as a Series Theme
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Safety as a Series-Long Theme")

principles = [
    ("Hallucination\nMitigation", "Part One: structured validation\npatterns for agent outputs"),
    ("PHI Protection", "Part One: least-privilege FHIR\naccess scoping for agents"),
    ("Human-in-the-Loop", "Part Two: human review checkpoints\nin multi-agent pipelines"),
    ("Output Validation", "Part One: JSON schema validation\nof agent tool calls"),
    ("Audit &\nAccountability", "Part One: audit logging of all\nagent actions and FHIR queries"),
    ("Grounding with\nSource Data", "RAG Lab: retrieval-augmented\ngeneration anchors output to sources"),
]
y = Inches(1.5)
for i, (principle, where) in enumerate(principles):
    col = i % 2
    row = i // 2
    x = Inches(0.7) + col * Inches(6.3)
    yy = Inches(1.5) + row * Inches(1.7)
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    x, yy, Inches(6.0), Inches(1.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(255, 245, 240)
    shape.line.color.rgb = RED_ACCENT
    shape.line.width = Pt(1.5)
    add_textbox(slide, x + Inches(0.2), yy + Inches(0.15), Inches(2.3), Inches(1.0),
                principle, font_size=15, bold=True, color=RED_ACCENT)
    add_textbox(slide, x + Inches(2.7), yy + Inches(0.15), Inches(3.1), Inches(1.0),
                where, font_size=14, color=DARK_GRAY)

add_textbox(slide, Inches(0.7), Inches(6.7), Inches(11.5), Inches(0.5),
            "Safety in healthcare AI is not about avoiding AI — it's about using it with "
            "the same rigor you apply to any clinical tool.",
            font_size=17, bold=True, italic=True, color=DARK_BLUE)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 29 — Key Takeaways
# ═══════════════════════════════════════════════════════════════════════
slide = content_slide("Key Takeaways")

takeaways = [
    "LLMs are powerful tools for healthcare — AND they require responsible use",
    "Raw FHIR bundles don't fit in context windows — structured extraction solves this",
    "Prompt engineering is a practical skill that directly affects output quality and safety",
    "The \"golden rule\": never trust LLM output without validating against source data",
    "Local models protect PHI — cloud models require BAAs and careful data handling",
    "Everything you learned today reappears in Parts One, Two, and the RAG Lab",
]
add_bullet_slide(slide, Inches(0.9), Inches(1.5), Inches(11), Inches(5.5),
                 takeaways, font_size=20, color=DARK_GRAY, spacing=Pt(16))


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 30 — Thank You / Q&A
# ═══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, BG_DARK)
add_textbox(slide, Inches(1), Inches(2.0), Inches(11), Inches(1.5),
            "Questions & Discussion",
            font_size=44, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)
add_textbox(slide, Inches(1), Inches(3.8), Inches(11), Inches(1.0),
            "Lesson 17: Using LLMs with FHIR",
            font_size=22, color=LIGHT_BLUE, alignment=PP_ALIGN.CENTER)
add_textbox(slide, Inches(1), Inches(5.0), Inches(11), Inches(1.5),
            "Next in the series: Agentic AI — Part One\n"
            "Agent architecture, MCP, tool-use, and structured validation",
            font_size=18, italic=True, color=RGBColor(180, 200, 220),
            alignment=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════
output_path = os.path.join(os.path.dirname(__file__),
                           "Using LLMs with FHIR - Lesson 17.pptx")
prs.save(output_path)
print(f"Saved: {output_path}")
print(f"Total slides: {len(prs.slides)}")
