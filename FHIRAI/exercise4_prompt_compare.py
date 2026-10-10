"""
Exercise 4: Prompt Engineering Comparison

Runs the same FHIR patient data through multiple prompting strategies to
demonstrate how prompt design affects LLM output. Also compares the effect
of temperature on output variability.

Students: try modifying the prompt constants below and re-running to see
how changes affect the summaries!

Usage:
    python exercise4_prompt_compare.py                # default patient ID "2"
    python exercise4_prompt_compare.py --patient 5    # specify patient ID
"""

import sys
import argparse

import ollama
import requests

from config import FHIR_BASE_URL, FHIR_USERNAME, FHIR_PASSWORD, LLM_MODEL
from exercise3_summarizer import group_resources, build_feature_text


# ---------------------------------------------------------------------------
# Prompt Strategies — edit these to experiment!
# ---------------------------------------------------------------------------

# Strategy 1: Minimal / Baseline — no guidance, let the LLM decide
PROMPT_MINIMAL_SYSTEM = "You are a helpful assistant."
PROMPT_MINIMAL_USER = "Summarize this patient data:\n\n{features}"

# Strategy 2: Structured system prompt — explicit role, format, safety
PROMPT_STRUCTURED_SYSTEM = """\
You are a clinical AI assistant reviewing patient data.
Summarize into these sections:
- Demographics
- Active Conditions
- Medications
- Recent Encounters
- Recent Observations
- Allergies
- Notable Findings

Do not invent information not present in the data.
If a section has no data, state 'None recorded.'"""
PROMPT_STRUCTURED_USER = "{features}"

# Strategy 3: Few-shot prompt — provide an example before the real data
PROMPT_FEWSHOT_SYSTEM = """\
You are a clinical AI assistant. Below is an example of how to summarize
patient data, followed by a new patient to summarize in the same style.

EXAMPLE INPUT:
Patient: Jane Doe, female, DOB 1985-06-15.
Conditions: Hypertension (active, onset 2018).
Medications: Lisinopril 10mg daily (active).
Observations: BP 138/85 mmHg (2024-01-10).

EXAMPLE OUTPUT:
Demographics: Jane Doe, 38F, DOB 1985-06-15
Active Conditions: Hypertension (since 2018)
Medications: Lisinopril 10mg daily
Recent Observations: BP 138/85 mmHg (Jan 2024)
Allergies: None recorded
Notable Findings: Blood pressure slightly elevated; monitor."""
PROMPT_FEWSHOT_USER = "Now summarize this patient in the same style:\n\n{features}"


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Send a chat request to Ollama and return the response text."""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": temperature},
    )
    return response.message.content or "(No response)"


def run_strategy(name: str, system_prompt: str, user_template: str,
                 features: str, temperature: float = 0.7) -> str | None:
    """Run one prompt strategy, print result, return the output text."""
    print(f"\nRunning {name} (temperature={temperature})...")
    try:
        result = call_llm(system_prompt, user_template.format(features=features), temperature)
    except ConnectionError:
        print("  Error: Cannot connect to Ollama. Please start Ollama.", file=sys.stderr)
        return None
    except ollama.ResponseError as e:
        if "not found" in str(e).lower():
            print(f"  Error: Model '{LLM_MODEL}' not found. Run: ollama pull {LLM_MODEL}",
                  file=sys.stderr)
        else:
            print(f"  Error: {e}", file=sys.stderr)
        return None

    print()
    print("=" * 60)
    print(f"  {name}")
    print("=" * 60)
    print()
    print(result)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prompt Engineering Comparison")
    parser.add_argument("--patient", "-p", default="2", help="FHIR Patient ID (default: 2)")
    args = parser.parse_args()
    patient_id = args.patient

    # ---- Data preparation (same for all strategies) ----
    print(f"Fetching and preparing data for patient {patient_id}...")
    try:
        response = requests.get(
            f"{FHIR_BASE_URL}/Patient/{patient_id}/$everything",
            headers={"Accept": "application/fhir+json"},
            auth=(FHIR_USERNAME, FHIR_PASSWORD),
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to FHIR server at {FHIR_BASE_URL}.", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Error: FHIR server returned {e.response.status_code}.", file=sys.stderr)
        sys.exit(1)

    bundle_dict = response.json()
    entries = bundle_dict.get("entry", [])
    if not entries:
        print(f"No data for patient {patient_id}.")
        sys.exit(0)

    resources = [e["resource"] for e in entries if "resource" in e]
    grouped = group_resources(resources)
    features = build_feature_text(grouped)
    print(f"Prepared {len(resources)} resources as extracted features.\n")

    # ---- Strategy comparisons ----

    print("=" * 60)
    print("  PART 1: Prompt Strategy Comparison")
    print("  (all at temperature 0.7)")
    print("=" * 60)

    run_strategy(
        "Strategy 1: Minimal Prompt (Baseline)",
        PROMPT_MINIMAL_SYSTEM,
        PROMPT_MINIMAL_USER,
        features,
    )
    # Baseline prompt often produces an unstructured wall of text because
    # the LLM has no guidance on format or role.

    run_strategy(
        "Strategy 2: Structured System Prompt",
        PROMPT_STRUCTURED_SYSTEM,
        PROMPT_STRUCTURED_USER,
        features,
    )
    # Structured prompt produces sections matching the requested format
    # because explicit instructions shape the output structure.

    run_strategy(
        "Strategy 3: Few-Shot Prompt",
        PROMPT_FEWSHOT_SYSTEM,
        PROMPT_FEWSHOT_USER,
        features,
    )
    # Few-shot example guides the format and style — the LLM mimics the
    # example output pattern without needing explicit format instructions.

    # ---- Temperature comparison ----

    print()
    print("=" * 60)
    print("  PART 2: Temperature Comparison")
    print("  (Strategy 2 at temperatures 0.1, 0.7, 1.0)")
    print("=" * 60)

    for temp in [0.1, 0.7, 1.0]:
        run_strategy(
            f"Temperature {temp}",
            PROMPT_STRUCTURED_SYSTEM,
            PROMPT_STRUCTURED_USER,
            features,
            temperature=temp,
        )
    # Low temperature (0.1)  → deterministic, focused output
    # Mid temperature (0.7)  → balanced creativity and consistency
    # High temperature (1.0) → more variable, creative phrasing
    # Run this script multiple times to see temp 1.0 vary while 0.1 stays stable.

    print()
    print("All strategies complete. Compare the outputs above!")
    print("Try editing the prompt constants at the top of this file and re-running.")


if __name__ == "__main__":
    main()
