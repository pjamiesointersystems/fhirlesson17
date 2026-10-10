"""
Exercise 3: Patient Summarization CLI Pipeline

Demonstrates the full AI+FHIR pipeline: retrieve a patient's FHIR bundle,
group resources by type, extract clinical features, and generate a
structured clinical summary via a local LLM.

This solves the problem shown in Exercise 2 — raw JSON is too large and
unstructured. Grouping by resource type and extracting clinical fields
produces compact, coherent input for the LLM.

Usage:
    python exercise3_summarizer.py                # default patient ID "2"
    python exercise3_summarizer.py --patient 5    # specify patient ID
"""

import json
import sys
import argparse

import ollama
import tiktoken
import requests

from config import FHIR_BASE_URL, FHIR_USERNAME, FHIR_PASSWORD, LLM_MODEL

SYSTEM_PROMPT = """You are a clinical AI assistant reviewing patient data.
Summarize the following patient data into these sections:
- Demographics
- Active Conditions
- Medications
- Recent Encounters
- Recent Observations
- Allergies
- Notable Findings

Do not invent information not present in the data.
If a section has no data, state 'None recorded.'"""


# ---------------------------------------------------------------------------
# Phase 2: Resource Grouping
# ---------------------------------------------------------------------------

def group_resources(resources: list) -> dict:
    """Group a list of FHIR resource dicts by resourceType."""
    grouped = {}
    for res in resources:
        rtype = res.get("resourceType", "Unknown")
        grouped.setdefault(rtype, []).append(res)
    return grouped


# ---------------------------------------------------------------------------
# Phase 3: Clinical Feature Extraction
# ---------------------------------------------------------------------------

def _safe(d, *keys, default=""):
    """Safely traverse nested dicts/lists."""
    val = d
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        elif isinstance(val, list) and isinstance(k, int) and k < len(val):
            val = val[k]
        else:
            return default
        if val is None:
            return default
    return val


def extract_patient(resources: list) -> str:
    """Extract demographics from Patient resources."""
    if not resources:
        return "No patient demographic data."
    p = resources[0]
    names = p.get("name", [])
    if names:
        name = f"{' '.join(names[0].get('given', []))} {names[0].get('family', '')}"
    else:
        name = "Unknown"
    gender = p.get("gender", "unknown")
    birth = p.get("birthDate", "unknown")
    return f"Name: {name}, Gender: {gender}, Date of Birth: {birth}"


def extract_conditions(resources: list) -> str:
    """Extract condition summaries."""
    if not resources:
        return "None recorded."
    lines = []
    for c in resources:
        display = (_safe(c, "code", "text")
                   or _safe(c, "code", "coding", 0, "display")
                   or "Unknown condition")
        status = _safe(c, "clinicalStatus", "coding", 0, "code", default="unknown")
        onset = _safe(c, "onsetDateTime", default="unknown onset")
        lines.append(f"- [{status}] {display} (onset: {onset[:10]})")
    return "\n".join(lines)


def extract_observations(resources: list, limit: int = 20) -> str:
    """Extract recent observations (most recent first, limited)."""
    if not resources:
        return "None recorded."
    # Sort by effectiveDateTime descending
    sorted_obs = sorted(
        resources,
        key=lambda o: o.get("effectiveDateTime", ""),
        reverse=True,
    )[:limit]
    lines = []
    for o in sorted_obs:
        display = (_safe(o, "code", "text")
                   or _safe(o, "code", "coding", 0, "display")
                   or "Unknown")
        value = _safe(o, "valueQuantity", "value", default="")
        unit = _safe(o, "valueQuantity", "unit", default="")
        date = _safe(o, "effectiveDateTime", default="")[:10]
        if value:
            lines.append(f"- {display}: {value} {unit} ({date})")
        else:
            lines.append(f"- {display} ({date})")
    if len(resources) > limit:
        lines.append(f"  ... and {len(resources) - limit} more observations")
    return "\n".join(lines)


def extract_medications(resources: list) -> str:
    """Extract medication request summaries."""
    if not resources:
        return "None recorded."
    lines = []
    for m in resources:
        display = (_safe(m, "medicationCodeableConcept", "text")
                   or _safe(m, "medicationCodeableConcept", "coding", 0, "display")
                   or "Unknown medication")
        status = m.get("status", "unknown")
        lines.append(f"- [{status}] {display}")
    return "\n".join(lines)


def extract_encounters(resources: list, limit: int = 10) -> str:
    """Extract recent encounter summaries."""
    if not resources:
        return "None recorded."
    sorted_enc = sorted(
        resources,
        key=lambda e: _safe(e, "period", "start", default=""),
        reverse=True,
    )[:limit]
    lines = []
    for e in sorted_enc:
        etype = (_safe(e, "type", 0, "text")
                 or _safe(e, "type", 0, "coding", 0, "display")
                 or "Unknown type")
        start = _safe(e, "period", "start", default="unknown date")[:10]
        status = e.get("status", "unknown")
        lines.append(f"- [{status}] {etype} ({start})")
    if len(resources) > limit:
        lines.append(f"  ... and {len(resources) - limit} more encounters")
    return "\n".join(lines)


def extract_allergies(resources: list) -> str:
    """Extract allergy/intolerance summaries."""
    if not resources:
        return "None recorded."
    lines = []
    for a in resources:
        substance = (_safe(a, "code", "text")
                     or _safe(a, "code", "coding", 0, "display")
                     or "Unknown substance")
        lines.append(f"- {substance}")
    return "\n".join(lines)


def build_feature_text(grouped: dict) -> str:
    """Build compact clinical feature text from grouped resources."""
    sections = [
        ("PATIENT DEMOGRAPHICS", extract_patient(grouped.get("Patient", []))),
        ("CONDITIONS", extract_conditions(grouped.get("Condition", []))),
        ("OBSERVATIONS (RECENT)", extract_observations(grouped.get("Observation", []))),
        ("MEDICATIONS", extract_medications(grouped.get("MedicationRequest", []))),
        ("ENCOUNTERS (RECENT)", extract_encounters(grouped.get("Encounter", []))),
        ("ALLERGIES", extract_allergies(grouped.get("AllergyIntolerance", []))),
    ]
    parts = []
    for heading, content in sections:
        parts.append(f"## {heading}\n{content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Token counting (same approximation as Exercise 2)
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Estimate token count using tiktoken cl100k_base (approximation for local models)."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FHIR Patient Summarization Pipeline")
    parser.add_argument("--patient", "-p", default="2", help="FHIR Patient ID (default: 2)")
    args = parser.parse_args()
    patient_id = args.patient

    # Phase 1: Retrieve FHIR bundle
    print(f"Fetching bundle for patient {patient_id}...")
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

    # Phase 2: Group by resource type
    grouped = group_resources(resources)
    total = len(resources)
    print(f"Grouping {total} resources by type...")
    for rtype, items in sorted(grouped.items(), key=lambda x: -len(x[1])):
        print(f"  {rtype}: {len(items)}")

    # Phase 3: Extract clinical features
    print("Extracting clinical features...")
    feature_text = build_feature_text(grouped)
    feature_tokens = count_tokens(feature_text)
    bundle_tokens = count_tokens(json.dumps(bundle_dict))
    print(f"  Extracted features: ~{feature_tokens:,} tokens "
          f"(vs. raw bundle: ~{bundle_tokens:,} tokens — "
          f"{(1 - feature_tokens / bundle_tokens) * 100:.0f}% reduction)")

    # Phase 4: Send to LLM
    print(f"Generating summary with {LLM_MODEL}...")
    try:
        llm_response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": feature_text},
            ],
        )
        summary = llm_response.message.content
    except ConnectionError:
        print("Error: Cannot connect to Ollama. Please start Ollama.", file=sys.stderr)
        sys.exit(1)
    except ollama.ResponseError as e:
        if "not found" in str(e).lower():
            print(f"Error: Model '{LLM_MODEL}' not found. Run: ollama pull {LLM_MODEL}",
                  file=sys.stderr)
        else:
            print(f"Error from Ollama: {e}", file=sys.stderr)
        sys.exit(1)

    if not summary:
        print("No response received from the LLM.")
        sys.exit(1)

    # Phase 5: Display summary
    print()
    print("=" * 60)
    print("  Patient Clinical Summary")
    print("=" * 60)
    print()
    print(summary)
    print()
    print("Summary complete.")


if __name__ == "__main__":
    main()
