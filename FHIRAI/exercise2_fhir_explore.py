"""
Exercise 2: FHIR Bundle Exploration

Retrieves a patient's complete FHIR record ($everything) and shows
concrete numbers about its size and structure, demonstrating why raw
FHIR JSON cannot be naively sent to an LLM.

Usage:
    python exercise2_fhir_explore.py              # default patient ID "2"
    python exercise2_fhir_explore.py --patient 5   # specify patient ID
"""

import json
import sys
import argparse

import tiktoken
import requests

from config import FHIR_BASE_URL, FHIR_USERNAME, FHIR_PASSWORD
from fhir_client import get_everything_for_patient, get_resource_counts

# Context window sizes for comparison
CONTEXT_WINDOW_TOKENS = 8_192  # typical small local model


def count_tokens(text: str) -> int:
    """
    Estimate token count using tiktoken cl100k_base encoding.

    Note: cl100k_base is the GPT-3.5/4 tokenizer. Local models (Llama, Mistral)
    use different tokenizers, so actual token count may vary by ~20%. This is
    sufficient for demonstrating the context window problem and is a teaching
    moment about tokenizer differences across LLMs.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def main():
    parser = argparse.ArgumentParser(description="Explore a FHIR patient bundle")
    parser.add_argument("--patient", default="2", help="FHIR Patient ID (default: 2)")
    args = parser.parse_args()
    patient_id = args.patient

    # Retrieve the bundle
    print(f"Fetching Patient/{patient_id}/$everything from {FHIR_BASE_URL} ...\n")
    try:
        response = requests.get(
            f"{FHIR_BASE_URL}/Patient/{patient_id}/$everything",
            headers={"Accept": "application/fhir+json"},
            auth=(FHIR_USERNAME, FHIR_PASSWORD),
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(
            f"Error: Cannot connect to FHIR server at {FHIR_BASE_URL}.\n"
            "Is IRIS running? See Readme setup instructions.",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Error: FHIR server returned {e.response.status_code}.", file=sys.stderr)
        sys.exit(1)

    bundle_dict = response.json()
    entries = bundle_dict.get("entry", [])

    if not entries:
        print(f"Patient {patient_id} not found or has no resources.")
        sys.exit(0)

    # Serialize to JSON for size measurements
    bundle_json = json.dumps(bundle_dict)
    char_count = len(bundle_json)
    token_count = count_tokens(bundle_json)

    # Resource counts
    counts = get_resource_counts(bundle_dict)
    total_resources = sum(counts.values())

    # --- Output ---

    print("=" * 50)
    print("  Bundle Statistics")
    print("=" * 50)
    print(f"  Patient ID:       {patient_id}")
    print(f"  Total characters: {char_count:,}")
    print(f"  Estimated tokens: ~{token_count:,} (tiktoken cl100k_base approximation)")
    print(f"  Total resources:  {total_resources}")

    print()
    print("=" * 50)
    print("  Resource Breakdown")
    print("=" * 50)
    for rtype, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {rtype}: {count}")

    print()
    print("=" * 50)
    print("  Context Window Analysis")
    print("=" * 50)
    print(f"  Bundle tokens:  ~{token_count:,}")
    print(f"  Context window: {CONTEXT_WINDOW_TOKENS:,} tokens (typical small local model)")
    print()

    if token_count <= CONTEXT_WINDOW_TOKENS:
        pct = token_count / CONTEXT_WINDOW_TOKENS * 100
        print(f"  Bundle fits within the context window ({pct:.0f}% utilization).")
    else:
        excess = token_count - CONTEXT_WINDOW_TOKENS
        pct = token_count / CONTEXT_WINDOW_TOKENS * 100
        print(f"  Bundle EXCEEDS capacity by ~{excess:,} tokens ({pct:.0f}% of limit).")
        print("  Raw JSON cannot be sent to the LLM without processing.")
        print("  -> Exercise 3 shows how to group and extract data first.")


if __name__ == "__main__":
    main()
