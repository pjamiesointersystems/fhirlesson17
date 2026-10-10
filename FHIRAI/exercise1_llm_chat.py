"""
Exercise 1: LLM Hello World — Local LLM Setup and First Chat

This exercise verifies your local LLM environment by sending a simple
healthcare-related prompt and printing the response. Run this first
before moving on to FHIR integration in Exercise 2.

Usage:
    python exercise1_llm_chat.py
"""

import sys
import ollama
from config import LLM_MODEL, LLM_PROVIDER

PROMPT = "Explain what an A1c test measures and why it matters for diabetes management."


def chat_with_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the response text."""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content


# --- LM Studio alternative (uncomment if LLM_PROVIDER is "lmstudio") ---
#
# import lmstudio as lms
#
# def chat_with_lmstudio(prompt: str) -> str:
#     """Send a prompt to LM Studio and return the response text."""
#     with lms.Client() as client:
#         model = client.llm.model(LLM_MODEL)
#         result = model.respond(prompt)
#         return str(result)


def main():
    print(f"Provider: {LLM_PROVIDER}")
    print(f"Model:    {LLM_MODEL}")
    print(f"\nPrompt:   {PROMPT}\n", flush=True)

    try:
        response_text = chat_with_ollama(PROMPT)
    except ConnectionError:
        print(
            "Error: Cannot connect to Ollama. "
            "Please start Ollama and try again.",
            file=sys.stderr,
        )
        sys.exit(1)
    except ollama.ResponseError as e:
        if "not found" in str(e).lower():
            print(
                f"Error: Model '{LLM_MODEL}' not found. "
                f"Run: ollama pull {LLM_MODEL}",
                file=sys.stderr,
            )
        else:
            print(f"Error from Ollama: {e}", file=sys.stderr)
        sys.exit(1)

    if not response_text:
        print("No response received from the LLM.")
        sys.exit(1)

    print(f"Response:\n{response_text}")


if __name__ == "__main__":
    main()
