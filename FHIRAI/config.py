"""
Configuration loader for Lesson 17 (Using LLMs with FHIR).

Loads settings from .env file in the FHIRAI/ directory.
If .env is missing, uses sensible defaults for the course environment.
Students should copy .env.example to .env and customize.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same directory as this file
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path)

if not _env_path.exists():
    print(
        "Warning: .env file not found. Using defaults. "
        "Copy .env.example to .env and customize for your environment.",
        file=sys.stderr,
    )

# FHIR Server Configuration
FHIR_BASE_URL = os.getenv(
    "FHIR_BASE_URL",
    "http://127.0.0.1:8080/csp/healthshare/demo/fhir/r4",
)
FHIR_USERNAME = os.getenv("FHIR_USERNAME", "_System")
FHIR_PASSWORD = os.getenv("FHIR_PASSWORD", "ISCDEMO")

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "lmstudio"
