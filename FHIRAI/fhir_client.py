"""
FHIR client module for Lesson 17 (Using LLMs with FHIR).

Provides functions to query an InterSystems IRIS for Health FHIR R4 server:
  - get_patient_from_server()      — fetch a single Patient by ID
  - get_patients_from_server()     — fetch all Patients
  - search_patients_get_ids()      — search Patients and extract IDs via FHIRPath
  - get_observations_for_patient() — fetch Observations for a patient
  - get_everything_for_patient()   — fetch Patient/$everything bundle
  - get_resource_counts()          — count resources by type in a bundle dict
  - print_fhir_resource()          — recursively print non-None fields

Configuration (FHIR server URL, credentials) is loaded from config.py.
"""

import requests
from fhirpathpy import evaluate as fhirpath
from fhir.resources.R4B.patient import Patient
from fhir.resources.R4B.observation import Observation
from config import FHIR_BASE_URL, FHIR_USERNAME, FHIR_PASSWORD

_HEADERS = {
    "Accept": "*/*",
    "Content-Type": "application/fhir+json",
    "Accept-Encoding": "gzip, deflate, br",
    "Prefer": "return=representation",
}


def get_patient_from_server(patient_id: str) -> Patient:
    """Fetch a single Patient resource by FHIR ID."""
    url = f"{FHIR_BASE_URL}/Patient/{patient_id}"
    response = requests.get(url, headers=_HEADERS, auth=(FHIR_USERNAME, FHIR_PASSWORD))
    response.raise_for_status()
    return Patient.model_validate(response.json())


def get_patients_from_server() -> list:
    """Fetch all Patient resources from the server."""
    url = f"{FHIR_BASE_URL}/Patient"
    response = requests.get(url, headers=_HEADERS, auth=(FHIR_USERNAME, FHIR_PASSWORD))
    response.raise_for_status()
    bundle_dict = response.json()
    raw = fhirpath(bundle_dict, "Bundle.entry.resource")
    return [Patient.model_validate(p) for p in raw if p.get("resourceType") == "Patient"]


def search_patients_get_ids(search_params: str) -> list:
    """Search for Patients and return a list of FHIR IDs via FHIRPath."""
    url = f"{FHIR_BASE_URL}/Patient?{search_params}"
    response = requests.get(url, headers=_HEADERS, auth=(FHIR_USERNAME, FHIR_PASSWORD))
    response.raise_for_status()
    bundle_dict = response.json()
    return fhirpath(bundle_dict, "Bundle.entry.resource.id")


def get_observations_for_patient(fhir_id: str) -> list:
    """Fetch all Observation resources for a given patient FHIR ID."""
    url = f"{FHIR_BASE_URL}/Observation?subject=Patient/{fhir_id}"
    response = requests.get(url, headers=_HEADERS, auth=(FHIR_USERNAME, FHIR_PASSWORD))
    response.raise_for_status()
    bundle_dict = response.json()
    raw = fhirpath(bundle_dict, "Bundle.entry.resource")
    return [Observation.model_validate(obs) for obs in raw if obs.get("resourceType") == "Observation"]


def get_everything_for_patient(fhir_id: str) -> list:
    """
    Fetch Patient/$everything and return a list of resource dicts.

    Returns the raw resource dicts (not parsed into fhir.resources objects)
    so callers can inspect resourceType and process flexibly.
    """
    url = f"{FHIR_BASE_URL}/Patient/{fhir_id}/$everything"
    response = requests.get(url, headers=_HEADERS, auth=(FHIR_USERNAME, FHIR_PASSWORD))
    response.raise_for_status()
    bundle_dict = response.json()
    return fhirpath(bundle_dict, "Bundle.entry.resource")


def get_resource_counts(bundle_dict: dict) -> dict:
    """
    Count resources by type in a FHIR Bundle.

    Args:
        bundle_dict: A FHIR Bundle as a Python dict (from response.json()).

    Returns:
        A dict mapping resource type to count, e.g.:
        {"Patient": 1, "Observation": 23, "Condition": 5}
    """
    resources = fhirpath(bundle_dict, "Bundle.entry.resource")
    counts = {}
    for res in resources:
        rtype = res.get("resourceType", "Unknown")
        counts[rtype] = counts.get(rtype, 0) + 1
    return counts


def print_fhir_resource(resource):
    """Recursively print all non-None fields from a FHIR resource."""
    def _print(data, prefix=""):
        if isinstance(data, dict):
            for key, value in data.items():
                if value is not None:
                    _print(value, prefix=f"{prefix}{key}.")
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if item is not None:
                    _print(item, prefix=f"{prefix}[{idx}].")
        else:
            print(f"{prefix[:-1]}: {data}")

    if hasattr(resource, "model_dump"):
        resource_dict = resource.model_dump()
    elif hasattr(resource, "dict"):
        resource_dict = resource.dict()
    else:
        resource_dict = resource
    _print(resource_dict)
