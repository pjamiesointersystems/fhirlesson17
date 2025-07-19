import requests
from pathlib import Path

# fhirpathpy library
from fhirpathpy import evaluate as fhirpath

# fhir.resources
from fhir.resources.patient import Patient
from fhir.resources.observation import Observation
from fhir.resources.bundle import Bundle


def get_patient_from_server(patient_id: str) -> Patient:
    """
    Fetches a Patient resource from the FHIR server
    using a REST GET request and returns a Patient object.
    """
    # FHIR base URL for individual Patient read
    url = f"http://127.0.0.1:8080/csp/healthshare/demo/fhir/r4/Patient/{patient_id}"
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/fhir+json",
        "Accept-Encoding": "gzip, deflate, br",
        "Prefer": "return=representation"
    }

    response = requests.get(url, headers=headers, auth=("_System", "ISCDEMO"))
    response.raise_for_status()

    # Convert JSON to Python dict
    json_data = response.json()
    # Parse into a fhir.resources Patient object
    patient = Patient.parse_obj(json_data)
    return patient

def get_patients_from_server() -> list:
    """
    Executes a request to retrieve all the patients from the FHIR server and uses fhirpathpy
    to extract an array of Patient.
    """
    # Construct search URL (e.g., /Patient?name=Smith)
    base_url = "http://127.0.0.1:8080/csp/healthshare/demo/fhir/r4"
    url = f"{base_url}/Patient"

    headers = {
        "Accept": "*/*",
        "Content-Type": "application/fhir+json",
        "Accept-Encoding": "gzip, deflate, br",
        "Prefer": "return=representation"
    }

    response = requests.get(url, headers=headers, auth=("_System", "ISCDEMO"))
    response.raise_for_status()

    # Parse response JSON into a Python dictionary
    bundle_dict = response.json()

    # Optional: parse into a Bundle resource for validation (not strictly required for fhirpath)
    # bundle_resource = Bundle.parse_obj(bundle_dict)

    # Use FHIRPath to gather all Patient objects
    # The path "Bundle.entry.resource.id" will collect each resource within the entries
    rawpatients = fhirpath(bundle_dict, "Bundle.entry.resource")
    # Parse into a fhir.resources Patient object
    
    patients = [Patient.parse_obj(p) for p in rawpatients if p.get("resourceType") == "Patient"]
    return patients

def get_patient_from_server(patient_id: str) -> Patient:
    """
    Fetches a Patient resource from the FHIR server
    using a REST GET request and returns a Patient object.
    """
    # FHIR base URL for individual Patient read
    url = f"http://127.0.0.1:8080/csp/healthshare/demo/fhir/r4/Patient/{patient_id}"
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/fhir+json",
        "Accept-Encoding": "gzip, deflate, br",
        "Prefer": "return=representation"
    }

    response = requests.get(url, headers=headers, auth=("_System", "ISCDEMO"))
    response.raise_for_status()

    # Convert JSON to Python dict
    json_data = response.json()
    # Parse into a fhir.resources Patient object
    patient = Patient.parse_obj(json_data)
    return patient

def search_patients_get_ids(search_params: str) -> list:
    """
    Executes a FHIR search for Patients based on `search_params`
    (e.g., "name=Smith"), retrieves a Bundle, and uses fhirpathpy
    to extract an array of Patient IDs.
    """
    # Construct search URL (e.g., /Patient?name=Smith)
    base_url = "http://127.0.0.1:8080/csp/healthshare/demo/fhir/r4"
    url = f"{base_url}/Patient?{search_params}"

    headers = {
        "Accept": "*/*",
        "Content-Type": "application/fhir+json",
        "Accept-Encoding": "gzip, deflate, br",
        "Prefer": "return=representation"
    }

    response = requests.get(url, headers=headers, auth=("_System", "ISCDEMO"))
    response.raise_for_status()

    # Parse response JSON into a Python dictionary
    bundle_dict = response.json()

    # Optional: parse into a Bundle resource for validation (not strictly required for fhirpath)
    # bundle_resource = Bundle.parse_obj(bundle_dict)

    # Use FHIRPath to gather all Patient IDs
    # The path "Bundle.entry.resource.id" will collect "id" from each resource within the entries
    patient_ids = fhirpath(bundle_dict, "Bundle.entry.resource.id")

    return patient_ids

def get_observations_for_patient(fhir_id: str) -> list:
    """
    Executes a request to retrieve all the observations from the FHIR server and uses fhirpathpy
    to extract an array of Observations. The parameter is the fhir_id of the Patient
    """
    # Construct search URL (e.g., /Patient?name=Smith)
    base_url = "http://127.0.0.1:8080/csp/healthshare/demo/fhir/r4"
    url = f"{base_url}/Observation?subject=Patient/{fhir_id}"

    headers = {
        "Accept": "*/*",
        "Content-Type": "application/fhir+json",
        "Accept-Encoding": "gzip, deflate, br",
        "Prefer": "return=representation"
    }

    response = requests.get(url, headers=headers, auth=("_System", "ISCDEMO"))
    response.raise_for_status()

    # Parse response JSON into a Python dictionary
    bundle_dict = response.json()

    # Optional: parse into a Bundle resource for validation (not strictly required for fhirpath)
    # bundle_resource = Bundle.parse_obj(bundle_dict)

    # Use FHIRPath to gather all Observation objects
    # The path "Bundle.entry.resource.id" will collect each resource within the entries
    rawobservations = fhirpath(bundle_dict, "Bundle.entry.resource")
    # Parse into a fhir.resources Observation object
    
    observations = [Observation.parse_obj(obs) for obs in rawobservations if obs.get("resourceType") == "Observation"]
    return observations


def get_everything_for_patient(fhir_id: str) -> list:
    """
    Executes a request to retrieve of $everything from the FHIR server and uses fhirpathpy
    to extract an array of Resources. The parameter is the fhir_id of the Patient
    """
    # Construct search URL (e.g., /Patient?name=Smith)
    base_url = "http://127.0.0.1:8080/csp/healthshare/demo/fhir/r4"
    url = f"{base_url}/Patient/{fhir_id}/$everything"

    headers = {
        "Accept": "*/*",
        "Content-Type": "application/fhir+json",
        "Accept-Encoding": "gzip, deflate, br",
        "Prefer": "return=representation"
    }

    response = requests.get(url, headers=headers, auth=("_System", "ISCDEMO"))
    response.raise_for_status()

    # Parse response JSON into a Python dictionary
    bundle_dict = response.json()

    # Optional: parse into a Bundle resource for validation (not strictly required for fhirpath)
    # bundle_resource = Bundle.parse_obj(bundle_dict)

    # Use FHIRPath to gather all Resource objects
    rawresources = fhirpath(bundle_dict, "Bundle.entry.resource")
    # Parse into a fhir.resources Observation object
    
    # observations = [Observation.parse_obj(obs) for obs in rawobservations if obs.get("resourceType") == "Observation"]
    return rawresources

def print_fhir_resource(resource):
    """
    Recursively prints all non-None fields (including nested fields)
    from a FHIR resource object, line by line.
    """
    def print_non_none(data, prefix=""):
        if isinstance(data, dict):
            for key, value in data.items():
                if value is not None:
                    print_non_none(value, prefix=f"{prefix}{key}.")
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if item is not None:
                    print_non_none(item, prefix=f"{prefix}[{idx}].")
        else:
            print(f"{prefix[:-1]}: {data}")

    # If it's an object, convert to dict; otherwise assume it's already dict
    if hasattr(resource, "dict"):
        resource_dict = resource.dict()
    else:
        resource_dict = resource

    # Start the recursive printing
    print_non_none(resource_dict)

# ---------- Run the functions ----------
if __name__ == "__main__":
    
    print("\n ===== Printing first resource from $everything request")
    allResources =  get_everything_for_patient("2")
    res1= allResources[0]
    print_fhir_resource(res1)