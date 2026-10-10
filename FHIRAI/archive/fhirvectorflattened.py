from fhir.resources.bundle import Bundle
from fhir.resources.patient import Patient
import os
import iris
import json
import tiktoken
from typing import List, Dict
import lmstudio as lms
import decimal
import sys
import asyncio
from fhirpathpy import evaluate as fhirpath
from getSearchPatients import get_everything_for_patient, search_patients_get_ids
from sentence_transformers import SentenceTransformer

VECTOR_TABLE = "PatientVectorsDemo"

RESOURCE_TYPES = [
    "Patient", "Condition", "MedicationRequest", "Observation", "Encounter",
    "Practitioner", "Procedure", "AllergyIntolerance", "Immunization",
    "DiagnosticReport", "DocumentReference", "CarePlan"
]


class FHIRVectors:
    def __init__(self):
        patientIds = search_patients_get_ids('')
        print("Patient Ids in the FHIR Repository")
        print(patientIds)
        print("")
        for patientId in patientIds:
            FHIRVector(patientId)
        print("All patients in the repository processed")

class FHIRVector:
    def __init__(self, ptFHIRid, **kwargs):
        super().__init__(**kwargs)
        self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        self.conn = self.get_connection()
        cursor = self.conn.cursor()
        self.ensure_patient_vectors_table(cursor)
        self.fhirId = ptFHIRid
        self.patientId = self.fhirId
        self.lastName = ''
        self.firstName = ''
        self.create_vectors()

    def get_connection(self):
        return iris.connect("127.0.0.1", 1972, "DEMO", "_SYSTEM", "ISCDEMO")

    def ensure_patient_vectors_table(self, cursor):
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = '{VECTOR_TABLE}'
        """)
        (count,) = cursor.fetchone()

        if count == 0:
            cursor.execute(f"""
                CREATE TABLE {VECTOR_TABLE} (
                    patient_id VARCHAR(75),
                    patient_lastname VARCHAR(75),
                    patient_firstname VARCHAR(75),
                    resource_type VARCHAR(50),
                    resource_id VARCHAR(75),
                    embedding VECTOR(DOUBLE, 768),
                    resourcetext VARCHAR(4000)
                )
            """)
            cursor.execute(f"""
            CREATE INDEX vector_index
            ON TABLE {VECTOR_TABLE} (embedding)
            AS HNSW(Distance='Cosine')
             """)
            print(f"Table '{VECTOR_TABLE}' and HNSW index created.")
        else:
            print(f"Table '{VECTOR_TABLE}' already exists.")

    def get_patient_bundle(self, patfhirid: str) -> list:
        return get_everything_for_patient(patfhirid)

    def extract_resources(self, bundle: list, resource_type: str) -> list:
        return fhirpath(bundle, f"where(resourceType = '{resource_type}')")

    def truncate_to_tokens(self, text: str, max_tokens: int = 1500) -> str:
        if not isinstance(text, str):
            raise TypeError(f"Expected a string, but got {type(text).__name__}: {text}")
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = enc.encode(text)
        return enc.decode(tokens[:max_tokens])

    def flatten_fhir_resource(self, resource: dict) -> str:
        def recurse(obj, prefix=""):
            lines = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    lines.extend(recurse(v, prefix + k.capitalize() + ": "))
            elif isinstance(obj, list):
                for item in obj:
                    lines.extend(recurse(item, prefix))
            else:
                lines.append(f"{prefix}{str(obj)}")
            return lines
        return "\n".join(recurse(resource))
    
    
    def create_one_vector(self, resource, text):
      cursor = self.conn.cursor()
      max_chars = 4000

      try:
        # Ensure text is UTF-8, max 4000 characters
        if not isinstance(text, str):
            text = str(text)

        text_bytes = text.encode('utf-8', errors='ignore')
        text = text_bytes[:max_chars].decode('utf-8', errors='ignore')

            # Skip blank or invalid text
        if not text.strip():
            print(f"❌ Skipping insert: text is empty.")
            return

        # Generate embedding
        embedding = self.model.encode(text).tolist()
        embedding_strs = [f"{x:.8f}" for x in embedding]
        embedding_csv = ",".join(embedding_strs)

        # Escape strings for SQL
        pid = self.patientId.replace("'", "''")
        last = self.lastName.replace("'", "''")
        first = self.firstName.replace("'", "''")
        rtype = resource['resourceType'].replace("'", "''")
        rid = resource['id'].replace("'", "''")
        text_sql = text.replace("'", "''")
        params = [pid, last, first, rtype, rid, embedding_csv, text_sql]

        # Final SQL INSERT
        sql = f"""
             INSERT INTO {VECTOR_TABLE} (patient_id, patient_lastname, patient_firstname, resource_type, resource_id, embedding, resourcetext) VALUES (?, ?, ?, ?, ?, TO_VECTOR(?,FLOAT), ?)
             """

        cursor.execute(sql, params)
        #print("DEBUG SQL PREVIEW:", sql)
        self.conn.commit()
        print(f"✅ Inserted {rtype}/{rid}")

      except Exception as e:
          print(f"❌ Failed to insert resource {resource.get('resourceType')}/{resource.get('id')}: {e}")
          print(f"    Text preview: {text[:200]}")

    

    def create_vectors(self) -> None:
        counter = 0
        bundle = self.get_patient_bundle(self.fhirId)
        patientResources = self.extract_resources(bundle, "Patient")
        if not patientResources:
            print("Can not find the patient resource, vector creation terminated")
            return
        elif len(patientResources) > 1:
            print("Found more than one patient resource, vector creation terminated")
            return
        else:
            pt = patientResources[0]
            name = pt['name'][0]
            self.firstName = name['given'][0] if name['given'] else ""
            self.lastName = name['family'] if name['family'] else ""

        for rtype in RESOURCE_TYPES:
            resources = self.extract_resources(bundle, rtype)
            if not resources:
                print(f"_No {rtype} resources found._ for {self.patientId}")
            else:
                for res in resources:
                    try:
                        flat_text = self.flatten_fhir_resource(res)
                        flat_text.encode('utf-8')  # validate encoding
                        chunked = self.truncate_to_tokens(flat_text)
                        self.create_one_vector(res, chunked)
                        counter += 1
                        if counter % 10 == 0:
                            print(f"{counter} vectors processed.")
                    except Exception as e:
                        print(f"❌ Skipping invalid resource {res.get('resourceType')}/{res.get('id')}: {e}")

        print(f"All vectors processed for patient with id = {self.patientId}")
        
if __name__ == '__main__':
    app = FHIRVectors()

