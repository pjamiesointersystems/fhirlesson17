from regex import R
import iris
import tiktoken
from sentence_transformers import SentenceTransformer
import lmstudio as lms
from typing import List, Tuple

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

IRIS_HOST    = "127.0.0.1"
IRIS_PORT    = 1972
IRIS_NAMESPACE = "DEMO"
IRIS_USER    = "_SYSTEM"
IRIS_PWD     = "ISCDEMO"
VECTOR_TABLE = "SQLUser.PatientVectors"
MODEL_NAME   = "nomic-ai/nomic-embed-text-v1.5"
TOP_K        = 5  # number of neighbors to return by default
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
VECTOR_DIM = 768

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def get_connection():
    return iris.connect(IRIS_HOST, IRIS_PORT, IRIS_NAMESPACE, IRIS_USER, IRIS_PWD)

def count_table_rows(conn):
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {VECTOR_TABLE}")
    return cur.fetchone()[0]

def embed_text(model, text: str) -> List[float]:
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    # optionally truncate tokens here
    vec = model.encode(text).tolist()
    return vec


client = lms.Client()
llm = client.llm.model("mistral-7b-instruct-v0.3")
# Embedding model
embedder = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
conn = iris.connect("127.0.0.1", 1972, "DEMO", "_SYSTEM", "ISCDEMO")


def main(fhir_id: str, query: str) -> str:
        # 1) embed the query
        vec = embed_text(embedder, query)
        csv = ",".join(f"{x:.8f}" for x in vec)

        # 2) retrieve top-K passages for this patient
        sql = f"""
          SELECT TOP {TOP_K}
            resource_id,
            patient_lastname,
            patient_firstname, 
            CAST(resourcetext AS VARCHAR(4000)), 
            VECTOR_COSINE(embedding, TO_VECTOR(?,DOUBLE)) AS score
          FROM {VECTOR_TABLE}
          WHERE patient_id = ? AND resource_type = 'Condition'
          ORDER BY score DESC
        """
        cur = conn.cursor()
        cur.execute(sql, [csv, fhir_id])
        id_sim_pairs = cur.fetchall()

        if not id_sim_pairs:
            return "No matching data found for that patient."
        
        cur = self.conn.cursor()
    
      
      
if __name__ == "__main__":
    main("2", "Does the patient have diabetes?")
