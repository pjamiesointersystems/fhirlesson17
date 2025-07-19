import iris
import tiktoken
from sentence_transformers import SentenceTransformer

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

IRIS_HOST    = "127.0.0.1"
IRIS_PORT    = 1972
IRIS_NAMESPACE = "DEMO"
IRIS_USER    = "_SYSTEM"
IRIS_PWD     = "ISCDEMO"
VECTOR_TABLE = "SQLUser.PatientVectors"
MODEL_NAME   = "nomic-ai/nomic-embed-text-v1.5"
TOP_K        = 5  # number of neighbors to return by default

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def get_connection():
    return iris.connect(IRIS_HOST, IRIS_PORT, IRIS_NAMESPACE, IRIS_USER, IRIS_PWD)

def count_table_rows(conn):
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {VECTOR_TABLE}")
    return cur.fetchone()[0]

def embed_text(model, text):
    # count tokens
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    token_count = len(tokens)
    # compute embedding
    vec = model.encode(text).tolist()
    return vec, token_count

def vector_search(conn, embedding, top_k=TOP_K):
    # turn Python list into comma-joined string
    emb_csv = ",".join(f"{x:.8f}" for x in embedding)
    sql = f"""
      SELECT TOP {top_k}
        patient_id,
        patient_lastname,
        patient_firstname,
        resource_type,
        resource_id,
        VECTOR_COSINE(embedding, ?) AS cosine_distance
      FROM {VECTOR_TABLE}
      ORDER BY cosine_distance DESC
    """
    cur = conn.cursor()
    cur.execute(sql, [emb_csv])
    return cur.fetchall()

def filter_top_per_patient(results):
    """
    From a list of rows like
      (patient_id, last, first, rtype, rid, distance)
    return only the best-scoring row per patient_id,
    sorted by descending distance.
    """
    best = {}
    for row in results:
        pid, last, first, rtype, rid, dist = row
        # Keep this row if we haven't seen pid yet, or if this dist is lower
        if pid not in best or dist < best[pid][-1]:
            best[pid] = row

    # Return the filtered rows sorted by distance
    return sorted(best.values(), key=lambda row: row[-1], reverse=True)
    

# ─── MAIN UTILITY ──────────────────────────────────────────────────────────────

def main():
    # load model once
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    conn  = get_connection()
    
    # count total vectors
    total = count_table_rows(conn)
    print(f"Total vectors in `{VECTOR_TABLE}`: {total}")
    
    # prompt query
    query = input("Enter test query: ").strip()
    if not query:
        print("No query entered, exiting.")
        return
    
    # embed + token count
    emb, tok_cnt = embed_text(model, query)
    print(f"Token count for your query: {tok_cnt}")
    
    # search
    #results = vector_search(conn, emb, top_k=10)
    #results = vector_search(conn, emb, top_k=10)
    results = vector_search(conn, emb, top_k=10)
    filtered = filter_top_per_patient(results)
    # Print
    print(f"\nTop {len(filtered)} unique-patient results:")
    for i, (pid, last, first, rtype, rid, dist) in enumerate(filtered, start=1):
      print(f" {i}. {pid}, {last}, {first}, {rtype}, {rid}, cosine_distance={dist:.4f}")
if __name__ == "__main__":
    main()
