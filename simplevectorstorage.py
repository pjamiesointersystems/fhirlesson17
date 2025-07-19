import os
import iris
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME= "nomic-ai/nomic-embed-text-v1.5"



ptAWellness = f"""
Mrs. Thompson is a healthy 45-year-old woman who presents for her annual wellness
examination. She reports no new complaints and feels well overall. 
Her past medical history is notable only for mild seasonal allergies
managed with over-the-counter antihistamines. 
She exercises three times per week with moderate intensity,
maintains a balanced diet, and does not smoke or drink alcohol.
Vital signs today are all within normal limits: 
BP 118/72 mm Hg, HR 72 bpm, RR 14 breaths/min, T 98.4 °F, BMI 23.
She is up to date on all age-appropriate screenings, 
including a mammogram last year and colonoscopy at age 40,
both with unremarkable findings. 
Physical exam is benign: clear lungs, regular heart sounds without murmurs,
soft non-tender abdomen, no peripheral edema.
Preventive recommendations include continued exercise,
a Mediterranean-style diet, and routine follow-up in one year.
"""

ptBRoutine = f"""
Mr. Patel is a 30-year-old male who comes in for a routine physical 
required by his employer. He denies any symptoms such as chest pain,
dyspnea, or gastrointestinal upset. He has no significant past medical history 
and takes no prescription medications.
He follows a plant-forward diet, jogs 5 km three times a week, 
and practices yoga on weekends. He does not smoke, drinks socially,
and works a desk job. Today’s vitals: BP 115/70 mm Hg, HR 68 bpm, RR 12 breaths/min,
T 97.9 °F, BMI 22. On exam, his cardiovascular, pulmonary, abdominal, 
and neurological exams are within normal limits. 
Laboratory tests drawn today include lipid panel, CMP, and CBC; 
results are expected to return within normal ranges. 
He is counseled on stress management techniques and schedules 
his next annual exam in 12 months.
"""

ptCDiabetes = f"""
Mr. Garcia is a 56-year-old man with a 5-year history of type 2 diabetes mellitus,
presenting for routine follow-up. He reports occasional fasting blood sugar readings 
in the 140–150 mg/dL range and postprandial values up to 200 mg/dL. 
He’s on metformin 1000 mg BID and glipizide 5 mg daily.
He admits to reduced physical activity over the past three months due to knee pain.
Diet adherence is variable, with frequent high-carbohydrate snacks.
Vitals: BP 132/80 mm Hg, HR 78 bpm, RR 16 breaths/min, T 98.2 °F, BMI 29.
Foot exam shows no ulcers but reduced sensation to monofilament testing bilaterally. 
Labs from last visit: A1c 8.2 %. Today, labs are drawn for A1c, 
lipid profile, and renal function. Plan: increase metformin to 1000 mg TID,
refer to physical therapy for knee strengthening, 
reinforce dietary counseling with a nutritionist, 
and schedule a diabetes education class. Follow-up in 3 months.
"""

ptDDiabetes = f"""
Ms. Lee is a 48-year-old woman referred for evaluation of newly
diagnosed type 2 diabetes. She reports polyuria, polydipsia, 
and unintentional 8-lb weight loss over the past two months. 
Family history is positive for diabetes in both parents. Her current medications
include lisinopril 10 mg daily for hypertension. 
She exercises infrequently and follows a high-carb diet.
Vital signs: BP 140/88 mm Hg, HR 82 bpm, RR 18 breaths/min, T 98.6 °F, BMI 31. 
Labs drawn today show fasting glucose 160 mg/dL, A1c 8.5 %. 
Exam reveals no acanthosis nigricans; foot exam is normal.
Plan: start metformin 500 mg BID, refer for dietary counseling,
encourage brisk walking 30 minutes/day, order repeat labs in 6 weeks, 
and schedule diabetes self-management education. 
Discussed signs of hypoglycemia and when to seek urgent care.
"""

ptECHF = f"""
Mr. Johnson is a 72-year-old man with chronic systolic congestive heart failure 
(EF 35 %) presenting for routine follow-up.
He reports mild exertional dyspnea when climbing two flights of stairs, 
no orthopnea or PND. He takes lisinopril 20 mg daily, 
carvedilol 12.5 mg BID, and furosemide 40 mg daily. 
He weighs himself daily; he notes a 2 lb increase over the past week.
Diet is low in sodium but occasional lapses. 
Vitals: BP 110/68 mm Hg, HR 64 bpm, RR 16 breaths/min, 
T 98.0 °F; weight 180 lb (was 178 last visit). Physical exam reveals 
mild bibasilar crackles and trace peripheral edema. 
Labs show stable renal function and potassium 4.2 mEq/L. 
Plan: increase furosemide to 60 mg daily for one week, 
reinforce sodium restriction, schedule echocardiogram in 3 months,
and arrange home health nursing to monitor weight and symptoms. Follow-up in 4 weeks.
"""
ptFCHF = f"""
Mrs. Davis is a 68-year-old woman with heart failure with reduced ejection fraction 
(LVEF 30 %) presenting with worsening dyspnea and lower extremity swelling
over the past five days. She notes orthopnea requiring two pillows and 2 + pitting edema
to mid-calves. Her regimen includes enalapril 10 mg BID,
metoprolol succinate 50 mg daily, and spironolactone 25 mg daily.
She admits to skipping her morning dose of furosemide last week.
Vitals: BP 100/60 mm Hg, HR 88 bpm, RR 20 breaths/min, T 98.3 °F; 
weight up 5 lb since last week. Exam: jugular venous distension, bibasilar crackles,
and 3 + edema bilaterally. Labs show mild hyponatremia and rising creatinine.
Plan: admit for IV diuresis, restart home diuretics upon discharge
with closer monitoring, educate on daily weights and medication adherence,
and schedule close outpatient follow-up.
"""

SUMMARIES = [
    {"id": "A", "text": ptAWellness},
    {"id": "B", "text": ptBRoutine},
    {"id": "C", "text": ptCDiabetes},
    {"id": "D", "text": ptDDiabetes},
    {"id": "E", "text": ptECHF},
    {"id": "F", "text": ptECHF},
]

def embed_text(model, text):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    vec = model.encode(text).tolist()
    return vec, len(tokens)

VECTOR_TABLE = "PatientSummaryVectors"

class PatientSummaryIndexer:
    MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
    VECTOR_DIM = 768

    def __init__(self,
                 iris_host="127.0.0.1", iris_port=1972,
                 namespace="DEMO", username="_SYSTEM", password="ISCDEMO"):
        # load model once
        self.model = SentenceTransformer(self.MODEL_NAME, trust_remote_code=True)

        # connect to IRIS
        self.conn = iris.connect(iris_host, iris_port, namespace, username, password)
        self._ensure_table()
        
    def _ensure_table(self):
        cur = self.conn.cursor()
        # check for existing table
        cur.execute(f"""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = '{VECTOR_TABLE}'
        """)
        (count,) = cur.fetchone()
        
        if count == 0:
            cur.execute(f"""
            CREATE TABLE {VECTOR_TABLE} (
                summary_id   VARCHAR(10)   PRIMARY KEY,
                summary_text VARCHAR(4000),
                embedding VECTOR(DOUBLE, 768)
                )
            """)
            
            cur.execute(f"""
            CREATE INDEX idx_summary_vectors
            ON {VECTOR_TABLE} (embedding)
            AS HNSW(Distance='Cosine')
            """)
            self.conn.commit()
            print(f"✅ Created {VECTOR_TABLE} + HNSW index")
        else:
            print(f"ℹ️  Table {VECTOR_TABLE} already exists")
            
    def load_summaries(self, summaries=SUMMARIES):
        """Embeds & bulk-inserts each summary into IRIS."""
        max_chars = 4000
        cur = self.conn.cursor()
        for s in summaries:
            if not isinstance(s['text'], str):
              s['text'] = str(s['text'])

            text_bytes = s['text'].encode('utf-8', errors='ignore')
            vec, _ = embed_text(self.model,  s['text'])
            # format as CSV of floats
            csv = ",".join(f"{v:.8f}" for v in vec)
            params = [s["id"],  s['text'], csv]
            sql = f"""
                  INSERT INTO {VECTOR_TABLE} (summary_id, summary_text, embedding)
                  VALUES (?, ?, TO_VECTOR(?,FLOAT))
                  """
            cur.execute(sql, params)
            print(f"Inserted/Updated summary {s['id']}")
        self.conn.commit()
        
    def search(self, query: str, top_k: int = 3):
        #Runs a vector‐similarity search in IRIS and returns top_k matches."""
        # 1) Embed the query to get a Python list of floats
          vec, _ = embed_text(self.model, query)
        
        # 2) Turn that list into the comma-joined string format IRIS expects
          emb_csv = ",".join(f"{x:.8f}" for x in vec)

        # 3) Execute select
          sql = f"""
          SELECT TOP {top_k}
            summary_id,
            VECTOR_COSINE(embedding, ?) AS similarity
          FROM {VECTOR_TABLE}
          ORDER BY similarity DESC
        """

          cur = self.conn.cursor()
          cur.execute(sql, [emb_csv])
          return cur.fetchall()
      
# 4) CLI / Demo
# -------------------------------------------------------------------
if __name__ == "__main__":
    idx = PatientSummaryIndexer(
        iris_host="127.0.0.1", iris_port=1972,
        namespace="DEMO", username="_SYSTEM", password="ISCDEMO"
    )

    # load the six summaries (only do this once; comment out after first run)
    idx.load_summaries()

    while True:
        q = input("\nEnter your query (or 'quit'): ").strip()
        if not q or q.lower() in ("q", "quit", "exit"):
            break
        results = idx.search(q, top_k=3)
        print("\nTop matches:")
        for sid, score in results:
            print(f" • {sid} (score={score}): ")
        
    
        