import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np
import logging


logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


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

summaries = [
    {"id": "A", "text": ptAWellness},
    {"id": "B", "text": ptBRoutine},
    {"id": "C", "text": ptCDiabetes},
    {"id": "D", "text": ptDDiabetes},
    {"id": "E", "text": ptECHF},
    {"id": "F", "text": ptECHF},
]

def embed_text(model, text):
    # count tokens
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    token_count = len(tokens)
    # compute embedding
    vec = model.encode(text).tolist()
    return vec, token_count


# Generate & store embeddings
def build_index(model, summaries):
    index = []
    for s in summaries:
        vec, count = embed_text(model, s["text"])
        index.append({
            "id": s["id"],
            "text": s["text"],
            "embedding": np.array(vec),
            "tokens": count
        })
    return index

model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
index = build_index(model, summaries)


# Cosine similarity helper
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Search function
def search(query: str, model, index, top_k=1):
    q_vec, _ = embed_text(model, query)
    q_vec = np.array(q_vec)
    # compute similarity against each indexed summary
    sims = [
        (item["id"], cosine_sim(q_vec, item["embedding"]))
        for item in index
    ]
    # sort by descending similarity
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return sims[:top_k]

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    results = search(user_query, model, index, top_k=5)
    for result in results:
        id, score = result
        print(f"Patient {id} has a match score of {score:.4f}")