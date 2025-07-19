import iris

# Establish connection
conn = iris.connect("127.0.0.1", 1972, "DEMO", "_SYSTEM", "ISCDEMO")
cursor = conn.cursor()

# Clean up old test row
cursor.execute("""
    DELETE FROM PatientVectors
    WHERE patient_id = 'test' AND resource_type = 'Condition' AND resource_id = 'test-001'
""")

# Insert a known good string
text = "This is a plain UTF-8 test string inserted at runtime for Condition."
embedding = "[" + ",".join(["0.0"] * 768) + "]"  # dummy 768-dim vector

print("Inserting test row...")
cursor.execute("""
    INSERT INTO PatientVectors (
        patient_id, patient_lastname, patient_firstname,
        resource_type, resource_id, resourcetext
    ) VALUES (?, ?, ?, ?, ?, ?)
""", ["test", "Test", "Case", "Condition", "test-001", text])
conn.commit()

# Read back
print("Retrieving test row...")
cursor.execute("""
    SELECT resourcetext FROM PatientVectors
    WHERE patient_id = 'test' AND resource_type = 'Condition' AND resource_id = 'test-001'
""")

rows = cursor.fetchall()
print(f"✅ Retrieved {len(rows)} rows.")
for i, row in enumerate(rows):
    val = row[0]
    print(f"[{i}] Type: {type(val)}, Value: {repr(val)}")

cursor.close()
conn.close()
