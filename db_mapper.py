import pandas as pd
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer, util

# 1. Connect to Postgres
engine = create_engine("postgresql+psycopg2://fhir_user:fhir_password@localhost:5432/fhir_terminology")

# 2. Load NAMASTE + ICD-11 data from DB
with engine.connect() as conn:
    # NAMASTE codes
    namaste = pd.read_sql(
        text("SELECT code, display, definition FROM code_system_entries WHERE system_uri LIKE '%NAMASTE%'"),
        conn
    )
    # ICD-11 codes (from concept_map_entries table)
    icd11 = pd.read_sql(
        text("SELECT DISTINCT target_code_or_uri AS code, comment AS display, '' AS definition FROM concept_map_entries"),
        conn
    )

print(f"✅ Loaded {len(namaste)} NAMASTE codes, {len(icd11)} ICD-11 codes")

# 3. Load Transformer Model
print("⏳ Loading Transformer model...")
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
print("✅ Model loaded!")

# 4. Create embeddings
print("Generating embeddings...")
namaste_embeddings = model.encode(namaste["definition"].fillna("").tolist(), convert_to_tensor=True)
# ICD-11 ka definition empty hai, toh 'display' use karenge
icd_embeddings = model.encode(icd11["display"].fillna("").tolist(), convert_to_tensor=True)
print("✅ Embeddings generated!")

# 5. Calculate similarities
cosine_scores = util.cos_sim(namaste_embeddings, icd_embeddings)

# 6. For each NAMASTE code, find best ICD-11 match
results = []
for i, row in namaste.iterrows():
    best_idx = int(cosine_scores[i].argmax())
    best_score = float(cosine_scores[i][best_idx])
    results.append({
        "namaste_code": row["code"],
        "namaste_term": row["display"],
        "icd_code": icd11.iloc[best_idx]["code"],
        "icd_term": icd11.iloc[best_idx]["display"],
        "similarity": round(best_score, 4)
    })

df_results = pd.DataFrame(results)
print("🔍 Sample Results:")
print(df_results.head())

# 7. Save results back to DB
df_results.to_sql("ml_mappings", engine, if_exists="replace", index=False)

print("🎉 Done! Mappings saved to table 'ml_mappings'")
