import pandas as pd
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer, util

# --- Database connection ---
engine = create_engine("postgresql+psycopg2://fhir_user:fhir_password@localhost:5432/fhir_terminology")

# --- Load NAMASTE (source codes) ---
namaste = pd.read_sql(
    text("SELECT DISTINCT source_code AS code, comment AS display FROM concept_map_entries"),
    engine
)

# --- Load ICD (target codes) ---
icd11 = pd.read_sql(
    text("SELECT DISTINCT target_code_or_uri AS code, comment AS display FROM concept_map_entries"),
    engine
)

# --- Load Manual Mappings ---
manual = pd.read_sql(
    text("SELECT source_code, comment, target_code_or_uri FROM concept_map_entries"),
    engine
)

print(f" Loaded {len(namaste)} NAMASTE codes, {len(icd11)} ICD-11 codes, {len(manual)} manual mappings")

# --- Transformer Model ---
print(" Loading Transformer model...")
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
print(" Model loaded!")

# --- Embeddings ---
print(" Generating embeddings...")
namaste_embeddings = model.encode(namaste["display"].tolist(), convert_to_tensor=True)
icd_embeddings = model.encode(icd11["display"].tolist(), convert_to_tensor=True)
print(" Embeddings generated!")

# --- Similarity Calculation ---
cosine_scores = util.cos_sim(namaste_embeddings, icd_embeddings)

results = []
for i, n_row in namaste.iterrows():
    top_match = cosine_scores[i].cpu().numpy().argmax()
    results.append({
        "namaste_code": n_row["code"],
        "namaste_term": n_row["display"],
        "icd_code": icd11.iloc[top_match]["code"],
        "icd_term": icd11.iloc[top_match]["display"],
        "similarity": float(cosine_scores[i][top_match])
    })

df_results = pd.DataFrame(results)
print(" Sample Transformer Results:")
print(df_results.head())

# --- Append Manual Mappings Below ---
manual_results = manual.rename(columns={
    "source_code": "namaste_code",
    "comment": "namaste_term",
    "target_code_or_uri": "icd_code"
})
manual_results["icd_term"] = manual_results["namaste_term"]
manual_results["similarity"] = 1.0  # Force as perfect match

# Merge both
final_df = pd.concat([df_results, manual_results], ignore_index=True)

# --- Save to DB ---
final_df.to_sql("ml_mappings", engine, if_exists="replace", index=False)
print(" Done! Combined transformer + manual mappings saved to 'ml_mappings'")
