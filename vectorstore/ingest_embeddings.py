# vectorstore/ingest_embeddings.py
import os, pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
#from utils.config import FAISS_INDEX_PATH, FACTS_PICKLE, EMB_MODEL
# Embeddings / FAISS
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "medintel_plus/vectorstore/med_faiss.index")
FACTS_PICKLE = os.getenv("FACTS_PICKLE", "medintel_plus/vectorstore/med_facts.pkl")
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "drug_data.csv")

def build_faiss():
    # Ensure vectorstore directory exists
    index_dir = os.path.dirname(FAISS_INDEX_PATH)
    os.makedirs(index_dir, exist_ok=True)
    
    print(f"📁 Using vectorstore directory: {index_dir}")
    
    # Load and prepare data
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Data file not found: {CSV_PATH}\n"
            "Please ensure drug_data.csv is present in the data directory."
        )
        
    df = pd.read_csv(CSV_PATH)
    facts = []
    for _, r in df.iterrows():
        fact = f"{r['Drug']} {r['Relation'].replace('_',' ')} {r['Target']}. {r.get('Note','')}"
        facts.append(fact)
        
    # Initialize model and compute embeddings
    model = SentenceTransformer(EMB_MODEL)
    print("⏳ Computing embeddings...")
    embeddings = model.encode(facts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FACTS_PICKLE, "wb") as f:
        pickle.dump(facts, f)
    print(f"✅ FAISS index saved to {FAISS_INDEX_PATH} with {len(facts)} facts.")

if __name__ == "__main__":
    build_faiss()
