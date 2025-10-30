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
        
    # Initialize model with device handling
    try:
        import torch
        if torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                print("🚀 Using GPU for embeddings generation")
            except Exception as e:
                print(f"⚠️ GPU available but failed to initialize: {e}")
                device = torch.device("cpu")
                print("🔄 Falling back to CPU")
        else:
            device = torch.device("cpu")
            print("💻 Using CPU for embeddings generation")

        model = SentenceTransformer(EMB_MODEL, device=device)
    except Exception as e:
        print(f"⚠️ Failed to initialize model with device selection: {e}")
        print("🔄 Attempting CPU initialization")
        model = SentenceTransformer(EMB_MODEL, device='cpu')

    print("⏳ Computing embeddings...")
    try:
        embeddings = model.encode(facts, show_progress_bar=True, convert_to_numpy=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        
        # Add vectors to index
        index.add(np.array(embeddings))
        
        # Save index
        print(f"💾 Saving FAISS index with {len(facts)} vectors of dimension {dim}")
        faiss.write_index(index, FAISS_INDEX_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Failed to compute embeddings or save FAISS index: {str(e)}\n"
            "This could be due to insufficient memory or disk space."
        ) from e
    with open(FACTS_PICKLE, "wb") as f:
        pickle.dump(facts, f)
    print(f"✅ FAISS index saved to {FAISS_INDEX_PATH} with {len(facts)} facts.")

if __name__ == "__main__":
    build_faiss()
