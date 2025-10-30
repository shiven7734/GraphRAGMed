# vectorstore/retriever.py
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from utils.config import FAISS_INDEX_PATH, FACTS_PICKLE, EMB_MODEL, TOP_K, PROJECT_ROOT
from utils.utils import normalize
from graph.graph_queries import find_entity_nodes_by_name, get_direct_relations

def get_faiss_cpu_index(index_path):
    """Load a FAISS index and ensure it runs on CPU."""
    try:
        # Try loading with standard FAISS (non-AVX2)
        import faiss.swigfaiss as cpu_faiss
        
        # Try direct CPU load
        try:
            print("🔄 Attempting direct CPU index load")
            index = cpu_faiss.read_index(index_path)
        except Exception as e1:
            print(f"⚠️ Direct CPU load failed: {str(e1)}")
            # Try loading normally then converting
            print("🔄 Trying load-then-convert approach")
            index = faiss.read_index(index_path)
            if faiss.get_num_gpus() > 0:
                print("🔄 Converting GPU index to CPU version")
                index = faiss.index_gpu_to_cpu(index)
        
        return index
    except Exception as e:
        raise RuntimeError(
            f"Failed to load FAISS index in CPU mode: {str(e)}\n"
            "This may indicate incompatible CPU instructions or a corrupted index."
        )

def safe_faiss_load(index_path):
    """Safely load a FAISS index with fallbacks."""
    try:
        # First try normal load using AVX2
        try:
            from faiss import swigfaiss_avx2 as swigfaiss
        except ImportError:
            print("⚠️ AVX2 FAISS not available, falling back to standard FAISS")
            from faiss import swigfaiss
        
        try:
            return faiss.read_index(index_path)
        except Exception as e:
            print(f"⚠️ Initial FAISS load failed: {str(e)}")
            print("🔄 Attempting CPU-only version...")
            return get_faiss_cpu_index(index_path)
    except Exception as e:
        print(f"❌ All FAISS loading attempts failed: {str(e)}")
        raise RuntimeError(
            "Failed to load FAISS index with both AVX2 and CPU methods. "
            "Try rebuilding the index on the deployment machine: "
            "python -m vectorstore.ingest_embeddings"
        ) from e

# Simple context compressor (keep highest scoring snippets and shorten them)
def compress_snippets(snippets, max_chars=1500):
    # naive compressor: join until limit
    out = []
    s = 0
    for text in snippets:
        if s + len(text) > max_chars: break
        out.append(text)
        s += len(text)
    return out

class Retriever:
    def __init__(self):
        try:
            # Initialize device
            import torch
            if torch.cuda.is_available():
                try:
                    device = torch.device("cuda")
                    print("🚀 Using GPU for inference")
                except Exception as e:
                    print(f"⚠️ GPU available but failed to initialize: {e}")
                    device = torch.device("cpu")
                    print("🔄 Falling back to CPU")
            else:
                device = torch.device("cpu")
                print("💻 Using CPU for inference")

            # Initialize model
            try:
                self.model = SentenceTransformer(EMB_MODEL, device=device)
            except Exception as e:
                print(f"⚠️ Failed to initialize model on {device}: {e}")
                print("🔄 Attempting CPU initialization")
                self.model = SentenceTransformer(EMB_MODEL, device='cpu')
                
            # Check if index and facts exist
            if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FACTS_PICKLE):
                print("⏳ FAISS index or facts not found. Building them...")
                from .ingest_embeddings import build_faiss
                build_faiss()

            # Print environment information for debugging
            print("\n🔍 Environment Information:")
            print(f"Working Directory: {os.getcwd()}")
            print(f"Project Root: {PROJECT_ROOT}")
            print(f"FAISS Index Path: {FAISS_INDEX_PATH}")
            
            # First check if the index exists at the configured path
            if os.path.exists(FAISS_INDEX_PATH):
                index_path = FAISS_INDEX_PATH
                print(f"✅ Found index at configured path: {index_path}")
            else:
                # Try alternative paths
                possible_paths = [
                    os.path.join(PROJECT_ROOT, "vectorstore/med_faiss.index"),
                    os.path.join(PROJECT_ROOT, "medintel_plus/vectorstore/med_faiss.index"),
                    os.path.join(os.getcwd(), "vectorstore/med_faiss.index"),
                    os.path.join(os.getcwd(), "medintel_plus/vectorstore/med_faiss.index"),
                ]
                
                # Try each path
                for path in possible_paths:
                    if os.path.exists(path):
                        index_path = path
                        print(f"✅ Found index at alternative path: {index_path}")
                        break
                else:
                    index_path = None
            
            if index_path is None:
                attempted_paths = [
                    FAISS_INDEX_PATH,
                    os.path.join(PROJECT_ROOT, "vectorstore/med_faiss.index"),
                    os.path.join(PROJECT_ROOT, "medintel_plus/vectorstore/med_faiss.index"),
                    os.path.join(os.getcwd(), "vectorstore/med_faiss.index"),
                    os.path.join(os.getcwd(), "medintel_plus/vectorstore/med_faiss.index")
                ]
                raise RuntimeError(
                    "FAISS index not found. Attempted paths:\n" +
                    "\n".join(f"{i+1}. {path}" for i, path in enumerate(attempted_paths)) +
                    "\n\nPlease ensure the vectorstore is properly initialized by running:\n"
                    "python -m vectorstore.ingest_embeddings"
                )
            
            # Load FAISS index
            try:
                print(f"📂 Loading FAISS index from {index_path}")
                self.index = safe_faiss_load(index_path)
                print("✅ FAISS index loaded successfully")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load FAISS index from {index_path}\n"
                    f"Error: {str(e)}\n\n"
                    "This might be due to:\n"
                    "1. Missing or corrupted index file\n"
                    "2. Incompatible CPU instructions\n"
                    "3. Insufficient permissions\n\n"
                    "Try rebuilding the index by running:\n"
                    "python -m vectorstore.ingest_embeddings"
                ) from e

            # Load facts
            try:
                with open(FACTS_PICKLE, "rb") as f:
                    self.facts = pickle.load(f)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load facts from {FACTS_PICKLE}. "
                    "Please ensure the vectorstore is properly initialized by running: "
                    "python -m vectorstore.ingest_embeddings"
                ) from e

        except NotImplementedError as e:
            msg = str(e)
            if "meta tensor" in msg:
                raise RuntimeError(
                    "Model loading failed due to a meta tensor error. "
                    "This usually means the model weights are missing or corrupted. "
                    "Please delete the model cache at C:/Users/<YourUsername>/.cache/huggingface/hub and re-run the app to re-download the model. "
                    "If you have a slow or unreliable internet connection, download the model manually from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2."
                ) from e
            else:
                raise
        except Exception as e:
            print(f"Error initializing Retriever: {str(e)}")
            raise


    def semantic_topk(self, query, top_k=TOP_K):
        try:
            # Generate query embedding
            q_emb = self.model.encode([query], convert_to_numpy=True)
            
            # Ensure embedding is in the correct format
            q_emb = np.array(q_emb).astype('float32')
            
            # Perform search
            D, I = self.index.search(q_emb, top_k)
            
            # Format results
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx < len(self.facts):
                    results.append({
                        "score": float(score), 
                        "text": self.facts[idx]
                    })
            return results
            
        except Exception as e:
            print(f"⚠️ Semantic search failed: {str(e)}")
            print("🔄 Falling back to keyword search")
            return [{"score": 1.0, "text": t} for t in self.keyword_fallback(query)]

    def keyword_fallback(self, query):
        # return facts that contain any token (very simple)
        tokens = [t for t in normalize(query).split() if len(t)>2]
        hits=[]
        for f in self.facts:
            lf = normalize(f)
            if any(tok in lf for tok in tokens[:6]):
                hits.append(f)
        return hits[:TOP_K]

    def graph_hits(self, query):
        ents = find_entity_nodes_by_name(query, limit=6)
        rels=[]
        for e in ents:
            rels.extend(get_direct_relations(e))
        # format
        return [f"{e['rel']} -> {e['target']} ({e.get('note','')})" for e in rels]

    def retrieve(self, query):
        sem = self.semantic_topk(query)
        kb_texts = [s["text"] for s in sem]
        if len(kb_texts) < 2:
            kb_texts += self.keyword_fallback(query)
        # compress
        compressed = compress_snippets(kb_texts)
        graph_texts = self.graph_hits(query)
        return {"snippets": compressed, "graph_facts": graph_texts, "sem_scores": [s["score"] for s in sem]}
