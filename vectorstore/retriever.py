# vectorstore/retriever.py
import pickle, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from utils.config import FAISS_INDEX_PATH, FACTS_PICKLE, EMB_MODEL, TOP_K
from utils.utils import normalize
from graph.graph_queries import find_entity_nodes_by_name, get_direct_relations

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
            # Initialize model on CPU first to avoid meta tensor errors
            self.model = SentenceTransformer(EMB_MODEL, device='cpu')

            # Move to GPU if available
            import torch
            if torch.cuda.is_available():
                self.model = self.model.to(torch.device("cuda"))

            # Load FAISS index
            self.index = faiss.read_index(FAISS_INDEX_PATH)

            # Load facts
            with open(FACTS_PICKLE, "rb") as f:
                self.facts = pickle.load(f)

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
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D,I = self.index.search(np.array(q_emb), top_k)
        results=[]
        for score, idx in zip(D[0], I[0]):
            if idx < len(self.facts):
                results.append({"score": float(score), "text": self.facts[idx]})
        return results

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
