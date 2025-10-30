# utils/config.py
import os
from dotenv import load_dotenv
load_dotenv()

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "test")

# Embeddings / FAISS
def get_project_root():
    """Get the absolute path to the project root directory."""
    # Try getting from environment variable first
    root_dir = os.getenv("PROJECT_ROOT")
    if root_dir:
        return root_dir
        
    # Otherwise calculate from current file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)  # Go up one level from utils/

# Ensure we use absolute paths
PROJECT_ROOT = get_project_root()
VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "vectorstore")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.join(VECTORSTORE_DIR, "med_faiss.index"))
FACTS_PICKLE = os.getenv("FACTS_PICKLE", os.path.join(VECTORSTORE_DIR, "med_facts.pkl"))
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")

# LLM backend
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai").lower()  # 'openai' or 'ollama'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Retriever settings
TOP_K = int(os.getenv("TOP_K", "5"))
GRAPH_HOPS = int(os.getenv("GRAPH_HOPS", "2"))

# Safety
DISALLOWED_ACTIONS = ["prescribe", "dosage", "administer"]  # simple guardrail
