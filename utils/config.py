# utils/config.py
import os
import sys
from dotenv import load_dotenv
load_dotenv()

def get_project_root():
    """Get the absolute path to the project root directory."""
    # First try environment variable
    root_dir = os.getenv("PROJECT_ROOT")
    if root_dir and os.path.exists(root_dir):
        print(f"Using PROJECT_ROOT from environment: {root_dir}")
        return root_dir
    
    # Then try to detect deployment paths
    if '/mount/src/graphragmed' in sys.path:
        root_dir = '/mount/src/graphragmed'
        print(f"Detected deployment path: {root_dir}")
        return root_dir
    
    # Finally use the local development path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)  # Go up one level from utils/
    print(f"Using local development path: {root_dir}")
    return root_dir

# Get project root before defining other paths
PROJECT_ROOT = get_project_root()
print(f"Project root set to: {PROJECT_ROOT}")

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "test")

# Define paths relative to project root
VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "vectorstore")
os.makedirs(VECTORSTORE_DIR, exist_ok=True)  # Ensure directory exists

# Default paths - can be overridden by environment variables
DEFAULT_FAISS_PATH = os.path.join(VECTORSTORE_DIR, "med_faiss.index")
DEFAULT_FACTS_PATH = os.path.join(VECTORSTORE_DIR, "med_facts.pkl")

# Try to find existing index files
if os.path.exists(os.path.join(PROJECT_ROOT, "medintel_plus/vectorstore/med_faiss.index")):
    DEFAULT_FAISS_PATH = os.path.join(PROJECT_ROOT, "medintel_plus/vectorstore/med_faiss.index")
    DEFAULT_FACTS_PATH = os.path.join(PROJECT_ROOT, "medintel_plus/vectorstore/med_facts.pkl")
    print(f"Found existing index in medintel_plus: {DEFAULT_FAISS_PATH}")

# Final paths (can be overridden by environment variables)
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", DEFAULT_FAISS_PATH)
FACTS_PICKLE = os.getenv("FACTS_PICKLE", DEFAULT_FACTS_PATH)
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")

print(f"Using FAISS index path: {FAISS_INDEX_PATH}")
print(f"Using facts pickle path: {FACTS_PICKLE}")

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
