# MedIntel+ — Graph RAG Health Copilot (Hackathon MVP)

## What
MedIntel+ is a demo Graph RAG system combining Neo4j (knowledge graph), FAISS (semantic retrieval), and an LLM (OpenAI or Ollama) to answer medical knowledge queries with explainable evidence.

## Quick setup (local)

1. Clone or copy the project to your machine and cd into it.

2. Create virtual env & install:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run Neo4j (Docker recommended):
```bash
docker run -d --name neo4j -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/test neo4j:latest
# open http://localhost:7474 and login neo4j/test
```

4. (Optional) create a .env file at repo root with:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASS=test
OPENAI_API_KEY=sk-...
LLM_BACKEND=openai   # or 'ollama'
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```

5. Build graph & vector store:
```bash
python graph/graph_build.py
python vectorstore/ingest_embeddings.py
```

6. Run UI:
```bash
streamlit run app/app.py
```

## Notes
- This is a demo; always recommend clinician verification.
- For offline demo, use Ollama and set LLM_BACKEND=ollama.

# GraphRAGMed
MedIntel+ MedIntel+ is an AI-powered health copilot that answers medical questions using a combination of knowledge graph reasoning and retrieval-augmented generation (RAG). It leverages both semantic search (vector embeddings) and graph-based evidence to provide trustworthy, explainable answers for drugs, diseases, side effects, and interactions
