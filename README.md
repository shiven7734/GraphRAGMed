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

## Deployment

1. Set up environment variables in your deployment environment:
   ```bash
   # Required
   NEO4J_URI=your-neo4j-uri
   NEO4J_USER=your-neo4j-user
   NEO4J_PASS=your-neo4j-password
   OPENAI_API_KEY=your-openai-key

   # Optional with defaults
   FAISS_INDEX_PATH=medintel_plus/vectorstore/med_faiss.index
   FACTS_PICKLE=medintel_plus/vectorstore/med_facts.pkl
   LLM_BACKEND=openai
   OLLAMA_URL=http://localhost:11434
   OLLAMA_MODEL=mistral
   ```

2. Initialize the vectorstore:
   ```bash
   python -m vectorstore.ingest_embeddings
   ```
   This will create the FAISS index and facts pickle file in the specified locations.

3. Initialize the knowledge graph:
   ```bash
   python -m graph.graph_build
   ```
   This will populate your Neo4j database with the medical knowledge graph.

4. Start the application:
   ```bash
   streamlit run app/app.py
   ```

Note: The app will automatically rebuild the vectorstore if the index files are missing.

# Notes
- This is a demo; always recommend clinician verification.
- For offline demo, use Ollama and set LLM_BACKEND=ollama.

# GraphRAGMed
MedIntel+ MedIntel+ is an AI-powered health copilot that answers medical questions using a combination of knowledge graph reasoning and retrieval-augmented generation (RAG). It leverages both semantic search (vector embeddings) and graph-based evidence to provide trustworthy, explainable answers for drugs, diseases, side effects, and interactions
