# app/app.py
import os
import sys
from pathlib import Path

# Add project root to Python path so we can import project modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import streamlit as st
from vectorstore.retriever import Retriever
from llm.llm_agent import LLMAgent
from explainability.evidence_scoring import compute_confidence, confidence_label
import streamlit.components.v1 as components
import time
from pyvis.network import Network
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "kakukaku123"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def build_pyvis_graph(center_entity, hops=2, max_nodes=100, out_path="app/graph.html"):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.toggle_physics(True)
    q = f"MATCH path=(n {{name:$name}})-[*1..{hops}]-(m) RETURN path LIMIT 200"
    nodes = {}
    edges = []
    with driver.session() as session:
        res = session.run(q, name=center_entity)
        for r in res:
            p = r["path"]
            for i in range(len(p.nodes)):
                n = p.nodes[i]["name"]
                if n not in nodes:
                    nodes[n] = {"id": len(nodes), "label": n}
            for i in range(len(p.nodes)-1):
                a = p.nodes[i]["name"]; b = p.nodes[i+1]["name"]
                rel = list(p.relationships)[i].type
                edges.append((nodes[a]["id"], nodes[b]["id"], rel))
    for n, v in nodes.items():
        net.add_node(v["id"], label=v["label"], title=v["label"])
    for a,b,rel in edges:
        net.add_edge(a,b, title=rel, label=rel, color="#0b62a4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    net.save_graph(out_path)
    return out_path

def simple_entity_tokens(q: str):
    tokens = [t for t in q.replace("/", " ").split() if len(t)>1]
    caps = [t for t in tokens if t[0].isupper()]
    return list(dict.fromkeys(caps))[:5]

st.set_page_config(page_title="MedIntel+", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .chat-user{background:#0b62a4;color:white;padding:10px;border-radius:12px}
    .chat-bot{background:#f1f1f1;padding:10px;border-radius:12px}
    .confidence-bar {height: 20px; border-radius: 8px; margin-bottom: 8px;}
    </style>
""", unsafe_allow_html=True)

tabs = st.tabs(["Ask", "History", "Graph", "About"])

with tabs[0]:
    st.title("MedIntel+ — Graph RAG Health Copilot")
    st.caption("Prototype: not for clinical use. Always consult a clinician.")
    st.sidebar.title("MedIntel+ Controls")
    backend = st.sidebar.selectbox("LLM Backend", ["openai", "ollama"], key="llm_backend_select")
    hops = st.sidebar.slider("Graph hops", 1, 3, 2, key="sidebar_hops")
    topk = st.sidebar.slider("Top-k snippets", 1, 8, 5, key="sidebar_topk")
    st.sidebar.markdown("**Demo Tips**: Ask exact drug names to hit graph.")

    # Example questions
    st.markdown("**Try an example question:**")
    example_questions = [
        "Does Metformin treat diabetes?",
        "What are the side effects of Aspirin?",
        "Is Ibuprofen safe with Peptic Ulcer?",
        "What does Lisinopril interact with?"
    ]
    ex_col = st.columns(len(example_questions))
    example_clicked = None
    for i, q in enumerate(example_questions):
        if ex_col[i].button(q, key=f"example_btn_{i}"):
            example_clicked = q

    if "history" not in st.session_state:
        st.session_state.history = []

    retriever = Retriever()
    agent = LLMAgent()

    query = st.text_input("Ask a health question (e.g. 'Can I take Metformin if I have kidney disease?')", value=example_clicked or "", key="main_query_input")
    run = st.button("Ask MedIntel+", key="ask_btn")
    if run and query.strip():
        with st.spinner("Thinking... retrieving evidence and graph"):
            try:
                res = retriever.retrieve(query)
                snippets = res["snippets"][:topk]
                graph_facts = res["graph_facts"]
                sem_scores = res.get("sem_scores", [])
                from utils.config import LLM_BACKEND
                import utils.config as cmod
                cmod.LLM_BACKEND = backend
                raw = agent.generate(query, snippets, graph_facts)
                if isinstance(raw, dict):
                    llm_out = raw
                else:
                    llm_out = {"answer": raw, "reasoning": "", "recommended_action": "", "evidence_indices": [], "graph_path_indices": [], "confidence": ""}
                conf_pct = compute_confidence(sem_scores, len(graph_facts))
                llm_out["confidence"] = f"{conf_pct}% ({confidence_label(conf_pct)})"
                st.session_state.history.append({"query": query, "answer": llm_out, "snippets": snippets, "graph": graph_facts})
            except Exception as e:
                st.error(f"Error: {e}")
    if st.session_state.history:
        last = st.session_state.history[-1]
        st.markdown("**Question:**")
        st.markdown(f"<div class='chat-user'>{last['query']}</div>", unsafe_allow_html=True)
        st.markdown("**Answer:**")
        st.markdown(f"<div class='chat-bot'>{last['answer'].get('answer','')}</div>", unsafe_allow_html=True)
        # Confidence bar
        conf = last["answer"].get("confidence", "0% (Low)")
        pct = float(conf.split('%')[0])
        color = "#4caf50" if pct >= 80 else "#ffc107" if pct >= 50 else "#f44336"
        st.markdown(f"""
            <div class='confidence-bar' style='width:{pct}%;background:{color}'></div>
            <b>Confidence:</b> {conf}
        """, unsafe_allow_html=True)
        with st.expander("Show reasoning & evidence"):
            st.write("**Reasoning:**")
            st.write(last["answer"].get("reasoning",""))
            st.write("**Recommended action:**")
            st.write(last["answer"].get("recommended_action",""))
            st.write("**Evidence snippets (indexed):**")
            for i,s in enumerate(last["snippets"]):
                st.markdown(f"- [{i}] {s}")
            st.write("**Graph facts used:**")
            for i,g in enumerate(last["graph"]):
                st.markdown(f"- [{i}] {g}")

with tabs[1]:
    st.subheader("Session history")
    if "history" in st.session_state and st.session_state.history:
        for i,h in enumerate(reversed(st.session_state.history[-10:])):
            st.markdown(f"**Q:** {h['query']}")
            st.markdown(f"**A:** {h['answer'].get('answer','')[:200]}...")
            st.write("---")
        st.subheader("Export")
        st.download_button("Export session JSON", data=str(st.session_state.history), file_name="medintel_session.json", key="history_export_btn")
    else:
        st.info("No session history yet.")

with tabs[2]:
    st.subheader("Graph Explorer")
    st.write("Explore the knowledge graph for a drug or entity.")
    graph_query = st.text_input("Enter a drug or entity name to visualize:", key="graph_query_input")
    graph_hops = st.slider("Graph hops", 1, 3, 2, key="graph_hops")
    if st.button("Show Graph", key="show_graph_btn") and graph_query:
        try:
            path = build_pyvis_graph(graph_query, hops=graph_hops)
            HtmlFile = open(path, 'r', encoding='utf-8')
            components.html(HtmlFile.read(), height=620)
        except Exception as e:
            st.error(f"Error building graph: {e}")

with tabs[3]:
    st.subheader("About")
    st.write("MedIntel+ is a hackathon project using graph RAG for health Q&A. Data sources: DrugBank, SIDER, MedlinePlus, etc.")
    st.write("Built with ❤️ using Streamlit, Neo4j, and modern LLMs.")

st.set_page_config(page_title="MedIntel+", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style> .chat-user{background:#0b62a4;color:white;padding:10px;border-radius:12px} .chat-bot{background:#f1f1f1;padding:10px;border-radius:12px} </style>", unsafe_allow_html=True)

# st.sidebar.title("MedIntel+ Controls")
# backend = st.sidebar.selectbox("LLM Backend", ["openai", "ollama"])
# hops = st.sidebar.slider("Graph hops", 1, 3, 2)
# topk = st.sidebar.slider("Top-k snippets", 1, 8, 5)
# st.sidebar.markdown("**Demo Tips**: Ask exact drug names to hit graph.")

# st.title("MedIntel+ — Graph RAG Health Copilot")
# st.caption("Prototype: not for clinical use. Always consult a clinician.")

# if "history" not in st.session_state:
#     st.session_state.history = []

retriever = Retriever()
agent = LLMAgent()

col1, col2 = st.columns([2,1])

with col1:
    query = st.text_input("Ask a health question (e.g. 'Can I take Metformin if I have kidney disease?')", "")
    run = st.button("Ask MedIntel+")
    if run and query.strip():
        # show loader
        with st.spinner("Thinking... retrieving evidence and graph"):
            # retrieve
            res = retriever.retrieve(query)
            snippets = res["snippets"][:topk]
            graph_facts = res["graph_facts"]
            sem_scores = res.get("sem_scores", [])
            # call llm
            # set backend from UI
            from utils.config import LLM_BACKEND
            # dynamical override (not persistent)
            import utils.config as cmod
            cmod.LLM_BACKEND = backend
            # generate
            raw = agent.generate(query, snippets, graph_facts)
            # If agent returns dict (preferred), use that; else adapt
            if isinstance(raw, dict):
                llm_out = raw
            else:
                llm_out = {"answer": raw, "reasoning": "", "recommended_action": "", "evidence_indices": [], "graph_path_indices": [], "confidence": ""}
            # compute confidence from sem_scores and graph facts count
            conf_pct = compute_confidence(sem_scores, len(graph_facts))
            llm_out["confidence"] = f"{conf_pct}% ({confidence_label(conf_pct)})"
            # store
            st.session_state.history.append({"query": query, "answer": llm_out, "snippets": snippets, "graph": graph_facts})
        # display
        last = st.session_state.history[-1]
        st.markdown("**Question:**")
        st.markdown(f"<div class='chat-user'>{last['query']}</div>", unsafe_allow_html=True)
        st.markdown("**Answer:**")
        st.markdown(f"<div class='chat-bot'>{last['answer'].get('answer','')}</div>", unsafe_allow_html=True)
        with st.expander("Show reasoning & evidence"):
            st.write("**Reasoning:**")
            st.write(last["answer"].get("reasoning",""))
            st.write("**Recommended action:**")
            st.write(last["answer"].get("recommended_action",""))
            st.write("**Evidence snippets (indexed):**")
            for i,s in enumerate(last["snippets"]):
                st.markdown(f"- [{i}] {s}")
            st.write("**Graph facts used:**")
            for i,g in enumerate(last["graph"]):
                st.markdown(f"- [{i}] {g}")
            st.write("**Confidence:**")
            st.write(last["answer"]["confidence"])
        # build graph visual
        if last["graph"]:
            st.subheader("Explore related graph")
            center = simple_entity_tokens(query)[0] if simple_entity_tokens(query) else None
            if center:
                path = build_pyvis_graph(center, hops=hops)
                HtmlFile = open(path, 'r', encoding='utf-8')
                components.html(HtmlFile.read(), height=620)
            else:
                st.write("No clear center entity found — try an exact drug name (e.g., 'Metformin').")
with col2:
    st.subheader("Session history")
    for i,h in enumerate(reversed(st.session_state.history[-10:])):
        st.markdown(f"**Q:** {h['query']}")
        st.markdown(f"**A:** {h['answer'].get('answer','')[:200]}...")
        st.write("---")
    st.subheader("Explainability panel")
    st.write("Shows evidence & confidence for last answer.")
    if st.session_state.history:
        last = st.session_state.history[-1]
        st.write("Confidence:", last["answer"]["confidence"])
        st.write("Number of snippets:", len(last["snippets"]))
        st.write("Number of graph facts:", len(last["graph"]))
    st.subheader("Export")
    st.download_button("Export session JSON", data=str(st.session_state.history), file_name="medintel_session.json")
