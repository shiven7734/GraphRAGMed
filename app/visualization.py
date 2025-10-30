# app/visualization.py
from pyvis.network import Network
from neo4j import GraphDatabase
from utils.config import NEO4J_URI, NEO4J_USER, NEO4J_PASS
import os

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def build_pyvis_graph(center_entity, hops=2, max_nodes=100, out_path="app/graph.html"):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.toggle_physics(True)
    # fetch paths
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
    # add nodes
    for n, v in nodes.items():
        net.add_node(v["id"], label=v["label"], title=v["label"])
    for a,b,rel in edges:
        net.add_edge(a,b, title=rel, label=rel, color="#0b62a4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    net.save_graph(out_path)
    return out_path
