# graph/graph_queries.py
from neo4j import GraphDatabase
import os
#from utils.config import NEO4J_URI, NEO4J_USER, NEO4J_PASS, GRAPH_HOPS
TOP_K = int(os.getenv("TOP_K", "5"))
GRAPH_HOPS = int(os.getenv("GRAPH_HOPS", "2"))
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "kakukaku123"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def find_entity_nodes_by_name(name, limit=10):
    q = "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($name) RETURN n.name LIMIT $limit"
    with driver.session() as session:
        res = session.run(q, name=name, limit=limit)
        return [r["n.name"] for r in res]

def multi_hop_paths(entity_name, hops=GRAPH_HOPS, limit=20):
    # return list of path summaries and raw path nodes/relationships
    query = f"""
    MATCH path = (n {{name:$name}})-[*1..{hops}]-(m)
    RETURN path LIMIT $limit
    """
    with driver.session() as session:
        res = session.run(query, name=entity_name, limit=limit)
        paths = []
        for r in res:
            p = r["path"]
            segments = []
            # relationships accessible via p.relationships; iterate nodes and relationships
            rels = list(p.relationships)
            for i in range(len(p.nodes)-1):
                a = p.nodes[i]["name"]
                b = p.nodes[i+1]["name"]
                rel_type = rels[i].type if i < len(rels) else ""
                note = rels[i].get("note", "") if i < len(rels) else ""
                segments.append({"from": a, "to": b, "rel": rel_type, "note": note})
            paths.append(segments)
    return paths

def get_direct_relations(entity_name, limit=50):
    q = """
    MATCH (a {name:$name})-[r]->(b)
    RETURN type(r) as rel, b.name as target, r.note as note LIMIT $limit
    """
    with driver.session() as session:
        res = session.run(q, name=entity_name, limit=limit)
        return [{"rel": r["rel"], "target": r["target"], "note": r["note"]} for r in res]
