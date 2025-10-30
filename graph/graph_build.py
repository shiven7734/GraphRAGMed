# graph/graph_build.py
import pandas as pd
from neo4j import GraphDatabase
#from utils.config import NEO4J_URI, NEO4J_USER, NEO4J_PASS
import os
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "kakukaku123"
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "drug_data.csv")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

CREATE_QUERY = """
MERGE (d:Drug {name: $drug})
MERGE (t:Entity {name: $target})
MERGE (d)-[r:RELATION {type: $relation}]->(t)
SET r.note = $note
"""

def build_graph():
    df = pd.read_csv(CSV_PATH)
    with driver.session() as session:
        for _, row in df.iterrows():
            session.execute_write(
                lambda tx, drug, relation, target, note: tx.run(CREATE_QUERY, drug=drug, relation=relation, target=target, note=note),
                row['Drug'], row['Relation'], row['Target'], str(row.get('Note', ''))
            )
    print("✅ Neo4j graph built.")

if __name__ == "__main__":
    build_graph()
