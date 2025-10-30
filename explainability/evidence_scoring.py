# explainability/evidence_scoring.py
import math

def compute_confidence(sem_scores, graph_paths_count):
    # sem_scores are L2 distances from FAISS: lower = closer. We'll invert and normalize.
    if not sem_scores:
        base = 0.2
    else:
        # convert distances to similarity-like score: s = 1/(1+d)
        s = sum([1.0/(1.0 + d) for d in sem_scores]) / len(sem_scores)
        base = s
    # graph bonus: paths increase confidence
    graph_bonus = min(0.3, 0.05*graph_paths_count)
    conf = base + graph_bonus
    # normalize into 0..1
    conf = max(0.0, min(1.0, conf))
    # return percentage
    return round(conf*100, 1)

def confidence_label(conf_pct):
    if conf_pct >= 80: return "High"
    if conf_pct >= 50: return "Medium"
    return "Low"
