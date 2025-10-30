# utils/utils.py
import re
import unidecode

def normalize(text: str) -> str:
    if not text: return ""
    text = unidecode.unidecode(text)
    text = re.sub(r"[^0-9a-zA-Z\s\-\,\.]", " ", text)
    return " ".join(text.split()).strip().lower()

def simple_entity_tokens(q: str):
    # crude split but effective for demo: return capitalized tokens & ngrams
    tokens = [t for t in q.replace("/", " ").split() if len(t)>1]
    caps = [t for t in tokens if t[0].isupper()]
    return list(dict.fromkeys(caps))[:5]
