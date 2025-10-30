# llm/llm_agent.py
import json
import requests
import openai
from llm.prompts import SYSTEM, TEMPLATE

# LLM backend & settings
LLM_BACKEND = "ollama"  # default backend
# OPENAI_API_KEY = "your-openai-api-key"
OLLAMA_MODEL = "mistral"
TOP_K = 5
GRAPH_HOPS = 2

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

class LLMAgent:
    def __init__(self):
        self.backend = LLM_BACKEND

    def _format_prompt(self, query, snippets, graph_facts):
        snips = "\n".join([f"[{i}] {s}" for i, s in enumerate(snippets)])
        gfs = "\n".join([f"[{i}] {g}" for i, g in enumerate(graph_facts)])
        return TEMPLATE.format(query=query, snippets=snips, graph_facts=gfs)

    def generate(self, query, snippets, graph_facts):
        prompt = self._format_prompt(query, snippets, graph_facts)
        if self.backend == "ollama":
            try:
                return self._call_ollama(prompt)
            except Exception as e:
                print(f"[Warning] Ollama failed: {e}. Falling back to OpenAI.")
                return self._call_openai(prompt)
        else:
            return self._call_openai(prompt)

    def _call_openai(self, prompt):
        try:
            # Using old API for openai 0.28.x
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":SYSTEM},
                          {"role":"user","content":prompt}],
                max_tokens=400,
                temperature=0.0
            )
            text = resp.choices[0].message.content
            try:
                return json.loads(text)
            except:
                return {"answer": text, "reasoning": "", "recommended_action": "",
                        "evidence_indices": [], "graph_path_indices": [], "confidence": ""}
        except Exception as e:
            return {"answer": "", "reasoning": f"OpenAI call failed: {e}",
                    "recommended_action": "", "evidence_indices": [], "graph_path_indices": [], "confidence": ""}

    def _call_ollama(self, prompt):
        url = f"{OLLAMA_URL}/api/generate"
        body = {
            "model": OLLAMA_MODEL,
            "messages":[{"role":"system","content":SYSTEM},
                        {"role":"user","content":prompt}],
            "max_tokens":400
        }
        try:
            r = requests.post(url, json=body, timeout=140)
            r.raise_for_status()
            data = r.json()
            out = data.get("response") or (data.get("generations") and data["generations"][0].get("text")) or str(data)
            try:
                return json.loads(out)
            except:
                return {"answer": out, "reasoning": "", "recommended_action": "",
                        "evidence_indices": [], "graph_path_indices": [], "confidence": ""}
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")
