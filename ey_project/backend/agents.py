import os
import json
import requests
from dotenv import load_dotenv

from backend.retriever import retrieve
from backend.prompts import (
    LIT_AGENT_PROMPT,
    SCORER_PROMPT,
    EXPERIMENT_PROMPT
)

load_dotenv()

# ---------------------------------------------------------------
# OLLAMA MODEL SELECTION
# ---------------------------------------------------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# ---------------------------------------------------------------
# LOW‑LEVEL LLM CALLER (Ollama only)
# ---------------------------------------------------------------
def llm(prompt: str) -> str:
    """
    Calls a local Ollama model running at:
        http://localhost:11434/api/generate

    Returns raw string text.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()

    except Exception as e:
        return f"OLLAMA_ERROR: {e}"

# ---------------------------------------------------------------
# LITERATURE → HYPOTHESES AGENT
# ---------------------------------------------------------------
def literature_agent(query: str):
    """
    Retrieves documents + asks the LLM to produce:
        - hypotheses
        - evidence mapping
        - JSON output ONLY

    Returns: (parsed_dict, docs_list)
    """

    docs = retrieve(query, k=5)

    doc_text = "\n\n".join(
        f"ID:{d['id']}\n{d['text']}"
        for d in docs
    )

    prompt = (
        LIT_AGENT_PROMPT
        + "\n\nYou are a local model running via Ollama. "
        + "Return ONLY VALID JSON. No extra words.\n\n"
        + f"QUERY: {query}\n\nDOCUMENTS:\n{doc_text}"
    )

    raw = llm(prompt)

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {
            "error": "JSON_PARSE_FAILED",
            "raw_response": raw
        }

    return parsed, docs

# ---------------------------------------------------------------
# SCORE A HYPOTHESIS USING EVIDENCE
# ---------------------------------------------------------------
def evidence_scorer(hypothesis: str, evidence):
    """
    Accepts a hypothesis + list[string] evidence.
    Returns a JSON score.
    """
    if isinstance(evidence, (list, tuple)):
        evidence_text = "\n\n".join(evidence)
    else:
        evidence_text = str(evidence)

    prompt = (
        SCORER_PROMPT
        + "\n\nYou are a local model running via Ollama. "
        + "Return ONLY JSON.\n\n"
        + f"HYPOTHESIS:\n{hypothesis}\n\nEVIDENCE:\n{evidence_text}"
    )

    raw = llm(prompt)

    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw, "error": "JSON_PARSE_FAILED"}

# ---------------------------------------------------------------
# EXPERIMENT RECOMMENDER
# ---------------------------------------------------------------
def experiment_recommender(hypothesis: str, evidence):
    """
    Returns a plain-text experimental protocol suggestion.
    """
    if isinstance(evidence, (list, tuple)):
        evidence_text = "\n\n".join(evidence)
    else:
        evidence_text = str(evidence)

    prompt = (
        EXPERIMENT_PROMPT
        + "\n\nYou are a local model running via Ollama. "
        + "You may respond in TEXT or MARKDOWN (no JSON needed).\n\n"
        + f"HYPOTHESIS:\n{hypothesis}\n\nEVIDENCE:\n{evidence_text}"
    )

    return llm(prompt)
