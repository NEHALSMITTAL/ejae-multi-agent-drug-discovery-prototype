import os
import json
import requests

from dotenv import load_dotenv

from backend.retriever import retrieve
from backend.prompts import LIT_AGENT_PROMPT, SCORER_PROMPT, EXPERIMENT_PROMPT

# Load variables from .env (if present)
load_dotenv()

# Read model name from environment, default to "llama3"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def llm(prompt: str) -> str:
    """
    Call a local Ollama model instead of OpenAI.

    Requires:
      - Ollama running locally (default: http://localhost:11434)
      - Model pulled with: `ollama pull llama3` (or whichever OLLAMA_MODEL you use)
    """
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,  # we just want one complete response
            },
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns {"response": "...", "done": true, ...}
        return data.get("response", "").strip()
    except Exception as e:
        # Fall back to a simple error string so the app doesn't crash
        return f"ERROR calling Ollama: {e}"


def literature_agent(query: str):
    """
    Run retrieval + LLM to produce hypotheses and evidence.
    Returns (parsed_json_or_error, retrieved_docs_list).
    """
    docs = retrieve(query, k=5)
    doc_text = "\n\n".join([f"ID:{d['id']}\n{d['text']}" for d in docs])

    prompt = (
        LIT_AGENT_PROMPT
        + "\n\nYou are a local model running via Ollama. "
        + "When you respond, output ONLY valid JSON, no extra text.\n\n"
        + f"QUERY: {query}\n\nDOCUMENTS:\n{doc_text}"
    )

    out = llm(prompt)

    try:
        parsed = json.loads(out)
    except Exception:
        parsed = {"error": "JSON parsing failed", "raw": out}

    return parsed, docs


def evidence_scorer(hypothesis: str, evidence):
    """
    Score the hypothesis based on a list of evidence snippets.
    `evidence` can be a single string or a list of strings.
    """
    if isinstance(evidence, (list, tuple)):
        evidence_text = "\n\n".join(str(e) for e in evidence)
    else:
        evidence_text = str(evidence)

    prompt = (
        SCORER_PROMPT
        + "\n\nYou are a local model running via Ollama. "
        + "When you respond, output ONLY valid JSON, no extra text.\n\n"
        + f"Hypothesis: {hypothesis}\nEvidence:\n{evidence_text}"
    )

    out = llm(prompt)
    try:
        return json.loads(out)
    except Exception:
        return {"raw": out}


def experiment_recommender(hypothesis: str, evidence):
    """
    Recommend an in-vitro experiment based on hypothesis + evidence.
    `evidence` can be a single string or a list of strings.
    """
    if isinstance(evidence, (list, tuple)):
        evidence_text = "\n\n".join(str(e) for e in evidence)
    else:
        evidence_text = str(evidence)

    prompt = (
        EXPERIMENT_PROMPT
        + "\n\nYou are a local model running via Ollama. "
        + "When you respond, output plain text or markdown.\n\n"
        + f"Hypothesis: {hypothesis}\nEvidence:\n{evidence_text}"
    )

    return llm(prompt)
