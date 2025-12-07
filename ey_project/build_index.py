import json
from pathlib import Path
from backend.embedder import Embedder

def load_docs():
    """
    Load documents from ey_project/data/docs.jsonl.

    Supports:
    - JSONL (one JSON object per line)
    - JSON array ([ {...}, {...} ])
    """

    # Build correct path no matter where script is run from
    data_path = Path(__file__).resolve().parent / "data" / "docs.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(f"docs.jsonl not found at: {data_path}")

    # Read whole file
    text = data_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("docs.jsonl is empty")

    docs = []

    # If file starts with array → parse normally
    if text[0] == "[":
        docs = json.loads(text)

    else:  # JSONL line-by-line
        for line in text.splitlines():
            clean = line.strip().rstrip(",")
            if clean:
                docs.append(json.loads(clean))

    return docs


if __name__ == "__main__":
    docs = load_docs()

    embedder = Embedder()
    embedder.create_collection()
    embedder.index_docs(docs)

    print("Indexing completed ✔")
