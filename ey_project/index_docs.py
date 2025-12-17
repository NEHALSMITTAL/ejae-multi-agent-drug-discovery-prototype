import json
from pathlib import Path
from backend.embedder import Embedder


# ---------------------------------------------------------------
# LOAD DOCUMENTS (JSON or JSONL)
# ---------------------------------------------------------------
def load_docs():
    """
    Loads documents from ey_project/data/docs.jsonl.
    Supports:
    - JSON array ([ {...}, {...} ])
    - JSONL (one JSON per line)
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / "docs.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(f"docs.jsonl not found at: {data_path}")

    raw = data_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("docs.jsonl is empty")

    docs = []

    # JSON array format
    if raw.startswith("["):
        docs = json.loads(raw)

    else:  # JSONL
        for line in raw.splitlines():
            clean = line.strip().rstrip(",")
            if clean:
                docs.append(json.loads(clean))

    return docs


# ---------------------------------------------------------------
# MAIN EXECUTION: INDEX DOCUMENTS
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("📄 Loading docs.jsonl ...")
    docs = load_docs()

    print(f"Loaded {len(docs)} documents.")

    embedder = Embedder()
    embedder.create_collection()

    print("🔄 Indexing into ChromaDB (this runs only once)...")
    embedder.index_docs(docs)

    print("✔ Indexing completed — ChromaDB is ready!")
