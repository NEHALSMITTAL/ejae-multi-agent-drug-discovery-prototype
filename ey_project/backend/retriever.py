from backend.embedder import Embedder

# ---------------------------------------------------------------
# Initialize embedder ONCE at module import
# This prevents reloading & rebuilding the DB on every Streamlit run
# ---------------------------------------------------------------
embedder = Embedder()
embedder.create_collection()


# ---------------------------------------------------------------
# RETRIEVE DOCUMENTS
# ---------------------------------------------------------------
def retrieve(query_text: str, k: int = 5):
    """
    Retrieve top-k semantically similar documents from ChromaDB.

    Returns:
        [
            {
                "id": "...",
                "text": "...",
                "meta": { ... }
            }
        ]
    """

    if not query_text:
        return []

    results = embedder.query(query_text, k)

    # Defensive fallback — Chroma sometimes returns empty lists
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    retrieved = []
    for i in range(len(ids)):
        retrieved.append({
            "id": ids[i],
            "text": docs[i],
            "meta": metas[i] if metas else {}
        })

    return retrieved
