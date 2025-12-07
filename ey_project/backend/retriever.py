from backend.embedder import Embedder

# Initialize embedder once at module load (fast + efficient)
embedder = Embedder()
embedder.create_collection()

def retrieve(query_text, k=5):
    """
    Retrieve top-k similar documents from the Chroma vector database.
    Returns a list of dicts: {id, text, meta}.
    """
    results = embedder.query(query_text, k)

    # Defensive checks in case Chroma returns empty lists
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    retrieved = []
    for i in range(len(ids)):
        retrieved.append({
            "id": ids[i],
            "text": docs[i],
            "meta": metas[i]
        })

    return retrieved
