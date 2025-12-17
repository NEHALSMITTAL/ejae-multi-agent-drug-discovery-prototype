import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------
# Embedder Class (Optimized A2 Version)
# ---------------------------------------------------------------
class Embedder:
    def __init__(self):
        """
        Loads model + initializes persistent ChromaDB client once.
        """
        # Load lightweight transformer model (fast)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Persistent local DB → no re-indexing needed
        project_root = Path(__file__).resolve().parents[1]
        db_path = project_root / "chroma_db"
        db_path.mkdir(exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings()
        )

        self.collection = None

    # -----------------------------------------------------------
    # Create or load collection
    # -----------------------------------------------------------
    def create_collection(self, name: str = "ej_docs"):
        """
        Creates or loads the Chroma collection.
        Embedding is handled automatically using a SentenceTransformer.
        """
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=ef
        )

    # -----------------------------------------------------------
    # Index documents (ONLY run once using index_docs.py)
    # -----------------------------------------------------------
    def index_docs(self, docs):
        """
        Adds documents into the vector database.
        docs must contain: id, text, title(optional)
        """
        if self.collection is None:
            self.create_collection()

        ids = [str(d["id"]) for d in docs]
        texts = [d["text"] for d in docs]
        metas = [{"title": d.get("title", None)} for d in docs]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metas
        )

    # -----------------------------------------------------------
    # QUERY DOCUMENTS
    # -----------------------------------------------------------
    def query(self, text: str, k: int = 5):
        """
        Returns top-k matching documents.
        """
        if self.collection is None:
            self.create_collection()

        return self.collection.query(
            query_texts=[text],
            n_results=k
        )
