import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


class Embedder:
    def __init__(self):
        # SentenceTransformer model (you may or may not use this directly)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Make sure the DB directory exists (at project root / chroma_db)
        project_root = Path(__file__).resolve().parents[1]
        db_path = project_root / "chroma_db"
        db_path.mkdir(exist_ok=True)

        # ✅ New-style Chroma client: PersistentClient instead of deprecated Client(...)
        # Settings is now optional; default is fine for local duckdb+parquet
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(),  # keep it simple; can tweak later if needed
        )

        self.collection = None

    def create_collection(self, name: str = "ej_docs"):
        # Use the same embedding function you had before
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=ef,
        )

    def index_docs(self, docs):
        if self.collection is None:
            self.create_collection()

        # Expect each doc to have at least 'id' and 'text'
        ids = [str(d["id"]) for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [{"title": d.get("title")} for d in docs]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

        # For PersistentClient, data is stored automatically,
        # but calling persist() is still OK for extra safety in older patterns.
        # NOTE: chromadb.PersistentClient does not expose .persist(),
        # so we don't call self.client.persist() here.

    def query(self, text, k: int = 5):
        if self.collection is None:
            self.create_collection()

        return self.collection.query(
            query_texts=[text],
            n_results=k,
        )
