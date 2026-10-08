"""Vector database management using ChromaDB + local sentence-transformers embeddings."""

from typing import Any, Dict, List, Optional, cast

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

from ..config import settings


class VectorStore:
    """Manage vector database for document embeddings (local embeddings, no API cost)."""

    # Collection name includes the model name so switching models
    # creates a fresh collection automatically (avoids dimension mismatches).
    _COLLECTION_PREFIX = "pdf_docs_"

    def __init__(self) -> None:
        self.db_path = settings.vector_db_path
        self.embedding_model_name = settings.embedding_model

        # Load local embedding model (downloads ~120 MB on first run)
        print(f"Cargando modelo de embeddings: {self.embedding_model_name}")
        self._embedder = SentenceTransformer(self.embedding_model_name)
        self._embedding_dim = self._embedder.get_sentence_embedding_dimension()

        # Ensure vector DB path exists
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Collection name tied to model to avoid dimension conflicts
        safe_name = (
            self.embedding_model_name.replace("/", "_").replace("-", "_").lower()
        )
        self.collection_name = f"{self._COLLECTION_PREFIX}{safe_name}"

        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document embeddings (local model)"},
        )

    def _refresh_collection(self) -> None:
        self.collection = self._get_or_create_collection()

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using local sentence-transformers model."""
        try:
            vector = self._embedder.encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            )
            return vector.tolist()
        except Exception as exc:
            print(f"Error generating embedding: {exc}")
            return []

    def query_by_embedding(
        self,
        embedding: List[float],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
    ):
        """Query collection by embedding with optional metadata filter."""
        kwargs: Dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            # Normalise to explicit $eq operator (required in chromadb ≥ 0.5)
            normalised = {
                k: ({"$eq": v} if not isinstance(v, dict) else v)
                for k, v in where.items()
            }
            kwargs["where"] = normalised
        try:
            return self.collection.query(**kwargs)
        except NotFoundError:
            self._refresh_collection()
            return self.collection.query(**kwargs)

    def get_distinct_filenames(self) -> List[str]:
        """Return all distinct filenames stored in the collection."""
        try:
            result = self.collection.get(include=["metadatas"])
            metadatas = result.get("metadatas") or []
            seen: set = set()
            names = []
            for m in metadatas:
                fn = m.get("filename", "") if isinstance(m, dict) else ""
                if fn and fn not in seen:
                    seen.add(fn)
                    names.append(fn)
            return names
        except Exception:
            return []

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to vector database."""
        if not chunks:
            print("No chunks to add")
            return

        print(f"Indexando {len(chunks)} fragmentos…")

        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for i, chunk in enumerate(chunks):
            content = str(chunk.get("content", ""))
            metadata = chunk.get("metadata", {})
            if not isinstance(metadata, dict):
                continue

            embedding = self.generate_embedding(content)
            if not embedding:
                continue

            documents.append(content)
            embeddings.append(embedding)
            metadatas.append(metadata)
            filename = str(metadata.get("filename", "unknown"))
            chunk_id = metadata.get("chunk_id", i)
            ids.append(f"{filename}_{chunk_id}")

        if documents:
            try:
                self.collection.add(
                    documents=documents,
                    embeddings=cast(Any, embeddings),
                    metadatas=cast(Any, metadatas),
                    ids=ids,
                )
            except NotFoundError:
                self._refresh_collection()
                self.collection.add(
                    documents=documents,
                    embeddings=cast(Any, embeddings),
                    metadatas=cast(Any, metadatas),
                    ids=ids,
                )
            print(f"Indexados {len(documents)} fragmentos correctamente.")

    def clear_collection(self) -> None:
        """Clear all documents from collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except NotFoundError:
            pass
        self.collection = self._get_or_create_collection()
        print("Base de datos vectorial limpiada.")

    def get_collection_count(self) -> int:
        """Get number of documents in collection."""
        try:
            return self.collection.count()
        except NotFoundError:
            self._refresh_collection()
            return self.collection.count()
