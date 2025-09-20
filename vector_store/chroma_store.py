"""ChromaDB Vector Store Implementation

Thin wrapper around ChromaDB client implementing the VectorStore interface.
Maintains lazy collection creation and basic error handling.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import time

try:
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore

from .base import VectorStore

class ChromaVectorStore(VectorStore):
    def __init__(self,
                 persist_dir: str = "./chroma_trading_db",
                 server_host: Optional[str] = None,
                 server_port: Optional[int] = None):
        if chromadb is None:
            raise RuntimeError("chromadb not installed. Add it to requirements.txt and install.")

        settings_cls = None
        try:  # Attempt to bind Settings class if available
            settings_cls = Settings  # type: ignore
        except Exception:  # pragma: no cover
            settings_cls = None

        if server_host and server_port:
            self.client = chromadb.HttpClient(host=server_host, port=server_port)
        else:
            if settings_cls is not None:
                self.client = chromadb.Client(settings_cls(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=persist_dir
                ))
            else:  # Fallback minimal client
                self.client = chromadb.Client()
        self._collections_cache: Dict[str, Any] = {}

    # Internal helper
    def _get_or_create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        if name in self._collections_cache:
            return self._collections_cache[name]
        existing = [c.name for c in self.client.list_collections()]
        if name in existing:
            col = self.client.get_collection(name)
        else:
            col = self.client.create_collection(name=name, metadata=metadata or {})
        self._collections_cache[name] = col
        return col

    def ensure_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._get_or_create_collection(name, metadata)

    def list_collections(self) -> List[str]:
        return [c.name for c in self.client.list_collections()]

    def add(self,
            collection: str,
            ids: List[str],
            embeddings: List[List[float]],
            metadatas: Optional[List[Dict[str, Any]]] = None,
            documents: Optional[List[str]] = None) -> None:
        col = self._get_or_create_collection(collection)
        col.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(self,
              collection: str,
              query_embeddings: List[List[float]],
              n_results: int = 10,
              where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        col = self._get_or_create_collection(collection)
        return col.query(query_embeddings=query_embeddings, n_results=n_results, where=where)

    def delete(self, collection: str, ids: List[str]) -> None:
        col = self._get_or_create_collection(collection)
        col.delete(ids=ids)

    def health(self) -> Dict[str, Any]:
        return {
            "collections": self.list_collections(),
            "timestamp": time.time()
        }
