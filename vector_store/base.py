"""Vector Store Abstraction

Defines the repository-style interface for embedding storage & similarity retrieval.
This allows swapping backends (ChromaDB, PGVector, Qdrant, etc.) without impacting
upstream consumers.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Sequence, Protocol

class EmbeddingGenerator(Protocol):
    def embed(self, texts: Sequence[str], model_version: Optional[str] = None) -> List[List[float]]: ...

class VectorStore(ABC):
    @abstractmethod
    def add(self,
            collection: str,
            ids: List[str],
            embeddings: List[List[float]],
            metadatas: Optional[List[Dict[str, Any]]] = None,
            documents: Optional[List[str]] = None) -> None:
        """Add embeddings to a collection."""
        raise NotImplementedError

    @abstractmethod
    def query(self,
              collection: str,
              query_embeddings: List[List[float]],
              n_results: int = 10,
              where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Similarity query returning ids, distances, metadata, documents."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, collection: str, ids: List[str]) -> None:
        """Delete vectors by id."""
        raise NotImplementedError

    @abstractmethod
    def ensure_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Create collection if it does not exist."""
        raise NotImplementedError

    @abstractmethod
    def list_collections(self) -> List[str]:
        """Return available collection names."""
        raise NotImplementedError

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """Return basic health / stats for monitoring."""
        raise NotImplementedError
