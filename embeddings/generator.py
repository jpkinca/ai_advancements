"""Embedding Generator Abstraction and Implementations.

Provides a unified interface for generating embeddings with pluggable backends.
Priority order: OpenAI API -> Local sentence-transformers model.
"""
from __future__ import annotations
from typing import List, Sequence, Optional
import os

OPENAI_MODEL_DEFAULT = "text-embedding-3-small"
LOCAL_MODEL_DEFAULT = "all-MiniLM-L6-v2"

class EmbeddingGenerator:
    def __init__(self,
                 openai_model: str = OPENAI_MODEL_DEFAULT,
                 local_model: str = LOCAL_MODEL_DEFAULT,
                 prefer_local: bool = False):
        self.openai_model = openai_model
        self.local_model_name = local_model
        self.prefer_local = prefer_local
        self._openai_client = None
        self._local_model = None

    def _ensure_openai(self):  # Lazy load OpenAI client
        if self._openai_client is not None:
            return
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot use OpenAI embeddings.")
        try:  # Late import
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("openai package not installed") from e
        self._openai_client = OpenAI(api_key=api_key)

    def _ensure_local(self):  # Lazy load local sentence-transformers model
        if self._local_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("sentence-transformers not installed") from e
        self._local_model = SentenceTransformer(self.local_model_name)

    def embed_openai(self, texts: Sequence[str]) -> List[List[float]]:
        self._ensure_openai()
        result_vectors: List[List[float]] = []
        # Simple sequential batching; can optimize later
        for text in texts:
            resp = self._openai_client.embeddings.create(model=self.openai_model, input=text)  # type: ignore
            result_vectors.append(resp.data[0].embedding)  # type: ignore
        return result_vectors

    def embed_local(self, texts: Sequence[str]) -> List[List[float]]:
        self._ensure_local()
        return self._local_model.encode(list(texts), convert_to_numpy=False)  # type: ignore

    def embed(self, texts: Sequence[str], model_version: Optional[str] = None) -> List[List[float]]:
        if self.prefer_local:
            try:
                return self.embed_local(texts)
            except Exception:
                return self.embed_openai(texts)
        else:
            try:
                return self.embed_openai(texts)
            except Exception:
                return self.embed_local(texts)
