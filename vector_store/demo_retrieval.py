"""Demo retrieval using ChromaVectorStore.

This is a simple smoke test and usage illustration; not a formal test.
"""
from __future__ import annotations
from vector_store.chroma_store import ChromaVectorStore
from embeddings.generator import EmbeddingGenerator
import uuid

EXAMPLE_DOCS = [
    "AAPL bullish momentum breakout with rising volume",
    "SPY consolidation near resistance with declining volume",
    "NVDA parabolic extension showing exhaustion signals",
    "EURUSD forming descending triangle under macro pressure"
]


def main():
    store = ChromaVectorStore()
    embedder = EmbeddingGenerator(prefer_local=True)

    embeddings = embedder.embed(EXAMPLE_DOCS)
    ids = [str(uuid.uuid4()) for _ in EXAMPLE_DOCS]

    store.ensure_collection("market_patterns")
    store.add("market_patterns", ids=ids, embeddings=embeddings, documents=EXAMPLE_DOCS)

    query_vec = embedder.embed(["bullish breakout with strong volume"])
    result = store.query("market_patterns", query_embeddings=query_vec, n_results=3)
    print("Top Matches:")
    for doc, dist in zip(result.get('documents', [[]])[0], result.get('distances', [[]])[0]):
        print(f"  similarity≈{1-dist:.3f} :: {doc}")

if __name__ == "__main__":
    main()
