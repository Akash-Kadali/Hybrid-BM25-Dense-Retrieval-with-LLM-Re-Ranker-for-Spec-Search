"""
RecommendationEngine — the main entry point.

Usage:
    engine = RecommendationEngine()
    engine.ingest("data/products.csv")
    engine.ingest("data/catalog.pdf")
    results = engine.recommend("best drug for headache with minimal side effects")
"""

import json
import os
import pickle
from pathlib import Path

from .chunker import SemanticChunker
from .document_loader import Document, load_directory, load_file
from .embedder import Embedder
from .retriever import HybridRetriever


class RecommendationEngine:
    def __init__(
        self,
        embedding_tier: str = "balanced",
        max_chunk_tokens: int = 256,
        chunk_overlap_tokens: int = 64,
        use_reranker: bool = True,
        device: str | None = None,
    ):
        """
        Initialize the recommendation engine.

        Args:
            embedding_tier: 'fast', 'balanced', or 'quality' — controls embedding model size.
            max_chunk_tokens: Maximum tokens per chunk.
            chunk_overlap_tokens: Token overlap between adjacent chunks.
            use_reranker: Whether to use cross-encoder re-ranking (slower but more accurate).
            device: 'cpu', 'cuda', or 'mps'. None for auto-detect.
        """
        self.chunker = SemanticChunker(
            max_tokens=max_chunk_tokens,
            overlap_tokens=chunk_overlap_tokens,
        )
        self.embedder = Embedder(model_tier=embedding_tier, device=device)
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            use_reranker=use_reranker,
        )
        self.raw_documents: list[Document] = []
        self.chunks: list[Document] = []
        self._indexed = False

    def ingest(self, path: str) -> int:
        """
        Ingest a file or directory. Supports CSV, Excel, Word, PDF.

        Returns the number of chunks created.
        """
        path = str(Path(path).resolve())

        if os.path.isdir(path):
            docs = load_directory(path)
        else:
            docs = load_file(path)

        self.raw_documents.extend(docs)
        new_chunks = self.chunker.chunk_documents(docs)
        self.chunks.extend(new_chunks)

        # Re-index with all chunks
        self._build_index()

        return len(new_chunks)

    def ingest_text(self, text: str, metadata: dict | None = None) -> int:
        """Ingest raw text directly."""
        doc = Document(text=text, metadata=metadata or {})
        self.raw_documents.append(doc)
        new_chunks = self.chunker.chunk_documents([doc])
        self.chunks.extend(new_chunks)
        self._build_index()
        return len(new_chunks)

    def recommend(
        self,
        query: str,
        top_k: int = 10,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
    ) -> list[dict]:
        """
        Get recommendations for a query.

        Returns list of dicts with 'text', 'score', and 'metadata' keys,
        sorted by relevance (best first).
        """
        if not self._indexed:
            raise RuntimeError("No documents indexed. Call ingest() first.")

        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

        return [
            {
                "text": doc.text,
                "score": round(score, 4),
                "metadata": doc.metadata,
                "rank": i + 1,
            }
            for i, (doc, score) in enumerate(results)
        ]

    def save(self, directory: str) -> None:
        """Save the engine state to disk for later reloading."""
        os.makedirs(directory, exist_ok=True)
        state = {
            "raw_documents": self.raw_documents,
            "chunks": self.chunks,
        }
        with open(os.path.join(directory, "engine_state.pkl"), "wb") as f:
            pickle.dump(state, f)

    def load(self, directory: str) -> None:
        """Load a previously saved engine state."""
        with open(os.path.join(directory, "engine_state.pkl"), "rb") as f:
            state = pickle.load(f)
        self.raw_documents = state["raw_documents"]
        self.chunks = state["chunks"]
        self._build_index()

    def stats(self) -> dict:
        """Return engine statistics."""
        return {
            "raw_documents": len(self.raw_documents),
            "chunks": len(self.chunks),
            "indexed": self._indexed,
            "embedding_dim": self.embedder.dimension,
        }

    def _build_index(self) -> None:
        if self.chunks:
            self.retriever.index(self.chunks)
            self._indexed = True
