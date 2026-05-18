"""
Hybrid retriever — combines BM25 (sparse) + FAISS (dense) + cross-encoder re-ranking.

This is the core of the recommendation engine. The hybrid approach consistently
outperforms either sparse or dense retrieval alone.

Pipeline:
  1. BM25 retrieves top-K candidates (keyword matching)
  2. FAISS retrieves top-K candidates (semantic similarity)
  3. Reciprocal Rank Fusion merges both result sets
  4. Cross-encoder re-ranks the merged candidates for precision
"""

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .document_loader import Document
from .embedder import Embedder


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        use_reranker: bool = True,
    ):
        self.embedder = embedder
        self.documents: list[Document] = []
        self.doc_texts: list[str] = []

        # Dense index
        self.faiss_index: faiss.IndexFlatIP | None = None

        # Sparse index
        self.bm25: BM25Okapi | None = None

        # Re-ranker
        self.use_reranker = use_reranker
        self.reranker: CrossEncoder | None = None
        if use_reranker:
            self.reranker = CrossEncoder(reranker_model)

    def index(self, documents: list[Document]) -> None:
        """Build both sparse and dense indices from documents."""
        self.documents = documents
        self.doc_texts = [doc.text for doc in documents]

        # --- Dense index (FAISS) ---
        embeddings = self.embedder.embed_documents(documents)
        dimension = embeddings.shape[1]

        # Use IndexFlatIP (inner product) since embeddings are L2-normalized
        # This is equivalent to cosine similarity
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings.astype(np.float32))

        # --- Sparse index (BM25) ---
        tokenized = [self._tokenize(text) for text in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        rerank_top_n: int = 50,
    ) -> list[tuple[Document, float]]:
        """
        Hybrid retrieval with optional re-ranking.

        Returns list of (document, score) tuples sorted by relevance.
        """
        if not self.documents:
            return []

        n_docs = len(self.documents)
        candidates_k = min(rerank_top_n, n_docs)

        # --- BM25 retrieval ---
        bm25_scores = self._bm25_retrieve(query, candidates_k)

        # --- Dense retrieval ---
        dense_scores = self._dense_retrieve(query, candidates_k)

        # --- Reciprocal Rank Fusion ---
        fused = self._reciprocal_rank_fusion(
            bm25_scores, dense_scores,
            bm25_weight=bm25_weight, dense_weight=dense_weight,
        )

        # Take top candidates for re-ranking
        sorted_indices = sorted(fused, key=fused.get, reverse=True)[:rerank_top_n]

        # --- Cross-encoder re-ranking ---
        if self.use_reranker and self.reranker and len(sorted_indices) > 0:
            pairs = [(query, self.doc_texts[i]) for i in sorted_indices]
            rerank_scores = self.reranker.predict(pairs)

            scored = list(zip(sorted_indices, rerank_scores))
            scored.sort(key=lambda x: x[1], reverse=True)
        else:
            scored = [(i, fused[i]) for i in sorted_indices]

        # Return top_k results
        results = []
        for doc_idx, score in scored[:top_k]:
            results.append((self.documents[doc_idx], float(score)))
        return results

    def _bm25_retrieve(self, query: str, top_k: int) -> dict[int, float]:
        """Get BM25 scores for query."""
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {int(i): float(scores[i]) for i in top_indices if scores[i] > 0}

    def _dense_retrieve(self, query: str, top_k: int) -> dict[int, float]:
        """Get FAISS dense retrieval scores."""
        query_vec = self.embedder.embed_query(query).reshape(1, -1).astype(np.float32)
        scores, indices = self.faiss_index.search(query_vec, top_k)

        return {
            int(indices[0][i]): float(scores[0][i])
            for i in range(len(indices[0]))
            if indices[0][i] >= 0
        }

    def _reciprocal_rank_fusion(
        self,
        bm25_scores: dict[int, float],
        dense_scores: dict[int, float],
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        k: int = 60,
    ) -> dict[int, float]:
        """
        Reciprocal Rank Fusion (RRF) — merges two ranked lists.
        RRF score = weight * 1/(k + rank)
        """
        fused: dict[int, float] = {}

        # Rank BM25 results
        bm25_ranked = sorted(bm25_scores, key=bm25_scores.get, reverse=True)
        for rank, doc_idx in enumerate(bm25_ranked):
            fused[doc_idx] = fused.get(doc_idx, 0) + bm25_weight / (k + rank + 1)

        # Rank dense results
        dense_ranked = sorted(dense_scores, key=dense_scores.get, reverse=True)
        for rank, doc_idx in enumerate(dense_ranked):
            fused[doc_idx] = fused.get(doc_idx, 0) + dense_weight / (k + rank + 1)

        return fused

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercased tokenization for BM25."""
        import re
        return re.findall(r"\w+", text.lower())
