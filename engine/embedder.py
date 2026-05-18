"""
Embedding engine — generates dense vector representations using sentence-transformers.

Uses all-MiniLM-L6-v2 by default (fast, good quality).
For max quality, switch to 'BAAI/bge-large-en-v1.5' or 'intfloat/e5-large-v2'.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from .document_loader import Document


class Embedder:
    # Ranked by quality vs speed tradeoff:
    MODELS = {
        "fast": "all-MiniLM-L6-v2",               # 384d, very fast
        "balanced": "all-mpnet-base-v2",            # 768d, good balance
        "quality": "BAAI/bge-large-en-v1.5",       # 1024d, top-tier
    }

    def __init__(self, model_tier: str = "balanced", device: str | None = None):
        model_name = self.MODELS.get(model_tier, model_tier)
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
            convert_to_numpy=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    def embed_documents(self, docs: list[Document], batch_size: int = 64) -> np.ndarray:
        texts = [doc.text for doc in docs]
        return self.embed_texts(texts, batch_size=batch_size)
