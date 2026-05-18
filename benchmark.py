"""
Benchmark: Compare our Hybrid engine against other recommendation approaches.

Engines compared:
  1. TF-IDF + Cosine Similarity  (classic baseline)
  2. BM25 Only                   (sparse retrieval)
  3. Dense Only (no reranker)    (semantic search)
  4. Hybrid (BM25 + Dense)       (no reranker)
  5. Hybrid + Cross-Encoder      (our full engine)

Metrics:
  - Precision@K
  - Recall@K
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - MRR (Mean Reciprocal Rank)
  - Latency (ms)
"""

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from engine.document_loader import Document, load_file
from engine.chunker import SemanticChunker
from engine.embedder import Embedder
from engine.retriever import HybridRetriever


# ========== Ground-truth test set ==========

EVAL_QUERIES = [
    {
        "query": "What drug is best for headache with minimal side effects?",
        "relevant": ["Acetaminophen", "Ibuprofen", "Aspirin"],
    },
    {
        "query": "I have acid reflux and stomach ulcers",
        "relevant": ["Omeprazole", "Pantoprazole"],
    },
    {
        "query": "Recommend something for anxiety and depression",
        "relevant": ["Fluoxetine", "Sertraline"],
    },
    {
        "query": "I need a blood thinner",
        "relevant": ["Aspirin", "Clopidogrel", "Warfarin"],
    },
    {
        "query": "Cheapest option for managing type 2 diabetes",
        "relevant": ["Metformin", "Insulin Glargine"],
    },
    {
        "query": "Non-drowsy allergy medication",
        "relevant": ["Loratadine", "Cetirizine"],
    },
    {
        "query": "I have back pain and muscle spasms",
        "relevant": ["Cyclobenzaprine", "Ibuprofen", "Naproxen", "Tramadol"],
    },
    {
        "query": "Medication for high blood pressure",
        "relevant": ["Lisinopril"],
    },
    {
        "query": "I need something for high cholesterol",
        "relevant": ["Atorvastatin"],
    },
    {
        "query": "Treatment for bacterial infection in throat",
        "relevant": ["Amoxicillin", "Azithromycin"],
    },
    {
        "query": "Asthma inhaler for bronchospasm",
        "relevant": ["Albuterol", "Montelukast"],
    },
    {
        "query": "Nerve pain and fibromyalgia treatment",
        "relevant": ["Gabapentin"],
    },
    {
        "query": "Anti-inflammatory for arthritis",
        "relevant": ["Ibuprofen", "Naproxen", "Aspirin"],
    },
    {
        "query": "Sleep aid that also helps with allergies",
        "relevant": ["Diphenhydramine"],
    },
    {
        "query": "Steroid for severe inflammation and autoimmune conditions",
        "relevant": ["Prednisone"],
    },
]


# ========== Engine wrappers ==========

class TFIDFEngine:
    """Classic TF-IDF + Cosine Similarity baseline."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.texts: list[str] = []
        self.matrix = None

    def index(self, docs: list[Document]):
        self.texts = [d.text for d in docs]
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]


class BM25Engine:
    """BM25-only retrieval."""

    def __init__(self):
        self.texts: list[str] = []
        self.bm25: BM25Okapi | None = None

    def index(self, docs: list[Document]):
        import re
        self.texts = [d.text for d in docs]
        tokenized = [re.findall(r"\w+", t.lower()) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        import re
        tokens = re.findall(r"\w+", query.lower())
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]


class DenseOnlyEngine:
    """Semantic search only (no BM25, no reranker)."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.texts: list[str] = []
        self.embeddings = None

    def index(self, docs: list[Document]):
        self.texts = [d.text for d in docs]
        self.embeddings = self.embedder.embed_documents(docs)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q_emb = self.embedder.embed_query(query)
        scores = self.embeddings @ q_emb
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]


class HybridNoRerankerEngine:
    """Hybrid BM25 + Dense, but without cross-encoder re-ranking."""

    def __init__(self, embedder: Embedder):
        self.retriever = HybridRetriever(embedder=embedder, use_reranker=False)
        self.docs: list[Document] = []

    def index(self, docs: list[Document]):
        self.docs = docs
        self.retriever.index(docs)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        results = self.retriever.retrieve(query, top_k=top_k)
        return [(self.docs.index(doc), score) for doc, score in results]


class HybridRerankerEngine:
    """Our full engine: Hybrid + Cross-Encoder Re-ranking."""

    def __init__(self, embedder: Embedder):
        self.retriever = HybridRetriever(embedder=embedder, use_reranker=True)
        self.docs: list[Document] = []

    def index(self, docs: list[Document]):
        self.docs = docs
        self.retriever.index(docs)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        results = self.retriever.retrieve(query, top_k=top_k)
        return [(self.docs.index(doc), score) for doc, score in results]


# ========== Metrics ==========

def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    top = retrieved_ids[:k]
    hits = sum(1 for i in top if i in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    top = retrieved_ids[:k]
    hits = sum(1 for i in top if i in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / np.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG
    ideal_rels = sorted([1.0] * min(len(relevant_ids), k) + [0.0] * max(0, k - len(relevant_ids)), reverse=True)
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """Mean Reciprocal Rank."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# ========== Main benchmark ==========

def run_benchmark():
    print("=" * 80)
    print("  RECOMMENDATION ENGINE BENCHMARK")
    print("  Comparing 5 approaches on 15 evaluation queries")
    print("=" * 80)
    print()

    # Load and chunk data
    from demo import create_sample_data
    data_path = create_sample_data()

    chunker = SemanticChunker(max_tokens=256, overlap_tokens=64)
    raw_docs = load_file(data_path)
    docs = chunker.chunk_documents(raw_docs)

    print(f"Dataset: {len(raw_docs)} records -> {len(docs)} chunks")
    print()

    # Map drug names to doc indices for ground truth
    doc_name_map: dict[int, str] = {}
    for i, doc in enumerate(docs):
        for line in doc.text.split("\n"):
            if line.startswith("Drug Name:"):
                doc_name_map[i] = line.split(":", 1)[1].strip()
                break

    def get_relevant_ids(relevant_names: list[str]) -> set[int]:
        ids = set()
        for idx, name in doc_name_map.items():
            if name in relevant_names:
                ids.add(idx)
        return ids

    # Initialize engines
    print("Initializing engines...")
    embedder = Embedder(model_tier="balanced")

    engines = {
        "TF-IDF Cosine": TFIDFEngine(),
        "BM25 Only": BM25Engine(),
        "Dense Only": DenseOnlyEngine(embedder),
        "Hybrid (no rerank)": HybridNoRerankerEngine(embedder),
        "Hybrid + Reranker": HybridRerankerEngine(embedder),
    }

    # Index all engines
    for name, eng in engines.items():
        eng.index(docs)
    print("All engines indexed.\n")

    K = 5  # evaluate at top-5

    # Run evaluation
    all_results: dict[str, dict[str, list[float]]] = {}

    for eng_name, eng in engines.items():
        metrics = {"P@K": [], "R@K": [], "NDCG@K": [], "MRR": [], "Latency_ms": []}

        for eq in EVAL_QUERIES:
            relevant_ids = get_relevant_ids(eq["relevant"])

            t0 = time.perf_counter()
            results = eng.retrieve(eq["query"], top_k=K)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            retrieved_ids = [r[0] for r in results]

            metrics["P@K"].append(precision_at_k(retrieved_ids, relevant_ids, K))
            metrics["R@K"].append(recall_at_k(retrieved_ids, relevant_ids, K))
            metrics["NDCG@K"].append(ndcg_at_k(retrieved_ids, relevant_ids, K))
            metrics["MRR"].append(mrr(retrieved_ids, relevant_ids))
            metrics["Latency_ms"].append(elapsed_ms)

        all_results[eng_name] = metrics

    # ========== Print results ==========

    print("=" * 80)
    print(f"  RESULTS (averaged over {len(EVAL_QUERIES)} queries, K={K})")
    print("=" * 80)
    print()

    # Build summary table
    rows = []
    for eng_name, metrics in all_results.items():
        rows.append({
            "Engine": eng_name,
            "P@5": np.mean(metrics["P@K"]),
            "R@5": np.mean(metrics["R@K"]),
            "NDCG@5": np.mean(metrics["NDCG@K"]),
            "MRR": np.mean(metrics["MRR"]),
            "Avg Latency (ms)": np.mean(metrics["Latency_ms"]),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("NDCG@5", ascending=False).reset_index(drop=True)

    # Display with formatting
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # ========== Head-to-head comparison ==========

    best = df.iloc[0]
    print("-" * 80)
    print(f"  WINNER: {best['Engine']}")
    print("-" * 80)
    print()

    for _, row in df.iterrows():
        name = row["Engine"]
        if name == best["Engine"]:
            continue

        print(f"  {best['Engine']}  vs  {name}")
        for metric in ["P@5", "R@5", "NDCG@5", "MRR"]:
            ours = best[metric]
            theirs = row[metric]
            diff = ours - theirs
            pct = (diff / theirs * 100) if theirs > 0 else float('inf')
            arrow = "+" if diff >= 0 else ""
            print(f"    {metric:10s}  {ours:.4f}  vs  {theirs:.4f}  ({arrow}{pct:.1f}%)")
        print()

    # ========== Per-query breakdown ==========

    print("=" * 80)
    print("  PER-QUERY BREAKDOWN (NDCG@5)")
    print("=" * 80)
    print()

    header = f"{'Query':<55s}"
    for eng_name in all_results:
        short = eng_name[:12]
        header += f" {short:>12s}"
    print(header)
    print("-" * len(header))

    for i, eq in enumerate(EVAL_QUERIES):
        q_short = eq["query"][:52] + "..." if len(eq["query"]) > 55 else eq["query"]
        line = f"{q_short:<55s}"
        scores_for_q = []
        for eng_name in all_results:
            val = all_results[eng_name]["NDCG@K"][i]
            scores_for_q.append(val)
            line += f" {val:>12.4f}"
        # Highlight the winner
        print(line)
    print()

    # ========== Final score card ==========

    print("=" * 80)
    print("  FINAL SCORE CARD")
    print("=" * 80)
    print()

    for _, row in df.iterrows():
        name = row["Engine"]
        composite = (row["P@5"] + row["R@5"] + row["NDCG@5"] + row["MRR"]) / 4 * 100
        bar_len = int(composite * 0.5)
        bar = "#" * bar_len
        is_best = " <-- BEST" if name == best["Engine"] else ""
        print(f"  {name:<25s} {composite:5.1f}/100  [{bar}]{is_best}")

    print()
    print("Composite = avg(P@5, R@5, NDCG@5, MRR) * 100")
    print()


if __name__ == "__main__":
    run_benchmark()
