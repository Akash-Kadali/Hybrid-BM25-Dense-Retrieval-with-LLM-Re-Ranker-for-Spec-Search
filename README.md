# Recommendation Engine

A production-grade, general-purpose recommendation engine that combines **BM25 (sparse retrieval)** + **Semantic Search (dense retrieval)** + **Cross-Encoder Re-ranking** — the same architecture used by Google Search and Glean.

Works with any domain: healthcare, e-commerce, legal, finance, HR — just feed it your data.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## How It Works

```
                    ┌─────────────────┐
                    │   Your Query    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌────────────────┐          ┌──────────────────┐
     │  BM25 (Sparse) │          │  FAISS (Dense)   │
     │  Keyword Match  │          │  Semantic Match  │
     └───────┬────────┘          └────────┬─────────┘
              │                            │
              └──────────┬─────────────────┘
                         ▼
              ┌─────────────────────┐
              │  Reciprocal Rank    │
              │  Fusion (RRF)       │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  Cross-Encoder      │
              │  Re-ranking         │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  Top-K Results      │
              └─────────────────────┘
```

| Component | What It Does | Used By |
|---|---|---|
| **BM25** | Exact keyword matching (sparse) | Google, Elasticsearch |
| **Sentence-Transformers** | Semantic understanding (dense) | Google, Glean, Cohere |
| **FAISS** | Billion-scale vector similarity search | Meta, Google |
| **Reciprocal Rank Fusion** | Merges sparse + dense results | Google Search |
| **Cross-Encoder Re-ranking** | Precision refinement on top candidates | Google, Bing |
| **tiktoken** | Token-accurate text chunking | OpenAI, GPT pipelines |

---

## Benchmark Results

Evaluated on 15 queries against a drug recommendation dataset (K=5):

```
Engine                     Composite   NDCG@5   MRR     P@5     R@5
─────────────────────────────────────────────────────────────────────
Hybrid + Reranker (ours)    75.4/100   0.886   0.933   0.320   0.878  <-- BEST
Dense Only                  71.8/100   0.826   0.847   0.320   0.878
Hybrid (no rerank)          70.8/100   0.819   0.850   0.307   0.856
TF-IDF Cosine               70.0/100   0.805   0.889   0.293   0.811
BM25 Only                   67.6/100   0.779   0.856   0.280   0.789
```

**+13.8% NDCG over BM25 | +10% over TF-IDF | +7.3% over Dense-only**

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Python API (3 lines)

```python
from engine import RecommendationEngine

engine = RecommendationEngine()
engine.ingest("data/products.csv")  # or .xlsx, .pdf, .docx, or a directory
results = engine.recommend("best drug for headache with minimal side effects")

for r in results:
    print(f"#{r['rank']} (score: {r['score']}) — {r['text'][:100]}")
```

### 3. Web UI + REST API

```bash
python -m uvicorn api:app --port 8000
# Open http://localhost:8000
```

### 4. Run the benchmark

```bash
python benchmark.py
```

### 5. CLI demo

```bash
python demo.py
```

---

## REST API

### `POST /api/ingest/file`

Upload a file (CSV, Excel, Word, PDF) for indexing.

```bash
curl -F "file=@data/catalog.csv" http://localhost:8000/api/ingest/file
```

```json
{ "filename": "catalog.csv", "chunks_created": 25, "total_chunks": 25 }
```

### `POST /api/ingest/text`

Ingest raw text directly.

```bash
curl -X POST http://localhost:8000/api/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Ibuprofen is an NSAID used for pain relief.", "metadata": {"source": "manual"}}'
```

### `POST /api/recommend`

Get recommendations for a query.

```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "non-drowsy allergy medication", "top_k": 5}'
```

```json
{
  "query": "non-drowsy allergy medication",
  "results": [
    { "rank": 1, "score": 5.6388, "text": "Drug Name: Loratadine\n...", "metadata": {} },
    { "rank": 2, "score": 0.118,  "text": "Drug Name: Cetirizine\n...", "metadata": {} }
  ],
  "total_indexed": 25
}
```

### `GET /api/stats`

Engine statistics.

### `POST /api/reset`

Clear all indexed data and reset the engine.

---

## Configuration

```python
engine = RecommendationEngine(
    embedding_tier="balanced",     # "fast" (384d) | "balanced" (768d) | "quality" (1024d)
    max_chunk_tokens=256,          # Chunk size in tokens
    chunk_overlap_tokens=64,       # Overlap between chunks
    use_reranker=True,             # Cross-encoder re-ranking (slower but more accurate)
    device=None,                   # "cpu", "cuda", "mps", or None (auto-detect)
)
```

| Tier | Model | Dimensions | Speed | Quality |
|---|---|---|---|---|
| `fast` | all-MiniLM-L6-v2 | 384 | Fastest | Good |
| `balanced` | all-mpnet-base-v2 | 768 | Medium | Better |
| `quality` | BAAI/bge-large-en-v1.5 | 1024 | Slower | Best |

---

## Supported File Types

| Format | Extension | Notes |
|---|---|---|
| CSV | `.csv` | Each row becomes a document |
| Excel | `.xlsx`, `.xls` | Multi-sheet support |
| Word | `.docx`, `.doc` | Paragraphs + tables extracted |
| PDF | `.pdf` | Text + table extraction via pdfplumber |

---

## Project Structure

```
recommendation-engine/
├── engine/
│   ├── __init__.py           # Public API
│   ├── document_loader.py    # Multi-format document ingestion
│   ├── chunker.py            # tiktoken-based semantic chunking
│   ├── embedder.py           # Sentence-transformer embeddings
│   ├── retriever.py          # Hybrid BM25 + FAISS + Cross-Encoder
│   └── core.py               # Main RecommendationEngine class
├── api.py                    # FastAPI REST backend
├── static/
│   └── index.html            # Web UI
├── benchmark.py              # Engine comparison benchmark
├── demo.py                   # CLI demo with sample data
├── requirements.txt
└── sample_data/
    └── drug_catalog.csv      # Sample healthcare dataset
```

---

## How to Use With Your Own Data

This engine is domain-agnostic. Just swap the data:

```python
engine = RecommendationEngine(embedding_tier="quality")

# E-commerce
engine.ingest("products.csv")
results = engine.recommend("wireless headphones under $50 with good bass")

# Legal
engine.ingest("contracts/")
results = engine.recommend("non-compete clause with 2 year restriction")

# HR
engine.ingest("resumes/")
results = engine.recommend("senior backend engineer with Kubernetes experience")

# Finance
engine.ingest("sec_filings.pdf")
results = engine.recommend("revenue growth drivers in Q4 2024")
```
