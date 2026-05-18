# Hybrid BM25-Dense Retrieval with LLM Re-Ranker for Spec Search

A hybrid retrieval system that combines **BM25 sparse retrieval**, **dense vector search**, and an **LLM-based re-ranker** to improve search relevance on complex technical document queries.

Although the use case here is construction spec search, the overall retrieval design is generic and can be adapted to any domain that needs both exact keyword matching and semantic retrieval.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

---

## Overview

This project implements a **hybrid BM25-dense retrieval pipeline** for document search.

It combines:

- **BM25 sparse retrieval** for exact keyword and term matching
- **Dense embedding retrieval** with **FAISS** for semantic similarity
- **LLM-based re-ranking** to refine the final result ordering
- A modular architecture that can support **search, retrieval, ranking, and downstream extraction workflows**

This setup is useful for technical search problems where queries may contain:
- exact keywords
- abbreviations
- product or entity names
- structured attributes
- semantically similar phrasing
- domain-specific terminology

---

## Architecture

```text
                    ┌────────────────────┐
                    │    User Query      │
                    └─────────┬──────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
    ┌──────────────────┐              ┌──────────────────┐
    │  BM25 Retrieval  │              │ Dense Retrieval  │
    │ (Elasticsearch)  │              │    (FAISS)       │
    └────────┬─────────┘              └────────┬─────────┘
             │                                  │
             └──────────────┬───────────────────┘
                            ▼
                 ┌──────────────────────┐
                 │  Hybrid Fusion Layer │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │  LLM-based Re-ranker │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │   Top Ranked Results │
                 └──────────────────────┘
```

---

## Tech Stack

- **Python**
- **PyTorch**
- **FAISS**
- **Elasticsearch**
- **LangChain**
- **Docker**

---

## Key Features

- Hybrid **BM25 + dense retrieval** pipeline for technical and document search
- **FAISS-based embedding search** for semantic matching
- **Elasticsearch BM25 retrieval** for exact terms and keyword-heavy queries
- **LLM re-ranking** to improve final candidate ordering
- Designed for unstructured and semi-structured technical documents
- Modular design for integration into broader **retrieval, search, and extraction systems**

---

## Results

On labeled benchmark data, this system achieved:

- **18% improvement in top-10 search relevance**
- **24% reduction in mean extraction errors**

These gains came from combining sparse and dense retrieval instead of relying on a single retrieval method, followed by LLM-based re-ranking on the fused candidate set.

---

## Example Query Types

Typical query patterns this system can handle include:

- exact keyword-based search
- semantically phrased technical requirements
- product or entity attribute matching
- standard or code-related lookup
- specification-style retrieval from long documents

This makes the system useful when source documents and user queries do not always use the same wording, but still refer to the same underlying requirement.

---

## Pipeline Flow

1. **Ingest document data**
2. **Index text for BM25 retrieval in Elasticsearch**
3. **Generate dense embeddings and store them in FAISS**
4. **Retrieve candidates from sparse and dense retrievers**
5. **Fuse candidate results**
6. **Apply LLM-based re-ranking**
7. **Return top-ranked results for retrieval or downstream processing**

---

## Why Hybrid Retrieval

Many search systems fail when they depend on only one retrieval method.

- **BM25** is strong for exact keywords, identifiers, and term overlap
- **Dense retrieval** helps capture semantic similarity beyond lexical matching
- **LLM re-ranking** improves precision by selecting the most contextually relevant results from the fused candidate pool

This combination creates a more robust retrieval system for long-form, technical, and domain-specific documents.

---

## Project Structure

```text
hybrid-retrieval-reranker/
├── data/                     # Source documents and benchmark files
├── ingestion/                # Parsing and preprocessing
├── retrieval/
│   ├── bm25.py               # Elasticsearch BM25 retrieval
│   ├── dense.py              # FAISS dense retrieval
│   ├── hybrid.py             # Fusion logic
│   └── reranker.py           # LLM-based reranking
├── evaluation/               # Relevance and extraction benchmark scripts
├── app/                      # Search service / API layer
├── notebooks/                # Experiments and analysis
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Summary

This project demonstrates a practical **hybrid BM25-dense retrieval system with LLM re-ranking** for document search.

The title reflects one target use case, but the architecture itself is generic and can be applied across different retrieval and ranking problems that require both lexical and semantic search.
