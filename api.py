"""
FastAPI backend for the Recommendation Engine.
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import RecommendationEngine

# Global engine instance
engine: RecommendationEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("Initializing recommendation engine...")
    engine = RecommendationEngine(
        embedding_tier="balanced",
        use_reranker=True,
    )
    print("Engine ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Recommendation Engine API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request / Response models ----------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    bm25_weight: float = 0.4
    dense_weight: float = 0.6


class TextIngestRequest(BaseModel):
    text: str
    metadata: dict | None = None


class Result(BaseModel):
    rank: int
    score: float
    text: str
    metadata: dict


class QueryResponse(BaseModel):
    query: str
    results: list[Result]
    total_indexed: int


class StatsResponse(BaseModel):
    raw_documents: int
    chunks: int
    indexed: bool
    embedding_dim: int


class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    total_chunks: int


# ---------- Endpoints ----------

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


@app.get("/api/stats")
async def get_stats() -> StatsResponse:
    return StatsResponse(**engine.stats())


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    suffix = Path(file.filename).suffix.lower()
    supported = {".csv", ".xlsx", ".xls", ".docx", ".doc", ".pdf"}
    if suffix not in supported:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Supported: {supported}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        n_chunks = engine.ingest(tmp_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to ingest file: {e}")
    finally:
        os.unlink(tmp_path)

    return IngestResponse(
        filename=file.filename,
        chunks_created=n_chunks,
        total_chunks=len(engine.chunks),
    )


@app.post("/api/ingest/text")
async def ingest_text(req: TextIngestRequest) -> IngestResponse:
    n_chunks = engine.ingest_text(req.text, metadata=req.metadata)
    return IngestResponse(
        filename="<text>",
        chunks_created=n_chunks,
        total_chunks=len(engine.chunks),
    )


@app.post("/api/recommend")
async def recommend(req: QueryRequest) -> QueryResponse:
    if not engine._indexed:
        raise HTTPException(400, "No documents indexed yet. Upload data first.")

    results = engine.recommend(
        query=req.query,
        top_k=req.top_k,
        bm25_weight=req.bm25_weight,
        dense_weight=req.dense_weight,
    )

    return QueryResponse(
        query=req.query,
        results=[Result(**r) for r in results],
        total_indexed=len(engine.chunks),
    )


@app.post("/api/reset")
async def reset_engine():
    global engine
    engine = RecommendationEngine(
        embedding_tier="balanced",
        use_reranker=True,
    )
    return {"status": "reset", "message": "Engine cleared."}


# Mount static files (after API routes so they don't shadow them)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
