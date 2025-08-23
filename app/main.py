from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import SearchRequest, SearchResponse, SearchItem
from .config import settings
from .search import Searcher
from .embedder import build_embeddings
from .indexer import ANNIndex, save_faiss, load_vectors
import os

app = FastAPI(title="Image Semantic Search", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

_searcher = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/index/build")
def build_index():
    embs, meta = build_embeddings()
    idx = ANNIndex(dim=embs.shape[1], use_faiss=True)
    idx.add(embs)
    save_faiss(idx)
    return {"embeddings": int(embs.shape[0]), "dim": int(embs.shape[1])}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    global _searcher
    if _searcher is None:
        _searcher = Searcher()
    res = _searcher.search(req.query, req.top_k)
    return SearchResponse(
        query=req.query,
        results=[SearchItem(**r) for r in res]
    )
