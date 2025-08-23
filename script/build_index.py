from app.embedder import build_embeddings
from app.indexer import ANNIndex, save_faiss
from app.config import settings
import numpy as np
import os

if __name__ == "__main__":
    embs, meta = build_embeddings()
    idx = ANNIndex(dim=embs.shape[1], use_faiss=True)
    idx.add(embs)
    save_faiss(idx)
    print(f"Built FAISS index at {settings.INDEX_PATH} with {len(meta)} items.")
