import os
import numpy as np
import pandas as pd
from PIL import Image
from .config import settings
from .embedder import embed_text
from .indexer import ANNIndex, load_vectors, load_faiss
from .explain import caption_image, explain_match

class Searcher:
    def __init__(self):
        self.meta = pd.read_parquet(settings.META_PATH)
        self.vecs = load_vectors()
        self.dim = self.vecs.shape[1]
        self.idx = load_faiss()
        if self.idx is None:
            self.idx = ANNIndex(self.dim, use_faiss=settings.USE_FAISS)
            self.idx.add(self.vecs)

    def search(self, query: str, top_k: int = None):
        top_k = top_k or settings.TOP_K
        q_vec = embed_text([query])
        sims, ids = self.idx.search(q_vec, top_k)
        sims = sims[0]
        ids = ids[0]
        results = []
        for rank, (i, s) in enumerate(zip(ids, sims), start=1):
            path = self.meta.iloc[i]["path"]
            # Generate explanation (could be cached)
            try:
                img = Image.open(path).convert("RGB")
                cap = caption_image(img)
            except Exception:
                cap = "No caption"
            expl = explain_match(query, cap, float(s))
            results.append({
                "rank": rank,
                "path": path,
                "similarity": float(s),
                "caption": cap,
                "explanation": expl
            })
        return results
