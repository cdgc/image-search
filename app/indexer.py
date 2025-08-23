import os
import numpy as np
import faiss
from sklearn.neighbors import NearestNeighbors
from .config import settings

class ANNIndex:
    def __init__(self, dim, use_faiss=True):
        self.dim = dim
        self.use_faiss = use_faiss
        if use_faiss:
            self.index = faiss.IndexFlatIP(dim)  # cosine if vectors normalized
        else:
            self.nn = NearestNeighbors(metric="cosine", n_neighbors=10)

    def add(self, vectors):
        if self.use_faiss:
            self.index.add(vectors)
        else:
            self.nn.fit(vectors)

    def search(self, queries, top_k):
        if self.use_faiss:
            D, I = self.index.search(queries, top_k)
            return D, I
        else:
            # sklearn returns distances; convert to similarity
            dist, idx = self.nn.kneighbors(queries, n_neighbors=top_k, return_distance=True)
            sims = 1 - dist
            return sims, idx

def load_vectors():
    path = os.path.join(settings.EMBED_DIR, "img_embs.npy")
    if not os.path.exists(path):
        raise FileNotFoundError("Embeddings not found. Run build_index first.")
    vecs = np.load(path)
    return vecs

def save_faiss(index):
    faiss.write_index(index.index, settings.INDEX_PATH)

def load_faiss():
    if not os.path.exists(settings.INDEX_PATH):
        return None
    idx = faiss.read_index(settings.INDEX_PATH)
    dim = idx.d
    wrapper = ANNIndex(dim, use_faiss=True)
    wrapper.index = idx
    return wrapper
