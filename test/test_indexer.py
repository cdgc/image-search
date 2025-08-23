import numpy as np
from app.indexer import ANNIndex

def test_ann_basic():
    vecs = np.random.randn(100, 32).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ann = ANNIndex(dim=32, use_faiss=False)
    ann.add(vecs)
    q = vecs[:1]
    sims, idx = ann.search(q, top_k=3)
    assert sims.shape == (1,3)
    assert idx.shape == (1,3)
    assert idx[0,0] in (0,1,2,3,4,5,6,7,8,9)
