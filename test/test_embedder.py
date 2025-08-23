import os
import numpy as np
from app.config import settings
from app.embedder import embed_text

def test_text_embedding_shape():
    q = ["a test query"]
    emb = embed_text(q)
    assert len(emb.shape) == 2
    assert emb.shape[0] == 1
    assert emb.shape[1] > 0
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-2)  # normalized
