import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
from .config import settings

def _load_model():
    model = SentenceTransformer(settings.MODEL_NAME, device=settings.DEVICE)
    return model

def list_images():
    pattern = os.path.join(settings.IMAGE_DIR, "*.jpg")
    files = sorted(glob.glob(pattern))
    return files

def load_image(path):
    return Image.open(path).convert("RGB")

def embed_images(batch_paths, model):
    imgs = [load_image(p) for p in batch_paths]
    # sentence-transformers can encode PIL images for CLIP models
    with torch.inference_mode():
        embs = model.encode(imgs, batch_size=len(imgs), convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    return embs

def build_embeddings():
    os.makedirs(settings.EMBED_DIR, exist_ok=True)
    model = _load_model()
    paths = list_images()
    if not paths:
        raise RuntimeError(f"No images found in {settings.IMAGE_DIR}")

    all_embs = []
    meta = []
    B = settings.BATCH_SIZE

    for i in tqdm(range(0, len(paths), B), desc="Embedding"):
        batch = paths[i:i+B]
        embs = embed_images(batch, model)
        all_embs.append(embs)
        for p in batch:
            meta.append({"path": p})

    embs = np.vstack(all_embs).astype("float32")
    meta_df = pd.DataFrame(meta)
    np.save(os.path.join(settings.EMBED_DIR, "img_embs.npy"), embs)
    meta_df.to_parquet(settings.META_PATH, index=False)
    return embs, meta_df

def embed_text(queries):
    model = _load_model()
    q_embs = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    return q_embs.astype("float32")
