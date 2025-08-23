import os

class Settings:
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    EMBED_DIR: str = os.getenv("EMBED_DIR", "./data/embeddings")
    IMAGE_DIR: str = os.path.join(DATA_DIR, "images")
    INDEX_PATH: str = os.path.join(EMBED_DIR, "faiss.index")
    META_PATH: str = os.path.join(EMBED_DIR, "meta.parquet")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "clip-ViT-B-32")  # sentence-transformers
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    DEVICE: str = os.getenv("DEVICE", "cuda" if os.environ.get("USE_CUDA") == "1" else "cpu")
    USE_FAISS: bool = os.getenv("USE_FAISS", "1") == "1"
    TOP_K: int = int(os.getenv("TOP_K", "5"))

settings = Settings()
