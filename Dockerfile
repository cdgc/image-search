FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for pillow/transformers/faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY app ./app
COPY scripts ./scripts
COPY data ./data

# Optional: prebuild index during image build (comment out if big dataset)
# RUN python scripts/build_index.py

EXPOSE 8000
ENV DATA_DIR=/app/data
ENV EMBED_DIR=/app/data/embeddings
ENV USE_FAISS=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
