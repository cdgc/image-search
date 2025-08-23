"""
Image Downloader for Photo Dataset (fixed)
-----------------------------------------
- Reads 'photos_url.csv' from DATA_DIR (env) or './data'
- Saves optimized JPEGs to ./data/images
"""

import os
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm

DATA_DIR = os.getenv("DATA_DIR", "./data")
CSV_PATH = os.path.join(DATA_DIR, "photos_url.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_images(num_images=None, target_size=(800, 800), timeout=10):
    df = pd.read_csv(CSV_PATH)
    if num_images:
        df = df.head(num_images)
    print(f"Downloading {len(df)} images to {OUTPUT_DIR}...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            url = row.get('photo_image_url') or row.get('image_url') or row.iloc[0]
            if not isinstance(url, str) or not url.startswith("http"):
                continue
            filename = f"{(idx+1):06d}.jpg"
            out_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.exists(out_path):
                continue

            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            img.save(out_path, 'JPEG', quality=85, optimize=True)
        except Exception as e:
            # soft-fail; log and continue
            print(f"Skip idx={idx+1}: {e}")

if __name__ == "__main__":
    # Change num_images to a small value for Colab demo if needed
    download_images(num_images=None)
