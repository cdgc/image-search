from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from functools import lru_cache

@lru_cache(maxsize=1)
def _load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

def caption_image(img: Image.Image) -> str:
    processor, model = _load_blip()
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_length=40)
    text = processor.decode(out[0], skip_special_tokens=True)
    return text

def explain_match(query: str, caption: str, score: float) -> str:
    return (
        f"This image is relevant because its caption mentions: '{caption}'. "
        f"The semantic match to your query '{query}' is high (similarity={score:.2f})."
    )
