from pydantic import BaseModel
from typing import List

class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None

class SearchItem(BaseModel):
    rank: int
    path: str
    similarity: float
    caption: str
    explanation: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchItem]
