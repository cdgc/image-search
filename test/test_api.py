from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_search_schema_smoke(monkeypatch):
    # monkeypatch Searcher to avoid heavy init in CI
    from app import main
    class DummySearcher:
        def search(self, q, k=None):
            return [{"rank":1,"path":"data/images/000001.jpg","similarity":0.99,
                     "caption":"a red car on street","explanation":"mock"}]
    main._searcher = DummySearcher()

    payload = {"query": "red car", "top_k": 3}
    r = client.post("/search", json=payload)
    assert r.status_code == 200
    js = r.json()
    assert js["query"] == "red car"
    assert isinstance(js["results"], list)
    assert js["results"][0]["rank"] == 1
