import json
from pathlib import Path
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer

VERSION = "combined"

# loaded ONCE at startup, so every search is fast
moments = json.loads(Path("data/moments.json").read_text())
emb = np.load(f"data/emb_{VERSION}.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

@app.get("/search")
def search(q: str, k: int = 5):
    qv = model.encode([q], normalize_embeddings=True)[0]
    scores = emb @ qv
    top = np.argsort(-scores)[:k]
    return [
        {"time": moments[i]["time"], "score": round(float(scores[i]), 2),
         "scene": moments[i]["scene"], "speech": moments[i]["speech"]}
        for i in top
    ]

app.mount("/media", StaticFiles(directory="data"), name="media")

@app.get("/")
def index():
    return FileResponse("web/index.html")