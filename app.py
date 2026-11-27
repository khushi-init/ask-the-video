import json
from pathlib import Path
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer

VERSION = "combined"   # which embeddings to search: "speech", "scene" or "combined"
MIN_SCORE = 0.30      # below this similarity, the moment probably isn't in the video

# loaded ONCE at startup, so every search is fast
moments = json.loads(Path("data/moments.json").read_text())
emb = np.load(f"data/emb_{VERSION}.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()


@app.get("/search")
def search(q: str, k: int = 5):
    qv = model.encode([q], normalize_embeddings=True)[0]
    scores = emb @ qv                      # cosine similarity with every moment
    top = np.argsort(-scores)[:k]          # indices of the k best matches
    print(f"query={q!r}  best score={scores[top[0]]:.2f}")   # handy for tuning MIN_SCORE
    return [
        {
            "time": moments[i]["time"],
            "score": round(float(scores[i]), 2),
            "scene": moments[i]["scene"],
            "speech": moments[i]["speech"],
        }
        for i in top
        if scores[i] >= MIN_SCORE
    ]


app.mount("/media", StaticFiles(directory="data"), name="media")


@app.get("/")
def index():
    return FileResponse("web/index.html")