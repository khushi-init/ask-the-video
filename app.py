import json
from pathlib import Path
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer

MIN_SCORE = 0.25   # retune this, see note below

# loaded ONCE at startup, so every search is fast
moments = json.loads(Path("data/moments.json").read_text())
emb_speech = np.load("data/emb_speech.npy")
emb_scene = np.load("data/emb_scene.npy")
has_speech = np.array([bool(m["speech"].strip()) for m in moments])
model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()


@app.get("/search")
def search(q: str, k: int = 5):
    qv = model.encode([q], normalize_embeddings=True)[0]
    # late fusion: score speech and scene separately, keep the better one
    speech_scores = np.where(has_speech, emb_speech @ qv, -1)
    scores = np.maximum(speech_scores, emb_scene @ qv)
    top = np.argsort(-scores)[:k]
    print(f"query={q!r}  best score={scores[top[0]]:.2f}")
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