import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

transcript = json.loads(Path("data/transcript.json").read_text())
scenes = json.loads(Path("data/scenes.json").read_text())
WINDOW = 2  # seconds each frame covers

# 1. merge: for every frame, find the speech that overlaps its 2-second window
moments = []
for s in sorted(scenes, key=lambda s: s["time"]):
    t = s["time"]
    speech = " ".join(seg["text"] for seg in transcript
                      if seg["start"] < t + WINDOW and seg["end"] > t)
    moments.append({"time": t, "speech": speech, "scene": s["scene"]})

Path("data/moments.json").write_text(json.dumps(moments, indent=2))
print(f"Merged {len(moments)} moments")

# 2. embed three versions
model = SentenceTransformer("all-MiniLM-L6-v2")
versions = {
    "speech":   [m["speech"] for m in moments],
    "scene":    [m["scene"] for m in moments],
    "combined": [f"{m['speech']} {m['scene']}" for m in moments],
}
for name, texts in versions.items():
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.save(f"data/emb_{name}.npy", emb)
    print(f"Saved data/emb_{name}.npy  shape={emb.shape}")