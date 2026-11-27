import sys, json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

VERSION = "combined"
moments = json.loads(Path("data/moments.json").read_text())
emb = np.load(f"data/emb_{VERSION}.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

query = " ".join(sys.argv[1:])
q = model.encode([query], normalize_embeddings=True)[0]
scores = emb @ q                       # cosine similarity (vectors are normalised)
for i in np.argsort(-scores)[:3]:
    m = moments[i]
    print(f"{m['time']:5d}s  score={scores[i]:.2f}  {m['scene'][:80]}")
    if m["speech"]:
        print(f"        💬 {m['speech'][:80]}")