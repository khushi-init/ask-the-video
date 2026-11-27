import json
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer

K = 3

moments = json.load(open("data/moments.json"))
times = np.array([m["time"] for m in moments])
queries = json.load(open("eval/queries.json"))

empty = [i for i, q in enumerate(queries) if not q["query"].strip()]
if empty:
    raise SystemExit(f"Fill in the query text first, entries {empty} are empty.")

model = SentenceTransformer("all-MiniLM-L6-v2")
query_emb = model.encode([q["query"] for q in queries], normalize_embeddings=True)
types = sorted({q["type"] for q in queries})

print(f"{'':10s}{'overall':>10s}" + "".join(f"{t:>10s}" for t in types))
emb_speech = np.load("data/emb_speech.npy")
emb_scene = np.load("data/emb_scene.npy")
emb_comb = np.load("data/emb_combined.npy")
has_speech = np.array([bool(m["speech"].strip()) for m in moments])

def scores_for(version, qv):
    if version == "speech":
        return emb_speech @ qv
    if version == "scene":
        return emb_scene @ qv
    if version == "combined":
        return emb_comb @ qv
    if version == "max":                       # late fusion
        s = np.where(has_speech, emb_speech @ qv, -1)   # ignore moments with no speech
        return np.maximum(s, emb_scene @ qv)

for version in ["speech", "scene", "combined", "max"]:
    hits, totals = defaultdict(int), defaultdict(int)
    for item, qv in zip(queries, query_emb):
        top = np.argsort(-scores_for(version, qv))[:K]
        hit = any(item["start"] <= times[i] <= item["end"] for i in top)
        for key in ("overall", item["type"]):
            totals[key] += 1
            hits[key] += hit
    row = f"{version:10s}{hits['overall'] / totals['overall']:>10.0%}"
    row += "".join(f"{hits[t] / totals[t]:>10.0%}" for t in types)
    print(row)