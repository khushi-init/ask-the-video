import whisper, json

model = whisper.load_model("base")
result = model.transcribe("data/tears_of_steel.mp4")

segments = [
    {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
    for s in result["segments"]
]

with open("data/transcript.json", "w") as f:
    json.dump(segments, f, indent=2)

for s in segments[:10]:
    print(f"{s['start']:6.1f}s  {s['text']}")
print(f"\n{len(segments)} segments saved.")