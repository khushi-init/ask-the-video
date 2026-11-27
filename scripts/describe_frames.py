import os, json, time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-3.1-flash-lite"  # check AI Studio for the current Flash model name
FRAMES_DIR = Path("data/frames")
OUT_FILE = Path("data/scenes.json")
SECONDS_PER_FRAME = 2

PROMPT = ("Describe this movie frame in one sentence: who or what is visible, "
          "what is happening, the setting, and the mood. Be specific.")

# resume support: keep what we already have
scenes = json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else []
done = {s["frame"] for s in scenes}

frames = sorted(FRAMES_DIR.glob("frame_*.jpg"))
for i, frame in enumerate(frames):
    if frame.name in done:
        continue

    image = types.Part.from_bytes(data=frame.read_bytes(), mime_type="image/jpeg")
    response = None
    for attempt in range(5):
        try:
            response = client.models.generate_content(model=MODEL, contents=[image, PROMPT])
            break
        except Exception as e:
            if "404" in str(e):
                raise                      # permanent error: stop immediately
            wait = 10 * 2 ** attempt       # 10s, 20s, 40s, 80s, 160s
            print(f"{frame.name}: {str(e)[:60]}… retrying in {wait}s")
            time.sleep(wait)
    if response is None:
        print(f"Skipping {frame.name} after 5 failed attempts")
        continue

    t = i * SECONDS_PER_FRAME
    scenes.append({"frame": frame.name, "time": t, "scene": response.text.strip()})
    OUT_FILE.write_text(json.dumps(scenes, indent=2))   # save after every frame
    print(f"{t:5d}s  {response.text.strip()}")

    time.sleep(4)   # stay under the free-tier rate limit

print(f"\nDone: {len(scenes)} scenes saved.")