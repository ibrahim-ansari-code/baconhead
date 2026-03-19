"""
capture/debug_ocr.py — One-shot OCR debug: saves crop and tries OCR with and without allowlist.
Run from project root: python -m capture.debug_ocr
"""
import sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import mss
import numpy as np
from PIL import Image
import yaml

with open(_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)["capture"]

roi = cfg["hud_roi"]

print("Capturing in 3 seconds — switch to Roblox now...")
time.sleep(3)

with mss.mss() as sct:
    monitor = sct.monitors[1]
    print(f"Monitor size: {monitor['width']}x{monitor['height']}")
    full = np.array(sct.grab(monitor))[:, :, :3]

print(f"Full frame shape: {full.shape}")  # height x width x 3

# Save full frame
Image.fromarray(full).save(_ROOT / "logs/debug_full.png")
print("Saved logs/debug_full.png")

# Save crop
t, l, h, w = roi["top"], roi["left"], roi["height"], roi["width"]
crop = full[t:t+h, l:l+w]
print(f"Crop region: top={t} left={l} height={h} width={w} → shape={crop.shape}")
Image.fromarray(crop).save(_ROOT / "logs/debug_crop.png")
print("Saved logs/debug_crop.png")

# Try OCR without allowlist
print("\nRunning EasyOCR WITHOUT allowlist...")
import easyocr
reader = easyocr.Reader(["en"], gpu=False)
results = reader.readtext(crop, detail=0)
print(f"  Result (no allowlist): {results}")

# Try OCR with allowlist
print("Running EasyOCR WITH allowlist '0123456789/'...")
results2 = reader.readtext(crop, allowlist="0123456789/", detail=0)
print(f"  Result (allowlist):    {results2}")

# Try on full frame with bounding boxes to find stage counter position
print("\nRunning EasyOCR on full frame with bounding boxes...")
results3 = reader.readtext(full)  # detail=1 (default) returns [[bbox, text, conf], ...]
for bbox, text, conf in results3:
    if "stage" in text.lower() or any(c.isdigit() for c in text):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        print(f"  text={text!r:30s} conf={conf:.2f}  x={int(min(xs))}-{int(max(xs))}  y={int(min(ys))}-{int(max(ys))}")
