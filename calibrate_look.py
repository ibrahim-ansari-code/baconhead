"""
Look calibration: drag the mouse right by increasing pixel amounts,
take before/after screenshots, save side-by-side for visual inspection.

Run with Roblox open, character standing still facing something distinctive.
"""

import os
import sys
import time

import mss
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from capture.screen import get_roblox_region, look_camera

OUT_DIR = "look_calib_shots"
os.makedirs(OUT_DIR, exist_ok=True)

# Pixel amounts to test — covers small nudge → large sweep
PIXEL_AMOUNTS = [50, 100, 150, 200, 250, 300, 400]

PAUSE_BEFORE  = 3.0   # seconds to get ready
PAUSE_BETWEEN = 2.5   # seconds between shots (time to look back)
DRAG_MS       = 400   # fixed drag duration for all tests


def snap(sct, region):
    raw = sct.grab(region)
    return np.array(raw)[..., :3][..., ::-1]   # BGR → RGB


def save_side_by_side(before, after, px, idx):
    h = max(before.shape[0], after.shape[0])
    w = before.shape[1] + after.shape[1] + 10
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 40
    canvas[:before.shape[0], :before.shape[1]] = before
    canvas[:after.shape[0], before.shape[1]+10:] = after
    path = os.path.join(OUT_DIR, f"{idx:02d}_px{px:03d}_before_after.png")
    Image.fromarray(canvas).save(path)
    print(f"  Saved: {path}", flush=True)
    return path


def main():
    with mss.mss() as sct:
        region = get_roblox_region()
        if region is None:
            print("ERROR: Roblox window not found.", flush=True)
            sys.exit(1)
        print(f"Roblox window: {region}", flush=True)

        print(f"\nStand still facing something distinctive (wall, building, landmark).")
        print(f"Starting in {PAUSE_BEFORE:.0f} seconds...\n", flush=True)
        time.sleep(PAUSE_BEFORE)

        results = []
        for i, px in enumerate(PIXEL_AMOUNTS):
            print(f"[{i+1}/{len(PIXEL_AMOUNTS)}] Dragging {px}px right ({DRAG_MS}ms)...", flush=True)

            before = snap(sct, region)

            # Do the look
            look_camera(px, DRAG_MS, region=region)
            time.sleep(0.3)

            after = snap(sct, region)
            path = save_side_by_side(before, after, px, i+1)
            results.append((px, path))

            # Pause so user can manually reset camera back to original facing
            if i < len(PIXEL_AMOUNTS) - 1:
                print(f"  → Manually rotate back to original facing. Continuing in {PAUSE_BETWEEN:.0f}s...", flush=True)
                time.sleep(PAUSE_BETWEEN)

        print(f"\nDone. {len(results)} images saved to '{OUT_DIR}/'.", flush=True)
        print("Look at the images and tell me which px amount looks like ~45° and ~90°.", flush=True)


if __name__ == "__main__":
    main()
