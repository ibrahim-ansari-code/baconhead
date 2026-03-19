#!/usr/bin/env python3
"""
scripts/calibrate_void.py — Void HSV calibration script.

Captures a live frame, displays it in a window. Click on void pixels to
sample their HSV values. Press 'q' or Enter when done — the script prints
the union HSV lower/upper bounds ready to paste into config.yaml.

Usage:
    python scripts/calibrate_void.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import mss
import numpy as np

# ---------------------------------------------------------------------------
# Globals for mouse callback
# ---------------------------------------------------------------------------

_samples: list[np.ndarray] = []  # list of mean HSV values from each click
_click_positions: list[tuple[int, int]] = []  # parallel list of (x, y) for each sample
_frame_hsv: np.ndarray | None = None
_frame_bgr: np.ndarray | None = None
_frame_bgr_orig: np.ndarray | None = None  # clean copy for redraw on undo

PATCH_RADIUS = 5  # 11x11 patch = radius 5


def _redraw() -> None:
    """Redraw all current sample markers onto a fresh copy of the original frame."""
    global _frame_bgr
    assert _frame_bgr_orig is not None
    _frame_bgr = _frame_bgr_orig.copy()
    for x, y in _click_positions:
        cv2.circle(_frame_bgr, (x, y), 8, (0, 255, 0), 2)
    cv2.imshow("Calibrate Void", _frame_bgr)


def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
    global _frame_hsv, _frame_bgr
    if event != cv2.EVENT_LBUTTONDOWN or _frame_hsv is None:
        return

    h, w = _frame_hsv.shape[:2]
    y1 = max(0, y - PATCH_RADIUS)
    y2 = min(h, y + PATCH_RADIUS + 1)
    x1 = max(0, x - PATCH_RADIUS)
    x2 = min(w, x + PATCH_RADIUS + 1)

    patch = _frame_hsv[y1:y2, x1:x2]
    mean_hsv = patch.mean(axis=(0, 1))
    _samples.append(mean_hsv)
    _click_positions.append((x, y))
    _redraw()

    print(f"  Sample {len(_samples)}: pixel ({x},{y}) → HSV mean [{mean_hsv[0]:.1f}, {mean_hsv[1]:.1f}, {mean_hsv[2]:.1f}]")


def main() -> None:
    global _frame_hsv, _frame_bgr, _frame_bgr_orig

    print("Capturing a live frame...")
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = np.array(sct.grab(monitor))

    _frame_bgr_orig = raw[:, :, :3].copy()
    _frame_bgr = _frame_bgr_orig.copy()
    _frame_hsv = cv2.cvtColor(_frame_bgr_orig, cv2.COLOR_BGR2HSV)

    print("Click on void pixels in the window.")
    print("Press 'u' to undo the last sample. Press 'q' or Enter when done.\n")

    cv2.imshow("Calibrate Void", _frame_bgr)
    cv2.setMouseCallback("Calibrate Void", _on_mouse)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 13, 10):  # q, Enter
            break
        if key == ord("u"):
            if _samples:
                removed = _samples.pop()
                _click_positions.pop()
                _redraw()
                print(f"  Undone. {len(_samples)} sample(s) remaining.")
            else:
                print("  Nothing to undo.")

    cv2.destroyAllWindows()

    if not _samples:
        print("\nNo samples collected. Exiting.")
        sys.exit(1)

    # Compute union bounds with tolerance
    all_hsv = np.array(_samples)
    h_min, s_min, v_min = all_hsv.min(axis=0)
    h_max, s_max, v_max = all_hsv.max(axis=0)

    # Apply tolerance and clamp to OpenCV ranges
    lower = [
        int(max(0, h_min - 10)),
        int(max(0, s_min - 40)),
        int(max(0, v_min - 40)),
    ]
    upper = [
        int(min(179, h_max + 10)),
        int(min(255, s_max + 40)),
        int(min(255, v_max + 40)),
    ]

    print(f"\n{'='*60}")
    print("VOID HSV CALIBRATION RESULT")
    print(f"{'='*60}")
    print(f"  Samples collected: {len(_samples)}")
    print(f"  Raw H range: [{h_min:.1f}, {h_max:.1f}]")
    print(f"  Raw S range: [{s_min:.1f}, {s_max:.1f}]")
    print(f"  Raw V range: [{v_min:.1f}, {v_max:.1f}]")
    print()
    print("  Paste into config.yaml lines 28-29:")
    print()
    print(f"  void_hsv_lower: {lower}")
    print(f"  void_hsv_upper: {upper}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
