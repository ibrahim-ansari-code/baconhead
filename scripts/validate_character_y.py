#!/usr/bin/env python3
"""
scripts/validate_character_y.py — Validates estimate_character_y() on live frames.

Runs two checks:
  1. Dry-run loop: prints normalized Y every second for 10 seconds so you can
     manually move the character and confirm Y decreases as you move forward.
  2. Position check: captures Y at start position vs. mid-obstacle and asserts
     that mid-obstacle Y is lower (further along = smaller Y).

Usage:
    python scripts/validate_character_y.py

Switch to Roblox before running. The script will countdown before each capture.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import mss
import numpy as np

from vision.perception import estimate_character_y


def _grab_frame() -> np.ndarray:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = np.array(sct.grab(monitor))
    return raw[:, :, :3]  # BGRA → BGR


def _countdown(seconds: int = 3) -> None:
    for i in range(seconds, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)


_passed = 0
_failed = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if condition:
        _passed += 1
    else:
        _failed += 1


def main() -> None:
    # ------------------------------------------------------------------
    # Part 1: dry-run loop — print Y every second for 10 seconds
    # ------------------------------------------------------------------
    print("=== Part 1: Live Y monitor (10 seconds) ===")
    print("Move the character forward through the obstacle.")
    print("Y should decrease as you move forward.")
    print()
    print("Switch to Roblox now — starting in:")
    _countdown(3)

    none_count = 0
    for i in range(10):
        frame = _grab_frame()
        y = estimate_character_y(frame)
        if y is None:
            none_count += 1
            print(f"  t={i+1:02d}s  y=None (character not detected)")
        else:
            bar_len = int((1.0 - y) * 40)  # longer bar = further along
            bar = "#" * bar_len + "." * (40 - bar_len)
            print(f"  t={i+1:02d}s  y={y:.4f}  [{bar}]")
        time.sleep(1)

    print()
    detection_rate = (10 - none_count) / 10
    _check(
        "1. Character detected in ≥ 70% of frames",
        detection_rate >= 0.7,
        f"detected {10 - none_count}/10 frames",
    )

    # ------------------------------------------------------------------
    # Part 2: start vs. mid-obstacle position check
    # ------------------------------------------------------------------
    print()
    print("=== Part 2: Start vs. mid-obstacle position check ===")

    print()
    print("Stand at the START of the obstacle, then press Enter...")
    input()
    print("Capturing start position — Switch to Roblox now:")
    _countdown(3)
    start_ys = []
    for _ in range(5):
        f = _grab_frame()
        y = estimate_character_y(f)
        if y is not None:
            start_ys.append(y)
        time.sleep(0.2)
    start_y = float(np.mean(start_ys)) if start_ys else None
    print(f"  Start Y: {start_y:.4f}" if start_y is not None else "  Start Y: None (not detected)")

    print()
    print("Now move the character FORWARD to mid-obstacle, then press Enter...")
    input()
    print("Capturing mid-obstacle position — Switch to Roblox now:")
    _countdown(3)
    mid_ys = []
    for _ in range(5):
        f = _grab_frame()
        y = estimate_character_y(f)
        if y is not None:
            mid_ys.append(y)
        time.sleep(0.2)
    mid_y = float(np.mean(mid_ys)) if mid_ys else None
    print(f"  Mid Y:   {mid_y:.4f}" if mid_y is not None else "  Mid Y: None (not detected)")

    print()
    _check(
        "2. Start Y detected",
        start_y is not None,
        f"y={start_y}",
    )
    _check(
        "3. Mid Y detected",
        mid_y is not None,
        f"y={mid_y}",
    )
    if start_y is not None and mid_y is not None:
        _check(
            "4. Mid Y < Start Y (further forward = lower Y)",
            mid_y < start_y,
            f"start={start_y:.4f} mid={mid_y:.4f} delta={start_y - mid_y:.4f}",
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    total = _passed + _failed
    print(f"CHARACTER Y VALIDATION: {_passed}/{total} passed, {_failed}/{total} failed")
    if _failed == 0:
        print("ALL CHECKS PASSED — estimate_character_y is working.")
    else:
        print("SOME CHECKS FAILED — review output above.")
    print("=" * 60)

    sys.exit(0 if _failed == 0 else 1)


if __name__ == "__main__":
    main()
