#!/usr/bin/env python3
"""
scripts/validate_perception.py — Phase 2 perception validation script.

Validates void detection, edge proximity, and directional bias on live frames.
Includes automated sanity checks and manual-assist checks.

Usage:
    python scripts/validate_perception.py

Requires Roblox to be visible on the primary monitor.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import mss
import numpy as np
import yaml

from vision.perception import compute_scene_state

# ---------------------------------------------------------------------------
# Screenshot saving
# ---------------------------------------------------------------------------

_SNAP_DIR = _PROJECT_ROOT / "logs" / "perception_validation"


def _save_frame(frame_bgr: np.ndarray, lower: np.ndarray, upper: np.ndarray, name: str) -> None:
    """Save the raw frame and a void-mask overlay side by side."""
    import cv2 as _cv2
    _SNAP_DIR.mkdir(parents=True, exist_ok=True)

    # Void mask overlay: void pixels tinted red
    hsv = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2HSV)
    mask = _cv2.inRange(hsv, lower, upper)
    overlay = frame_bgr.copy()
    overlay[mask > 0] = [0, 0, 220]  # red tint on void pixels

    # Resize both to same height for side-by-side (cap at 540px tall)
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, 540 / h)
    new_h, new_w = int(h * scale), int(w * scale)
    left = _cv2.resize(frame_bgr, (new_w, new_h))
    right = _cv2.resize(overlay, (new_w, new_h))
    combined = np.concatenate([left, right], axis=1)

    path = _SNAP_DIR / f"{name}.png"
    _cv2.imwrite(str(path), combined)
    print(f"     Saved: {path}")

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


def _load_void_hsv() -> tuple[np.ndarray, np.ndarray, float]:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)["capture"]
    lower = np.array(cfg["void_hsv_lower"], dtype=np.uint8)
    upper = np.array(cfg["void_hsv_upper"], dtype=np.uint8)
    threshold = float(cfg["death_void_ratio_threshold"])
    return lower, upper, threshold


def _grab_frame() -> np.ndarray:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = np.array(sct.grab(monitor))
    return raw[:, :, :3]  # BGRA → BGR


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

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
    global _passed, _failed

    lower, upper, death_threshold = _load_void_hsv()
    print(f"Void HSV lower: {lower.tolist()}")
    print(f"Void HSV upper: {upper.tolist()}")
    print(f"Death void ratio threshold: {death_threshold}")
    print()

    # ------------------------------------------------------------------
    # Automated checks on a single live frame
    # ------------------------------------------------------------------
    print("=== Automated checks (single frame) ===")
    frame = _grab_frame()
    state = compute_scene_state(frame, lower, upper)
    _save_frame(frame, lower, upper, "check_1234_baseline")

    print(f"  Scene state: {state}")
    print()

    _check(
        "1. void_ratio not degenerate",
        0.01 < state["void_ratio"] < 0.95,
        f"void_ratio={state['void_ratio']:.4f}",
    )
    _check(
        "2. edge_proximity in [0, 1]",
        0.0 <= state["edge_proximity"] <= 1.0,
        f"edge_proximity={state['edge_proximity']:.4f}",
    )
    _check(
        "3. direction_bias valid",
        state["direction_bias"] in {"left", "right", "center"},
        f"direction_bias={state['direction_bias']}",
    )
    _check(
        "4. platform ratios valid and sum > 0",
        0.0 <= state["platform_left"] <= 1.0
        and 0.0 <= state["platform_right"] <= 1.0
        and (state["platform_left"] + state["platform_right"]) > 0,
        f"left={state['platform_left']:.4f} right={state['platform_right']:.4f}",
    )

    # ------------------------------------------------------------------
    # Stability check — 10 frames while standing still
    # ------------------------------------------------------------------
    print()
    print("=== Stability check (10 frames, stand still) ===")
    ratios = []
    for i in range(10):
        f = _grab_frame()
        s = compute_scene_state(f, lower, upper)
        ratios.append(s["void_ratio"])
        if i == 0:
            _save_frame(f, lower, upper, "check_5_stability_frame0")
        time.sleep(0.1)

    std = float(np.std(ratios))
    _check(
        "5. void_ratio std dev < 0.15 (standing still)",
        std < 0.15,
        f"std={std:.4f}, values={[f'{r:.3f}' for r in ratios]}",
    )

    # ------------------------------------------------------------------
    # Manual-assist checks
    # ------------------------------------------------------------------
    print()
    print("=== Manual-assist checks ===")

    input("  6. Walk to a platform edge, then press Enter...")
    frame = _grab_frame()
    state = compute_scene_state(frame, lower, upper)
    _save_frame(frame, lower, upper, "check_6_edge_proximity")
    _check(
        "6. edge_proximity > 0.3 at platform edge",
        state["edge_proximity"] > 0.3,
        f"edge_proximity={state['edge_proximity']:.4f}",
    )

    input("  7. Jump into the void, press Enter after dying...")
    print("     Grabbing frames for 5 seconds...")
    max_void = 0.0
    max_frame = None
    end_time = time.monotonic() + 5.0
    while time.monotonic() < end_time:
        f = _grab_frame()
        s = compute_scene_state(f, lower, upper)
        if s["void_ratio"] > max_void:
            max_void = s["void_ratio"]
            max_frame = f
        time.sleep(0.1)
    if max_frame is not None:
        _save_frame(max_frame, lower, upper, "check_7_death_peak")

    _check(
        "7. void_ratio > threshold during death",
        max_void > death_threshold,
        f"max_void_ratio={max_void:.4f}, threshold={death_threshold}",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    total = _passed + _failed
    print(f"PERCEPTION VALIDATION: {_passed}/{total} passed, {_failed}/{total} failed")
    if _failed == 0:
        print("ALL CHECKS PASSED — Phase 2 perception gate met.")
    else:
        print("SOME CHECKS FAILED — review output above.")
    print("=" * 60)

    sys.exit(0 if _failed == 0 else 1)


if __name__ == "__main__":
    main()
