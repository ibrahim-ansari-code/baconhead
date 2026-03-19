"""
scripts/record_demo.py — Record human demonstration for behavioral cloning.

Records (screenshot, action) pairs at 5fps. Saves to demos/run_NNN/.

Usage:
    python scripts/record_demo.py

Storage per run:
    frames.npz   — compressed N frames of shape (84, 84) float32
    actions.npy  — (N,) int array, action index per frame
    meta.json    — timestamp, fps, num_frames, duration, action_distribution
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import mss
import numpy as np
from pynput import keyboard

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.preprocess import preprocess_frame
from control.actions import ACTION_NAMES, NUM_ACTIONS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FPS = 5
FRAME_INTERVAL = 1.0 / FPS
COUNTDOWN_SECONDS = 5
DEMOS_DIR = Path(__file__).resolve().parent.parent / "demos"

# Key-to-action mapping (priority order matches control/actions.py)
# 0: forward (W), 1: left (A), 2: right (D), 3: jump (Space),
# 4: forward_jump (W+Space), 5: idle (nothing)

# ---------------------------------------------------------------------------
# Key tracking state
# ---------------------------------------------------------------------------

held_keys: set[str] = set()
space_latch = threading.Event()
stop_event = threading.Event()


def _on_press(key):
    try:
        if key == keyboard.Key.space:
            space_latch.set()
            held_keys.add("space")
        elif hasattr(key, "char") and key.char in ("w", "a", "d"):
            held_keys.add(key.char)
    except AttributeError:
        pass


def _on_release(key):
    try:
        if key == keyboard.Key.space:
            held_keys.discard("space")
        elif hasattr(key, "char") and key.char in ("w", "a", "d"):
            held_keys.discard(key.char)
    except AttributeError:
        pass


def get_action() -> int:
    """Map currently held keys to an action index. Consumes the space latch."""
    has_space = space_latch.is_set()
    space_latch.clear()

    has_w = "w" in held_keys
    has_a = "a" in held_keys
    has_d = "d" in held_keys

    # Priority order per plan
    if has_w and has_space:
        return 4  # forward_jump
    if has_space:
        return 3  # jump
    if has_w:
        return 0  # forward
    if has_a:
        return 1  # left
    if has_d:
        return 2  # right
    return 5  # idle


def next_run_dir() -> Path:
    """Find the next available run_NNN directory."""
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(DEMOS_DIR.glob("run_*"))
    if not existing:
        return DEMOS_DIR / "run_001"
    last_num = max(int(d.name.split("_")[1]) for d in existing if d.name.split("_")[1].isdigit())
    return DEMOS_DIR / f"run_{last_num + 1:03d}"


def main():
    # Signal handler for clean stop
    def _signal_handler(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)

    run_dir = next_run_dir()
    print(f"Recording to: {run_dir}")
    print(f"FPS: {FPS}")
    print(f"Actions: {ACTION_NAMES}")
    print()
    print("Controls: W/A/D to move, Space to jump. Ctrl+C to stop recording.")
    print()

    # Countdown
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"  Starting in {i}...", flush=True)
        time.sleep(1.0)
    print("  GO! Recording...\n")

    # Start keyboard listener
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    frames: list[np.ndarray] = []
    actions: list[int] = []
    start_time = time.time()

    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # primary monitor

            while not stop_event.is_set():
                loop_start = time.time()

                # Capture screen
                raw = sct.grab(monitor)
                frame_bgr = np.array(raw)[:, :, :3]  # BGRA -> BGR

                # Preprocess
                processed = preprocess_frame(frame_bgr)
                frames.append(processed)

                # Read action from held keys
                action = get_action()
                actions.append(action)

                # Frame rate limiter
                elapsed = time.time() - loop_start
                if elapsed < FRAME_INTERVAL:
                    time.sleep(FRAME_INTERVAL - elapsed)

    except Exception as e:
        print(f"\nError during recording: {e}")
    finally:
        listener.stop()

    duration = time.time() - start_time
    num_frames = len(frames)

    if num_frames == 0:
        print("No frames captured. Exiting.")
        return

    # Save
    run_dir.mkdir(parents=True, exist_ok=True)

    frames_arr = np.array(frames, dtype=np.float32)
    actions_arr = np.array(actions, dtype=np.int64)

    np.savez_compressed(run_dir / "frames.npz", frames=frames_arr)
    np.save(run_dir / "actions.npy", actions_arr)

    # Action distribution
    action_counts = {name: 0 for name in ACTION_NAMES}
    for a in actions_arr:
        action_counts[ACTION_NAMES[a]] = action_counts.get(ACTION_NAMES[a], 0) + 1

    meta = {
        "timestamp": datetime.now().isoformat(),
        "fps": FPS,
        "num_frames": num_frames,
        "duration_seconds": round(duration, 2),
        "action_distribution": action_counts,
    }

    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Summary
    print(f"\nRecording complete!")
    print(f"  Frames: {num_frames}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Saved to: {run_dir}")
    print(f"\n  Action distribution:")
    for name, count in action_counts.items():
        pct = (count / num_frames) * 100 if num_frames > 0 else 0
        flag = " ⚠️  0%!" if count == 0 else ""
        print(f"    {name:15s}: {count:5d} ({pct:5.1f}%){flag}")


if __name__ == "__main__":
    main()
