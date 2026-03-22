"""
Collect training data for GameSense by playing the game.

Captures frames while you play and auto-labels them:
  - playing:  you're pressing keys, game is active
  - dead:     bright white / black screen
  - menu:     static screen with no input
  - danger:   frames right before a death event

Post-hoc relabeling marks frames in the seconds before death as "danger"
so the model learns to predict danger BEFORE it happens.

Usage:
  python -m vision.collect --seconds 120 --out game_data
"""

import argparse
import os
import time
from typing import Optional

import mss
import numpy as np
from PIL import Image

from capture.screen import get_roblox_region, capture_region
from reward.input_state import start_listener, is_active
from vision.game_sense import heuristic_state, STATE_LABELS, STATE_TO_IDX

FRAME_SIZE = 224
SAMPLE_FPS = 2.0
DANGER_LOOKBACK = 3.0  # seconds before death → relabel as danger


def collect(
    out_dir: str,
    total_seconds: float = 120.0,
    region: Optional[dict] = None,
    sample_fps: float = SAMPLE_FPS,
):
    os.makedirs(out_dir, exist_ok=True)
    start_listener()
    sct = mss.mss()

    frames_list = []
    labels_list = []
    timestamps_list = []

    interval = 1.0 / sample_fps
    t_start = time.perf_counter()
    t_next = t_start
    prev_small = None
    count = 0

    print(
        f"[collect] Recording for {total_seconds:.0f}s at {sample_fps} fps. "
        "Play the game!",
        flush=True,
    )

    try:
        while True:
            now = time.perf_counter()
            if now - t_start >= total_seconds:
                break
            if now < t_next:
                time.sleep(min(0.05, t_next - now))
                continue

            raw = capture_region(region=region, sct=sct)
            small = np.array(
                Image.fromarray(raw.astype(np.uint8)).resize(
                    (FRAME_SIZE, FRAME_SIZE), Image.Resampling.LANCZOS
                )
            )

            user_active = is_active(2.0)
            state = _auto_label(small, user_active, prev_small)

            frames_list.append(small)
            labels_list.append(STATE_TO_IDX[state])
            timestamps_list.append(now - t_start)

            count += 1
            if count % 20 == 0:
                elapsed = now - t_start
                print(
                    f"[collect] {count} frames | {elapsed:.0f}s | last={state}",
                    flush=True,
                )

            prev_small = small
            t_next += interval
            if t_next < now:
                t_next = now + interval

    except KeyboardInterrupt:
        pass
    finally:
        sct.close()

    if not frames_list:
        print("[collect] No frames captured.", flush=True)
        return

    frames = np.stack(frames_list, axis=0)
    labels = np.array(labels_list, dtype=np.int8)
    timestamps = np.array(timestamps_list, dtype=np.float64)

    # Post-hoc: relabel frames before death as danger
    labels = _relabel_danger(labels, timestamps, lookback=DANGER_LOOKBACK)

    # Append to existing data
    npz_path = os.path.join(out_dir, "data.npz")
    if os.path.isfile(npz_path):
        existing = np.load(npz_path, allow_pickle=False)
        frames = np.concatenate([existing["frames"], frames], axis=0)
        labels = np.concatenate([existing["labels"], labels], axis=0)
        timestamps = np.concatenate(
            [existing["timestamps"], timestamps], axis=0
        )
        old_n = len(existing["labels"])
        print(f"[collect] Appended to existing ({old_n} + {count})", flush=True)

    np.savez(npz_path, frames=frames, labels=labels, timestamps=timestamps)

    print(f"[collect] Saved {len(labels)} total frames → {npz_path}", flush=True)
    for i, name in enumerate(STATE_LABELS):
        n = int((labels == i).sum())
        print(f"  {name}: {n}", flush=True)


def _auto_label(
    frame: np.ndarray,
    user_active: bool,
    prev_frame: Optional[np.ndarray],
) -> str:
    """Heuristic auto-labeling for bootstrap training."""
    state, _ = heuristic_state(frame)
    if state in ("dead", "danger"):
        return state

    # Menu: static screen + no user activity
    if not user_active and prev_frame is not None:
        diff = float(
            np.abs(frame.astype(float) - prev_frame.astype(float)).mean()
        )
        if diff < 3.0:
            return "menu"

    return "playing"


def _relabel_danger(
    labels: np.ndarray,
    timestamps: np.ndarray,
    lookback: float = 3.0,
) -> np.ndarray:
    """Relabel frames in the N seconds before each 'dead' frame as 'danger'."""
    dead_idx = STATE_TO_IDX["dead"]
    danger_idx = STATE_TO_IDX["danger"]
    labels = labels.copy()

    dead_times = timestamps[labels == dead_idx]
    for dt in dead_times:
        mask = (
            (timestamps >= dt - lookback)
            & (timestamps < dt)
            & (labels != dead_idx)
        )
        labels[mask] = danger_idx

    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Collect GameSense training data"
    )
    parser.add_argument("--out", type=str, default="game_data")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--fps", type=float, default=SAMPLE_FPS)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--full-screen", action="store_true")
    args = parser.parse_args()

    region = None
    if args.region:
        parts = [int(x.strip()) for x in args.region.split(",")]
        if len(parts) != 4:
            raise ValueError("--region must be left,top,width,height")
        region = {
            "left": parts[0],
            "top": parts[1],
            "width": parts[2],
            "height": parts[3],
        }
    elif args.full_screen:
        with mss.mss() as m:
            mon = m.monitors[0]
        region = {
            "left": mon["left"],
            "top": mon["top"],
            "width": mon["width"],
            "height": mon["height"],
        }
    else:
        region = get_roblox_region()
        if region:
            print(f"[collect] Roblox window: {region}", flush=True)
        else:
            print("[collect] Roblox not found, using primary monitor.", flush=True)

    collect(
        out_dir=args.out,
        total_seconds=args.seconds,
        region=region,
        sample_fps=args.fps,
    )


if __name__ == "__main__":
    main()
