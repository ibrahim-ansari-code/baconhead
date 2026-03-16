"""
Collect (state, label) data for the reward model.
Runs capture + keyboard listener; saves downsampled frames and active/idle labels.
"""

import argparse
import json
import os
import time
import threading
from typing import Optional

import numpy as np

from capture.screen import get_roblox_region, capture_loop
from reward.input_state import start_listener, is_active


def run_collect(
    out_dir: str,
    region=None,
    fps: float = 10.0,
    sample_every: int = 3,
    frame_height: int = 84,
    frame_width: int = 84,
    active_window_seconds: float = 2.0,
    max_samples: int = 0,
    stop_event: Optional[threading.Event] = None,
):
    """
    Run capture and key listener; every sample_every frames, save (resized frame, active_label).
    active_label = 1 if user pressed a key in the last active_window_seconds, else 0.
    Saves to out_dir/data.npz (frames, labels) and out_dir/config.json.
    """
    os.makedirs(out_dir, exist_ok=True)
    start_listener()

    frames_list = []
    labels_list = []
    n = [0]
    log_every = 30  # log every N samples

    def on_frame(frame, timestamp):
        n[0] += 1
        if n[0] % sample_every != 0:
            return
        if max_samples > 0 and len(frames_list) >= max_samples:
            if stop_event:
                stop_event.set()
            return
        # Resize frame to (frame_height, frame_width, 3)
        from PIL import Image
        pil = Image.fromarray(frame.astype(np.uint8))
        pil = pil.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
        small = np.array(pil)  # (H, W, 3)
        label = 1 if is_active(active_window_seconds) else 0
        frames_list.append(small)
        labels_list.append(label)
        # Log what we're deciding (active vs idle)
        num = len(frames_list)
        if num % log_every == 0:
            n_active = sum(1 for l in labels_list if l == 1)
            print(f"  [collect] sample {num}: label={int(label)} ({'active' if label == 1 else 'idle'}) | total active={n_active}/{num}")

    try:
        capture_loop(region=region, fps=fps, callback=on_frame, stop_event=stop_event)
    except KeyboardInterrupt:
        pass

    if not frames_list:
        print("No samples collected.")
        return

    frames = np.stack(frames_list, axis=0)
    labels = np.array(labels_list, dtype=np.float32)
    np.savez(
        os.path.join(out_dir, "data.npz"),
        frames=frames,
        labels=labels,
    )
    config = {
        "fps": fps,
        "sample_every": sample_every,
        "frame_height": frame_height,
        "frame_width": frame_width,
        "active_window_seconds": active_window_seconds,
        "n_samples": len(frames_list),
        "n_active": int(labels.sum()),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved {len(frames_list)} samples to {out_dir} (active: {config['n_active']})")


def main():
    parser = argparse.ArgumentParser(description="Collect gameplay data for reward model")
    parser.add_argument("--out-dir", type=str, default="reward_data", help="Output directory")
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--sample-every", type=int, default=3, help="Save every N frames")
    parser.add_argument("--size", type=int, default=84, help="Frame height and width")
    parser.add_argument("--active-window", type=float, default=2.0, help="Seconds of no keys before label=idle")
    parser.add_argument("--max-samples", type=int, default=0, help="Stop after N samples (0 = unlimited)")
    parser.add_argument("--seconds", type=float, default=None, help="Run for N seconds then stop")
    parser.add_argument("--no-window-detect", action="store_true")
    parser.add_argument("--region", type=str, default=None, help="left,top,width,height")
    args = parser.parse_args()

    region = None
    if args.region:
        parts = [int(x.strip()) for x in args.region.split(",")]
        if len(parts) != 4:
            raise ValueError("--region must be left,top,width,height")
        region = {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
    elif not args.no_window_detect:
        region = get_roblox_region()
        if region:
            print("Using Roblox window:", region)
    if not region:
        print("Using primary monitor.")

    stop = threading.Event()
    if args.seconds is not None:
        def stop_after():
            time.sleep(args.seconds)
            stop.set()
        threading.Thread(target=stop_after, daemon=True).start()

    print("Collecting. Play the game; press keys to mark 'active' frames. Ctrl+C to stop.")
    print("(On Mac: if you see 'not trusted', add Terminal or Python to System Settings → Privacy & Security → Accessibility, then restart Terminal.)")
    run_collect(
        out_dir=args.out_dir,
        region=region,
        fps=args.fps,
        sample_every=args.sample_every,
        frame_height=args.size,
        frame_width=args.size,
        active_window_seconds=args.active_window,
        max_samples=args.max_samples,
        stop_event=stop,
    )


if __name__ == "__main__":
    main()
