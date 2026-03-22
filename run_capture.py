#!/usr/bin/env python3
"""
Capture Roblox gameplay to screen.

Usage:
  python run_capture.py
  python run_capture.py --report --report-every 15 --seconds 30
"""

import argparse
import threading
import time

from capture.screen import get_roblox_region, capture_loop


def main():
    parser = argparse.ArgumentParser(description="Capture Roblox gameplay")
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--no-window-detect", action="store_true")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show GameSense state predictions every N frames",
    )
    parser.add_argument("--report-every", type=int, default=15)
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
    elif not args.no_window_detect:
        region = get_roblox_region()
        if region:
            print("Using Roblox window:", region)
        else:
            print("Roblox window not found; using primary monitor.")

    # Load GameSense for --report if available
    game_sense = None
    if args.report:
        try:
            from vision.game_sense import load_game_sense

            game_sense = load_game_sense("game_sense.pt")
            print("GameSense model loaded for reporting.")
        except Exception:
            print("No GameSense model — using heuristic for --report.")

    fps = args.fps
    print(f"Starting capture at {fps} FPS. Ctrl+C to stop.")

    n = [0]
    t0 = [time.perf_counter()]
    stop = threading.Event()

    if args.seconds is not None:
        def stop_after():
            time.sleep(args.seconds)
            stop.set()

        threading.Thread(target=stop_after, daemon=True).start()

    def on_frame(frame, timestamp):
        n[0] += 1
        if n[0] == 1:
            print(f"Frame shape: {frame.shape} (H, W, C)")
        if n[0] % 50 == 0:
            elapsed = time.perf_counter() - t0[0]
            print(f"Captured {n[0]} frames | {n[0] / elapsed:.1f} FPS")
        if args.report and n[0] % args.report_every == 0:
            if game_sense:
                state, conf = game_sense.predict(frame)
            else:
                from vision.game_sense import heuristic_state

                state, conf = heuristic_state(frame)
            print(f"  [{n[0]}] State: {state} ({conf:.2f})")

    try:
        capture_loop(region=region, fps=fps, callback=on_frame, stop_event=stop)
    except KeyboardInterrupt:
        pass
    elapsed = time.perf_counter() - t0[0]
    print(f"Done. Total frames: {n[0]}, avg FPS: {n[0] / elapsed:.1f}")


if __name__ == "__main__":
    main()
