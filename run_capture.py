#!/usr/bin/env python3
"""
Run gameplay capture from Roblox.
- Tries to find Roblox window on Mac; otherwise captures primary monitor (or region from config).
- Optional: report what we see (image caption) every N frames via --report.

Usage:
  pip install -r requirements.txt
  python run_capture.py
  python run_capture.py --report --report-every 15 --seconds 30
"""

import argparse
import threading
import time

from capture.screen import get_roblox_region, capture_loop


def main():
    parser = argparse.ArgumentParser(description="Capture Roblox gameplay")
    parser.add_argument("--fps", type=float, default=10, help="Capture FPS (default 10)")
    parser.add_argument("--region", type=str, default=None, help="Optional: left,top,width,height")
    parser.add_argument("--no-window-detect", action="store_true", help="Skip Roblox window detection; use full monitor")
    parser.add_argument("--seconds", type=float, default=None, help="Run for N seconds then exit (default: forever)")
    parser.add_argument("--report", action="store_true", help="Run vision model and print what we see")
    parser.add_argument("--report-every", type=int, default=15, help="Report every N frames when --report (default 15)")
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
        else:
            print("Roblox window not found; using primary monitor. Use --region l,t,w,h to capture a specific area.")

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
            from vision.report import describe_frame
            desc = describe_frame(frame)
            print(f"  [{n[0]}] What we see: {desc}")

    try:
        capture_loop(region=region, fps=fps, callback=on_frame, stop_event=stop)
    except KeyboardInterrupt:
        pass
    elapsed = time.perf_counter() - t0[0]
    print(f"Done. Total frames: {n[0]}, avg FPS: {n[0] / elapsed:.1f}")


if __name__ == "__main__":
    main()
