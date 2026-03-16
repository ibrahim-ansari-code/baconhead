#!/usr/bin/env python3
"""
One-shot live test: find Roblox, capture frame, run CEM (real Scout if API key set),
focus window, execute one action. Use this to verify capture + Scout + input with your obby open.

If Roblox isn't detected: run with --full-screen to use primary monitor, or --region left,top,w,h.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from capture.screen import get_roblox_region, capture_region, focus_roblox_and_click
from llm_agent.cem import run_cem, execute_action_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-screen", action="store_true", help="Use full primary monitor if Roblox not found")
    parser.add_argument("--region", type=str, help="left,top,width,height (overrides Roblox detection)")
    args = parser.parse_args()

    print("=== Live test (Roblox obby) ===\n")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set. Using mock Scout.", flush=True)

    # 1) Find region (Roblox, --region, or --full-screen)
    print("1) Getting capture region...")
    region = None
    if args.region:
        parts = [int(x.strip()) for x in args.region.split(",")]
        if len(parts) != 4:
            print("   FAIL: --region must be left,top,width,height", file=sys.stderr)
            sys.exit(1)
        region = {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
        print(f"   Using --region: {region}")
    if region is None:
        region = get_roblox_region()
        if region:
            print(f"   Roblox window: {region}")
        else:
            if args.full_screen:
                with __import__("mss").mss() as m:
                    mon = m.monitors[0]
                    region = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}
                print(f"   Roblox not found; using full primary monitor: {region}")
            else:
                print("   FAIL: Roblox window not found.", file=sys.stderr)
                print("   Tip: Open Roblox, or use --full-screen or --region left,top,width,height", file=sys.stderr)
                sys.exit(1)
    print(f"   OK: region {region}")

    # 2) Capture one frame
    print("2) Capturing one frame...")
    sct = __import__("mss").mss()
    try:
        frame = capture_region(region=region, sct=sct)
    finally:
        sct.close()
    print(f"   OK: shape {frame.shape} dtype {frame.dtype}")

    # 3) Run CEM (real Scout or mock)
    print("3) Running CEM (one decision)...")
    t0 = time.perf_counter()
    best_action, scores, r, objectives, _ = run_cem(
        frame,
        scout_api_key=api_key,
        use_scout=bool(api_key),
        use_reward_model=False,
        last_actions=[],
        last_objective=None,
        mock_scout_result=None if api_key else ([0.7, 0.3, 0.3, 0.3, 0.2, 0.1, 0.7, 0.3, 0.3, 0.3], 0.0, "move forward"),
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"   OK: best={best_action!r} r={r:.3f} in {elapsed_ms:.0f} ms")
    if objectives:
        print(f"   objectives: {objectives[:80]}...")
    print(f"   scores: {[f'{s:.2f}' for s in scores]}")

    # 4) Focus Roblox and execute one action
    print("4) Focusing Roblox and executing action...")
    if focus_roblox_and_click():
        print("   Focus + click OK.")
    else:
        print("   Focus failed (non-Mac?). Keys may not reach game.")
    duration_ms = 600
    print(f"   Executing {best_action!r} for {duration_ms} ms...")
    execute_action_ms(best_action, duration_ms=duration_ms)
    print("   Done.")

    print("\n=== Live test passed. Check that the character moved in Roblox. ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
