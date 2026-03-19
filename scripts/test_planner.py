"""
scripts/test_planner.py — Validation script for GeminiPlanner.

Runs 10 tick() calls against a live Roblox screen.
Calls 2, 4, 6, 8, 10 wait 1.6s first so they hit the live API.
Calls 1, 3, 5, 7, 9 return the cache (fast).

Expected output: PASS (10/10)

Requirements:
    - GEMINI_API_KEY in .env or environment
    - Roblox open and visible on screen
    - pip install google-generativeai python-dotenv
"""

import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from capture.screen import Capturer
from agent.planner import GeminiPlanner

VALID_INSTRUCTIONS = {"forward", "left", "right", "jump", "forward_jump", "idle"}
TOTAL = 10
MAX_ELAPSED = 4.0  # seconds — each tick must complete within this


def main() -> None:
    print("Initialising Capturer...")
    capturer = Capturer()

    print("Initialising GeminiPlanner...")
    planner = GeminiPlanner()

    print("Switch to Roblox now...")
    for s in range(3, 0, -1):
        print(f"  {s}...", flush=True)
        time.sleep(1)
    print("Starting!")

    passes = 0
    failures = 0
    results = []

    for i in range(1, TOTAL + 1):
        is_api_round = (i % 2 == 0) or (i == 1)
        # Every even call: wait 1.6s so the throttle expires → hits live API
        if i % 2 == 0:
            time.sleep(1.6)

        # Capture frame
        capturer.tick()
        frame = capturer.last_frame
        if frame is None:
            print(f"[{i}/{TOTAL}] FAIL — capturer returned no frame")
            failures += 1
            results.append({"was_api_call": is_api_round, "reason": "no_frame"})
            continue

        # Time the planner tick
        t0 = time.time()
        result = planner.tick(frame)
        elapsed = time.time() - t0

        # Validate result
        ok = True
        fail_reasons = []

        if not isinstance(result, dict):
            ok = False
            fail_reasons.append(f"result is {type(result).__name__}, expected dict")

        instruction = result.get("instruction", "")
        if instruction not in VALID_INSTRUCTIONS:
            ok = False
            fail_reasons.append(f"invalid instruction {instruction!r}")

        confidence = result.get("confidence", -1)
        if not (isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0):
            ok = False
            fail_reasons.append(f"confidence out of range: {confidence!r}")

        reason = result.get("reason", "")
        if not isinstance(reason, str) or not reason.strip():
            ok = False
            fail_reasons.append("reason is empty or not a string")

        if elapsed >= MAX_ELAPSED:
            ok = False
            fail_reasons.append(f"elapsed {elapsed:.3f}s >= {MAX_ELAPSED}s limit")

        status = "PASS" if ok else "FAIL"
        cache_label = "cache" if i % 2 == 1 and i > 1 else "api"
        print(
            f"[{i}/{TOTAL}] {status} ({cache_label}) "
            f"instruction={instruction!r} confidence={confidence:.2f} "
            f"elapsed={elapsed:.3f}s"
        )
        if reason:
            print(f"          reason: {reason}")
        if fail_reasons:
            for r in fail_reasons:
                print(f"          ! {r}")

        results.append({"was_api_call": is_api_round, "reason": reason})

        if ok:
            passes += 1
        else:
            failures += 1

    # Liveness check: at least one API round must have returned a real response
    api_rounds = [r for r in results if r["was_api_call"] and r["reason"] != "init"]
    print()
    if not api_rounds:
        print("FAIL — no API round returned a real response (all fell back to cache)")
        sys.exit(1)

    if failures == 0:
        print(f"PASS ({passes}/{TOTAL})")
        sys.exit(0)
    else:
        print(f"FAIL ({passes}/{TOTAL})")
        sys.exit(1)


if __name__ == "__main__":
    main()
