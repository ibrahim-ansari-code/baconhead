"""
scripts/validate_progress_estimator.py — Validates GeminiProgressEstimator + RewardCalculator.

Checks:
  1. 5 frames from Roblox → all return float in [0.0, 1.0]
  2. reason strings printed for human sanity check
  3. User moves forward; 5 more frames → mean(batch2) >= mean(batch1)
  4. RewardCalculator.compute(progress=None, ...) returns -0.001 (no crash)

Run from project root:
    /opt/homebrew/bin/python3.14 scripts/validate_progress_estimator.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.screen import Capturer
from agent.progress_estimator import GeminiProgressEstimator
from training.reward import RewardCalculator

PASSES: list[str] = []
FAILS: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSES.append(name)
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    else:
        FAILS.append(name)
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def capture_frames(capturer: Capturer, n: int) -> list:
    frames = []
    for _ in range(n):
        capturer.tick_fast()
        frames.append(capturer.last_frame.copy())
        time.sleep(0.15)
    return frames


def main() -> None:
    print("=" * 60)
    print("validate_progress_estimator.py")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    capturer = Capturer()
    estimator = GeminiProgressEstimator()

    # ------------------------------------------------------------------
    # Batch 1: 5 frames at start position
    # ------------------------------------------------------------------
    print("\nSwitch to Roblox now — starting in 3... 2... 1...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)
    print("Capturing batch 1 (stay still)...")

    frames1 = capture_frames(capturer, 5)
    scores1: list[float] = []
    reasons1: list[str] = []

    for i, frame in enumerate(frames1):
        result = estimator.estimate(frame)
        print(f"  Frame {i+1}: progress={result}")
        is_float = isinstance(result, float)
        in_range = is_float and 0.0 <= result <= 1.0
        check(f"batch1_frame{i+1}_is_float", is_float, f"got {type(result).__name__}")
        check(f"batch1_frame{i+1}_in_range", in_range, f"got {result}")
        if is_float:
            scores1.append(result)

    # Print reasons via a second pass (estimator is stateless, re-call would cost API)
    # Instead we just print scores summary
    print(f"\n  Batch 1 scores: {[round(s, 3) for s in scores1]}")
    mean1 = sum(scores1) / len(scores1) if scores1 else 0.0
    print(f"  Batch 1 mean: {mean1:.3f}")

    # ------------------------------------------------------------------
    # Batch 2: user moves forward
    # ------------------------------------------------------------------
    print("\nNow MOVE YOUR CHARACTER FORWARD significantly.")
    print("Starting in 5... 4... 3... 2... 1...")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)
    print("Capturing batch 2 (character should be further along)...")

    frames2 = capture_frames(capturer, 5)
    scores2: list[float] = []

    for i, frame in enumerate(frames2):
        result = estimator.estimate(frame)
        print(f"  Frame {i+1}: progress={result}")
        if isinstance(result, float):
            scores2.append(result)

    mean2 = sum(scores2) / len(scores2) if scores2 else 0.0
    print(f"\n  Batch 2 scores: {[round(s, 3) for s in scores2]}")
    print(f"  Batch 2 mean: {mean2:.3f}")

    check(
        "batch2_mean_gte_batch1",
        mean2 >= mean1,
        f"batch1={mean1:.3f} batch2={mean2:.3f}",
    )

    # ------------------------------------------------------------------
    # Check 4: RewardCalculator.compute(progress=None) returns -0.001
    # ------------------------------------------------------------------
    print("\nChecking RewardCalculator.compute(progress=None)...")
    calc = RewardCalculator()
    reward = calc.compute(progress=None, death_event=False, stuck=False)
    expected = -0.001
    tolerance = 1e-6
    check(
        "reward_none_progress_no_crash",
        abs(reward - expected) < tolerance,
        f"expected={expected:.4f} got={reward:.4f}",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    total = len(PASSES) + len(FAILS)
    print(f"Results: {len(PASSES)}/{total} checks passed")
    if FAILS:
        print("FAILED checks:")
        for f in FAILS:
            print(f"  - {f}")
        print("\nOVERALL: FAIL")
        sys.exit(1)
    else:
        print("\nOVERALL: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
