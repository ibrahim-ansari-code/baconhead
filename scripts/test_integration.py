#!/usr/bin/env python3
"""
scripts/test_integration.py — Phase 3 integration test.

Validates the full ObbyEnv pipeline against the real Roblox environment:
  - reset() returns a correctly-shaped observation
  - 20 steps with random actions all return correctly-shaped observations
  - reward, terminated, and truncated are the correct types

Requires Roblox to be running and visible on screen.

Usage:
    python scripts/test_integration.py

Pass/fail output is printed after every step. Exits 0 on success, 1 on failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from agent.env import ObbyEnv

STEPS = 20
PASS = "PASS"
FAIL = "FAIL"

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    tag = PASS if condition else FAIL
    print(f"  [{tag}] {label}")
    if not condition:
        failures.append(label)


def main() -> None:
    print("=" * 60)
    print("Phase 3 integration test — ObbyEnv")
    print("=" * 60)
    print("Requires Roblox to be running and visible on screen.")
    print()

    env = ObbyEnv()

    # --- reset ---
    print("reset()")
    obs, info = env.reset()
    check(f"obs shape == (4, 84, 84)  got {obs.shape}", obs.shape == (4, 84, 84))
    check(f"obs dtype == float32  got {obs.dtype}", obs.dtype == np.float32)
    check(f"obs min >= 0.0  got {obs.min():.4f}", float(obs.min()) >= 0.0)
    check(f"obs max <= 1.0  got {obs.max():.4f}", float(obs.max()) <= 1.0)
    check(f"info contains 'stage'", "stage" in info)
    print()

    # --- steps ---
    print(f"Running {STEPS} steps with random actions...")
    print(f"  {'step':>4}  {'action':>6}  {'obs shape':>12}  {'reward':>8}  {'term':>6}  {'trunc':>6}  {'stage':>6}")
    print("  " + "-" * 58)

    for i in range(1, STEPS + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        shape_ok = obs.shape == (4, 84, 84)
        dtype_ok = obs.dtype == np.float32
        range_ok = float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0
        reward_ok = isinstance(reward, (int, float))
        term_ok = isinstance(terminated, (bool, np.bool_))
        trunc_ok = isinstance(truncated, (bool, np.bool_))

        step_ok = shape_ok and dtype_ok and range_ok and reward_ok and term_ok and trunc_ok
        tag = PASS if step_ok else FAIL

        print(
            f"  [{tag}] {i:>4}  {action:>6}  {str(obs.shape):>12}"
            f"  {reward:>8.2f}  {str(terminated):>6}  {str(truncated):>6}"
            f"  {info.get('stage', '?'):>6}"
        )

        if not shape_ok:
            failures.append(f"step {i}: obs shape {obs.shape}")
        if not dtype_ok:
            failures.append(f"step {i}: obs dtype {obs.dtype}")
        if not range_ok:
            failures.append(f"step {i}: obs range [{obs.min():.4f}, {obs.max():.4f}]")
        if not reward_ok:
            failures.append(f"step {i}: reward type {type(reward)}")
        if not term_ok:
            failures.append(f"step {i}: terminated type {type(terminated)}")
        if not trunc_ok:
            failures.append(f"step {i}: truncated type {type(truncated)}")

        if terminated or truncated:
            print(f"         Episode ended (terminated={terminated}, truncated={truncated})")
            break

    env.close()
    print()

    # --- summary ---
    print("=" * 60)
    if failures:
        print(f"FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("PASS — all checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
