#!/usr/bin/env python3
"""
Run all tests in a loop indefinitely. Every LIVE_EVERY_N cycles, run a short live test
(capture + CEM + one action) against the open Roblox obby. Popup dismissal is handled in takeover.
Press Ctrl+C to stop.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

LIVE_EVERY_N = 3  # run live test every N full test cycles
LIVE_MAX_DECISIONS = 2  # for takeover live run: stop after 2 decisions


def run_all_tests():
    from run_all_tests import TEST_FILES
    python = sys.executable
    failed = []
    for path in TEST_FILES:
        name = os.path.basename(path)
        r = __import__("subprocess").run([python, path], cwd=ROOT, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            failed.append(name)
            if r.stderr:
                print(r.stderr, file=sys.stderr)
    return failed


def run_live_cycle():
    """One quick live test: run_live_test --full-screen (capture + CEM + one action)."""
    import subprocess
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "run_live_test.py"), "--full-screen"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r.returncode == 0


def main():
    cycle = 0
    print("Running tests indefinitely. Every", LIVE_EVERY_N, "cycles run live obby test. Ctrl+C to stop.\n")
    try:
        while True:
            cycle += 1
            print(f"\n========== Cycle {cycle} ==========")
            failed = run_all_tests()
            if failed:
                print(f"FAILED: {failed}", file=sys.stderr)
            else:
                print("All tests passed.")
            if cycle % LIVE_EVERY_N == 0:
                print("\n--- Live obby test ---")
                if run_live_cycle():
                    print("Live test OK.")
                else:
                    print("Live test had non-zero exit (check Roblox / API).")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
