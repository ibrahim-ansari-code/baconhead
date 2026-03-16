#!/usr/bin/env python3
"""Run all offline test modules. Exit 0 if all pass, 1 otherwise."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# Run each test file as script so __main__ runs
TEST_FILES = [
    os.path.join(ROOT, "tests", "test_cem_offline.py"),
    os.path.join(ROOT, "tests", "test_reward_offline.py"),
    os.path.join(ROOT, "tests", "test_capture_config.py"),
    os.path.join(ROOT, "tests", "test_avoids_offline.py"),
    os.path.join(ROOT, "tests", "test_scout_offline.py"),
    os.path.join(ROOT, "tests", "test_input_state_offline.py"),
    os.path.join(ROOT, "tests", "test_cem_extended.py"),
    os.path.join(ROOT, "tests", "test_capture_extended.py"),
    os.path.join(ROOT, "tests", "test_reward_extended.py"),
    os.path.join(ROOT, "tests", "test_takeover_config.py"),
    os.path.join(ROOT, "tests", "test_misc.py"),
]


def main():
    python = sys.executable
    failed = []
    for path in TEST_FILES:
        name = os.path.basename(path)
        print(f"\n--- {name} ---", flush=True)
        r = subprocess.run([python, path], cwd=ROOT)
        if r.returncode != 0:
            failed.append(name)
    if failed:
        print("\nFailed:", failed, file=sys.stderr)
        sys.exit(1)
    print("\nAll tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
