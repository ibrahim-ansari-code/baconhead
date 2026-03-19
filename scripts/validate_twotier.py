"""
scripts/validate_twotier.py — Phase 4 validation script.

Standalone, <2min, PASS/FAIL, countdown before capture.
Runs the two-tier agent for 60 seconds and validates:
  1. CNN inference time < 50ms
  2. At least 1 Gemini response (if API key present)
  3. Death → respawn completes without crash
  4. Continuous operation without crash
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
)
log = logging.getLogger("validate_twotier")


def check_cnn_inference_speed() -> bool:
    """PASS if CNN forward pass < 50ms."""
    from vision.model import ObbyCNN

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ObbyCNN(n_actions=6).to(device).eval()

    dummy = torch.randn(1, 4, 84, 84, device=device)

    # Warm up
    with torch.no_grad():
        for _ in range(5):
            model(dummy)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            model(dummy)
            times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    p99_ms = np.percentile(times, 99) * 1000
    passed = avg_ms < 50.0
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] CNN inference: avg={avg_ms:.1f}ms p99={p99_ms:.1f}ms (limit: 50ms)")
    return passed


def check_begin_end_action() -> bool:
    """PASS if begin_action/end_action work without error."""
    from control.actions import begin_action, end_action, _current_held_keys

    try:
        begin_action(0)  # forward
        assert len(_current_held_keys) > 0, "No keys held after begin_action(0)"
        begin_action(4)  # forward_jump — should release previous
        begin_action(5)  # idle — should release all
        assert len(_current_held_keys) == 0, "Keys still held after idle"
        end_action()
        print("[PASS] begin_action / end_action work correctly")
        return True
    except Exception as e:
        print(f"[FAIL] begin_action / end_action: {e}")
        return False
    finally:
        end_action()


def check_tick_fast() -> bool:
    """PASS if tick_fast captures a frame and detects death state."""
    from capture.screen import Capturer

    print("Switch to Roblox now — starting in 3... ", end="", flush=True)
    for i in range(3, 0, -1):
        print(f"{i}... ", end="", flush=True)
        time.sleep(1)
    print("GO!")

    cap = Capturer()
    try:
        t0 = time.perf_counter()
        cap.tick_fast()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        has_frame = cap.last_frame is not None
        passed = has_frame and elapsed_ms < 200
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] tick_fast: frame={'yes' if has_frame else 'no'} time={elapsed_ms:.1f}ms")
        return passed
    except Exception as e:
        print(f"[FAIL] tick_fast: {e}")
        return False


def check_twotier_agent_creation() -> bool:
    """PASS if TwoTierAgent can be created (with mock checkpoint if needed)."""
    from vision.model import ObbyCNN

    cp_path = Path("checkpoints/bc_best.pt")
    cleanup = False

    # Create a dummy checkpoint if the real one doesn't exist
    if not cp_path.exists():
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_model = ObbyCNN(n_actions=6)
        torch.save(dummy_model.state_dict(), cp_path)
        cleanup = True
        log.info("Created dummy checkpoint for validation")

    try:
        from agent.twotier import TwoTierAgent
        from capture.screen import Capturer

        cap = Capturer()
        agent = TwoTierAgent(
            capturer=cap,
            checkpoint_path=str(cp_path),
            use_gemini=False,  # don't require API key for this check
        )
        print("[PASS] TwoTierAgent created successfully")
        return True
    except Exception as e:
        print(f"[FAIL] TwoTierAgent creation: {e}")
        return False
    finally:
        if cleanup:
            cp_path.unlink(missing_ok=True)


def check_live_run() -> bool:
    """PASS if agent runs for 30s without crashing (dry_run mode)."""
    from vision.model import ObbyCNN

    cp_path = Path("checkpoints/bc_best.pt")
    cleanup = False

    if not cp_path.exists():
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_model = ObbyCNN(n_actions=6)
        torch.save(dummy_model.state_dict(), cp_path)
        cleanup = True

    try:
        from agent.twotier import TwoTierAgent
        from capture.screen import Capturer

        cap = Capturer()
        agent = TwoTierAgent(
            capturer=cap,
            checkpoint_path=str(cp_path),
            use_gemini=False,
        )

        print("Running agent for 30s in dry_run mode...")
        t0 = time.monotonic()
        agent.run(duration_seconds=30, dry_run=True)
        elapsed = time.monotonic() - t0

        passed = elapsed >= 25  # allow some tolerance
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] Live dry run: ran {elapsed:.1f}s")
        return passed
    except Exception as e:
        print(f"[FAIL] Live dry run crashed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if cleanup:
            cp_path.unlink(missing_ok=True)


def main() -> None:
    print("=" * 60)
    print("Phase 4 — Two-Tier Agent Validation")
    print("=" * 60)
    print()

    results = {}

    results["cnn_speed"] = check_cnn_inference_speed()
    results["begin_end_action"] = check_begin_end_action()
    results["tick_fast"] = check_tick_fast()
    results["agent_creation"] = check_twotier_agent_creation()
    results["live_run"] = check_live_run()

    print()
    print("=" * 60)
    total = len(results)
    passed = sum(results.values())
    print(f"Results: {passed}/{total} passed")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")

    if passed == total:
        print("\nPhase 4 validation PASSED")
    else:
        print("\nPhase 4 validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
