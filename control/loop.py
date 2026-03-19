"""
control/loop.py — Main observation-action loop.

Ties together capture (screenshot → OCR) and control (action → input).
This is the Phase 2 main loop. In Phase 3 the agent module will supply
the action selection logic; for now, actions are random for validation.

Usage:
    python -m control.loop          # run with random actions for validation
    python -m control.loop --dry    # dry run: capture only, no inputs sent
"""

from __future__ import annotations

import atexit
import logging
import random
import signal
import time

from capture.screen import Capturer
from control.actions import (
    ACTION_NAMES,
    NUM_ACTIONS,
    execute_action,
    release_all,
    set_camera_angle,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frame rate limiter
# ---------------------------------------------------------------------------

TARGET_FPS = 5
FRAME_DURATION = 1.0 / TARGET_FPS


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(
    action_fn=None,
    dry_run: bool = False,
    max_frames: int = 0,
    capturer: Capturer | None = None,
) -> None:
    """
    Main observation-action loop.

    Args:
        action_fn: Callable(stage, death) -> int.  If None, random actions.
        dry_run:   If True, capture frames but send no inputs.
        max_frames: Stop after this many frames (0 = run forever).
        capturer:  Optional pre-built Capturer. If None, one is created.
    """
    # Safety: release everything on exit no matter what
    atexit.register(release_all)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if capturer is None:
        capturer = Capturer()

    if not dry_run:
        log.info("Setting camera angle...")
        time.sleep(2)  # give user time to focus the Roblox window
        set_camera_angle()
        time.sleep(0.5)

    log.info(
        "Loop starting — fps=%d dry_run=%s max_frames=%s",
        TARGET_FPS,
        dry_run,
        max_frames or "unlimited",
    )

    frame_count = 0
    loop_start = time.time()

    try:
        while True:
            step_start = time.time()

            # 1. Observe
            capturer.tick()
            stage = capturer.current_stage
            died = capturer.death_event

            # 2. Decide
            if action_fn is not None:
                action = action_fn(stage, died)
            else:
                action = random.randint(0, NUM_ACTIONS - 1)

            # 3. Act
            if died:
                log.info("Death detected — releasing all inputs")
                release_all()
            elif not dry_run:
                execute_action(action)

            frame_count += 1

            # Periodic status log
            if frame_count % (TARGET_FPS * 10) == 0:  # every ~10 seconds
                elapsed_total = time.time() - loop_start
                log.info(
                    "frame=%d  stage=%d  action=%s  uptime=%.0fs",
                    frame_count,
                    stage,
                    ACTION_NAMES[action],
                    elapsed_total,
                )

            if 0 < max_frames <= frame_count:
                log.info("Reached max_frames=%d — stopping", max_frames)
                break

            # 4. Frame rate limiter
            elapsed = time.time() - step_start
            if elapsed < FRAME_DURATION:
                time.sleep(FRAME_DURATION - elapsed)

    finally:
        release_all()
        elapsed_total = time.time() - loop_start
        log.info("Loop ended — %d frames in %.1fs", frame_count, elapsed_total)


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _signal_handler(signum, frame):
    """Ensure inputs are released on interrupt/termination."""
    release_all()
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Phase 2 validation loop")
    parser.add_argument("--dry", action="store_true", help="Capture only, no inputs")
    parser.add_argument("--frames", type=int, default=0, help="Max frames (0=forever)")
    args = parser.parse_args()

    run_loop(dry_run=args.dry, max_frames=args.frames)
