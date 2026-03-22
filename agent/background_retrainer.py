"""
agent/background_retrainer.py — Daemon thread that retrains BC when enough self-demos accumulate.

Fires scripts/train_bc.py as a subprocess when enough new self-demo runs
have been collected, then writes a sentinel flag for hot-swap.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

log = logging.getLogger(__name__)


class BackgroundRetrainer:
    """
    Background daemon that retrains BC model when enough self-demo runs accumulate.

    Args:
        demos_dir: Path to demos/ directory.
        checkpoint_dir: Path to checkpoints/ directory.
        min_new_runs: Minimum new self-demo runs before triggering retrain.
        retrain_interval_minutes: Minimum minutes between retrains.
    """

    def __init__(
        self,
        demos_dir: str | Path = "demos",
        checkpoint_dir: str | Path = "checkpoints",
        min_new_runs: int = 1,
        retrain_interval_minutes: float = 5.0,
    ) -> None:
        self._demos_dir = Path(demos_dir)
        self._checkpoint_dir = Path(checkpoint_dir)
        self._min_new_runs = min_new_runs
        self._retrain_interval = retrain_interval_minutes * 60  # seconds

        self._pending_runs: int = 0
        self._lock = threading.Lock()
        self._training_active: bool = False
        self._last_retrain_time: float = 0.0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Track previous val_accuracy for regression guard
        self._prev_val_accuracy: float | None = self._read_current_val_accuracy()

    def notify_new_run(self, run_path: Path) -> None:
        """Increment pending runs counter when a new self-demo is saved."""
        with self._lock:
            self._pending_runs += 1
            log.info(
                "Retrainer notified: %s (pending=%d/%d)",
                run_path.name, self._pending_runs, self._min_new_runs,
            )

    def start(self) -> None:
        """Start the background retrainer daemon thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._daemon_loop, daemon=True)
        self._thread.start()
        log.info("Background retrainer started (min_runs=%d, interval=%.0fm)",
                 self._min_new_runs, self._retrain_interval / 60)

    def stop(self) -> None:
        """Signal the daemon to stop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            log.info("Background retrainer stopped")

    def _daemon_loop(self) -> None:
        """Check every 60s whether to kick off a retrain."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=30.0)
            if self._stop_event.is_set():
                break

            with self._lock:
                ready = (
                    self._pending_runs >= self._min_new_runs
                    and not self._training_active
                    and (time.monotonic() - self._last_retrain_time) >= self._retrain_interval
                )
                if ready:
                    self._training_active = True
                    self._pending_runs = 0

            if ready:
                self._run_retrain()

    def _run_retrain(self) -> None:
        """Run train_bc.py as a subprocess."""
        train_script = Path(__file__).resolve().parent.parent / "scripts" / "train_bc.py"
        log.info("Starting BC retrain: %s", train_script)

        try:
            result = subprocess.run(
                [sys.executable, str(train_script)],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode == 0:
                log.info("BC retrain completed successfully")
                self._check_regression_and_signal()
            else:
                log.error("BC retrain failed (rc=%d): %s", result.returncode, result.stderr[-500:])

        except subprocess.TimeoutExpired:
            log.error("BC retrain timed out after 600s")
        except Exception:
            log.exception("BC retrain error")
        finally:
            with self._lock:
                self._training_active = False
                self._last_retrain_time = time.monotonic()

    def _check_regression_and_signal(self) -> None:
        """Check if new model regressed, and write hot-swap flag if not."""
        new_val_acc = self._read_current_val_accuracy()

        if new_val_acc is not None and self._prev_val_accuracy is not None:
            if new_val_acc < self._prev_val_accuracy:
                log.warning(
                    "Regression detected: new val_accuracy=%.3f < previous=%.3f — skipping hot-swap",
                    new_val_acc, self._prev_val_accuracy,
                )
                return

        # Write hot-swap sentinel
        flag_path = self._checkpoint_dir / "bc_pending.flag"
        checkpoint_path = str(self._checkpoint_dir / "bc_best.pt")
        flag_path.write_text(checkpoint_path)
        log.info("Hot-swap flag written: %s", flag_path)

        if new_val_acc is not None:
            self._prev_val_accuracy = new_val_acc

    def _read_current_val_accuracy(self) -> float | None:
        """Read val_accuracy from the current bc_best.pt checkpoint."""
        cp_path = self._checkpoint_dir / "bc_best.pt"
        if not cp_path.exists():
            return None
        try:
            state = torch.load(cp_path, map_location="cpu", weights_only=False)
            return state.get("val_accuracy")
        except Exception:
            return None
