"""
agent/logger.py — Per-frame session logger for the heuristic hybrid agent.

Writes one line per frame to logs/session.log with:
    timestamp, current_stage, death_event, action_taken, llm_decision, motion_mask_mean
"""

from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).parent.parent / "logs"
SESSION_LOG = LOGS_DIR / "session.log"

FIELDNAMES = [
    "timestamp",
    "current_stage",
    "death_event",
    "action_taken",
    "llm_decision",
    "motion_mask_mean",
]


class SessionLogger:
    """Append-only CSV logger that writes one row per frame."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or SESSION_LOG
        self._path.parent.mkdir(parents=True, exist_ok=True)

        write_header = not self._path.exists() or self._path.stat().st_size == 0
        self._file = open(self._path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

        log.info("Session logger writing to %s", self._path)

    def log_frame(
        self,
        current_stage: int,
        death_event: bool,
        action_taken: str,
        llm_decision: str | None,
        motion_mask_mean: float,
    ) -> None:
        self._writer.writerow(
            {
                "timestamp": f"{time.time():.3f}",
                "current_stage": current_stage,
                "death_event": death_event,
                "action_taken": action_taken,
                "llm_decision": llm_decision or "",
                "motion_mask_mean": f"{motion_mask_mean:.4f}",
            }
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()
