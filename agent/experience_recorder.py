"""
agent/experience_recorder.py — Records successful agent runs as new demo data.

Buffers (frame, action) pairs during TwoTierAgent runs and saves
stage-advancing sequences as new demo runs in BCDataset-compatible format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from vision.preprocess import preprocess_frame

log = logging.getLogger(__name__)


class ExperienceRecorder:
    """
    Records agent experience and flushes successful runs as new demo data.

    Only flushes when the agent achieves >= min_stage_advances distinct
    stage advances in a single run, to prevent low-quality data from
    polluting the training set.

    Args:
        demos_dir: Path to demos/ directory.
        min_stage_advances: Minimum distinct stage advances required to flush.
        buffer_max_frames: Maximum frames to buffer before dropping oldest.
        death_exclusion_frames: Frames to roll back on death (likely mistakes).
    """

    def __init__(
        self,
        demos_dir: str | Path = "demos",
        min_stage_advances: int = 2,
        buffer_max_frames: int = 4000,
        death_exclusion_frames: int = 10,
    ) -> None:
        self._demos_dir = Path(demos_dir)
        self._min_stage_advances = min_stage_advances
        self._buffer_max_frames = buffer_max_frames
        self._death_exclusion_frames = death_exclusion_frames

        # Buffer: list of (processed_frame, action, stage) tuples
        self._buffer: list[tuple[np.ndarray, int, int]] = []
        self._last_advance_idx: int = 0
        self._stage_advance_count: int = 0
        self._last_stage: int | None = None

    def record(self, frame_bgr: np.ndarray, action: int, stage: int) -> None:
        """Record a single (frame, action) pair. Preprocesses frame immediately."""
        processed = preprocess_frame(frame_bgr)

        if len(self._buffer) >= self._buffer_max_frames:
            # Drop oldest frames but preserve advance cursor validity
            drop = len(self._buffer) - self._buffer_max_frames + 1
            self._buffer = self._buffer[drop:]
            self._last_advance_idx = max(0, self._last_advance_idx - drop)

        self._buffer.append((processed, action, stage))

        if self._last_stage is None:
            self._last_stage = stage

    def on_stage_advance(self, new_stage: int) -> None:
        """Called when the agent advances to a new stage."""
        self._last_advance_idx = len(self._buffer)
        self._stage_advance_count += 1
        self._last_stage = new_stage
        log.info(
            "Stage advance #%d to stage %d (buffer=%d frames)",
            self._stage_advance_count, new_stage, len(self._buffer),
        )

    def on_death(self) -> None:
        """Roll back the advance cursor to exclude pre-death mistake frames."""
        rollback = min(self._death_exclusion_frames, self._last_advance_idx)
        self._last_advance_idx = max(0, self._last_advance_idx - rollback)
        log.debug("Death rollback: advance cursor -> %d", self._last_advance_idx)

    def flush_run(self) -> Path | None:
        """
        Flush buffered data as a new demo run if quality threshold is met.

        Returns:
            Path to the new run directory, or None if quality gate not met.
        """
        if self._stage_advance_count < self._min_stage_advances:
            log.info(
                "Flush skipped: only %d stage advances (need %d)",
                self._stage_advance_count, self._min_stage_advances,
            )
            self._reset_buffer()
            return None

        # Only save frames up to the last advance cursor
        good_data = self._buffer[: self._last_advance_idx]
        if len(good_data) == 0:
            log.info("Flush skipped: no good frames after cursor trim")
            self._reset_buffer()
            return None

        # Find next run number
        run_dir = self._next_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Unpack and save in BCDataset-compatible format
        frames = np.stack([t[0] for t in good_data], axis=0)  # (N, 84, 84) float32
        actions = np.array([t[1] for t in good_data], dtype=np.int64)  # (N,)

        np.savez(run_dir / "frames.npz", frames=frames)
        np.save(run_dir / "actions.npy", actions)

        meta = {
            "source": "self_play",
            "num_frames": len(good_data),
            "stage_advances": self._stage_advance_count,
            "final_stage": good_data[-1][2],
        }
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        log.info(
            "Flushed self-demo: %s (%d frames, %d stage advances)",
            run_dir.name, len(good_data), self._stage_advance_count,
        )

        self._reset_buffer()
        return run_dir

    def _next_run_dir(self) -> Path:
        """Find the next available run_NNN_self/ directory."""
        existing = sorted(self._demos_dir.glob("run_*"))
        max_num = 0
        for d in existing:
            # Parse run_NNN or run_NNN_self
            parts = d.name.split("_")
            if len(parts) >= 2:
                try:
                    max_num = max(max_num, int(parts[1]))
                except ValueError:
                    pass
        return self._demos_dir / f"run_{max_num + 1:03d}_self"

    def _reset_buffer(self) -> None:
        """Clear the buffer and counters for the next run."""
        self._buffer.clear()
        self._last_advance_idx = 0
        self._stage_advance_count = 0
        self._last_stage = None
