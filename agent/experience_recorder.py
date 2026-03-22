"""
agent/experience_recorder.py — Records successful agent runs as new demo data.

Buffers (frame, action) pairs during TwoTierAgent runs and saves
long-survival sequences as new demo runs in BCDataset-compatible format.
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

    Segments runs into "lives" (frames between deaths). Only saves lives
    where the agent survived at least min_survival_frames, trimming
    pre-death mistake frames from the end of each life.

    Args:
        demos_dir: Path to demos/ directory.
        min_survival_frames: Minimum frames in a life to consider it good data.
        buffer_max_frames: Maximum frames to buffer before dropping oldest.
        death_exclusion_frames: Frames to trim from end of each life (pre-death mistakes).
    """

    def __init__(
        self,
        demos_dir: str | Path = "demos",
        min_survival_frames: int = 600,
        buffer_max_frames: int = 4000,
        death_exclusion_frames: int = 10,
    ) -> None:
        self._demos_dir = Path(demos_dir)
        self._min_survival_frames = min_survival_frames
        self._buffer_max_frames = buffer_max_frames
        self._death_exclusion_frames = death_exclusion_frames

        # Buffer: list of (processed_frame, action, stage) tuples
        self._buffer: list[tuple[np.ndarray, int, int]] = []
        self._current_life_start: int = 0
        self._good_segments: list[tuple[int, int]] = []

    def record(self, frame_bgr: np.ndarray, action: int, stage: int) -> None:
        """Record a single (frame, action) pair. Preprocesses frame immediately."""
        processed = preprocess_frame(frame_bgr)

        if len(self._buffer) >= self._buffer_max_frames:
            drop = len(self._buffer) - self._buffer_max_frames + 1
            self._buffer = self._buffer[drop:]
            # Adjust all cursors
            self._current_life_start = max(0, self._current_life_start - drop)
            self._good_segments = [
                (max(0, s - drop), max(0, e - drop))
                for s, e in self._good_segments
                if e - drop > 0
            ]

        self._buffer.append((processed, action, stage))

    def on_death(self) -> None:
        """End the current life and record it if it was long enough."""
        life_len = len(self._buffer) - self._current_life_start
        end = len(self._buffer) - self._death_exclusion_frames
        if life_len >= self._min_survival_frames and end > self._current_life_start:
            self._good_segments.append((self._current_life_start, end))
            log.info(
                "Good life recorded: %d frames (trimmed to %d)",
                life_len, end - self._current_life_start,
            )
        else:
            log.debug(
                "Short life discarded: %d frames (need %d)",
                life_len, self._min_survival_frames,
            )
        self._current_life_start = len(self._buffer)

    def flush_run(self) -> Path | None:
        """
        Flush buffered data as a new demo run if any good segments exist.

        Returns:
            Path to the new run directory, or None if no good segments.
        """
        # Include the final life (from last death to end of buffer) if it qualifies
        final_life_len = len(self._buffer) - self._current_life_start
        if final_life_len >= self._min_survival_frames:
            end = len(self._buffer) - self._death_exclusion_frames
            if end > self._current_life_start:
                self._good_segments.append((self._current_life_start, end))
                log.info("Final life qualified: %d frames", end - self._current_life_start)

        if not self._good_segments:
            log.info(
                "Flush skipped: no lives exceeded %d frames",
                self._min_survival_frames,
            )
            self._reset_buffer()
            return None

        # Concatenate all good segments
        good_data: list[tuple[np.ndarray, int, int]] = []
        for start, end in self._good_segments:
            good_data.extend(self._buffer[start:end])

        if len(good_data) == 0:
            log.info("Flush skipped: no frames after segment assembly")
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
            "quality_gate": "survival_time",
            "num_frames": len(good_data),
            "num_segments": len(self._good_segments),
            "min_survival_frames": self._min_survival_frames,
            "final_stage": good_data[-1][2],
        }
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        log.info(
            "Flushed self-demo: %s (%d frames from %d good lives)",
            run_dir.name, len(good_data), len(self._good_segments),
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
        self._current_life_start = 0
        self._good_segments.clear()
