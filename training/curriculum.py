"""
training/curriculum.py — Curriculum learning tracker for the Roblox obby agent.

Tracks death clusters and computes an informational curriculum start stage.
Cannot actually teleport the agent (pixel-only constraint), so get_start_stage()
is used for logging and monitoring only.

See training/CLAUDE.md for the full curriculum specification.
"""

from __future__ import annotations

import collections
import logging
from statistics import mode

log = logging.getLogger(__name__)


class CurriculumTracker:
    """Tracks death clusters and computes a recommended start stage."""

    def __init__(
        self, death_window: int = 10, reward_window: int = 3
    ) -> None:
        self._death_history: collections.deque[int] = collections.deque(
            maxlen=death_window
        )
        self._reward_history: collections.deque[float] = collections.deque(
            maxlen=reward_window
        )
        self._max_stage_history: collections.deque[int] = collections.deque(
            maxlen=death_window
        )
        self._curriculum_start: int = 1
        self._prev_max_stage_peak: int = 0

    def on_episode_end(
        self,
        death_stage: int,
        max_stage: int,
        episode_reward: float,
    ) -> None:
        self._death_history.append(death_stage)
        self._max_stage_history.append(max_stage)
        self._reward_history.append(episode_reward)

        # Compute death cluster (mode, tie-break by lowest stage)
        if self._death_history:
            self._curriculum_start = max(1, self._death_cluster() - 1)

        # Advance: if max stage across window improved, move start back by 1
        current_peak = max(self._max_stage_history)
        if current_peak > self._prev_max_stage_peak:
            self._curriculum_start = max(1, self._curriculum_start - 1)
            self._prev_max_stage_peak = current_peak

        # Regression reset: 3 consecutive reward drops → reset to stage 1
        if len(self._reward_history) == self._reward_history.maxlen:
            rewards = list(self._reward_history)
            if all(
                rewards[i] < rewards[i - 1]
                for i in range(1, len(rewards))
            ):
                log.info(
                    "Reward regression detected (last 3: %s), "
                    "resetting curriculum to stage 1",
                    rewards,
                )
                self._curriculum_start = 1

    def get_start_stage(self) -> int:
        return self._curriculum_start

    def _death_cluster(self) -> int:
        """Mode of death history, tie-break by lowest stage."""
        counts: dict[int, int] = {}
        for stage in self._death_history:
            counts[stage] = counts.get(stage, 0) + 1
        max_count = max(counts.values())
        # Among stages with max count, pick the lowest
        return min(s for s, c in counts.items() if c == max_count)
