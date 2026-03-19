"""
training/callbacks.py — SB3 callback for episode logging and curriculum updates.

Logs per-episode stats to CSV and updates the CurriculumTracker at episode
boundaries. Designed for use with stable_baselines3.PPO.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from training.curriculum import CurriculumTracker

log = logging.getLogger(__name__)

CSV_COLUMNS = [
    "episode",
    "total_reward",
    "max_stage",
    "num_deaths",
    "episode_length",
    "end_reason",
    "curriculum_start",
]


class EpisodeLogger(BaseCallback):
    """Logs episode stats and drives curriculum updates."""

    def __init__(
        self,
        curriculum: CurriculumTracker,
        log_path: str | Path = "logs/training.csv",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._curriculum = curriculum
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Per-episode accumulators
        self._episode_reward: float = 0.0
        self._episode_deaths: int = 0
        self._max_stage: int = 1
        self._episode_steps: int = 0
        self._episode_count: int = 0

        # Write CSV header
        self._csv_file = open(self._log_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(CSV_COLUMNS)
        self._csv_file.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        if not infos:
            return True

        info = infos[0]
        reward = float(rewards[0]) if len(rewards) > 0 else 0.0

        # Accumulate episode stats
        self._episode_reward += reward
        self._episode_steps += 1

        stage = info.get("stage", 1)
        if stage > self._max_stage:
            self._max_stage = stage

        if info.get("death", False):
            self._episode_deaths += 1

        # Episode boundary
        if dones[0]:
            self._episode_count += 1
            end_reason = "stuck" if info.get("stuck", False) else "death"

            # Update curriculum
            self._curriculum.on_episode_end(
                death_stage=self._max_stage,
                max_stage=self._max_stage,
                episode_reward=self._episode_reward,
            )

            curriculum_start = self._curriculum.get_start_stage()

            # Write CSV row
            self._csv_writer.writerow([
                self._episode_count,
                round(self._episode_reward, 4),
                self._max_stage,
                self._episode_deaths,
                self._episode_steps,
                end_reason,
                curriculum_start,
            ])
            self._csv_file.flush()

            log.info(
                "Episode %d: reward=%.2f max_stage=%d deaths=%d "
                "steps=%d end=%s curriculum=%d",
                self._episode_count,
                self._episode_reward,
                self._max_stage,
                self._episode_deaths,
                self._episode_steps,
                end_reason,
                curriculum_start,
            )

            # Reset accumulators for next episode
            self._episode_reward = 0.0
            self._episode_deaths = 0
            self._max_stage = 1
            self._episode_steps = 0

        return True

    def _on_training_end(self) -> None:
        self._csv_file.close()
