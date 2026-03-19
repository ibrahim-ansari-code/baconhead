"""
training/reward.py — Reward function for the Roblox obby RL agent.

Computes per-step reward from stage progress, deaths, and stuck state.
See training/CLAUDE.md for the full reward specification.
"""

from __future__ import annotations


class RewardCalculator:
    """Stateless reward calculator. Call compute() each step."""

    def compute(
        self,
        prev_stage: int,
        curr_stage: int,
        death_event: bool,
        stuck: bool,
    ) -> float:
        reward = 0.0

        # Checkpoint reward: only on single-step stage increase
        if curr_stage == prev_stage + 1:
            if curr_stage <= 25:
                reward += 1.0
            elif curr_stage <= 50:
                reward += 1.5
            elif curr_stage <= 100:
                reward += 2.0
            else:
                reward += 2.5

            # Milestone bonus: every 25 stages
            if curr_stage % 25 == 0:
                reward += 3.0

        # Death penalty (once per death, not per frame)
        if death_event:
            reward -= 1.0

        # Step penalty (every frame)
        reward -= 0.001

        # Stuck penalty
        if stuck:
            reward -= 0.5

        return reward
