"""
training/reward.py — Reward function for the Roblox obby RL agent.

Punishment-only reward: step penalty discourages stalling, death penalty
discourages dying. No positive reward signal needed.
See training/CLAUDE.md for spec.
"""

from __future__ import annotations


class RewardCalculator:
    """Stateless reward calculator. Call compute() each step."""

    def compute(
        self,
        progress: float | None,
        death_event: bool,
        stuck: bool,
    ) -> float:
        reward = 0.0

        if death_event:
            reward -= 1.0

        reward -= 0.001  # step penalty

        if stuck:
            reward -= 0.5

        return reward
