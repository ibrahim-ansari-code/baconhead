"""
agent/env.py — Gymnasium environment wrapper for the Roblox obby agent.

Wraps screen capture, vision preprocessing, frame stacking, and control
into a standard gymnasium.Env so Stable Baselines3 can train on it.
"""

from __future__ import annotations

import logging
import time

import gymnasium as gym
import numpy as np

from capture.screen import Capturer
from control.actions import execute_action, release_all, set_camera_angle, NUM_ACTIONS
from training.reward import RewardCalculator
from vision.preprocess import preprocess_frame
from vision.stacker import FrameStacker

log = logging.getLogger(__name__)

# Seconds to wait for Roblox respawn animation to finish
RESPAWN_WAIT = 3.0

# Seconds without stage progress before the agent is considered stuck
STUCK_TIMEOUT = 30.0


class ObbyEnv(gym.Env):
    """
    Gymnasium environment for Roblox obby RL training.

    Observation: (4, 84, 84) float32 stacked grayscale frames.
    Actions:     Discrete(6) — see control/CLAUDE.md.
    """

    metadata = {"render_modes": []}

    def __init__(self, stuck_timeout: float = STUCK_TIMEOUT) -> None:
        super().__init__()

        # SB3 CnnPolicy expects channel-last uint8 [0,255] images.
        # We return (84, 84, 4) uint8; SB3 normalizes and transposes internally.
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 4), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        self._capturer = Capturer()
        self._stacker = FrameStacker(stack_size=4)
        self._reward_calc = RewardCalculator()
        self._stuck_timeout = stuck_timeout

        # Episode state (reset in reset())
        self._prev_stage: int = 1
        self._last_stage_change_time: float = 0.0
        self._episode_deaths: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.

        Releases all held keys, waits for Roblox respawn, sets the camera,
        captures the first frame, and fills the frame stack.
        """
        super().reset(seed=seed)

        release_all()

        # Wait for Roblox respawn animation to complete
        log.info("Waiting %.1fs for respawn...", RESPAWN_WAIT)
        time.sleep(RESPAWN_WAIT)

        # Set camera to standard angle
        set_camera_angle()

        # Capture initial frame and fill the stack
        self._capturer.tick()
        frame = self._capturer.last_frame
        processed = preprocess_frame(frame)
        stacked = self._stacker.reset(processed)
        # (4,84,84) float [0,1] → (84,84,4) uint8 [0,255] for SB3
        observation = (np.transpose(stacked, (1, 2, 0)) * 255).astype(np.uint8)

        # Initialize episode state
        self._prev_stage = self._capturer.current_stage
        self._last_stage_change_time = time.monotonic()
        self._episode_deaths = 0

        info = {
            "stage": self._capturer.current_stage,
        }

        return observation, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one action and return the new observation.

        Args:
            action: Index into the discrete action space (0-5).

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Send action to Roblox
        execute_action(action)

        # Capture new frame
        self._capturer.tick()
        frame = self._capturer.last_frame
        processed = preprocess_frame(frame)
        stacked = self._stacker.push(processed)
        # (4,84,84) float [0,1] → (84,84,4) uint8 [0,255] for SB3
        observation = (np.transpose(stacked, (1, 2, 0)) * 255).astype(np.uint8)

        curr_stage = self._capturer.current_stage
        death = self._capturer.death_event

        # Track stage progress for stuck detection
        if curr_stage > self._prev_stage:
            self._last_stage_change_time = time.monotonic()

        # Stuck detection
        stuck = (
            time.monotonic() - self._last_stage_change_time
        ) > self._stuck_timeout

        # Reward
        reward = self._reward_calc.compute(
            self._prev_stage, curr_stage, death, stuck
        )

        # Termination conditions
        terminated = death
        truncated = stuck

        # Update episode state
        if death:
            self._episode_deaths += 1
        self._prev_stage = curr_stage

        info = {
            "stage": curr_stage,
            "death": death,
            "deaths": self._episode_deaths,
            "stuck": stuck,
        }

        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """Release all held inputs on environment shutdown."""
        release_all()
