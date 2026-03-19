# agent/CLAUDE.md — Gymnasium environment wrapper

## Responsibility
This module contains the gymnasium environment wrapper that the RL policy plugs into. It wraps the full observation loop as a `gymnasium.Env` for Stable Baselines3.

---

## Inputs (from other modules)
- `current_stage: int` — from capture module
- `death_event: bool` — from capture module
- `stacked_frames: np.ndarray` shape (4, 84, 84) — from vision module

---

## Gymnasium environment wrapper (phase 4)

```python
import gymnasium as gym
import numpy as np

class ObbyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4, 84, 84), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(6)  # see control/CLAUDE.md

    def reset(self, seed=None):
        # wait for Roblox respawn, return initial stacked frames
        ...

    def step(self, action):
        # send action via control module
        # grab new frame, update stack
        # compute reward via training module
        # check termination conditions
        ...
        return observation, reward, terminated, truncated, info
```

### Termination conditions
- `terminated = True` when death_event is True
- `truncated = True` when stuck detection timeout fires (see training/CLAUDE.md)
- Episode ends on either condition

### Observation
Return the stacked frames tensor (4, 84, 84) as the observation.
