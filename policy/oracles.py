"""
Oracles: given a screenshot (and optional context), return the "ideal" action index for training.
Multiple implementations so we can compare and monitor what works best.
"""

import os
import random
from typing import Callable, Optional

import numpy as np

# 8 actions for policy (no duplicates); index -> name
ACTION_NAMES = ["W", "A", "S", "D", "space", "none", "look_left", "look_right"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_NAMES)}


def _frame_resized(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image
    pil = Image.fromarray(frame.astype(np.uint8))
    pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    return np.array(pil)


def oracle_avoid_only(
    frame: np.ndarray,
    use_vision: bool = True,
    **_
) -> int:
    """
    If avoid (danger/death) detected -> none; else W.
    No API calls if use_vision=False and caption provided; else uses BLIP.
    """
    from reward.avoids import get_avoid_penalty
    penalty = get_avoid_penalty(frame, caption=None, use_vision=use_vision)
    if penalty >= 0.5:
        return ACTION_TO_IDX["none"]
    return ACTION_TO_IDX["W"]


def oracle_random(**_) -> int:
    """Random action (baseline)."""
    return random.randint(0, len(ACTION_NAMES) - 1)


def oracle_forward(**_) -> int:
    """Always W (forward). Baseline to see if learning beats constant forward."""
    return ACTION_TO_IDX["W"]


def oracle_scout(
    frame: np.ndarray,
    api_key: Optional[str] = None,
    reward_model=None,
    device=None,
    **_
) -> int:
    """
    Use CEM/Scout to pick best action; return its index in ACTION_NAMES.
    Requires GROQ_API_KEY if api_key is None.
    """
    from llm_agent.cem import run_cem
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return oracle_avoid_only(frame)
    default_actions = ["W", "A", "S", "D", "space", "none", "look_left", "look_right"]
    best_action, _, _, _, _ = run_cem(
        frame,
        reward_model=reward_model,
        device=device,
        scout_api_key=api_key,
        use_scout=True,
        avoid_weight=1.0,
    )
    return ACTION_TO_IDX.get(best_action, ACTION_TO_IDX["W"])


def oracle_cem_no_scout(
    frame: np.ndarray,
    reward_model=None,
    device=None,
    **_
) -> int:
    """
    CEM without Scout: equal scores minus avoid penalty. So avoid -> none, else first action (W).
    Same as avoid_only when we don't have per-action scores.
    """
    from llm_agent.cem import run_cem
    best_action, _, _, _, _ = run_cem(
        frame,
        reward_model=reward_model,
        device=device,
        scout_api_key=None,
        use_scout=False,
        avoid_weight=1.0,
    )
    return ACTION_TO_IDX.get(best_action, ACTION_TO_IDX["W"])


def get_oracle(
    name: str,
    api_key: Optional[str] = None,
    reward_model=None,
    device=None,
) -> Callable[..., int]:
    """
    Return an oracle callable that takes (frame, **kwargs) and returns action index in [0, 7].
    name: "scout" | "avoid_only" | "cem_no_scout" | "random" | "forward"
    """
    def fn(frame: np.ndarray, **kwargs) -> int:
        if name == "scout":
            return oracle_scout(frame, api_key=api_key, reward_model=reward_model, device=device, **kwargs)
        if name == "avoid_only":
            return oracle_avoid_only(frame, **kwargs)
        if name == "cem_no_scout":
            return oracle_cem_no_scout(frame, reward_model=reward_model, device=device, **kwargs)
        if name == "random":
            return oracle_random(**kwargs)
        if name == "forward":
            return oracle_forward(**kwargs)
        return oracle_avoid_only(frame, **kwargs)
    return fn
