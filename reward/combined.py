"""
Combined reward: learned RewardNet r(s) minus preset avoid penalty.
r_total(s) = clamp(r_cnn(s) - avoid_penalty(s), 0, 1) or allow negative.
"""

from typing import Optional

import numpy as np
import torch

from reward.avoids import get_avoid_penalty


def frame_to_tensor(frame: np.ndarray, height: int = 84, width: int = 84) -> torch.Tensor:
    """Convert RGB frame (H,W,3) to (1,3,H,W) tensor in [0,1]."""
    from PIL import Image
    pil = Image.fromarray(frame.astype(np.uint8))
    pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    x = np.array(pil).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    return x


def combined_reward(
    frame: np.ndarray,
    reward_model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    caption: Optional[str] = None,
    avoid_weight: float = 1.0,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> float:
    """
    r_total = r_cnn(s) - avoid_weight * avoid_penalty(s), then clamp.
    If reward_model is None, only avoid penalty is used (0 or -avoid_weight then clamped).
    """
    r_cnn = 0.0
    if reward_model is not None and device is not None:
        with torch.no_grad():
            x = frame_to_tensor(frame).to(device)
            r_cnn = reward_model(x).item()
            r_cnn = max(0.0, min(1.0, r_cnn))
    penalty = get_avoid_penalty(frame, caption=caption, use_vision=(caption is None))
    r_total = r_cnn - avoid_weight * penalty
    return float(max(clamp_min, min(clamp_max, r_total)))
