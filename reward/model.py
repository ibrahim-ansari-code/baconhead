"""
Reward model r(s): takes a state (frame) and outputs a scalar reward.
Trained so that states where the user was actively playing get higher reward than idle states.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import numpy as np


class RewardNet(nn.Module):
    """Small CNN that maps a single frame (C, H, W) to a scalar reward."""

    def __init__(
        self,
        in_channels: int = 3,
        height: int = 84,
        width: int = 84,
        hidden: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        # Simple conv stack then fc
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        # Compute feature size by running a dummy
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            flat_size = self.conv(dummy).view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(flat_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h).squeeze(-1)  # (B,)


def load_reward_model(path: str, device: Optional[torch.device] = None) -> RewardNet:
    """Load a trained RewardNet from a checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(path, map_location=device, weights_only=True)
    if isinstance(data, dict) and "model_state" in data:
        state = data["model_state"]
        cfg = data.get("config", {})
    else:
        state = data
        cfg = {}
    model = RewardNet(
        in_channels=cfg.get("in_channels", 3),
        height=cfg.get("height", 84),
        width=cfg.get("width", 84),
        hidden=cfg.get("hidden", 256),
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def save_reward_model(model: RewardNet, path: str, config: Optional[dict] = None):
    """Save RewardNet and optional config."""
    cfg = config or {}
    if not cfg:
        cfg = {
            "in_channels": model.in_channels,
            "height": model.height,
            "width": model.width,
            "hidden": model.fc[0].out_features,
        }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg}, path)
