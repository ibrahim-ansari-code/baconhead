"""
vision/model.py — Nature DQN CNN architecture.

Matches the exact spec in vision/CLAUDE.md:
    Input:  (batch, 4, 84, 84)
    Conv1:  32 filters, 8×8 kernel, stride 4, ReLU  → (batch, 32, 20, 20)
    Conv2:  64 filters, 4×4 kernel, stride 2, ReLU  → (batch, 64,  9,  9)
    Conv3:  64 filters, 3×3 kernel, stride 1, ReLU  → (batch, 64,  7,  7)
    Flatten → 3136
    Linear: 3136 → 512, ReLU
    Output: 512 → n_actions
"""

import torch
import torch.nn as nn


class ObbyCNN(nn.Module):
    def __init__(self, n_actions: int = 6) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Batch of stacked frames, shape (batch, 4, 84, 84), float32.

        Returns:
            Action logits / Q-values, shape (batch, n_actions).
        """
        return self.fc(self.conv(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the 512-d feature vector before the output layer.

        Useful for passing features to the RL policy head.
        """
        h = self.conv(x)
        # Apply all FC layers except the last linear
        h = self.fc[0](h)   # Flatten
        h = self.fc[1](h)   # Linear 3136→512
        h = self.fc[2](h)   # ReLU
        return h
