"""
vision/siglip_model.py — SigLIP2 vision encoder with classification head.

Freezes the pretrained SigLIP2 ViT and trains only a lightweight
2-layer MLP head (~200K params) for action classification.

Input:  (batch, 3, 224, 224) — ImageNet-normalized RGB
Output: (batch, 6) — action logits
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class SigLIPObbyModel(nn.Module):
    def __init__(self, n_actions: int = 6, encoder_name: str = "google/siglip2-base-patch16-224") -> None:
        super().__init__()
        full_model = AutoModel.from_pretrained(encoder_name)
        self.encoder = full_model.vision_model
        self.encoder.requires_grad_(False)

        hidden_dim = full_model.config.vision_config.hidden_size  # 768 for base
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 224, 224) ImageNet-normalized RGB frames.

        Returns:
            Action logits, shape (batch, n_actions).
        """
        with torch.no_grad():
            features = self.encoder(pixel_values=x).pooler_output  # (batch, 768)
        return self.head(features)
