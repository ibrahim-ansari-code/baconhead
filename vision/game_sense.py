"""
GameSense — learned game-state classifier.

Fine-tuned ViT that replaces all hardcoded game detection (death screens,
danger zones, menus, etc.) with a single trained model.

States:
  0 = playing   normal gameplay, character moving around
  1 = dead       death / respawn / game over / loading / white flash
  2 = menu       lobby / shop / inventory / chat overlay
  3 = danger     near map edge, low health, about to die

Architecture:
  ViT-base-patch16-224 → 768-d CLS token
  → Linear(768, 256) → ReLU → Dropout → Linear(256, 4)

Usage:
  # Predict
  model = load_game_sense("game_sense.pt")
  state, confidence = model.predict(frame)

  # Fallback when no model trained yet
  state, confidence = heuristic_state(frame)
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


STATE_LABELS = ["playing", "dead", "menu", "danger"]
STATE_TO_IDX = {s: i for i, s in enumerate(STATE_LABELS)}
NUM_STATES = len(STATE_LABELS)


class GameSense(nn.Module):
    """ViT-based game state classifier."""

    def __init__(
        self,
        backbone: str = "google/vit-base-patch16-224",
        num_states: int = NUM_STATES,
    ):
        super().__init__()
        from transformers import ViTModel, ViTImageProcessor

        self.processor = ViTImageProcessor.from_pretrained(backbone)
        self.vit = ViTModel.from_pretrained(backbone)
        hidden = self.vit.config.hidden_size  # 768
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_states),
        )
        self._num_states = num_states
        self._backbone = backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) → (B, num_states) logits."""
        out = self.vit(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)

    def freeze_backbone(self):
        for p in self.vit.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.vit.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def predict(
        self, frame: np.ndarray, device: Optional[torch.device] = None
    ) -> Tuple[str, float]:
        """
        Single-frame prediction.

        frame: (H, W, 3) uint8 numpy
        Returns: (state_label, confidence)
        """
        from PIL import Image

        device = device or next(self.parameters()).device
        pil = Image.fromarray(frame.astype(np.uint8))
        inputs = self.processor(images=pil, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        self.eval()
        logits = self.forward(pv)
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax())
        return STATE_LABELS[idx], float(probs[idx])

    @torch.no_grad()
    def predict_probs(
        self, frame: np.ndarray, device: Optional[torch.device] = None
    ) -> dict:
        """Return probability for every state."""
        from PIL import Image

        device = device or next(self.parameters()).device
        pil = Image.fromarray(frame.astype(np.uint8))
        inputs = self.processor(images=pil, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        self.eval()
        logits = self.forward(pv)
        probs = torch.softmax(logits, dim=-1)[0]
        return {STATE_LABELS[i]: float(probs[i]) for i in range(self._num_states)}


# ── Save / Load ──────────────────────────────────────────────────────────────


def save_game_sense(model: GameSense, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone": model._backbone,
            "num_states": model._num_states,
        },
        path,
    )
    print(f"[game_sense] Saved → {path}", flush=True)


def load_game_sense(
    path: str, device: Optional[torch.device] = None
) -> GameSense:
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GameSense(
        backbone=ckpt.get("backbone", "google/vit-base-patch16-224"),
        num_states=ckpt.get("num_states", NUM_STATES),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


# ── Heuristic fallback (no trained model) ────────────────────────────────────


def heuristic_state(frame: np.ndarray) -> Tuple[str, float]:
    """
    Simple pixel-based fallback when GameSense hasn't been trained yet.
    Returns (state_label, confidence).
    """
    mean_bright = float(frame.mean())
    screen_std = float(frame.std())

    # White respawn / game-over screen
    if mean_bright > 210 and screen_std < 25:
        return "dead", 0.9

    # Black loading / transition
    if mean_bright < 15:
        return "dead", 0.8

    if frame.ndim == 3:
        h = frame.shape[0]
        body = frame[int(h * 0.1) : int(h * 0.9)]

        # Underwater / void — blue dominant, no red
        blue_mean = float(body[..., 2].mean())
        red_mean = float(body[..., 0].mean())
        if blue_mean > 180 and red_mean < 80:
            return "danger", 0.7

        # Map edge — middle strip is mostly water/sky
        h2, w2 = frame.shape[:2]
        mid = frame[h2 // 3 : 2 * h2 // 3, w2 // 4 : 3 * w2 // 4]
        blue = mid[..., 2].astype(float)
        red = mid[..., 0].astype(float)
        green = mid[..., 1].astype(float)
        water_pct = float(
            ((blue > 120) & (blue > red + 20) & (blue > green + 20)).mean()
        )
        if water_pct > 0.35:
            return "danger", 0.6

    return "playing", 0.5
