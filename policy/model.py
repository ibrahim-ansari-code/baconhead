"""
Policy model: Hugging Face vision backbone + classification head for 8 actions.
"""

import os
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn

from policy.oracles import ACTION_NAMES


# Default HF vision model (ViT small for speed; 224x224 input)
DEFAULT_BACKBONE = "google/vit-base-patch16-224"


class PolicyNet(nn.Module):
    """
    Vision backbone + linear head -> logits over ACTION_NAMES.
    Uses Hugging Face AutoModelForImageClassification with num_labels=len(ACTION_NAMES).
    """

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        num_actions: int = None,
        freeze_backbone_epochs: int = 0,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_actions = num_actions or len(ACTION_NAMES)
        self.freeze_backbone_epochs = freeze_backbone_epochs
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.processor = AutoImageProcessor.from_pretrained(backbone)
        self.model = AutoModelForImageClassification.from_pretrained(
            backbone,
            num_labels=self.num_actions,
            ignore_mismatched_sizes=True,
        )
        self._frozen = False

    def freeze_backbone(self):
        for name, p in self.model.named_parameters():
            if "classifier" in name or "fc" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        self._frozen = True

    def unfreeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = True
        self._frozen = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits

    def predict_action(self, pixel_values: torch.Tensor):
        """Single or batch; returns action index (int) or list of indices."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(pixel_values)
            if logits.dim() == 1:
                return int(logits.argmax().item())
            out = logits.argmax(dim=1).cpu().numpy()
            return int(out[0]) if out.size == 1 else out.tolist()


def frame_to_tensor(
    frame: np.ndarray,
    processor,
    device: torch.device,
) -> torch.Tensor:
    """Convert RGB numpy (H,W,3) to batch of 1 with processor normalization."""
    from PIL import Image
    pil = Image.fromarray(frame.astype(np.uint8))
    inputs = processor(images=pil, return_tensors="pt")
    return inputs["pixel_values"].to(device)


def load_policy_model(
    path: str,
    device: Optional[torch.device] = None,
) -> PolicyNet:
    """Load PolicyNet from a directory (saved with save_policy_model)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    import json
    config_path = os.path.join(path, "config.json") if os.path.isdir(path) else path.replace(".pt", "_config.json")
    if not os.path.isfile(config_path):
        config_path = path.replace(".pt", ".json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    backbone = config.get("backbone", DEFAULT_BACKBONE)
    num_actions = config.get("num_actions", len(ACTION_NAMES))
    model = PolicyNet(backbone=backbone, num_actions=num_actions)
    state_path = os.path.join(path, "pytorch_model.bin") if os.path.isdir(path) else path
    if not state_path.endswith(".bin") and not state_path.endswith(".pt"):
        state_path = os.path.join(path, "pytorch_model.bin")
    if os.path.isfile(state_path):
        state = torch.load(state_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def save_policy_model(model: PolicyNet, path: str, config: Optional[dict] = None):
    """Save model state and config."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cfg = config or {}
    cfg.setdefault("backbone", model.backbone_name)
    cfg.setdefault("num_actions", model.num_actions)
    if path.endswith(".pt"):
        torch.save({"model_state": model.model.state_dict(), "config": cfg}, path)
        config_path = path.replace(".pt", ".json")
    else:
        os.makedirs(path, exist_ok=True)
        torch.save(model.model.state_dict(), os.path.join(path, "pytorch_model.bin"))
        config_path = os.path.join(path, "config.json")
    import json
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
