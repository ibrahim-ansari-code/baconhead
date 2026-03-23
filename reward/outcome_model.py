"""
Outcome predictor: (frame, spatial_features, key_sequence) → P(survived next 10s).

Architecture:
  Frame encoder:   ViT-base-16 frozen → 768-dim CLS token
  Spatial encoder: edge_distances (4) + flow_mag (9) + flow_dir (9) = 22 dims → Linear(22→64) → ReLU
  Key encoder:     per-key stats (8 keys × 3 stats = 24 dims) → Linear(24→64) → ReLU
  Fusion MLP:      concat(768+64+64=896) → Linear(256) → ReLU → Dropout(0.3) → Linear(1)
  Loss:            BCE (with optional class weighting for imbalanced data)

Key vocabulary (must match collect_episodes.py):
  [w, a, s, d, space, shift, q, e]  → indices 0-7

Separate train script at bottom (python -m reward.train_outcome).
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

KEY_VOCAB      = ["w", "a", "s", "d", "space", "shift", "q", "e"]
N_KEYS         = len(KEY_VOCAB)
MAX_KEY_EVENTS = 64
EPISODE_MS     = 10_000.0   # normalisation denominator
N_BUCKETS      = 9


# ── key-event feature encoding ─────────────────────────────────────────────────

def encode_key_events(key_events: np.ndarray) -> np.ndarray:
    """
    key_events: (MAX_KEY_EVENTS, 3) float32  [key_idx, t_down_ms, t_up_ms]
    Returns 24-dim float32 feature vector (N_KEYS × 3 stats):
      [total_hold_ms, press_count, max_press_ms] per key, normalised by EPISODE_MS.
    """
    feats = np.zeros((N_KEYS, 3), dtype=np.float32)
    for row in key_events:
        kidx, tdown, tup = int(row[0]), float(row[1]), float(row[2])
        if tdown == 0 and tup == 0:
            break   # padding row
        if not (0 <= kidx < N_KEYS):
            continue
        dur = max(0.0, tup - tdown)
        feats[kidx, 0] += dur            # total hold
        feats[kidx, 1] += 1.0            # count
        feats[kidx, 2]  = max(feats[kidx, 2], dur)  # max
    feats[:, 0] /= EPISODE_MS
    feats[:, 2] /= EPISODE_MS
    # press_count: normalise by max plausible presses (50 in 10s)
    feats[:, 1] = np.clip(feats[:, 1] / 50.0, 0.0, 1.0)
    return feats.flatten()


def encode_key_events_batch(key_events_batch: np.ndarray) -> np.ndarray:
    """(B, MAX_KEY_EVENTS, 3) → (B, 24) float32."""
    B = key_events_batch.shape[0]
    out = np.zeros((B, N_KEYS * 3), dtype=np.float32)
    for i in range(B):
        out[i] = encode_key_events(key_events_batch[i])
    return out


def encode_spatial_features(
    edge_distances: np.ndarray,  # (4,) or (B, 4)
    flow_mag: np.ndarray,         # (9,) or (B, 9)
    flow_dir: np.ndarray,         # (9,) or (B, 9)
    physics: Optional[dict] = None,
) -> np.ndarray:
    """
    Concatenate [edge_distances, flow_mag, flow_dir] → (22,) or (B, 22) float32.
    Optionally normalise flow_mag by w_px_per_ms from physics.json.
    """
    squeeze = edge_distances.ndim == 1
    if squeeze:
        edge_distances = edge_distances[None]
        flow_mag       = flow_mag[None]
        flow_dir       = flow_dir[None]

    # Normalise flow_mag: divide by reference px/ms * EPISODE_MS (≈ total pixels over window)
    if physics and physics.get("w_px_per_ms", 0) > 0:
        ref = float(physics["w_px_per_ms"]) * EPISODE_MS / N_BUCKETS
        flow_mag_norm = flow_mag / max(ref, 1e-6)
    else:
        flow_mag_norm = flow_mag / max(float(flow_mag.max()) + 1e-6, 1.0)

    # flow_dir is already in [0, 2π]; normalise to [0, 1]
    flow_dir_norm = flow_dir / (2.0 * np.pi + 1e-6)

    # edge_distances already in [0, 1] (normalised by frame dim in collect_episodes)
    spatial = np.concatenate([
        edge_distances.astype(np.float32),
        flow_mag_norm.astype(np.float32),
        flow_dir_norm.astype(np.float32),
    ], axis=-1)   # (B, 22)
    return spatial[0] if squeeze else spatial


# ── model ──────────────────────────────────────────────────────────────────────

class OutcomeModel(nn.Module):
    """
    (frame_pixels, spatial_feats, key_feats) → logit (scalar).
    Apply sigmoid for P(survived).
    """

    def __init__(self, backbone: str = "google/vit-base-patch16-224"):
        super().__init__()
        from transformers import ViTModel, ViTImageProcessor
        self.processor = ViTImageProcessor.from_pretrained(backbone)
        self._vit = ViTModel.from_pretrained(backbone)
        # Freeze backbone
        for p in self._vit.parameters():
            p.requires_grad = False
        vit_dim = self._vit.config.hidden_size  # 768

        self.spatial_encoder = nn.Sequential(
            nn.Linear(22, 64),
            nn.ReLU(),
        )
        self.key_encoder = nn.Sequential(
            nn.Linear(N_KEYS * 3, 64),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(vit_dim + 64 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,   # (B, 3, 224, 224) in [0,1]
        spatial_feats: torch.Tensor,  # (B, 22)
        key_feats: torch.Tensor,      # (B, 24)
    ) -> torch.Tensor:
        with torch.no_grad():
            vit_out = self._vit(pixel_values=pixel_values)
        cls = vit_out.last_hidden_state[:, 0, :]  # (B, 768)
        sp  = self.spatial_encoder(spatial_feats)  # (B, 64)
        kf  = self.key_encoder(key_feats)          # (B, 64)
        x   = torch.cat([cls, sp, kf], dim=1)      # (B, 896)
        return self.fusion(x).squeeze(-1)           # (B,)

    def predict_survival(
        self,
        frame: np.ndarray,             # (224, 224, 3) uint8
        edge_distances: np.ndarray,    # (4,)
        flow_mag: np.ndarray,          # (9,)
        flow_dir: np.ndarray,          # (9,)
        key_events: np.ndarray,        # (MAX_KEY_EVENTS, 3)
        device: torch.device,
        physics: Optional[dict] = None,
    ) -> float:
        """Convenience: single example → P(survived) float."""
        from PIL import Image as _Image
        pil = _Image.fromarray(frame.astype(np.uint8))
        inputs = self.processor(images=pil, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)

        sp = encode_spatial_features(edge_distances, flow_mag, flow_dir, physics)
        kf = encode_key_events(key_events)

        sp_t = torch.from_numpy(sp).unsqueeze(0).to(device)
        kf_t = torch.from_numpy(kf).unsqueeze(0).to(device)

        self.eval()
        with torch.no_grad():
            logit = self.forward(pv, sp_t, kf_t).item()
        return float(torch.sigmoid(torch.tensor(logit)).item())


def save_outcome_model(model: OutcomeModel, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "backbone": "google/vit-base-patch16-224"}, path)
    print(f"[outcome_model] Saved to {path}", flush=True)


def load_outcome_model(path: str, device: Optional[torch.device] = None) -> OutcomeModel:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "google/vit-base-patch16-224")
    model = OutcomeModel(backbone=backbone)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


# ── dataset ────────────────────────────────────────────────────────────────────

class EpisodeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frames: np.ndarray,          # (N, 224, 224, 3) uint8
        edge_distances: np.ndarray,  # (N, 4)
        flow_mag: np.ndarray,        # (N, 9)
        flow_dir: np.ndarray,        # (N, 9)
        key_events: np.ndarray,      # (N, MAX_KEY_EVENTS, 3)
        survived: np.ndarray,        # (N,) int8
        processor,
        physics: Optional[dict] = None,
    ):
        self.frames         = frames
        self.edge_distances = edge_distances
        self.flow_mag       = flow_mag
        self.flow_dir       = flow_dir
        self.key_events     = key_events
        self.survived       = torch.from_numpy(survived.astype(np.float32))
        self.processor      = processor
        self.physics        = physics

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        from PIL import Image as _Image
        pil = _Image.fromarray(self.frames[i].astype(np.uint8))
        inputs = self.processor(images=pil, return_tensors="pt")
        pv = inputs["pixel_values"].squeeze(0)   # (3, 224, 224)

        sp = encode_spatial_features(
            self.edge_distances[i], self.flow_mag[i], self.flow_dir[i], self.physics
        )
        kf = encode_key_events(self.key_events[i])

        return (
            pv,
            torch.from_numpy(sp),
            torch.from_numpy(kf),
            self.survived[i],
        )


# ── training script ─────────────────────────────────────────────────────────────

def train(
    data_dir: str,
    out_path: str,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    backbone: str = "google/vit-base-patch16-224",
):
    import json
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    npz_path = os.path.join(data_dir, "data.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"No data at {npz_path}. Run reward/collect_episodes.py first.")

    data = np.load(npz_path, allow_pickle=False)
    frames         = data["frames"]
    edge_distances = data["edge_distances"]
    flow_mag       = data["flow_mag"]
    flow_dir       = data["flow_dir"]
    key_events     = data["key_events"]
    survived       = data["survived"]

    n = len(survived)
    n_survived = int(survived.sum())
    n_fell     = n - n_survived
    print(f"[train_outcome] {n} episodes: {n_survived} survived, {n_fell} fell", flush=True)
    if n < 4:
        raise ValueError("Need at least 4 episodes to train.")

    physics = None
    physics_path = os.path.join(data_dir, "physics.json")
    if os.path.isfile(physics_path):
        with open(physics_path) as f:
            physics = json.load(f)
        print(f"[train_outcome] Loaded physics from {physics_path}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = OutcomeModel(backbone=backbone).to(device)

    dataset = EpisodeDataset(frames, edge_distances, flow_mag, flow_dir, key_events, survived,
                             model.processor, physics)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Class weighting: up-weight the minority class
    if n_survived > 0 and n_fell > 0:
        pos_weight = torch.tensor([n_fell / n_survived], dtype=torch.float32).to(device)
    else:
        pos_weight = None

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for pv, sp, kf, labels in loader:
            pv, sp, kf, labels = pv.to(device), sp.to(device), kf.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(pv, sp, kf)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        # Eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for pv, sp, kf, labels in loader:
                pv, sp, kf, labels = pv.to(device), sp.to(device), kf.to(device), labels.to(device)
                probs = torch.sigmoid(model(pv, sp, kf))
                preds = (probs >= 0.5).float()
                correct += (preds == labels).sum().item()
        acc = correct / n
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_outcome_model(model, out_path)
        print(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  acc={acc:.3f}  (best saved)", flush=True)

    print(f"[train_outcome] Done. Best model at {out_path}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train the episode outcome model")
    parser.add_argument("--data",       type=str, default="episode_data")
    parser.add_argument("--out",        type=str, default="outcome_model.pt")
    parser.add_argument("--epochs",     type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--backbone",   type=str, default="google/vit-base-patch16-224")
    args = parser.parse_args()
    train(
        data_dir=args.data,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
    )


if __name__ == "__main__":
    main()
