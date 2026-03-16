"""
Train the reward model on collected (frame, active/idle) data.
Uses BCE loss: model(s) -> sigmoid -> probability of "active"; label 1 = active, 0 = idle.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from reward.model import RewardNet, save_reward_model


class RewardDataset(Dataset):
    def __init__(self, frames: np.ndarray, labels: np.ndarray):
        # frames (N, H, W, C) uint8
        self.frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return self.frames[i], self.labels[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="reward_data", help="Directory containing data.npz and config.json")
    parser.add_argument("--out", type=str, default="reward_model.pt", help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    args = parser.parse_args()

    data_path = os.path.join(args.data, "data.npz")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"No data at {data_path}. Run reward/collect.py first.")

    data = np.load(data_path)
    frames = data["frames"]
    labels = data["labels"]
    n, h, w, c = frames.shape

    config_path = os.path.join(args.data, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = RewardNet(in_channels=c, height=h, width=w, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    dataset = RewardDataset(frames, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        n_batches = len(loader)
        avg_loss = total_loss / n_batches

        # Log what the reward model decides: mean r(s) for active vs idle
        model.eval()
        with torch.no_grad():
            all_logits = []
            all_labels = []
            for x, y in loader:
                x = x.to(device)
                all_logits.append(model(x).cpu())
                all_labels.append(y)
            logits_cat = torch.cat(all_logits, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)
            probs = torch.sigmoid(logits_cat).numpy()
            labels_np = labels_cat.numpy()
            active_mask = labels_np == 1
            idle_mask = ~active_mask
            mean_r_active = probs[active_mask].mean() if active_mask.any() else float("nan")
            mean_r_idle = probs[idle_mask].mean() if idle_mask.any() else float("nan")
        model.train()

        print(f"Epoch {epoch + 1}/{args.epochs} loss={avg_loss:.4f} | reward model: mean r(active)={mean_r_active:.3f} mean r(idle)={mean_r_idle:.3f}")
        # Log 5 random decisions
        idx = np.random.choice(len(probs), size=min(5, len(probs)), replace=False)
        for i in idx:
            print(f"    sample {i}: label={int(labels_np[i])} ({'active' if labels_np[i] == 1 else 'idle'}) -> r(s)={probs[i]:.3f}")

    save_config = {
        "in_channels": c,
        "height": h,
        "width": w,
        "hidden": args.hidden,
    }
    save_reward_model(model, args.out, config=save_config)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
