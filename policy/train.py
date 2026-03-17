"""
Train the policy model on collected (frame, action_id) data.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from policy.model import PolicyNet, save_policy_model, DEFAULT_BACKBONE
from policy.oracles import ACTION_NAMES


class PolicyDataset(Dataset):
    def __init__(self, frames: np.ndarray, actions: np.ndarray, processor):
        # frames (N, H, W, C) uint8
        self.frames = frames
        self.actions = torch.from_numpy(actions).long()
        self.processor = processor

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        from PIL import Image
        pil = Image.fromarray(self.frames[i].astype(np.uint8))
        inputs = self.processor(images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, self.actions[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="policy_data", help="Directory with data.npz and config.json")
    parser.add_argument("--out", type=str, default="policy_model.pt", help="Output path (.pt or dir)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2, help="Train only head for this many epochs")
    args = parser.parse_args()

    data_path = os.path.join(args.data, "data.npz")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"No data at {data_path}. Run policy/collect.py first.")

    data = np.load(data_path)
    frames = data["frames"]
    actions = data["actions"]
    n, h, w, c = frames.shape

    config_path = os.path.join(args.data, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            data_config = json.load(f)
        action_names = data_config.get("action_names", ACTION_NAMES)
        num_actions = len(action_names)
    else:
        num_actions = len(ACTION_NAMES)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = PolicyNet(backbone=args.backbone, num_actions=num_actions).to(device)
    processor = model.processor
    if args.freeze_backbone_epochs > 0:
        model.freeze_backbone()

    dataset = PolicyDataset(frames, actions, processor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        if epoch == args.freeze_backbone_epochs and args.freeze_backbone_epochs > 0:
            model.unfreeze_all()
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1)
        model.train()
        total_loss = 0.0
        n_batches = 0
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            logits = model(pixel_values)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        # Eval accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for pixel_values, labels in loader:
                pixel_values = pixel_values.to(device)
                logits = model(pixel_values)
                pred = logits.argmax(dim=1)
                correct += (pred == labels.to(device)).sum().item()
                total += labels.size(0)
        acc = correct / total if total else 0
        print(f"Epoch {epoch + 1}/{args.epochs} loss={avg_loss:.4f} acc={acc:.3f}", flush=True)

    save_config = {
        "backbone": args.backbone,
        "num_actions": num_actions,
        "frame_height": h,
        "frame_width": w,
    }
    save_policy_model(model, args.out, config=save_config)
    print(f"Saved policy to {args.out}", flush=True)


if __name__ == "__main__":
    main()
