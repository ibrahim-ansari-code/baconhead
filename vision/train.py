"""
Train GameSense model.

Two-phase training:
  Phase 1 — frozen ViT backbone, train classification head only (fast)
  Phase 2 — unfreeze backbone, fine-tune everything with lower LR

Prints per-class precision / recall at the end.

Usage:
  python -m vision.train --data game_data --out game_sense.pt --epochs 10
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from vision.game_sense import (
    GameSense,
    save_game_sense,
    load_game_sense,
    STATE_LABELS,
    NUM_STATES,
)


class GameSenseDataset(Dataset):
    def __init__(self, frames, labels, processor):
        self.frames = frames
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.processor = processor

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        from PIL import Image

        pil = Image.fromarray(self.frames[i].astype(np.uint8))
        inputs = self.processor(images=pil, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), self.labels[i]


# ── Training ──────────────────────────────────────────────────────────────────


def train(
    data_dir: str,
    out_path: str,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    freeze_epochs: int = 3,
    backbone: str = "google/vit-base-patch16-224",
):
    npz_path = os.path.join(data_dir, "data.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"No data at {npz_path}. "
            f"Run: python -m vision.collect --seconds 120 --out {data_dir}"
        )

    data = np.load(npz_path, allow_pickle=False)
    frames = data["frames"]
    labels = data["labels"]
    n = len(labels)

    print(f"[train] {n} samples", flush=True)
    for i, name in enumerate(STATE_LABELS):
        count = int((labels == i).sum())
        print(f"  {name}: {count}", flush=True)

    if n < 10:
        raise ValueError("Need at least 10 samples. Collect more data first.")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[train] Device: {device}", flush=True)

    model = GameSense(backbone=backbone).to(device)
    dataset = GameSenseDataset(frames, labels, model.processor)

    # 80/20 split
    n_val = max(1, int(n * 0.2))
    n_train = n - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Class weighting for imbalanced data
    counts = np.bincount(labels.astype(int), minlength=NUM_STATES).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = (1.0 / counts) * n / NUM_STATES
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0

    # ── Phase 1: frozen backbone ──────────────────────────────────────────────

    print(f"\n[train] Phase 1: frozen backbone ({freeze_epochs} epochs)", flush=True)
    model.freeze_backbone()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr * 10
    )

    for epoch in range(freeze_epochs):
        avg_loss = _train_epoch(model, train_loader, criterion, opt, device)
        val_acc = _eval_acc(model, val_loader, device)
        print(
            f"  epoch {epoch + 1}/{freeze_epochs}  "
            f"loss={avg_loss:.4f}  val_acc={val_acc:.3f}",
            flush=True,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_game_sense(model, out_path)

    # ── Phase 2: fine-tune everything ─────────────────────────────────────────

    remaining = epochs - freeze_epochs
    if remaining > 0:
        print(
            f"\n[train] Phase 2: fine-tune all ({remaining} epochs)", flush=True
        )
        model.unfreeze_backbone()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=remaining
        )

        for epoch in range(freeze_epochs, epochs):
            avg_loss = _train_epoch(
                model, train_loader, criterion, opt, device
            )
            scheduler.step()
            val_acc = _eval_acc(model, val_loader, device)
            print(
                f"  epoch {epoch + 1}/{epochs}  "
                f"loss={avg_loss:.4f}  val_acc={val_acc:.3f}",
                flush=True,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_game_sense(model, out_path)

    # ── Final eval ────────────────────────────────────────────────────────────

    print(f"\n[train] Best val accuracy: {best_val_acc:.3f}", flush=True)
    best_model = load_game_sense(out_path, device)
    _eval_detailed(best_model, val_loader, device)
    print(f"[train] Model saved → {out_path}", flush=True)


def _train_epoch(model, loader, criterion, opt, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for pv, lbl in loader:
        pv, lbl = pv.to(device), lbl.to(device)
        opt.zero_grad()
        logits = model(pv)
        loss = criterion(logits, lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _eval_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for pv, lbl in loader:
            pv, lbl = pv.to(device), lbl.to(device)
            preds = model(pv).argmax(dim=-1)
            correct += (preds == lbl).sum().item()
            total += len(lbl)
    return correct / max(total, 1)


def _eval_detailed(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for pv, lbl in loader:
            pv, lbl = pv.to(device), lbl.to(device)
            preds = model(pv).argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(lbl.cpu().tolist())

    print("\n[eval] Per-class results:", flush=True)
    for i, name in enumerate(STATE_LABELS):
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == i and l == i)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == i and l != i)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != i and l == i)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        print(f"  {name:10s}  prec={prec:.3f}  recall={rec:.3f}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train GameSense")
    parser.add_argument("--data", type=str, default="game_data")
    parser.add_argument("--out", type=str, default="game_sense.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument(
        "--backbone", type=str, default="google/vit-base-patch16-224"
    )
    args = parser.parse_args()

    train(
        data_dir=args.data,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_epochs=args.freeze_epochs,
        backbone=args.backbone,
    )


if __name__ == "__main__":
    main()
