"""
scripts/train_bc.py — Train behavioral cloning model.

Supports two model types:
  - cnn:    Nature DQN CNN on 84x84 grayscale 4-frame stacks
  - siglip: SigLIP2 ViT (frozen) + MLP head on 224x224 RGB single frames

Usage:
    python scripts/train_bc.py                    # default: CNN
    python scripts/train_bc.py --model-type siglip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.model import ObbyCNN
from training.bc_dataset import BCDataset
from control.actions import ACTION_NAMES, NUM_ACTIONS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = _ROOT / "checkpoints"

# CNN defaults
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 50
PATIENCE = 10  # early stopping

# SigLIP defaults
SIGLIP_LR = 5e-5
SIGLIP_BATCH_SIZE = 32
SIGLIP_MAX_EPOCHS = 30
SIGLIP_PATIENCE = 8


def compute_class_weights(dataset: BCDataset) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced action distribution."""
    counts = dataset.action_counts.astype(np.float64)
    # Avoid division by zero for actions not present
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.sum() * NUM_ACTIONS
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Compute loss, accuracy, and per-class accuracy on a data loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = np.zeros(NUM_ACTIONS)
    class_total = np.zeros(NUM_ACTIONS)

    with torch.no_grad():
        for stacks, actions in loader:
            stacks = stacks.to(device)
            actions = actions.to(device)

            logits = model(stacks)
            loss = criterion(logits, actions)
            total_loss += loss.item() * stacks.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == actions).sum().item()
            total += stacks.size(0)

            for c in range(NUM_ACTIONS):
                mask = actions == c
                class_total[c] += mask.sum().item()
                class_correct[c] += ((preds == c) & mask).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    per_class = np.where(class_total > 0, class_correct / class_total, 0.0)

    return avg_loss, accuracy, per_class


def main():
    parser = argparse.ArgumentParser(description="Train behavioral cloning model")
    parser.add_argument(
        "--model-type", choices=["cnn", "siglip"], default="cnn",
        help="Model backbone: 'cnn' (Nature DQN) or 'siglip' (SigLIP2 ViT)",
    )
    args = parser.parse_args()

    use_siglip = args.model_type == "siglip"
    DEMOS_DIR = _ROOT / ("demos_rgb" if use_siglip else "demos_cnn")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")

    # Dataset mode
    ds_mode = "rgb224" if use_siglip else "grayscale"
    batch_size = SIGLIP_BATCH_SIZE if use_siglip else BATCH_SIZE
    lr = SIGLIP_LR if use_siglip else LR
    max_epochs = SIGLIP_MAX_EPOCHS if use_siglip else MAX_EPOCHS
    patience = SIGLIP_PATIENCE if use_siglip else PATIENCE

    # Load datasets (drop idle frames to prevent class imbalance)
    print("Loading datasets...")
    train_ds = BCDataset(DEMOS_DIR, split="train", augment=True, drop_idle=True, mode=ds_mode)
    val_ds = BCDataset(DEMOS_DIR, split="val", augment=False, drop_idle=True, mode=ds_mode)
    print(f"  Train: {len(train_ds)} samples from {train_ds.num_runs} runs "
          f"({train_ds._idle_dropped} idle frames dropped)")
    print(f"  Val:   {len(val_ds)} samples from {val_ds.num_runs} runs "
          f"({val_ds._idle_dropped} idle frames dropped)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Class weights
    class_weights = compute_class_weights(train_ds).to(device)
    print(f"  Class weights: {class_weights.cpu().numpy().round(2)}")

    # Model
    if use_siglip:
        from vision.siglip_model import SigLIPObbyModel
        model = SigLIPObbyModel(n_actions=NUM_ACTIONS).to(device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  SigLIP2 params: {total:,} total, {trainable:,} trainable")
        checkpoint_name = "siglip_best.pt"
    else:
        model = ObbyCNN(n_actions=NUM_ACTIONS).to(device)
        checkpoint_name = "bc_best.pt"

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if use_siglip:
        optimizer = torch.optim.AdamW(
            model.head.parameters(), lr=lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = None

    # Training loop
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nTraining for up to {max_epochs} epochs (patience={patience})...\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>10}  {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, actions in train_loader:
            inputs = inputs.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (logits.argmax(dim=1) == actions).sum().item()
            train_total += inputs.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        if scheduler is not None:
            scheduler.step()

        # Validate
        val_loss, val_acc, per_class_acc = evaluate(model, val_loader, criterion, device)

        print(f"{epoch:5d}  {train_loss:10.4f}  {train_acc:8.1%}  {val_loss:10.4f}  {val_acc:6.1%}")

        # Per-class accuracy
        for i, name in enumerate(ACTION_NAMES):
            print(f"        {name:15s}: {per_class_acc[i]:5.1%}")

        # Early stopping / checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_type": args.model_type,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "class_weights": class_weights.cpu().numpy().tolist(),
                "action_names": ACTION_NAMES,
            }
            torch.save(checkpoint, CHECKPOINT_DIR / checkpoint_name)
            print(f"        ✓ Saved best checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        print()

    # Final summary
    best = torch.load(CHECKPOINT_DIR / checkpoint_name, weights_only=False)
    print(f"\nTraining complete.")
    print(f"  Best epoch: {best['epoch']}")
    print(f"  Best val loss: {best['val_loss']:.4f}")
    print(f"  Best val accuracy: {best['val_accuracy']:.1%}")

    if best["val_accuracy"] >= 0.70:
        print("\n  ✓ GATE PASSED: Val accuracy >= 70%")
    elif best["val_accuracy"] >= 0.60:
        print("\n  ⚠ WARNING: Val accuracy 60-70% — consider collecting more data")
    else:
        print("\n  ✗ GATE FAILED: Val accuracy < 60% — collect more data before continuing")


if __name__ == "__main__":
    main()
