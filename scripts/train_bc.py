"""
scripts/train_bc.py — Train behavioral cloning CNN.

Trains ObbyCNN on recorded demonstrations using cross-entropy loss
with inverse-frequency class weights.

Usage:
    python scripts/train_bc.py
"""

from __future__ import annotations

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

DEMOS_DIR = Path(__file__).resolve().parent.parent / "demos"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 50
PATIENCE = 10  # early stopping


def compute_class_weights(dataset: BCDataset) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced action distribution."""
    counts = dataset.action_counts.astype(np.float64)
    # Avoid division by zero for actions not present
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.sum() * NUM_ACTIONS
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model: ObbyCNN, loader: DataLoader, criterion: nn.Module, device: torch.device):
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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load datasets (drop idle frames to prevent class imbalance)
    print("Loading datasets...")
    train_ds = BCDataset(DEMOS_DIR, split="train", augment=True, drop_idle=True)
    val_ds = BCDataset(DEMOS_DIR, split="val", augment=False, drop_idle=True)
    print(f"  Train: {len(train_ds)} samples from {train_ds.num_runs} runs "
          f"({train_ds._idle_dropped} idle frames dropped)")
    print(f"  Val:   {len(val_ds)} samples from {val_ds.num_runs} runs "
          f"({val_ds._idle_dropped} idle frames dropped)")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Class weights
    class_weights = compute_class_weights(train_ds).to(device)
    print(f"  Class weights: {class_weights.cpu().numpy().round(2)}")

    # Model
    model = ObbyCNN(n_actions=NUM_ACTIONS).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Training loop
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nTraining for up to {MAX_EPOCHS} epochs (patience={PATIENCE})...\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>10}  {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for stacks, actions in train_loader:
            stacks = stacks.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            logits = model(stacks)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * stacks.size(0)
            train_correct += (logits.argmax(dim=1) == actions).sum().item()
            train_total += stacks.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

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
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "class_weights": class_weights.cpu().numpy().tolist(),
                "action_names": ACTION_NAMES,
            }
            # bc_best.pt is the default CNN loaded by run_twotier.py
            torch.save(checkpoint, CHECKPOINT_DIR / "bc_best.pt")
            print(f"        ✓ Saved best checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

        print()

    # Final summary
    best = torch.load(CHECKPOINT_DIR / "bc_best.pt", weights_only=True)
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
