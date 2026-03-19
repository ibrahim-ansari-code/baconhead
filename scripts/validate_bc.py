"""
scripts/validate_bc.py — Validate behavioral cloning model.

Standalone validation script per project testing standards.
Runs all checks in < 2 minutes with clear PASS/FAIL output.

Usage:
    python scripts/validate_bc.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.model import ObbyCNN
from vision.preprocess import preprocess_frame
from training.bc_dataset import BCDataset
from control.actions import ACTION_NAMES, NUM_ACTIONS

DEMOS_DIR = Path(__file__).resolve().parent.parent / "demos"
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "bc_best.pt"

results: list[tuple[str, str, str]] = []  # (name, status, detail)


def record(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    symbol = "✓" if passed else "✗"
    print(f"  [{status}] {symbol} {name}" + (f" — {detail}" if detail else ""))


def record_warn(name: str, detail: str = ""):
    results.append((name, "WARN", detail))
    print(f"  [WARN] ⚠ {name}" + (f" — {detail}" if detail else ""))


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print()

    # ------------------------------------------------------------------
    # Check 1: Dataset loads, shapes correct
    # ------------------------------------------------------------------
    print("Check 1: Dataset loading")
    try:
        val_ds = BCDataset(DEMOS_DIR, split="val", drop_idle=True)
        stack, action = val_ds[0]
        shape_ok = stack.shape == (4, 84, 84)
        dtype_ok = stack.dtype == torch.float32
        action_ok = 0 <= action < NUM_ACTIONS
        record(
            "Dataset loads, shape (4, 84, 84)",
            shape_ok and dtype_ok and action_ok,
            f"shape={tuple(stack.shape)}, dtype={stack.dtype}, action={action}",
        )
    except Exception as e:
        record("Dataset loads", False, str(e))
        val_ds = None

    # ------------------------------------------------------------------
    # Check 2: Checkpoint loads into ObbyCNN
    # ------------------------------------------------------------------
    print("\nCheck 2: Checkpoint loading")
    model = None
    try:
        if not CHECKPOINT_PATH.exists():
            record("Checkpoint loads", False, f"File not found: {CHECKPOINT_PATH}")
        else:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
            model = ObbyCNN(n_actions=NUM_ACTIONS).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            record(
                "Checkpoint loads into ObbyCNN",
                True,
                f"epoch={checkpoint.get('epoch')}, val_acc={checkpoint.get('val_accuracy', 0):.1%}",
            )
    except Exception as e:
        record("Checkpoint loads", False, str(e))

    # ------------------------------------------------------------------
    # Check 3: Inference speed
    # ------------------------------------------------------------------
    print("\nCheck 3: Inference speed (100 forward passes)")
    if model is not None:
        dummy = torch.randn(1, 4, 84, 84, device=device)
        # Warmup
        for _ in range(5):
            model(dummy)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                model(dummy)
            times.append((time.perf_counter() - start) * 1000)

        mean_ms = np.mean(times)
        record(
            "Inference < 100ms",
            mean_ms < 100,
            f"mean={mean_ms:.1f}ms, max={max(times):.1f}ms",
        )
    else:
        record("Inference speed", False, "No model loaded")

    # ------------------------------------------------------------------
    # Check 4: Val accuracy >= 70%
    # ------------------------------------------------------------------
    print("\nCheck 4: Validation accuracy")
    if model is not None and val_ds is not None and len(val_ds) > 0:
        loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for stacks, actions in loader:
                stacks = stacks.to(device)
                actions = actions.to(device)
                preds = model(stacks).argmax(dim=1)
                correct += (preds == actions).sum().item()
                total += stacks.size(0)

        acc = correct / max(total, 1)
        if acc >= 0.70:
            record("Val accuracy >= 70%", True, f"{acc:.1%}")
        elif acc >= 0.60:
            record_warn("Val accuracy 60-70%", f"{acc:.1%} — consider more data")
        else:
            record("Val accuracy >= 70%", False, f"{acc:.1%}")
    else:
        record("Val accuracy", False, "No model or dataset available")

    # ------------------------------------------------------------------
    # Check 5: Action coverage (>= 4 distinct actions predicted on val)
    # ------------------------------------------------------------------
    print("\nCheck 5: Action coverage")
    if model is not None and val_ds is not None and len(val_ds) > 0:
        all_preds = []
        loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
        with torch.no_grad():
            for stacks, _ in loader:
                stacks = stacks.to(device)
                preds = model(stacks).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())

        distinct = len(set(all_preds))
        pred_counts = np.bincount(all_preds, minlength=NUM_ACTIONS)
        detail_parts = [f"{ACTION_NAMES[i]}={pred_counts[i]}" for i in range(NUM_ACTIONS)]
        record(
            "Action coverage >= 4 distinct",
            distinct >= 4,
            f"{distinct} distinct — {', '.join(detail_parts)}",
        )
    else:
        record("Action coverage", False, "No model or dataset available")

    # ------------------------------------------------------------------
    # Check 6: Live inference (informational)
    # ------------------------------------------------------------------
    print("\nCheck 6: Live inference (informational)")
    if model is not None:
        try:
            import mss

            with mss.mss() as sct:
                monitor = sct.monitors[1]
                print("  Capturing 10 frames from screen...")
                for i in range(10):
                    raw = sct.grab(monitor)
                    frame_bgr = np.array(raw)[:, :, :3]
                    processed = preprocess_frame(frame_bgr)
                    # Build a simple stack (same frame repeated 4x for demo)
                    stack = np.stack([processed] * 4, axis=0)
                    tensor = torch.from_numpy(stack).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(tensor)
                        pred = logits.argmax(dim=1).item()
                    print(f"    Frame {i+1}: predicted={ACTION_NAMES[pred]} (idx={pred})")
                    time.sleep(0.1)
            print("  [INFO] Live inference complete")
        except Exception as e:
            print(f"  [INFO] Live inference skipped: {e}")
    else:
        print("  [INFO] Live inference skipped: no model loaded")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    passes = sum(1 for _, s, _ in results if s == "PASS")
    fails = sum(1 for _, s, _ in results if s == "FAIL")
    warns = sum(1 for _, s, _ in results if s == "WARN")
    total_checks = passes + fails + warns

    for name, status, detail in results:
        symbol = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}[status]
        print(f"  [{status}] {symbol} {name}")

    print()
    print(f"  {passes}/{total_checks} PASS, {fails} FAIL, {warns} WARN")

    if fails > 0:
        print("\n  ✗ VALIDATION FAILED")
        sys.exit(1)
    elif warns > 0:
        print("\n  ⚠ VALIDATION PASSED WITH WARNINGS")
    else:
        print("\n  ✓ ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
