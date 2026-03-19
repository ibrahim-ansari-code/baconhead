"""
training/bc_dataset.py — PyTorch Dataset for behavioral cloning.

Loads demonstration runs from demos/, splits by run (not frame),
and returns (4-frame stack, action) pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BCDataset(Dataset):
    """
    Behavioral cloning dataset that loads recorded demos and produces
    4-frame stacks paired with action labels.

    Splits by run (not by frame) to prevent data leakage between
    train and val sets.

    Args:
        demos_dir: Path to demos/ directory containing run_NNN/ subdirs.
        stack_size: Number of frames to stack (default 4).
        split: 'train' or 'val'.
        val_ratio: Fraction of runs reserved for validation.
        seed: Random seed for reproducible train/val split.
        augment: If True, apply brightness jitter (train only).
        drop_idle: If True, remove all frames labeled idle (action 5) after loading.
    """

    def __init__(
        self,
        demos_dir: str | Path = "demos",
        stack_size: int = 4,
        split: str = "train",
        val_ratio: float = 0.15,
        seed: int = 42,
        augment: bool = False,
        drop_idle: bool = False,
    ) -> None:
        super().__init__()
        self.stack_size = stack_size
        self.split = split
        self.augment = augment and (split == "train")

        demos_path = Path(demos_dir)
        if not demos_path.exists():
            raise FileNotFoundError(f"Demos directory not found: {demos_path}")

        # Discover all valid runs (must have frames.npz and actions.npy)
        all_runs = sorted(
            d for d in demos_path.iterdir()
            if d.is_dir() and (d / "frames.npz").exists() and (d / "actions.npy").exists()
        )

        if len(all_runs) == 0:
            raise FileNotFoundError(f"No valid runs found in {demos_path}")

        # Split by run
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_runs))
        n_val = max(1, int(len(all_runs) * val_ratio))
        val_indices = set(indices[:n_val])
        train_indices = set(indices[n_val:])

        selected_indices = val_indices if split == "val" else train_indices
        selected_runs = [all_runs[i] for i in sorted(selected_indices)]

        # Load all selected runs into memory
        self._frames: list[np.ndarray] = []  # each: (N, 84, 84) float32
        self._actions: list[np.ndarray] = []  # each: (N,) int
        self._run_offsets: list[tuple[int, int]] = []  # (start_idx, length) per run

        total = 0
        for run_dir in selected_runs:
            data = np.load(run_dir / "frames.npz")
            frames = data["frames"]  # (N, 84, 84)
            actions = np.load(run_dir / "actions.npy")  # (N,)

            assert frames.shape[0] == actions.shape[0], (
                f"Frame/action count mismatch in {run_dir}: "
                f"{frames.shape[0]} vs {actions.shape[0]}"
            )

            self._frames.append(frames)
            self._actions.append(actions)
            self._run_offsets.append((total, frames.shape[0]))
            total += frames.shape[0]

        self._total_samples = total
        self._idle_dropped = 0

        if drop_idle:
            filtered_frames: list[np.ndarray] = []
            filtered_actions: list[np.ndarray] = []
            filtered_offsets: list[tuple[int, int]] = []
            new_total = 0
            for frames, actions in zip(self._frames, self._actions):
                mask = actions != 5
                kept_frames = frames[mask]
                kept_actions = actions[mask]
                self._idle_dropped += int((~mask).sum())
                filtered_frames.append(kept_frames)
                filtered_actions.append(kept_actions)
                filtered_offsets.append((new_total, kept_frames.shape[0]))
                new_total += kept_frames.shape[0]
            self._frames = filtered_frames
            self._actions = filtered_actions
            self._run_offsets = filtered_offsets
            self._total_samples = new_total

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Find which run this index belongs to
        run_idx, frame_t = self._locate(idx)
        frames = self._frames[run_idx]
        action = int(self._actions[run_idx][frame_t])

        # Build 4-frame stack: [t-3, t-2, t-1, t]
        # For early frames, pad by repeating frame 0 (matches FrameStacker behavior)
        stack_indices = [max(0, frame_t - (self.stack_size - 1 - i)) for i in range(self.stack_size)]
        stack = np.stack([frames[j] for j in stack_indices], axis=0)  # (4, 84, 84)

        # Optional brightness jitter
        if self.augment:
            jitter = np.random.uniform(0.9, 1.1)
            stack = np.clip(stack * jitter, 0.0, 1.0)

        return torch.from_numpy(stack.astype(np.float32)), action

    def _locate(self, idx: int) -> tuple[int, int]:
        """Convert flat index to (run_index, frame_within_run)."""
        for run_i, (offset, length) in enumerate(self._run_offsets):
            if idx < offset + length:
                return run_i, idx - offset
        raise IndexError(f"Index {idx} out of range (total={self._total_samples})")

    @property
    def num_runs(self) -> int:
        return len(self._frames)

    @property
    def action_counts(self) -> np.ndarray:
        """Count occurrences of each action across all loaded runs."""
        counts = np.zeros(6, dtype=np.int64)
        for actions in self._actions:
            for a in actions:
                counts[a] += 1
        return counts
