#!/usr/bin/env python3
"""Offline tests for reward model and frame_to_tensor. No Roblox."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from reward.combined import frame_to_tensor
    from reward.model import RewardNet
else:
    frame_to_tensor = None
    RewardNet = None


def _fake_frame(h=84, w=84):
    return np.zeros((h, w, 3), dtype=np.uint8) + 128


def test_frame_to_tensor_shape():
    if frame_to_tensor is None:
        return  # skip when no torch
    frame = _fake_frame(100, 200)
    x = frame_to_tensor(frame, height=84, width=84)
    assert x.shape == (1, 3, 84, 84)
    assert float(x.min()) >= 0 and float(x.max()) <= 1.01


def test_frame_to_tensor_values():
    if frame_to_tensor is None:
        return
    frame = np.ones((84, 84, 3), dtype=np.uint8) * 255
    x = frame_to_tensor(frame)
    assert x.shape == (1, 3, 84, 84)
    assert float(x.mean()) > 0.9


def test_reward_net_forward():
    if RewardNet is None or torch is None:
        return
    net = RewardNet(in_channels=3, height=84, width=84, hidden=64)
    x = torch.zeros(2, 3, 84, 84)
    out = net(x)
    assert out.shape == (2,)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()


def test_reward_net_forward_single():
    if RewardNet is None or torch is None:
        return
    net = RewardNet(in_channels=3, height=84, width=84, hidden=64)
    x = torch.zeros(1, 3, 84, 84)
    out = net(x)
    assert out.shape == (1,) or out.dim() == 1
    r = out.item() if out.numel() == 1 else out[0].item()
    assert isinstance(r, (float, np.floating))


def test_reward_pipeline_tensor():
    """frame -> frame_to_tensor -> RewardNet forward (no saved model)."""
    if frame_to_tensor is None or RewardNet is None or torch is None:
        return
    frame = _fake_frame(360, 640)
    x = frame_to_tensor(frame)
    net = RewardNet(in_channels=3, height=84, width=84, hidden=64)
    with torch.no_grad():
        r = net(x)
    r = r.item() if r.numel() == 1 else r[0].item()
    assert isinstance(r, (float, np.floating))


if __name__ == "__main__":
    for name in (
        "test_frame_to_tensor_shape",
        "test_frame_to_tensor_values",
        "test_reward_net_forward",
        "test_reward_net_forward_single",
        "test_reward_pipeline_tensor",
    ):
        fn = globals()[name]
        fn()
        print(name, "OK")
    if torch is None:
        print("torch not installed; reward tests skipped.")
    else:
        print("All reward offline tests passed.")
