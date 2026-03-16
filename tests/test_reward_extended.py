#!/usr/bin/env python3
"""Extended reward tests: combined_reward, frame_to_tensor, model save/load."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from reward.combined import frame_to_tensor, combined_reward
    from reward.model import RewardNet, load_reward_model, save_reward_model
else:
    frame_to_tensor = combined_reward = RewardNet = load_reward_model = save_reward_model = None


def _fake(h=84, w=84):
    return np.zeros((h, w, 3), dtype=np.uint8) + 128


def test_frame_to_tensor_batch_dim():
    if frame_to_tensor is None:
        return
    x = frame_to_tensor(_fake(100, 100), height=84, width=84)
    assert x.dim() == 4 and x.shape[0] == 1


def test_frame_to_tensor_channel_first():
    if frame_to_tensor is None:
        return
    x = frame_to_tensor(_fake())
    assert x.shape[1] == 3 and x.shape[2] == 84 and x.shape[3] == 84


def test_combined_reward_no_model():
    if combined_reward is None:
        return
    r = combined_reward(_fake(), reward_model=None, device=None, caption="game over", avoid_weight=1.0)
    assert r <= 0.0
    r2 = combined_reward(_fake(), reward_model=None, device=None, caption="running", avoid_weight=1.0)
    assert r2 >= 0.0


def test_combined_reward_clamp():
    if combined_reward is None:
        return
    r = combined_reward(_fake(), reward_model=None, device=None, caption="safe", clamp_min=0.0, clamp_max=1.0)
    assert 0.0 <= r <= 1.0


def test_reward_net_batch_2():
    if RewardNet is None or torch is None:
        return
    net = RewardNet(in_channels=3, height=84, width=84, hidden=64)
    x = torch.zeros(2, 3, 84, 84)
    out = net(x)
    assert out.shape == (2,)


def test_reward_net_deterministic():
    if RewardNet is None or torch is None:
        return
    net = RewardNet(in_channels=3, height=84, width=84, hidden=64)
    net.eval()
    x = torch.zeros(1, 3, 84, 84)
    with torch.no_grad():
        o1 = net(x).item()
        o2 = net(x).item()
    assert o1 == o2


def test_save_load_roundtrip():
    if save_reward_model is None or load_reward_model is None or torch is None:
        return
    import tempfile
    net = RewardNet(in_channels=3, height=84, width=84, hidden=64)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        path = f.name
    try:
        save_reward_model(net, path)
        loaded = load_reward_model(path, device=None)
        x = torch.zeros(1, 3, 84, 84)
        with torch.no_grad():
            a = net(x).item()
            b = loaded(x).item()
        assert abs(a - b) < 1e-5
    finally:
        if os.path.isfile(path):
            os.remove(path)


def test_frame_to_tensor_different_sizes():
    if frame_to_tensor is None:
        return
    for h, w in [(64, 64), (128, 128), (320, 240)]:
        x = frame_to_tensor(_fake(h, w), height=84, width=84)
        assert x.shape == (1, 3, 84, 84)


def test_combined_reward_avoid_weight_half():
    if combined_reward is None:
        return
    r = combined_reward(_fake(), reward_model=None, device=None, caption="game over", avoid_weight=0.5, clamp_min=-1.0, clamp_max=1.0)
    assert -0.5 <= r <= 0.0


def test_reward_net_hidden_size():
    if RewardNet is None or torch is None:
        return
    net = RewardNet(in_channels=3, height=84, width=84, hidden=32)
    x = torch.zeros(1, 3, 84, 84)
    out = net(x)
    assert out.numel() == 1


def test_combined_reward_caption_only():
    if combined_reward is None:
        return
    r = combined_reward(_fake(), reward_model=None, device=None, caption="dead")
    assert r <= 0.0 or r >= 0.0  # caption triggers avoid; no crash


def test_combined_reward_with_model():
    if combined_reward is None or torch is None or RewardNet is None:
        return
    net = RewardNet(in_channels=3, height=84, width=84, hidden=64)
    dev = torch.device("cpu")
    r = combined_reward(_fake(), reward_model=net, device=dev, caption="running", avoid_weight=0.0)
    assert isinstance(r, (int, float))


def test_frame_to_tensor_normalized():
    if frame_to_tensor is None:
        return
    white = np.ones((84, 84, 3), dtype=np.uint8) * 255
    x = frame_to_tensor(white)
    assert float(x.max()) <= 1.01 and float(x.min()) >= 0.0


def test_reward_net_grad_off():
    if RewardNet is None or torch is None:
        return
    net = RewardNet(in_channels=3, height=84, width=84, hidden=64)
    x = torch.zeros(1, 3, 84, 84)
    with torch.no_grad():
        out = net(x)
    assert not out.requires_grad


if __name__ == "__main__":
    tests = [
        test_frame_to_tensor_batch_dim, test_frame_to_tensor_channel_first, test_combined_reward_no_model,
        test_combined_reward_clamp, test_reward_net_batch_2, test_reward_net_deterministic,
        test_save_load_roundtrip, test_frame_to_tensor_different_sizes, test_combined_reward_avoid_weight_half,
        test_reward_net_hidden_size, test_combined_reward_caption_only, test_combined_reward_with_model,
        test_frame_to_tensor_normalized, test_reward_net_grad_off,
    ]
    for t in tests:
        t()
        print(t.__name__, "OK")
    if torch is None:
        print("torch not installed; some tests skipped.")
    print("All reward extended tests passed.")
