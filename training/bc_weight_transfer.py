"""
training/bc_weight_transfer.py — Transfer weights between ObbyCNN and SB3 PPO.

Provides bidirectional weight transfer:
- transfer_bc_to_ppo: ObbyCNN → SB3 PPO policy (warm-start RL from BC)
- export_ppo_to_bc: SB3 PPO policy → ObbyCNN format (export RL-improved weights)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)

# ObbyCNN key → SB3 CnnPolicy key mapping
# SB3 stores 3 copies of the feature extractor: shared, pi (policy), vf (value)
_BC_TO_SB3_CONV_MAP = {
    "conv.0.weight": "cnn.0.weight",
    "conv.0.bias": "cnn.0.bias",
    "conv.2.weight": "cnn.2.weight",
    "conv.2.bias": "cnn.2.bias",
    "conv.4.weight": "cnn.4.weight",
    "conv.4.bias": "cnn.4.bias",
}

_BC_TO_SB3_FC_MAP = {
    "fc.1.weight": "linear.0.weight",
    "fc.1.bias": "linear.0.bias",
}

# Feature extractor prefixes in SB3 policy
_FE_PREFIXES = [
    "features_extractor.",
    "pi_features_extractor.",
    "vf_features_extractor.",
]

# Action head mapping (no value head — leave at Xavier init)
_BC_TO_SB3_ACTION_MAP = {
    "fc.3.weight": "action_net.weight",
    "fc.3.bias": "action_net.bias",
}

# Reverse mapping for export
_SB3_TO_BC_CONV_MAP = {v: k for k, v in _BC_TO_SB3_CONV_MAP.items()}
_SB3_TO_BC_FC_MAP = {v: k for k, v in _BC_TO_SB3_FC_MAP.items()}
_SB3_TO_BC_ACTION_MAP = {v: k for k, v in _BC_TO_SB3_ACTION_MAP.items()}


def transfer_bc_to_ppo(bc_checkpoint_path: str | Path, ppo_model) -> dict:
    """
    Transfer ObbyCNN weights into an SB3 PPO policy.

    Args:
        bc_checkpoint_path: Path to bc_best.pt checkpoint.
        ppo_model: SB3 PPO model instance.

    Returns:
        Dict with keys:
            - transferred_keys: list of SB3 keys that were updated
            - bc_metadata: dict of non-weight metadata from the BC checkpoint
    """
    bc_checkpoint_path = Path(bc_checkpoint_path)
    state = torch.load(bc_checkpoint_path, map_location="cpu", weights_only=False)
    bc_sd = state.get("model_state_dict", state)

    # Extract metadata
    bc_metadata = {k: v for k, v in state.items() if k != "model_state_dict"}

    policy_sd = ppo_model.policy.state_dict()
    policy_keys = set(policy_sd.keys())
    transferred_keys = []

    # Transfer conv layers to all 3 feature extractor copies
    for bc_key, sb3_suffix in _BC_TO_SB3_CONV_MAP.items():
        if bc_key not in bc_sd:
            log.warning("BC key %s not found in checkpoint", bc_key)
            continue
        bc_tensor = bc_sd[bc_key]
        for prefix in _FE_PREFIXES:
            sb3_key = prefix + sb3_suffix
            if sb3_key in policy_keys:
                assert policy_sd[sb3_key].shape == bc_tensor.shape, (
                    f"Shape mismatch: {sb3_key} {policy_sd[sb3_key].shape} vs {bc_key} {bc_tensor.shape}"
                )
                policy_sd[sb3_key] = bc_tensor.clone()
                transferred_keys.append(sb3_key)
            else:
                log.warning("SB3 key %s not found in policy", sb3_key)

    # Transfer FC (linear) layer to all 3 feature extractor copies
    for bc_key, sb3_suffix in _BC_TO_SB3_FC_MAP.items():
        if bc_key not in bc_sd:
            log.warning("BC key %s not found in checkpoint", bc_key)
            continue
        bc_tensor = bc_sd[bc_key]
        for prefix in _FE_PREFIXES:
            sb3_key = prefix + sb3_suffix
            if sb3_key in policy_keys:
                assert policy_sd[sb3_key].shape == bc_tensor.shape, (
                    f"Shape mismatch: {sb3_key} {policy_sd[sb3_key].shape} vs {bc_key} {bc_tensor.shape}"
                )
                policy_sd[sb3_key] = bc_tensor.clone()
                transferred_keys.append(sb3_key)
            else:
                log.warning("SB3 key %s not found in policy", sb3_key)

    # Transfer action head
    for bc_key, sb3_key in _BC_TO_SB3_ACTION_MAP.items():
        if bc_key not in bc_sd:
            log.warning("BC key %s not found in checkpoint", bc_key)
            continue
        bc_tensor = bc_sd[bc_key]
        if sb3_key in policy_keys:
            assert policy_sd[sb3_key].shape == bc_tensor.shape, (
                f"Shape mismatch: {sb3_key} {policy_sd[sb3_key].shape} vs {bc_key} {bc_tensor.shape}"
            )
            policy_sd[sb3_key] = bc_tensor.clone()
            transferred_keys.append(sb3_key)
        else:
            log.warning("SB3 key %s not found in policy", sb3_key)

    # If any key was not found, dump all policy keys for debugging
    if len(transferred_keys) == 0:
        log.error("No keys transferred! Available policy keys: %s", sorted(policy_keys))

    # Load updated state dict
    ppo_model.policy.load_state_dict(policy_sd)

    log.info("Transferred %d keys from BC to PPO", len(transferred_keys))
    return {"transferred_keys": transferred_keys, "bc_metadata": bc_metadata}


def export_ppo_to_bc(ppo_model, output_path: str | Path, metadata: dict | None = None) -> None:
    """
    Export SB3 PPO policy weights back to ObbyCNN checkpoint format.

    Reads from the shared features_extractor.* and action_net.* keys.
    Saves in the same format as train_bc.py: {"model_state_dict": ..., ...}

    Args:
        ppo_model: SB3 PPO model instance.
        output_path: Path to save the checkpoint (e.g. checkpoints/bc_best.pt).
        metadata: Additional metadata dict to include in checkpoint.
    """
    output_path = Path(output_path)
    policy_sd = ppo_model.policy.state_dict()
    bc_sd = {}

    # Transfer conv layers (from shared features_extractor)
    prefix = "features_extractor."
    for sb3_suffix, bc_key in _SB3_TO_BC_CONV_MAP.items():
        sb3_key = prefix + sb3_suffix
        if sb3_key in policy_sd:
            bc_sd[bc_key] = policy_sd[sb3_key].clone()
        else:
            log.warning("SB3 key %s not found during export", sb3_key)

    # Transfer FC layer
    for sb3_suffix, bc_key in _SB3_TO_BC_FC_MAP.items():
        sb3_key = prefix + sb3_suffix
        if sb3_key in policy_sd:
            bc_sd[bc_key] = policy_sd[sb3_key].clone()
        else:
            log.warning("SB3 key %s not found during export", sb3_key)

    # Transfer action head
    for sb3_key, bc_key in _SB3_TO_BC_ACTION_MAP.items():
        if sb3_key in policy_sd:
            bc_sd[bc_key] = policy_sd[sb3_key].clone()
        else:
            log.warning("SB3 key %s not found during export", sb3_key)

    checkpoint = {"model_state_dict": bc_sd}
    if metadata:
        checkpoint.update(metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    log.info("Exported PPO weights to BC format: %s (%d keys)", output_path, len(bc_sd))
