"""
vision/perception.py — Scene perception from HSV void detection.

Pure function module — no state, no side effects, no imports from capture/.

Public interface:
    compute_scene_state(frame_bgr, void_hsv_lower, void_hsv_upper) -> dict
"""

from __future__ import annotations

import cv2
import numpy as np


def _void_mask(frame_hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Binary mask where 255 = void pixel."""
    return cv2.inRange(frame_hsv, lower, upper)


def _compute_void_ratio(mask: np.ndarray) -> float:
    """Fraction of pixels that are void in the given mask region."""
    return float(np.count_nonzero(mask) / mask.size)


def _compute_edge_proximity(mask: np.ndarray) -> float:
    """Void ratio of the bottom 20% height, center 40% width strip."""
    h, w = mask.shape[:2]
    top = int(h * 0.8)
    left = int(w * 0.3)
    right = int(w * 0.7)
    strip = mask[top:, left:right]
    if strip.size == 0:
        return 0.0
    return float(np.count_nonzero(strip) / strip.size)


def _compute_directional_bias(mask: np.ndarray) -> tuple[float, float, str]:
    """
    Compare non-void ratio in left vs right halves.
    Returns (platform_left, platform_right, direction_bias).
    Uses 0.05 dead zone to prevent jitter.
    """
    h, w = mask.shape[:2]
    mid = w // 2

    left_half = mask[:, :mid]
    right_half = mask[:, mid:]

    # Non-void ratio = fraction of pixels that are NOT void
    platform_left = float(1.0 - np.count_nonzero(left_half) / left_half.size) if left_half.size > 0 else 0.0
    platform_right = float(1.0 - np.count_nonzero(right_half) / right_half.size) if right_half.size > 0 else 0.0

    diff = platform_left - platform_right
    if abs(diff) < 0.05:
        bias = "center"
    elif diff > 0:
        bias = "left"
    else:
        bias = "right"

    return platform_left, platform_right, bias


def compute_scene_state(
    frame_bgr: np.ndarray,
    void_hsv_lower: np.ndarray,
    void_hsv_upper: np.ndarray,
) -> dict:
    """
    Compute perception signals from a single BGR frame.

    Args:
        frame_bgr: Full game frame in BGR format.
        void_hsv_lower: Lower HSV bound for void color (shape (3,), dtype uint8).
        void_hsv_upper: Upper HSV bound for void color (shape (3,), dtype uint8).

    Returns:
        dict with keys: void_ratio, edge_proximity, platform_left,
        platform_right, direction_bias.
    """
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = _void_mask(frame_hsv, void_hsv_lower, void_hsv_upper)

    void_ratio = _compute_void_ratio(mask)
    edge_proximity = _compute_edge_proximity(mask)
    platform_left, platform_right, direction_bias = _compute_directional_bias(mask)

    return {
        "void_ratio": void_ratio,
        "edge_proximity": edge_proximity,
        "platform_left": platform_left,
        "platform_right": platform_right,
        "direction_bias": direction_bias,
    }
