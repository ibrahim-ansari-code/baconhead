"""
vision/preprocess.py — Frame preprocessing pipeline.

Converts raw BGR frames to 84x84 grayscale normalized tensors
ready for CNN input.
"""

import cv2
import numpy as np


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess a raw BGR frame for the CNN.

    Pipeline: grayscale → resize 84×84 → normalize to [0.0, 1.0].

    Args:
        frame_bgr: Raw BGR frame from screen capture (any resolution).

    Returns:
        np.ndarray of shape (84, 84), dtype float32, values in [0.0, 1.0].
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def compute_motion_mask(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold: int = 25,
) -> np.ndarray:
    """
    Binary motion mask via frame differencing.

    Both inputs must be preprocessed (84×84 float32 [0,1]).
    Returns a binary uint8 mask of shape (84, 84).
    """
    # Convert back to uint8 range for absdiff + threshold
    prev_u8 = (prev_frame * 255).astype(np.uint8)
    curr_u8 = (curr_frame * 255).astype(np.uint8)
    diff = cv2.absdiff(prev_u8, curr_u8)
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return binary
