"""
vision/stacker.py — Frame stacking with a fixed-length deque.

Maintains the last 4 preprocessed frames and stacks them into a
single (4, 84, 84) tensor for CNN input.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class FrameStacker:
    """
    Stacks the last N preprocessed frames along axis 0.

    At startup the buffer is filled with copies of the first frame
    so the CNN always receives a full (N, 84, 84) observation.
    """

    def __init__(self, stack_size: int = 4) -> None:
        self._stack_size = stack_size
        self._buffer: deque[np.ndarray] = deque(maxlen=stack_size)

    def reset(self, initial_frame: np.ndarray) -> np.ndarray:
        """
        Clear the buffer and fill it with copies of *initial_frame*.

        Args:
            initial_frame: Preprocessed frame, shape (84, 84), float32.

        Returns:
            Stacked observation, shape (4, 84, 84), float32.
        """
        self._buffer.clear()
        for _ in range(self._stack_size):
            self._buffer.append(initial_frame)
        return self.get()

    def push(self, frame: np.ndarray) -> np.ndarray:
        """
        Append a new frame, evicting the oldest, and return the stack.

        Args:
            frame: Preprocessed frame, shape (84, 84), float32.

        Returns:
            Stacked observation, shape (4, 84, 84), float32.
        """
        self._buffer.append(frame)
        return self.get()

    def get(self) -> np.ndarray:
        """Return the current stacked frames as shape (4, 84, 84)."""
        return np.stack(list(self._buffer), axis=0)
