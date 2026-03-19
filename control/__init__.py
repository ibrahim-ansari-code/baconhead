"""control — Input emulation, action space, and main loop."""

from control.actions import (
    ACTION_NAMES,
    NUM_ACTIONS,
    execute_action,
    release_all,
    set_camera_angle,
)
from control.loop import run_loop

__all__ = [
    "ACTION_NAMES",
    "NUM_ACTIONS",
    "execute_action",
    "release_all",
    "run_loop",
    "set_camera_angle",
]
