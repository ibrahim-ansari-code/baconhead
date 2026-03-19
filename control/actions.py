"""
control/actions.py — Discrete action space and pynput input emulation.

Public interface:
    ACTION_NAMES      : list[str]  — human-readable names for each action index
    NUM_ACTIONS       : int        — size of action space (6)
    execute_action(i) : None       — send the input for action index i
    set_camera_angle(): None       — one-time camera setup at episode start
    release_all()     : None       — safety release of all held keys/buttons

Camera is third-person, fixed at episode start. Camera rotation is not part
of the action space — set_camera_angle() is called once and never again.
"""

from __future__ import annotations

import logging
import time

from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Controllers (module-level singletons)
# ---------------------------------------------------------------------------

keyboard = KeyboardController()
mouse = MouseController()

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

NUM_ACTIONS = 6

ACTION_NAMES = [
    "forward",         # 0: W held
    "left",            # 1: A held
    "right",           # 2: D held
    "jump",            # 3: Space tap
    "forward_jump",    # 4: W held + Space tap
    "idle",            # 5: no input
]

# Key hold duration per step (seconds). At 5 fps each step is 200ms.
HOLD_DURATION = 0.200


def execute_action(action: int) -> None:
    """Execute a discrete action by index. Blocks for HOLD_DURATION."""
    if action == 0:
        _hold_key("w")
    elif action == 1:
        _hold_key("a")
    elif action == 2:
        _hold_key("d")
    elif action == 3:
        _tap_key(Key.space)
    elif action == 4:
        _hold_and_tap("w", Key.space)
    elif action == 5:
        time.sleep(HOLD_DURATION)
    else:
        log.warning("Unknown action index %d — treating as idle", action)
        time.sleep(HOLD_DURATION)


# ---------------------------------------------------------------------------
# Camera control
# ---------------------------------------------------------------------------

def set_camera_angle(
    window_x: int = 0,
    window_y: int = 0,
    window_w: int = 1920,
    window_h: int = 1080,
) -> None:
    """
    One-time camera setup at episode start.
    Moves mouse to game center, right-click drags to pitch down slightly.
    """
    cx = window_x + window_w // 2
    cy = window_y + window_h // 2
    mouse.position = (cx, cy)
    time.sleep(0.1)
    mouse.click(Button.left)   # give Roblox window focus
    time.sleep(0.1)
    mouse.press(Button.right)
    time.sleep(0.3)            # hold long enough for Roblox to enter drag mode
    mouse.move(0, -60)         # move up → pitches camera down ~25°
    time.sleep(0.1)
    mouse.release(Button.right)
    log.info("Camera angle set (center=%d,%d)", cx, cy)


# ---------------------------------------------------------------------------
# Input safety
# ---------------------------------------------------------------------------

def release_all() -> None:
    """Release every key and mouse button the agent might be holding."""
    end_action()
    mouse.release(Button.right)
    log.debug("All inputs released")


# ---------------------------------------------------------------------------
# Non-blocking action API (for 20fps two-tier agent)
# ---------------------------------------------------------------------------

_current_held_keys: set = set()


def begin_action(action: int) -> None:
    """Press keys for action without blocking. Releases previous keys first."""
    end_action()
    if action == 0:      # forward
        keyboard.press("w")
        _current_held_keys.add("w")
    elif action == 1:    # left
        keyboard.press("a")
        _current_held_keys.add("a")
    elif action == 2:    # right
        keyboard.press("d")
        _current_held_keys.add("d")
    elif action == 3:    # jump
        keyboard.press(Key.space)
        _current_held_keys.add(Key.space)
    elif action == 4:    # forward_jump
        keyboard.press("w")
        _current_held_keys.add("w")
        keyboard.press(Key.space)
        _current_held_keys.add(Key.space)
    # action == 5 (idle): nothing pressed


def end_action() -> None:
    """Release all currently held keys."""
    for key in list(_current_held_keys):
        keyboard.release(key)
    _current_held_keys.clear()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hold_key(key: str) -> None:
    """Press key, hold for HOLD_DURATION, release."""
    keyboard.press(key)
    time.sleep(HOLD_DURATION)
    keyboard.release(key)


def _tap_key(key) -> None:
    """Quick press-release (tap), then wait remainder of HOLD_DURATION."""
    keyboard.press(key)
    time.sleep(0.05)
    keyboard.release(key)
    time.sleep(HOLD_DURATION - 0.05)


def _hold_and_tap(hold_key: str, tap_key) -> None:
    """Hold one key while tapping another, then release both."""
    keyboard.press(hold_key)
    time.sleep(0.05)
    keyboard.press(tap_key)
    time.sleep(0.05)
    keyboard.release(tap_key)
    time.sleep(HOLD_DURATION - 0.10)
    keyboard.release(hold_key)
