"""
Track keyboard state for reward data collection.
Uses pynput to record which keys are currently held; used to label "active" vs "idle" frames.
Also keeps a rolling log of recent key events for user-habit summary (USER_PATTERN for Scout).
On macOS, grant Accessibility permission if you need to capture keys while another app is focused.
"""

import threading
import time
from collections import deque
from typing import Set, Optional, Tuple

_keys_down: Set[str] = set()
_key_press_times: dict = {}  # key -> time.perf_counter() when pressed
_last_key_time: Optional[float] = None
_last_key_duration: Optional[Tuple[str, float]] = None  # (key_name, duration_ms) on release
_lock = threading.Lock()
# Rolling log of (timestamp, key_name) for user-activity summary (max ~2 min at 10 events/s)
_activity_log: deque = deque(maxlen=1200)

# Set this True while the bot is executing an action so pynput events from
# pyautogui keypresses are ignored for idle detection purposes.
_bot_pressing: bool = False


def set_bot_pressing(active: bool) -> None:
    """Call with True before bot executes an action, False after. Thread-safe."""
    global _bot_pressing
    with _lock:
        _bot_pressing = active


def _on_press(key):
    global _last_key_time
    try:
        k = key.char if hasattr(key, "char") and key.char else key.name
    except Exception:
        k = getattr(key, "name", str(key))
    with _lock:
        if not _bot_pressing:
            now = time.perf_counter()
            _keys_down.add(k)
            _key_press_times[k] = now
            _last_key_time = now
            _activity_log.append((_last_key_time, k))


def _on_release(key):
    global _last_key_time, _last_key_duration
    try:
        k = key.char if hasattr(key, "char") and key.char else key.name
    except Exception:
        k = getattr(key, "name", str(key))
    with _lock:
        now = time.perf_counter()
        if not _bot_pressing:
            # Only remove from _keys_down if WE added it (i.e. not bot-pressed)
            press_time = _key_press_times.pop(k, now)
            duration_ms = (now - press_time) * 1000.0
            _last_key_duration = (k, duration_ms)
            _keys_down.discard(k)
            _last_key_time = now
            _activity_log.append((_last_key_time, k))
        else:
            # Bot released a key — don't touch _keys_down or _last_key_time,
            # but do clean up _key_press_times if the bot somehow added an entry
            _key_press_times.pop(k, None)


def get_current_keys() -> Set[str]:
    """Return a copy of the set of keys currently held."""
    with _lock:
        return set(_keys_down)


def get_last_key_time() -> Optional[float]:
    """Time of last key press or release (for active vs idle labeling)."""
    with _lock:
        return _last_key_time


def get_last_key_duration() -> Optional[Tuple[str, float]]:
    """(key_name, duration_ms) for the most recently released key. None if no release yet."""
    with _lock:
        return _last_key_duration


def _system_idle_seconds() -> float:
    """
    Return how many seconds since the user last touched the keyboard or mouse,
    using macOS CGEventSource. This works regardless of which app is focused
    and does NOT require Accessibility permission.

    Falls back to pynput-based tracking if Quartz is unavailable.
    """
    try:
        from Quartz import (
            CGEventSourceSecondsSinceLastEventType,
            kCGEventSourceStateHIDSystemState,
            kCGAnyInputEventType,
        )
        return CGEventSourceSecondsSinceLastEventType(
            kCGEventSourceStateHIDSystemState, kCGAnyInputEventType
        )
    except Exception:
        return None


def is_active(active_window_seconds: float) -> bool:
    """
    True if the user is currently active (touching keyboard or mouse recently).

    Primary method: macOS CGEventSource system idle time — works even when
    Roblox has focus and pynput can't see keypresses. No Accessibility permission needed.

    Fallback: pynput _keys_down / _last_key_time (used if Quartz unavailable).

    Bot's own pyautogui events are excluded via set_bot_pressing().
    """
    # Skip system idle check while bot is pressing — its own events would reset
    # the system idle timer and make it look like the user is active
    with _lock:
        bot_is_pressing = _bot_pressing

    if not bot_is_pressing:
        idle = _system_idle_seconds()
        if idle is not None:
            return idle < active_window_seconds

    # Fallback: pynput-based tracking
    with _lock:
        if _keys_down:
            return True
        t = _last_key_time
    if t is None:
        return True
    return (time.perf_counter() - t) < active_window_seconds


def get_recent_activity_summary(seconds: float = 120.0, max_keys: int = 40) -> str:
    """Return a short summary of recent key activity for Scout USER_PATTERN (e.g. 'Recent keys: w,a,s,d,e')."""
    with _lock:
        now = time.perf_counter()
        recent = [(t, k) for t, k in _activity_log if (now - t) <= seconds]
    if not recent:
        return ""
    keys = [k for _, k in recent[-max_keys:]]
    return "Recent keys (last {:.0f}s): {}".format(seconds, ",".join(keys[:max_keys]))


def start_listener():
    """Start the keyboard listener in a daemon thread. Call once at start of collection."""
    from pynput import keyboard
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.daemon = True
    listener.start()
    return listener
