"""
Track keyboard state for reward data collection.
Uses pynput to record which keys are currently held; used to label "active" vs "idle" frames.
Also keeps a rolling log of recent key events for user-habit summary (USER_PATTERN for Scout).
On macOS, grant Accessibility permission if you need to capture keys while another app is focused.
"""

import threading
import time
from collections import deque
from typing import Set, Optional

_keys_down: Set[str] = set()
_lock = threading.Lock()
_last_key_time: Optional[float] = None
# Rolling log of (timestamp, key_name) for user-activity summary (max ~2 min at 10 events/s)
_activity_log: deque = deque(maxlen=1200)


def _on_press(key):
    global _last_key_time
    try:
        k = key.char if hasattr(key, "char") and key.char else key.name
    except Exception:
        k = getattr(key, "name", str(key))
    with _lock:
        _keys_down.add(k)
        _last_key_time = time.perf_counter()
        _activity_log.append((_last_key_time, k))


def _on_release(key):
    try:
        k = key.char if hasattr(key, "char") and key.char else key.name
    except Exception:
        k = getattr(key, "name", str(key))
    with _lock:
        _keys_down.discard(k)
        _last_key_time = time.perf_counter()
        _activity_log.append((_last_key_time, k))


def get_current_keys() -> Set[str]:
    """Return a copy of the set of keys currently held."""
    with _lock:
        return set(_keys_down)


def get_last_key_time() -> Optional[float]:
    """Time of last key press or release (for active vs idle labeling)."""
    with _lock:
        return _last_key_time


def is_active(active_window_seconds: float) -> bool:
    """True if user pressed or released a key in the last active_window_seconds."""
    t = get_last_key_time()
    if t is None:
        return False
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
