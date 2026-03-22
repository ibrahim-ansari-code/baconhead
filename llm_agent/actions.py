"""
Action execution for the Roblox bot.

Handles keyboard holds, Quartz mouse camera rotation, and combo actions
(e.g. W+space, look_left+W) with millisecond-accurate timing.
"""

import time
import threading

from llm_agent.physics import LOOK_PX_PER_MS


KEY_MAP = {
    "w": "w",
    "a": "a",
    "s": "s",
    "d": "d",
    "space": "space",
}


def _normalize_key(part: str):
    p = part.strip().lower()
    return KEY_MAP.get(p, p) if p in KEY_MAP else None


def execute_action_ms(action: str, duration_ms: int = 5000) -> None:
    """
    Execute an action for duration_ms milliseconds.

    Supports:
      - Single keys: W, A, S, D, space
      - Combos: W+space (hold both simultaneously)
      - Camera: look_left, look_right (Quartz mouse drag)
      - Mixed: look_right+W (camera + movement at same time)
      - none / None: just sleep
    """
    duration_ms = max(1, int(duration_ms))

    if action is None:
        time.sleep(duration_ms / 1000.0)
        return

    action_str = action.strip()
    if not action_str or action_str.lower() == "none":
        time.sleep(duration_ms / 1000.0)
        return

    # Parse action into look + key components
    parts = [p.strip() for p in action_str.split("+") if p.strip()]
    look_part = None
    key_parts = []

    for p in parts:
        pl = p.lower()
        if pl in ("look_left", "look_right", "look_up", "look_down"):
            look_part = pl
        else:
            k = _normalize_key(p) or (pl if pl in KEY_MAP else None)
            if k:
                key_parts.append(k)

    import pyautogui
    from capture.screen import look_camera

    try:
        if look_part and key_parts:
            # Simultaneous look + movement
            look_ms = min(duration_ms, 3000)
            look_px = int(LOOK_PX_PER_MS * look_ms)
            look_dx_val = (
                look_px
                if look_part == "look_right"
                else -look_px
                if look_part == "look_left"
                else 0
            )
            done = threading.Event()

            def _hold():
                for key in key_parts:
                    pyautogui.keyDown(key)
                done.wait(timeout=duration_ms / 1000.0 + 0.5)
                for key in key_parts:
                    pyautogui.keyUp(key)

            t = threading.Thread(target=_hold, daemon=True)
            t.start()
            look_camera(look_dx_val, look_ms)
            time.sleep(max(0, (duration_ms - look_ms) / 1000.0))
            done.set()
            t.join(timeout=0.5)

        elif look_part:
            # Camera rotation only
            look_ms = min(duration_ms, 3000)
            look_px = int(LOOK_PX_PER_MS * look_ms)
            look_dx_val = (
                look_px
                if look_part == "look_right"
                else -look_px
                if look_part == "look_left"
                else 0
            )
            look_camera(look_dx_val, look_ms)
            time.sleep(max(0, (duration_ms - look_ms) / 1000.0))

        elif key_parts:
            # Keyboard only
            for key in key_parts:
                pyautogui.keyDown(key)
            try:
                time.sleep(duration_ms / 1000.0)
            finally:
                for key in key_parts:
                    pyautogui.keyUp(key)

        else:
            time.sleep(duration_ms / 1000.0)

    except Exception:
        # Safety: release everything on error
        try:
            pyautogui.mouseUp(button="right")
            for key in ("w", "a", "s", "d", "space"):
                try:
                    pyautogui.keyUp(key)
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(duration_ms / 1000.0)
