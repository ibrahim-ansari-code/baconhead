"""
Screen capture for Roblox gameplay.
Uses mss for fast capture. Can target a configurable region or try to find the Roblox window (Mac).
"""

import time
import sys
from typing import Optional

import mss
import numpy as np


def get_roblox_region() -> Optional[dict]:
    """
    Try to get the bounding box of the Roblox window.
    On macOS uses AppleScript to find a window whose name contains "Roblox".
    Returns None if not found or not on Mac.
    """
    if sys.platform != "darwin":
        return None
    try:
        import subprocess
        # Get frontmost window or search for Roblox
        script = '''
        tell application "System Events"
            set wins to every window of every process whose name contains "Roblox"
            if (count of wins) > 0 then
                set w to item 1 of wins
                set b to position of w
                set s to size of w
                return (item 1 of b) & "," & (item 2 of b) & "," & (item 1 of s) & "," & (item 2 of s)
            end if
        end tell
        '''
        out = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        parts = [int(x.strip()) for x in out.stdout.strip().split(",")]
        if len(parts) != 4:
            return None
        left, top, width, height = parts
        # Retina: AppleScript can return logical coords; we use them as-is and mss will grab that region
        return {"left": left, "top": top, "width": width, "height": height}
    except Exception:
        return None


def focus_roblox() -> bool:
    """
    On macOS: bring the Roblox window to front so key presses go to the game.
    Returns True if we focused (or tried), False if not Mac.
    """
    if sys.platform != "darwin":
        return False
    try:
        import subprocess
        script = '''
        tell application "System Events"
            set procs to every process whose name contains "Roblox"
            if (count of procs) > 0 then
                set frontmost of item 1 of procs to true
                return "ok"
            end if
        end tell
        '''
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=2)
        return out.returncode == 0 and "ok" in (out.stdout or "")
    except Exception:
        return False


def focus_roblox_and_click() -> bool:
    """
    Focus Roblox, then click the center of its window so the game captures keyboard/mouse.
    Many games only accept input after a click inside the window.
    """
    if not focus_roblox():
        return False
    time.sleep(0.2)
    region = get_roblox_region()
    if not region:
        return True  # focus worked, no region for click
    try:
        import pyautogui
        cx = region["left"] + region["width"] // 2
        cy = region["top"] + region["height"] // 2
        pyautogui.click(cx, cy)
        time.sleep(0.15)
        return True
    except Exception:
        return True  # focus already done


def capture_region(
    region: Optional[dict] = None,
    monitor: int = 0,
    sct: Optional[mss.mss.MSS] = None,
) -> np.ndarray:
    """
    Capture a screen region (or full monitor) and return as numpy array (RGB, height x width x 3).
    If region is None, captures the given monitor (default primary).
    """
    own_sct = False
    if sct is None:
        sct = mss.mss()
        own_sct = True
    try:
        if region is not None:
            box = region
        else:
            mon = sct.monitors[monitor]
            box = {
                "left": mon["left"],
                "top": mon["top"],
                "width": mon["width"],
                "height": mon["height"],
            }
        raw = sct.grab(box)
        # mss returns BGRA; convert to RGB
        img = np.array(raw)[:, :, :3][:, :, ::-1]  # BGR -> RGB
        return img
    finally:
        if own_sct:
            sct.close()


def capture_loop(
    region: Optional[dict] = None,
    monitor: int = 0,
    fps: float = 10.0,
    callback=None,
    stop_event=None,
):
    """
    Run a capture loop at the given FPS.
    - region: optional {left, top, width, height}; if None uses monitor.
    - monitor: monitor index when region is None.
    - fps: target capture rate.
    - callback: called as callback(frame: np.ndarray, timestamp: float) each frame. If None, frames are discarded (useful for FPS test).
    - stop_event: optional threading.Event or similar; when set, loop exits.
    """
    interval = 1.0 / fps
    sct = mss.mss()
    try:
        if region is None:
            mon = sct.monitors[monitor]
            region = {
                "left": mon["left"],
                "top": mon["top"],
                "width": mon["width"],
                "height": mon["height"],
            }
        next_ts = time.perf_counter()
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            now = time.perf_counter()
            if now >= next_ts:
                frame = capture_region(region=region, sct=sct)
                if callback is not None:
                    callback(frame, now)
                next_ts = next_ts + interval
                if next_ts < now:
                    next_ts = now + interval
            else:
                time.sleep(min(interval * 0.5, next_ts - now))
    finally:
        sct.close()
