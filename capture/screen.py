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
    On macOS: first try Quartz (CGWindowListCopyWindowInfo) which sees Roblox's windows;
    if that fails, try AppleScript (System Events). Roblox often exposes 0 windows to
    System Events, so Quartz is the reliable path.
    Returns None if not found or not on Mac.
    """
    if sys.platform != "darwin":
        return None
    region = _get_roblox_region_quartz()
    if region is not None:
        return region
    return _get_roblox_region_applescript()


def _get_roblox_region_quartz() -> Optional[dict]:
    """Use Quartz CGWindowListCopyWindowInfo to find Roblox window bounds. Works when System Events sees 0 windows."""
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListExcludeDesktopElements,
            kCGNullWindowID,
            CGDisplayBounds,
            CGMainDisplayID,
        )
        # Get main display height for Y flip (Quartz uses bottom-left origin)
        main_id = CGMainDisplayID()
        bounds = CGDisplayBounds(main_id)
        # Quartz uses bottom-left origin; convert window Y to top-left for mss
        screen_height = int(getattr(bounds.size, "height", 1080))
        wl = CGWindowListCopyWindowInfo(kCGWindowListExcludeDesktopElements, kCGNullWindowID)
        best = None
        best_area = 0
        for w in wl:
            owner = (w.get("kCGWindowOwnerName") or "")
            if "roblox" not in owner.lower():
                continue
            b = w.get("kCGWindowBounds")
            if not b:
                continue
            try:
                x = int(b.get("X", 0))
                y = int(b.get("Y", 0))
                width = int(b.get("Width", 0))
                height = int(b.get("Height", 0))
            except (TypeError, ValueError):
                continue
            # Skip tiny windows (menu bar, etc.)
            if width < 100 or height < 100:
                continue
            area = width * height
            if area > best_area:
                best_area = area
                # Convert from bottom-left origin to top-left (for mss)
                top = screen_height - y - height
                best = {"left": x, "top": top, "width": width, "height": height}
        return best
    except Exception:
        return None


def _get_roblox_region_applescript() -> Optional[dict]:
    """Fallback: AppleScript System Events (often 0 windows for Roblox)."""
    try:
        import subprocess
        for process_name in ("Roblox", "RobloxPlayer"):
            script = f'''
            tell application "System Events"
                set wins to every window of every process whose name contains "{process_name}"
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
            if out.returncode == 0 and out.stdout.strip():
                parts = [int(x.strip()) for x in out.stdout.strip().split(",")]
                if len(parts) == 4:
                    left, top, width, height = parts
                    return {"left": left, "top": top, "width": width, "height": height}
    except Exception:
        pass
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


def look_camera(look_dx: int, duration_ms: int, region: Optional[dict] = None) -> None:
    """
    Rotate the Roblox camera by posting Quartz kCGEventRightMouseDragged events
    with explicit kCGMouseEventDeltaX values.

    This is the ONLY method that works reliably in Roblox on macOS — the game
    reads raw delta fields from Quartz events, not absolute cursor positions.

    look_dx > 0  → look right
    look_dx < 0  → look left
    duration_ms  → spread the drag over this many ms (smoother = more realistic)
    """
    if sys.platform != "darwin":
        # Fallback for non-Mac: pyautogui drag
        try:
            import pyautogui
            if region:
                cx = region["left"] + region["width"] // 2
                cy = region["top"] + region["height"] // 2
                pyautogui.moveTo(cx, cy)
            pyautogui.dragRel(look_dx, 0, button="right", duration=duration_ms / 1000.0)
        except Exception:
            pass
        return

    try:
        import Quartz
    except ImportError:
        return

    if region:
        cx = float(region["left"] + region["width"]  // 2)
        cy = float(region["top"]  + region["height"] // 2)
    else:
        cx, cy = 960.0, 540.0

    steps    = max(20, abs(look_dx) // 6)
    dx_step  = look_dx / steps
    step_dur = duration_ms / 1000.0 / steps

    x, y = cx, cy

    # RightMouseDown
    evt = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventRightMouseDown, Quartz.CGPoint(x, y), 1
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)
    time.sleep(0.08)

    # Dragged events — delta fields are what Roblox actually reads
    for _ in range(steps):
        x += dx_step
        evt = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventRightMouseDragged, Quartz.CGPoint(x, y), 1
        )
        Quartz.CGEventSetIntegerValueField(evt, Quartz.kCGMouseEventDeltaX, int(round(dx_step)))
        Quartz.CGEventSetIntegerValueField(evt, Quartz.kCGMouseEventDeltaY, 0)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)
        time.sleep(step_dur)

    # RightMouseUp
    evt = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventRightMouseUp, Quartz.CGPoint(x, y), 0
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)
    time.sleep(0.05)


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
