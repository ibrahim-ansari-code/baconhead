"""
Quick test: try several camera-rotation methods and save before/after screenshots.
Run while Roblox is open (Brookhaven).

Methods tested:
  1. Quartz CGEvent - kCGEventRightMouseDragged with delta fields set
  2. Quartz CGEvent - kCGEventRightMouseDragged without explicit delta
  3. pyautogui dragRel
  4. pyautogui mouseDown + moveRel loop

Usage:
    python test_look.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import mss
import numpy as np
from PIL import Image
from capture.screen import get_roblox_region, capture_region, focus_roblox_and_click

OUT_DIR = "episode_data/calib_screenshots"
os.makedirs(OUT_DIR, exist_ok=True)


def save(raw, name):
    Image.fromarray(raw.astype(np.uint8)).save(f"{OUT_DIR}/{name}.png")
    print(f"  saved → {OUT_DIR}/{name}.png", flush=True)


# ── Method 1: Quartz with delta fields ──────────────────────────────────────

def look_quartz_delta(look_dx: int, duration_ms: int, cx: float, cy: float):
    """Post kCGEventRightMouseDragged events with explicit DeltaX set."""
    import Quartz
    steps     = max(20, abs(look_dx) // 6)
    dx_step   = look_dx / steps
    step_dur  = duration_ms / 1000.0 / steps
    x, y      = cx, cy

    evt = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventRightMouseDown, Quartz.CGPoint(x, y), 1
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)
    time.sleep(0.08)

    for _ in range(steps):
        x += dx_step
        evt = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventRightMouseDragged, Quartz.CGPoint(x, y), 1
        )
        Quartz.CGEventSetIntegerValueField(evt, Quartz.kCGMouseEventDeltaX, int(round(dx_step)))
        Quartz.CGEventSetIntegerValueField(evt, Quartz.kCGMouseEventDeltaY, 0)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)
        time.sleep(step_dur)

    evt = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventRightMouseUp, Quartz.CGPoint(x, y), 0
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)
    time.sleep(0.05)


# ── Method 2: Quartz WITHOUT explicit delta ──────────────────────────────────

def look_quartz_nodelta(look_dx: int, duration_ms: int, cx: float, cy: float):
    """Same as above but don't set delta fields — let Quartz compute them."""
    import Quartz
    steps     = max(20, abs(look_dx) // 6)
    dx_step   = look_dx / steps
    step_dur  = duration_ms / 1000.0 / steps
    x, y      = cx, cy

    evt = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventRightMouseDown, Quartz.CGPoint(x, y), 1
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)
    time.sleep(0.08)

    for _ in range(steps):
        x += dx_step
        evt = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventRightMouseDragged, Quartz.CGPoint(x, y), 1
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)
        time.sleep(step_dur)

    evt = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventRightMouseUp, Quartz.CGPoint(x, y), 0
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)
    time.sleep(0.05)


# ── Method 3: pyautogui dragRel ──────────────────────────────────────────────

def look_pyautogui_drag(look_dx: int, duration_ms: int, cx: float, cy: float):
    import pyautogui
    pyautogui.moveTo(cx, cy)
    time.sleep(0.05)
    pyautogui.dragRel(look_dx, 0, button="right", duration=duration_ms / 1000.0)
    time.sleep(0.05)


# ── Method 4: pyautogui mouseDown + moveRel ──────────────────────────────────

def look_pyautogui_manual(look_dx: int, duration_ms: int, cx: float, cy: float):
    import pyautogui
    pyautogui.moveTo(cx, cy)
    time.sleep(0.05)
    steps    = max(20, abs(look_dx) // 6)
    dx_step  = look_dx // steps
    step_dur = duration_ms / 1000.0 / steps
    pyautogui.mouseDown(button="right")
    try:
        for _ in range(steps):
            pyautogui.moveRel(dx_step, 0)
            time.sleep(step_dur)
    finally:
        pyautogui.mouseUp(button="right")
    time.sleep(0.05)


# ── Main ─────────────────────────────────────────────────────────────────────

def test_method(name, fn, look_dx, duration_ms, cx, cy, sct, region):
    print(f"\n── {name} (dx={look_dx}, {duration_ms}ms) ──", flush=True)
    raw = capture_region(region=region, sct=sct)
    save(raw, f"test_{name}_before")

    focus_roblox_and_click()
    time.sleep(0.6)

    fn(look_dx, duration_ms, cx, cy)
    time.sleep(0.4)

    raw = capture_region(region=region, sct=sct)
    save(raw, f"test_{name}_after_right")

    # Reset
    fn(-look_dx, duration_ms, cx, cy)
    time.sleep(0.6)

    raw = capture_region(region=region, sct=sct)
    save(raw, f"test_{name}_after_reset")


def main():
    region = get_roblox_region()
    print(f"Region: {region}", flush=True)

    sct = mss.mss()
    cx = float(region["left"] + region["width"]  // 2) if region else 960.0
    cy = float(region["top"]  + region["height"] // 2) if region else 540.0

    LOOK_DX   = 280
    DUR_MS    = 700

    focus_roblox_and_click()
    time.sleep(1.5)

    methods = [
        ("quartz_delta",   look_quartz_delta),
        ("quartz_nodelta", look_quartz_nodelta),
        ("pyautogui_drag", look_pyautogui_drag),
        ("pyautogui_manual", look_pyautogui_manual),
    ]

    for name, fn in methods:
        test_method(name, fn, LOOK_DX, DUR_MS, cx, cy, sct, region)
        time.sleep(2.5)   # pause between methods

    sct.close()
    print("\nAll done — check episode_data/calib_screenshots/test_*.png", flush=True)


if __name__ == "__main__":
    main()
