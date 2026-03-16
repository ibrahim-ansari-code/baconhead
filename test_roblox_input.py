#!/usr/bin/env python3
"""
Test that Roblox actually receives input after we focus it.
Run with Roblox open and in-game. You should see:
  - Roblox window come to front, then a click in the center (so game captures input)
  - Character move forward (W), then back (S), then strafe (A then D), then jump (space)
  - Camera move (mouse moved right then left)
"""
import sys
import time

from capture.screen import focus_roblox_and_click


def send_key(key_name: str, duration_s: float):
    """Press and hold key for duration_s seconds. Uses pyautogui (reaches focused app on Mac)."""
    key_map = {"W": "w", "A": "a", "S": "s", "D": "d", "space": "space"}
    k = key_map.get(key_name, key_name.lower())
    try:
        import pyautogui
        key = "space" if k == "space" else k
        pyautogui.keyDown(key)
        time.sleep(duration_s)
        pyautogui.keyUp(key)
        print(f"  Sent {key_name} for {duration_s}s")
    except Exception as e:
        print(f"  Key {key_name}: {e}")


def move_mouse(dx: int, dy: int):
    """Move mouse relative (for camera look in Roblox). Uses pyautogui."""
    try:
        import pyautogui
        pyautogui.move(dx, dy)
        print(f"  Mouse moved by ({dx}, {dy})")
    except Exception as e:
        print(f"  Mouse: {e}")


def main():
    print("Roblox input test. Have Roblox open and in-game.")
    print("Starting in 3s...")
    time.sleep(3)

    print("Focus Roblox + click center (so game captures keys)...")
    if not focus_roblox_and_click():
        print("Could not focus Roblox. Aborting.")
        sys.exit(1)
    print("Focus + click OK. Sending keys...")
    time.sleep(0.25)

    print("\n1) W (forward) 1.5s")
    send_key("W", 1.5)
    time.sleep(0.3)

    print("2) S (back) 1s")
    send_key("S", 1.0)
    time.sleep(0.3)

    print("3) A (left) 1s")
    send_key("A", 1.0)
    time.sleep(0.3)

    print("4) D (right) 1s")
    send_key("D", 1.0)
    time.sleep(0.3)

    print("5) Mouse: look right then left (camera)")
    move_mouse(120, 0)
    time.sleep(0.2)
    move_mouse(-120, 0)
    time.sleep(0.3)

    print("6) Space (jump) 0.8s")
    send_key("space", 0.8)

    print("\nDone. Did your character move and camera pan?")
    print("If NOT: add this app to System Settings → Privacy & Security → Accessibility:")
    print("  - If you ran from Terminal: add 'Terminal'. If from Cursor: add 'Cursor'.")
    print("  Then turn the toggle ON, quit and reopen the app, and run this test again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
