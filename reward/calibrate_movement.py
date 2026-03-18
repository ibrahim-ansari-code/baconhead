"""
Comprehensive movement physics calibration for Roblox (Brookhaven or large open world).

Tests single keys, simultaneous combos (W+Space together, not separately),
diagonals, look-only, and look+move at the same time.

Per combo it:
  1. Holds all keys simultaneously (keyDown all, THEN start timing)
  2. Optionally right-drag the mouse concurrently for look combos
  3. Captures frames at FLOW_FPS during the hold
  4. Computes mean optical-flow magnitude
  5. Resets the camera after any look action
  6. Saves a debug PNG screenshot before+after each combo group

Writes episode_data/physics.json with per-combo flow curves and px/ms slopes.
Saves debug screenshots to episode_data/calib_screenshots/.

Expected runtime with default settings: ~18-22 minutes.

Usage:
    python -m reward.calibrate_movement --out episode_data/physics.json --pause 2.5 --reps 4
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import mss
import numpy as np

from capture.screen import get_roblox_region, capture_region, focus_roblox_and_click, look_camera

FRAME_SIZE = 224
FLOW_FPS   = 10.0
SCREENSHOT_DIR = "episode_data/calib_screenshots"

# ---------------------------------------------------------------------------
# Action combos:  (name, keys_to_hold_simultaneously, look_dx, durations_ms)
#   keys:    list of keys pressed with keyDown at the SAME time
#   look_dx: if nonzero → hold right-click and drag this many px while keys are held
#             negative = look left, positive = look right
# ---------------------------------------------------------------------------
ACTION_COMBOS: List[Tuple[str, List[str], int, List[int]]] = [
    # --- single movement ---
    ("w",            ["w"],              0,     [200, 400, 600, 800, 1000, 1500, 2000]),
    ("a",            ["a"],              0,     [200, 400, 600, 800, 1000, 1500]),
    ("d",            ["d"],              0,     [200, 400, 600, 800, 1000, 1500]),
    ("s",            ["s"],              0,     [200, 400, 600, 800]),

    # --- jump: space alone, then W+space simultaneously ---
    ("space",        ["space"],          0,     [150, 200, 300, 400, 500]),
    ("w_space",      ["w", "space"],     0,     [200, 400, 600, 800, 1000]),
    ("a_space",      ["a", "space"],     0,     [200, 400, 600, 800]),
    ("d_space",      ["d", "space"],     0,     [200, 400, 600, 800]),

    # --- diagonals ---
    ("w_a",          ["w", "a"],         0,     [200, 400, 600, 800]),
    ("w_d",          ["w", "d"],         0,     [200, 400, 600, 800]),

    # --- look only (right-drag, no movement keys) ---
    ("look_right",   [],                 220,   [200, 400, 600]),
    ("look_left",    [],                -220,   [200, 400, 600]),

    # --- forward + simultaneous look (turn while walking) ---
    ("w_look_right", ["w"],              220,   [400, 600, 800, 1000]),
    ("w_look_left",  ["w"],             -220,   [400, 600, 800, 1000]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_gray(frame: np.ndarray) -> np.ndarray:
    r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def _mean_flow(frames: List[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    import cv2
    total, count = 0.0, 0
    for i in range(len(frames) - 1):
        g1 = _to_gray(frames[i])
        g2 = _to_gray(frames[i + 1])
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        total += float(mag.mean())
        count += 1
    return total / count if count else 0.0


def _resize(raw: np.ndarray) -> np.ndarray:
    from PIL import Image as _Image
    return np.array(
        _Image.fromarray(raw.astype(np.uint8)).resize(
            (FRAME_SIZE, FRAME_SIZE), _Image.Resampling.LANCZOS
        )
    )


def _reset_camera(look_dx: int, region: Optional[dict]) -> None:
    """After a look action, rotate back the opposite direction."""
    if look_dx == 0:
        return
    look_camera(-look_dx, 500, region)
    time.sleep(0.3)


def _save_screenshot(raw: np.ndarray, name: str) -> None:
    """Save full-res frame as PNG for manual inspection."""
    try:
        from PIL import Image as _Image
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        path = os.path.join(SCREENSHOT_DIR, f"{name}.png")
        _Image.fromarray(raw.astype(np.uint8)).save(path)
    except Exception as e:
        print(f"[calib] screenshot save failed: {e}", flush=True)


def _run_combo(
    keys: List[str],
    duration_ms: int,
    look_dx: int,
    region: Optional[dict],
    sct,
    pause_before: float = 0.35,
) -> List[np.ndarray]:
    """
    Execute a combo (simultaneous key holds + optional look drag) and capture frames.
    Keys are pressed DOWN at the SAME time before timing starts.

    For look combos (look_dx != 0):
      Uses Quartz kCGEventRightMouseDragged with explicit delta fields —
      the only method confirmed to rotate Roblox's camera on macOS.
      Any movement keys are held simultaneously in a side thread.

    For movement-only combos:
      Holds all keys simultaneously and captures frames at FLOW_FPS.
    """
    import pyautogui
    import threading

    frames: List[np.ndarray] = []
    interval = 1.0 / FLOW_FPS

    # Baseline frame
    raw = capture_region(region=region, sct=sct)
    frames.append(_resize(raw))

    time.sleep(pause_before)

    if look_dx != 0:
        # ----------------------------------------------------------------
        # Look combo: Quartz kCGEventRightMouseDragged with delta fields.
        # Simultaneously hold any movement keys in a side thread.
        # ----------------------------------------------------------------
        drag_done = threading.Event()

        def _hold_keys_during_drag():
            for key in keys:
                pyautogui.keyDown(key)
            drag_done.wait(timeout=duration_ms / 1000.0 + 1.0)
            for key in keys:
                pyautogui.keyUp(key)

        if keys:
            t = threading.Thread(target=_hold_keys_during_drag, daemon=True)
            t.start()

        # Quartz delta method — confirmed to rotate Roblox camera
        look_camera(look_dx, duration_ms, region)
        drag_done.set()

        if keys:
            t.join(timeout=0.5)

        # Two frames bracketing the look for flow computation
        time.sleep(0.05)
        raw = capture_region(region=region, sct=sct)
        frames.append(_resize(raw))
        time.sleep(0.15)
        raw = capture_region(region=region, sct=sct)
        frames.append(_resize(raw))

    else:
        # ----------------------------------------------------------------
        # Pure movement combo: hold all keys simultaneously, capture frames
        # ----------------------------------------------------------------
        t_start      = time.perf_counter()
        t_end        = t_start + duration_ms / 1000.0
        t_next_frame = t_start

        for key in keys:
            pyautogui.keyDown(key)
        try:
            while time.perf_counter() < t_end:
                now = time.perf_counter()
                if now >= t_next_frame:
                    raw = capture_region(region=region, sct=sct)
                    frames.append(_resize(raw))
                    t_next_frame += interval
                else:
                    time.sleep(min(0.02, t_next_frame - now))
        finally:
            for key in keys:
                pyautogui.keyUp(key)

        time.sleep(0.1)
        raw = capture_region(region=region, sct=sct)
        frames.append(_resize(raw))

    return frames


# ---------------------------------------------------------------------------
# Main calibration loop
# ---------------------------------------------------------------------------

def calibrate(
    out_path: str,
    region: Optional[dict],
    pause_between: float = 2.5,
    n_reps: int = 4,
) -> dict:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    print("[calib] Focusing Roblox...", flush=True)
    focus_roblox_and_click()
    time.sleep(1.5)

    sct = mss.mss()
    results: Dict[str, Dict[str, float]] = {}
    action_count = 0

    total_combos = sum(len(durs) * n_reps for _, _, _, durs in ACTION_COMBOS)
    print(f"[calib] Total runs planned: {total_combos}  (~18-22 min)", flush=True)

    for combo_name, keys, look_dx, durations_ms in ACTION_COMBOS:
        combo_results: Dict[str, float] = {}
        print(f"\n[calib] ===== {combo_name.upper()} (keys={keys}, look_dx={look_dx}) =====", flush=True)

        # Screenshot BEFORE this combo group
        raw_before = capture_region(region=region, sct=sct)
        _save_screenshot(raw_before, f"{combo_name}_GROUP_before")

        # Center mouse before any look combo
        if look_dx != 0:
            _center_mouse(region)
            time.sleep(0.1)

        for dur_ms in durations_ms:
            rep_flows: List[float] = []
            for rep in range(n_reps):
                print(
                    f"[calib]   {combo_name} {dur_ms}ms rep {rep+1}/{n_reps}",
                    flush=True,
                )

                # Re-focus every 10 actions to prevent Roblox losing focus
                action_count += 1
                if action_count % 10 == 0:
                    focus_roblox_and_click()
                    time.sleep(0.5)

                # Center mouse before look combos
                if look_dx != 0:
                    _center_mouse(region)

                frames = _run_combo(
                    keys=keys,
                    duration_ms=dur_ms,
                    look_dx=look_dx,
                    region=region,
                    sct=sct,
                    pause_before=0.25,
                )

                flow_val = _mean_flow(frames)
                rep_flows.append(flow_val)
                print(f"[calib]   → flow={flow_val:.4f}  frames={len(frames)}", flush=True)

                # Reset camera after look actions
                if look_dx != 0:
                    _reset_camera(look_dx, region)

                # Save mid-action screenshot for the FIRST rep only
                if rep == 0 and len(frames) > 1:
                    mid_idx = len(frames) // 2
                    _save_screenshot(
                        capture_region(region=region, sct=sct),
                        f"{combo_name}_{dur_ms}ms_rep1_after",
                    )

                time.sleep(pause_between)

            combo_results[str(dur_ms)] = float(np.mean(rep_flows))

        results[combo_name] = combo_results

        # Screenshot AFTER this combo group
        raw_after = capture_region(region=region, sct=sct)
        _save_screenshot(raw_after, f"{combo_name}_GROUP_after")

    sct.close()

    # -----------------------------------------------------------------------
    # Compute px/ms slopes (linear fit through origin) for key combos
    # -----------------------------------------------------------------------
    def _slope(name: str) -> float:
        if name not in results:
            return 0.0
        durs = sorted(int(k) for k in results[name])
        if not durs:
            return 0.0
        d_arr = np.array(durs, dtype=np.float32)
        f_arr = np.array([results[name][str(d)] for d in durs], dtype=np.float32)
        denom = float(np.dot(d_arr, d_arr))
        return float(np.dot(f_arr, d_arr) / denom) if denom > 0 else 0.0

    physics = {
        "w_px_per_ms":         _slope("w"),
        "a_px_per_ms":         _slope("a"),
        "d_px_per_ms":         _slope("d"),
        "s_px_per_ms":         _slope("s"),
        "space_px_per_ms":     _slope("space"),
        "w_space_px_per_ms":   _slope("w_space"),
        "look_right_px_per_ms": _slope("look_right"),
        "look_left_px_per_ms": _slope("look_left"),
        "w_look_right_px_per_ms": _slope("w_look_right"),
        "w_look_left_px_per_ms":  _slope("w_look_left"),
        # Legacy keys expected by scout.py
        "space_px_per_ms_legacy": _slope("space"),
        "look_px_per_ms":      (_slope("look_right") + _slope("look_left")) / 2.0,
        "w_jump_px":           max(0.0, _slope("w_space") - _slope("w")),
        "per_combo": results,
    }

    with open(out_path, "w") as f:
        json.dump(physics, f, indent=2)

    print("\n[calib] ===== RESULTS =====", flush=True)
    for k, v in physics.items():
        if k not in ("per_combo",):
            print(f"  {k}: {v:.5f}", flush=True)
    print(f"[calib] Saved → {out_path}", flush=True)
    print(f"[calib] Screenshots → {SCREENSHOT_DIR}/", flush=True)
    return physics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Roblox movement physics calibration"
    )
    parser.add_argument("--out",    type=str,   default="episode_data/physics.json")
    parser.add_argument("--pause",  type=float, default=2.5,
                        help="Seconds between individual reps")
    parser.add_argument("--reps",   type=int,   default=4,
                        help="Repetitions per duration per combo")
    parser.add_argument("--region", type=str,   default=None,
                        help="left,top,width,height override")
    args = parser.parse_args()

    region = None
    if args.region:
        parts = [int(x.strip()) for x in args.region.split(",")]
        region = {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
    else:
        region = get_roblox_region()
        if region:
            print(f"[calib] Using Roblox window: {region}", flush=True)
        else:
            print("[calib] WARNING: Could not detect Roblox window; capturing full screen", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    calibrate(out_path=args.out, region=region, pause_between=args.pause, n_reps=args.reps)


if __name__ == "__main__":
    main()
