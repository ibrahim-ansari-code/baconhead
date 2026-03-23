"""
Street-aware Episode 1 calibration.

Character starts facing forward on a linear street.
Rules:
  - W  = forward along street → always follow with S to return
  - "Left"  = look_left first, then W forward in new dir, then reset camera
  - "Right" = look_right first, then W forward in new dir, then reset camera
  - Jumps = W+space along street, then S back
  - Looks = standalone camera rotation (Quartz delta method), then reset
  - After every direction group, return to origin with opposite key

Saves flow measurements to episode_data/physics.json (merges with existing).
Screenshots saved to episode_data/calib_screenshots/.

Usage:
    python run_street_calib.py
"""

import json, os, sys, time, threading
sys.path.insert(0, os.path.dirname(__file__))

import mss
import numpy as np
from PIL import Image

from capture.screen import (
    get_roblox_region, capture_region,
    focus_roblox_and_click, look_camera,
)
from reward.calibrate_movement import (
    _resize, _mean_flow, _save_screenshot, SCREENSHOT_DIR,
)

OUT       = "episode_data/physics.json"
PAUSE_BW  = 2.5   # seconds between individual reps
N_REPS    = 4
LOOK_DX   = 220   # pixels for a full look turn

# ── helpers ──────────────────────────────────────────────────────────────────

def snap(sct, region) -> np.ndarray:
    return capture_region(region=region, sct=sct)

def save(img, name):
    _save_screenshot(img, name)

def hold_keys(keys, duration_ms):
    import pyautogui
    for k in keys:
        pyautogui.keyDown(k)
    time.sleep(duration_ms / 1000.0)
    for k in keys:
        pyautogui.keyUp(k)

def hold_keys_with_frames(keys, duration_ms, region, sct):
    """Hold keys simultaneously and collect frames at 10 fps."""
    import pyautogui
    frames = []
    interval = 0.1
    t_end = time.perf_counter() + duration_ms / 1000.0
    t_next = time.perf_counter()
    for k in keys:
        pyautogui.keyDown(k)
    try:
        while time.perf_counter() < t_end:
            now = time.perf_counter()
            if now >= t_next:
                frames.append(_resize(snap(sct, region)))
                t_next += interval
            else:
                time.sleep(min(0.02, t_next - now))
    finally:
        for k in keys:
            pyautogui.keyUp(k)
    time.sleep(0.1)
    frames.append(_resize(snap(sct, region)))
    return frames

def do_look(dx, duration_ms, region):
    look_camera(dx, duration_ms, region)
    time.sleep(0.15)

def reset_look(dx, region):
    """Undo a look by rotating back."""
    look_camera(-dx, 450, region)
    time.sleep(0.3)

def dismiss_popups():
    """Click the left portion of the screen where Brookhaven invite/decline
    buttons typically appear, dismissing them without touching Roblox settings."""
    import pyautogui
    pyautogui.click(200, 420)   # ~where Decline buttons show up
    time.sleep(0.1)
    pyautogui.click(200, 480)   # second click in case there are two stacked
    time.sleep(0.15)

def refocus(n, region):
    if n % 8 == 0:
        focus_roblox_and_click()
        time.sleep(0.3)

# ── run one rep, return flow ──────────────────────────────────────────────────

def run_rep_movement(keys, duration_ms, region, sct):
    """Press keys simultaneously, capture frames, return mean flow."""
    dismiss_popups()          # clear any invite/vehicle popup first
    baseline = _resize(snap(sct, region))
    time.sleep(0.3)
    frames = [baseline] + hold_keys_with_frames(keys, duration_ms, region, sct)
    return _mean_flow(frames)

def run_rep_look(look_dx, duration_ms, region, sct):
    """Do a look, capture before+after, return mean flow."""
    dismiss_popups()
    before = _resize(snap(sct, region))
    time.sleep(0.3)
    do_look(look_dx, duration_ms, region)
    time.sleep(0.1)
    after = _resize(snap(sct, region))
    return _mean_flow([before, after])

# ── combo definitions for a street ──────────────────────────────────────────
#
# Each entry: (name, fn, durations_ms, reset_fn)
#   fn(dur_ms, region, sct) -> flow float
#   reset_fn(dur_ms, region, sct) -> None   (returns character/camera to origin)

def main():
    region = get_roblox_region()
    print(f"[street_calib] Region: {region}", flush=True)
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    os.makedirs("episode_data", exist_ok=True)

    print("[street_calib] Focusing Roblox...", flush=True)
    focus_roblox_and_click()
    time.sleep(1.5)
    dismiss_popups()
    time.sleep(0.3)

    sct = mss.mss()
    results = {}
    action_n = [0]

    def tick():
        action_n[0] += 1
        refocus(action_n[0], region)

    # ── W: forward, then S back ───────────────────────────────────────────────
    print("\n[street_calib] ===== W (forward + S return) =====", flush=True)
    dismiss_popups()
    save(snap(sct, region), "w_GROUP_before")
    w_flows = {}
    for dur in [200, 400, 600, 800, 1000, 1500, 2000]:
        reps = []
        for rep in range(N_REPS):
            tick()
            print(f"  w {dur}ms rep {rep+1}/{N_REPS}", flush=True)
            f = run_rep_movement(["w"], dur, region, sct)
            reps.append(f)
            print(f"  → flow={f:.4f}", flush=True)
            # Return to origin
            hold_keys(["s"], dur)
            time.sleep(PAUSE_BW)
        w_flows[str(dur)] = float(np.mean(reps))
    results["w"] = w_flows
    save(snap(sct, region), "w_GROUP_after")

    # ── S: backward, then W back ──────────────────────────────────────────────
    print("\n[street_calib] ===== S (backward + W return) =====", flush=True)
    save(snap(sct, region), "s_GROUP_before")
    s_flows = {}
    for dur in [200, 400, 600, 800]:
        reps = []
        for rep in range(N_REPS):
            tick()
            print(f"  s {dur}ms rep {rep+1}/{N_REPS}", flush=True)
            f = run_rep_movement(["s"], dur, region, sct)
            reps.append(f)
            print(f"  → flow={f:.4f}", flush=True)
            hold_keys(["w"], dur)
            time.sleep(PAUSE_BW)
        s_flows[str(dur)] = float(np.mean(reps))
    results["s"] = s_flows
    save(snap(sct, region), "s_GROUP_after")

    # ── W+Space: forward jump, then S back ───────────────────────────────────
    print("\n[street_calib] ===== W+Space (forward jump + S return) =====", flush=True)
    save(snap(sct, region), "w_space_GROUP_before")
    ws_flows = {}
    for dur in [200, 400, 600, 800, 1000]:
        reps = []
        for rep in range(N_REPS):
            tick()
            print(f"  w_space {dur}ms rep {rep+1}/{N_REPS}", flush=True)
            f = run_rep_movement(["w", "space"], dur, region, sct)
            reps.append(f)
            print(f"  → flow={f:.4f}", flush=True)
            hold_keys(["s"], dur)
            time.sleep(PAUSE_BW)
        ws_flows[str(dur)] = float(np.mean(reps))
    results["w_space"] = ws_flows
    save(snap(sct, region), "w_space_GROUP_after")

    # ── Space alone: jump in place ────────────────────────────────────────────
    print("\n[street_calib] ===== Space (jump in place) =====", flush=True)
    save(snap(sct, region), "space_GROUP_before")
    sp_flows = {}
    for dur in [150, 200, 300, 400, 500]:
        reps = []
        for rep in range(N_REPS):
            tick()
            print(f"  space {dur}ms rep {rep+1}/{N_REPS}", flush=True)
            f = run_rep_movement(["space"], dur, region, sct)
            reps.append(f)
            print(f"  → flow={f:.4f}", flush=True)
            time.sleep(PAUSE_BW)
        sp_flows[str(dur)] = float(np.mean(reps))
    results["space"] = sp_flows
    save(snap(sct, region), "space_GROUP_after")

    # ── "Left": look_left → W forward → reset camera, then W+S return ────────
    print("\n[street_calib] ===== Left (look_left + W + reset) =====", flush=True)
    save(snap(sct, region), "left_GROUP_before")
    left_flows = {}
    for dur in [200, 400, 600, 800, 1000]:
        reps = []
        for rep in range(N_REPS):
            tick()
            print(f"  left {dur}ms rep {rep+1}/{N_REPS}", flush=True)
            # Turn left
            do_look(-LOOK_DX, 500, region)
            time.sleep(0.2)
            # Walk forward in the new direction
            f = run_rep_movement(["w"], dur, region, sct)
            reps.append(f)
            print(f"  → flow={f:.4f}", flush=True)
            # Come back (S in turned direction)
            hold_keys(["s"], dur)
            # Reset camera
            reset_look(-LOOK_DX, region)
            time.sleep(PAUSE_BW)
        left_flows[str(dur)] = float(np.mean(reps))
    results["look_left_then_w"] = left_flows
    save(snap(sct, region), "left_GROUP_after")

    # ── "Right": look_right → W forward → reset camera, then S return ────────
    print("\n[street_calib] ===== Right (look_right + W + reset) =====", flush=True)
    save(snap(sct, region), "right_GROUP_before")
    right_flows = {}
    for dur in [200, 400, 600, 800, 1000]:
        reps = []
        for rep in range(N_REPS):
            tick()
            print(f"  right {dur}ms rep {rep+1}/{N_REPS}", flush=True)
            do_look(LOOK_DX, 500, region)
            time.sleep(0.2)
            f = run_rep_movement(["w"], dur, region, sct)
            reps.append(f)
            print(f"  → flow={f:.4f}", flush=True)
            hold_keys(["s"], dur)
            reset_look(LOOK_DX, region)
            time.sleep(PAUSE_BW)
        right_flows[str(dur)] = float(np.mean(reps))
    results["look_right_then_w"] = right_flows
    save(snap(sct, region), "right_GROUP_after")

    # ── Left jump: look_left → W+space → reset ───────────────────────────────
    print("\n[street_calib] ===== Left jump (look_left + W+space + reset) =====", flush=True)
    save(snap(sct, region), "left_jump_GROUP_before")
    lj_flows = {}
    for dur in [200, 400, 600, 800]:
        reps = []
        for rep in range(N_REPS):
            tick()
            print(f"  left_jump {dur}ms rep {rep+1}/{N_REPS}", flush=True)
            do_look(-LOOK_DX, 500, region)
            time.sleep(0.2)
            f = run_rep_movement(["w", "space"], dur, region, sct)
            reps.append(f)
            print(f"  → flow={f:.4f}", flush=True)
            hold_keys(["s"], dur)
            reset_look(-LOOK_DX, region)
            time.sleep(PAUSE_BW)
        lj_flows[str(dur)] = float(np.mean(reps))
    results["look_left_space"] = lj_flows
    save(snap(sct, region), "left_jump_GROUP_after")

    # ── Right jump: look_right → W+space → reset ─────────────────────────────
    print("\n[street_calib] ===== Right jump (look_right + W+space + reset) =====", flush=True)
    save(snap(sct, region), "right_jump_GROUP_before")
    rj_flows = {}
    for dur in [200, 400, 600, 800]:
        reps = []
        for rep in range(N_REPS):
            tick()
            print(f"  right_jump {dur}ms rep {rep+1}/{N_REPS}", flush=True)
            do_look(LOOK_DX, 500, region)
            time.sleep(0.2)
            f = run_rep_movement(["w", "space"], dur, region, sct)
            reps.append(f)
            print(f"  → flow={f:.4f}", flush=True)
            hold_keys(["s"], dur)
            reset_look(LOOK_DX, region)
            time.sleep(PAUSE_BW)
        rj_flows[str(dur)] = float(np.mean(reps))
    results["look_right_space"] = rj_flows
    save(snap(sct, region), "right_jump_GROUP_after")

    # ── Pure look right, then pure look left ─────────────────────────────────
    print("\n[street_calib] ===== Look only (right then left) =====", flush=True)
    save(snap(sct, region), "look_GROUP_before")
    lr_flows, ll_flows = {}, {}
    for dur in [200, 400, 600, 800]:
        rr, rl = [], []
        for rep in range(N_REPS):
            tick()
            print(f"  look_right {dur}ms rep {rep+1}/{N_REPS}", flush=True)
            f = run_rep_look(LOOK_DX, dur, region, sct)
            rr.append(f)
            print(f"  → flow={f:.4f}", flush=True)
            reset_look(LOOK_DX, region)
            time.sleep(PAUSE_BW * 0.6)

            tick()
            print(f"  look_left {dur}ms rep {rep+1}/{N_REPS}", flush=True)
            f = run_rep_look(-LOOK_DX, dur, region, sct)
            rl.append(f)
            print(f"  → flow={f:.4f}", flush=True)
            reset_look(-LOOK_DX, region)
            time.sleep(PAUSE_BW * 0.6)

        lr_flows[str(dur)] = float(np.mean(rr))
        ll_flows[str(dur)] = float(np.mean(rl))
    results["look_right"] = lr_flows
    results["look_left"]  = ll_flows
    save(snap(sct, region), "look_GROUP_after")

    # ── W + simultaneous look (walk and turn at same time) ───────────────────
    print("\n[street_calib] ===== W+look_right simultaneous =====", flush=True)
    save(snap(sct, region), "w_look_right_GROUP_before")
    wlr_flows = {}
    for dur in [400, 600, 800, 1000]:
        reps = []
        for rep in range(N_REPS):
            tick()
            print(f"  w_look_right {dur}ms rep {rep+1}/{N_REPS}", flush=True)
            # Simultaneously hold W and look right
            import pyautogui
            done = threading.Event()
            def _walk():
                pyautogui.keyDown("w")
                done.wait(timeout=dur/1000.0 + 0.5)
                pyautogui.keyUp("w")
            t = threading.Thread(target=_walk, daemon=True)
            t.start()
            before = _resize(snap(sct, region))
            look_camera(LOOK_DX, dur, region)
            done.set(); t.join(0.3)
            after = _resize(snap(sct, region))
            f = _mean_flow([before, after])
            reps.append(f)
            print(f"  → flow={f:.4f}", flush=True)
            # Return: S back + reset camera
            hold_keys(["s"], dur)
            reset_look(LOOK_DX, region)
            time.sleep(PAUSE_BW)
        wlr_flows[str(dur)] = float(np.mean(reps))
    results["w_look_right"] = wlr_flows
    save(snap(sct, region), "w_look_right_GROUP_after")

    print("\n[street_calib] ===== W+look_left simultaneous =====", flush=True)
    save(snap(sct, region), "w_look_left_GROUP_before")
    wll_flows = {}
    for dur in [400, 600, 800, 1000]:
        reps = []
        for rep in range(N_REPS):
            tick()
            print(f"  w_look_left {dur}ms rep {rep+1}/{N_REPS}", flush=True)
            done = threading.Event()
            def _walk2():
                pyautogui.keyDown("w")
                done.wait(timeout=dur/1000.0 + 0.5)
                pyautogui.keyUp("w")
            t = threading.Thread(target=_walk2, daemon=True)
            t.start()
            before = _resize(snap(sct, region))
            look_camera(-LOOK_DX, dur, region)
            done.set(); t.join(0.3)
            after = _resize(snap(sct, region))
            f = _mean_flow([before, after])
            reps.append(f)
            print(f"  → flow={f:.4f}", flush=True)
            hold_keys(["s"], dur)
            reset_look(-LOOK_DX, region)
            time.sleep(PAUSE_BW)
        wll_flows[str(dur)] = float(np.mean(reps))
    results["w_look_left"] = wll_flows
    save(snap(sct, region), "w_look_left_GROUP_after")

    sct.close()

    # ── Save physics.json ────────────────────────────────────────────────────
    physics = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            physics = json.load(f)

    def slope(name):
        d = results.get(name, {})
        if not d: return 0.0
        durs = sorted(int(k) for k in d)
        da = np.array(durs, dtype=np.float32)
        fa = np.array([d[str(x)] for x in durs], dtype=np.float32)
        denom = float(np.dot(da, da))
        return float(np.dot(fa, da) / denom) if denom > 0 else 0.0

    physics.update({
        "w_px_per_ms":              slope("w"),
        "s_px_per_ms":              slope("s"),
        "space_px_per_ms":          slope("space"),
        "w_space_px_per_ms":        slope("w_space"),
        "look_left_w_px_per_ms":    slope("look_left_then_w"),
        "look_right_w_px_per_ms":   slope("look_right_then_w"),
        "look_right_px_per_ms":     slope("look_right"),
        "look_left_px_per_ms":      slope("look_left"),
        "look_px_per_ms":           (slope("look_right") + slope("look_left")) / 2.0,
        "w_look_right_px_per_ms":   slope("w_look_right"),
        "w_look_left_px_per_ms":    slope("w_look_left"),
        "w_jump_px":                max(0.0, slope("w_space") - slope("w")),
        "per_combo": {**physics.get("per_combo", {}), **results},
    })

    with open(OUT, "w") as f:
        json.dump(physics, f, indent=2)

    print("\n[street_calib] ===== FINAL =====", flush=True)
    for k, v in physics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.5f}", flush=True)
    print(f"[street_calib] Saved → {OUT}", flush=True)
    print(f"[street_calib] Screenshots → {SCREENSHOT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
