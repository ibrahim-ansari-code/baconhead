"""
Run calibration for ONLY look combos (look_left, look_right, w_look_right, w_look_left).
Appends results to existing physics.json if it exists.
Usage:
    python run_look_calib.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import mss
import numpy as np
from capture.screen import get_roblox_region, capture_region, focus_roblox_and_click, look_camera
from reward.calibrate_movement import (
    _resize, _mean_flow, _run_combo, _reset_camera,
    _save_screenshot, SCREENSHOT_DIR, ACTION_COMBOS,
)

OUT = "episode_data/physics.json"

LOOK_COMBOS = [c for c in ACTION_COMBOS if c[2] != 0]  # only combos with look_dx

def main():
    region = get_roblox_region()
    print(f"[look_calib] Region: {region}", flush=True)

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    os.makedirs("episode_data", exist_ok=True)

    print("[look_calib] Focusing Roblox...", flush=True)
    focus_roblox_and_click()
    time.sleep(1.5)

    sct = mss.mss()
    results = {}

    n_reps       = 4
    pause_bw     = 2.5
    action_count = 0
    total = sum(len(durs) * n_reps for _, _, _, durs in LOOK_COMBOS)
    print(f"[look_calib] {len(LOOK_COMBOS)} combos, {total} total runs", flush=True)

    for combo_name, keys, look_dx, durations_ms in LOOK_COMBOS:
        combo_results = {}
        print(f"\n[look_calib] ===== {combo_name.upper()} =====", flush=True)

        raw_before = capture_region(region=region, sct=sct)
        _save_screenshot(raw_before, f"{combo_name}_GROUP_before")

        for dur_ms in durations_ms:
            rep_flows = []
            for rep in range(n_reps):
                print(f"[look_calib]   {combo_name} {dur_ms}ms rep {rep+1}/{n_reps}", flush=True)

                action_count += 1
                if action_count % 8 == 0:
                    focus_roblox_and_click()
                    time.sleep(0.5)

                frames = _run_combo(
                    keys=keys,
                    duration_ms=dur_ms,
                    look_dx=look_dx,
                    region=region,
                    sct=sct,
                    pause_before=0.35,
                )
                flow_val = _mean_flow(frames)
                rep_flows.append(flow_val)
                print(f"[look_calib]   → flow={flow_val:.4f}  frames={len(frames)}", flush=True)

                # Reset camera after each look
                _reset_camera(look_dx, region)

                if rep == 0:
                    _save_screenshot(
                        capture_region(region=region, sct=sct),
                        f"{combo_name}_{dur_ms}ms_rep1_after",
                    )

                time.sleep(pause_bw)

            combo_results[str(dur_ms)] = float(np.mean(rep_flows))

        results[combo_name] = combo_results

        raw_after = capture_region(region=region, sct=sct)
        _save_screenshot(raw_after, f"{combo_name}_GROUP_after")

    sct.close()

    # Merge into existing physics.json
    physics = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            physics = json.load(f)

    def _slope(name):
        if name not in results:
            return 0.0
        durs = sorted(int(k) for k in results[name])
        if not durs:
            return 0.0
        d = np.array(durs, dtype=np.float32)
        fl = np.array([results[name][str(x)] for x in durs], dtype=np.float32)
        denom = float(np.dot(d, d))
        return float(np.dot(fl, d) / denom) if denom > 0 else 0.0

    physics["look_right_px_per_ms"]     = _slope("look_right")
    physics["look_left_px_per_ms"]      = _slope("look_left")
    physics["w_look_right_px_per_ms"]   = _slope("w_look_right")
    physics["w_look_left_px_per_ms"]    = _slope("w_look_left")
    physics["look_px_per_ms"]           = (_slope("look_right") + _slope("look_left")) / 2.0
    physics["per_combo"] = {**physics.get("per_combo", {}), **results}

    with open(OUT, "w") as f:
        json.dump(physics, f, indent=2)

    print(f"\n[look_calib] look_right={physics['look_right_px_per_ms']:.5f} px/ms", flush=True)
    print(f"[look_calib] look_left={physics['look_left_px_per_ms']:.5f} px/ms", flush=True)
    print(f"[look_calib] Saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
