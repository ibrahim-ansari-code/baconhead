#!/usr/bin/env python3
"""
Idle takeover: when the user stops playing, Scout plans the next 10 seconds and the bot
executes them. Optionally scores competing plans with the outcome model.

Single execution path:
  1. Wait for idle.
  2. Capture frame.
  3. Call plan_next_10s (Scout / Llama-4 vision) → list of (action, ms).
  4. If outcome_model loaded: generate 2 variant plans and pick the one with highest P(survived).
  5. Execute the plan step-by-step, focusing Roblox before every look action.
  6. Resume watching when user becomes active again.

Usage:
    python run_takeover.py --idle 3
    python run_takeover.py --idle 3 --outcome-model outcome_model.pt
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from typing import List, Optional, Tuple

import mss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from capture.screen import capture_region, get_roblox_region, focus_roblox_and_click
from reward.input_state import start_listener, is_active, get_recent_activity_summary
from llm_agent.cem import execute_action_ms
from llm_agent.scout import plan_next_10s, _default_plan_10s


def _compute_spatial(frame: np.ndarray, prev_frame: Optional[np.ndarray]):
    """Return (edge_distances (4,), flow_mean float) from a frame pair."""
    try:
        from reward.collect_episodes import compute_edge_distances, compute_flow_features, N_BUCKETS
        import cv2
        from PIL import Image as _Image
        frame_224 = np.array(_Image.fromarray(frame.astype(np.uint8)).resize((224, 224), _Image.Resampling.LANCZOS))
        edge_dists = compute_edge_distances(frame_224)
        if prev_frame is not None:
            prev_224 = np.array(_Image.fromarray(prev_frame.astype(np.uint8)).resize((224, 224), _Image.Resampling.LANCZOS))
            gray1 = cv2.cvtColor(prev_224, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame_224, cv2.COLOR_RGB2GRAY)
            flow  = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag   = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_mean = float(mag.mean())
        else:
            flow_mean = 0.0
        return edge_dists, flow_mean
    except Exception:
        return None, None


# ── helpers ────────────────────────────────────────────────────────────────────

def click_close(region: Optional[dict]) -> None:
    """Click the X then the No button when a popup is detected."""
    if not region:
        return
    try:
        import pyautogui
        x = region["left"] + region["width"] - 40
        y = region["top"] + 25
        pyautogui.click(x, y)
        time.sleep(0.15)
        no_x = int(region["left"] + region["width"] * 0.72)
        no_y = int(region["top"] + region["height"] * 0.65)
        pyautogui.click(no_x, no_y)
        time.sleep(0.1)
    except Exception:
        pass


def _focus_for_look(region: Optional[dict]) -> None:
    """Ensure the Roblox window is focused and the mouse is inside it before a camera drag."""
    try:
        focus_roblox_and_click()
    except Exception:
        pass


def _is_look_action(action: str) -> bool:
    return "look_" in action.lower()


# ── outcome-model scoring ──────────────────────────────────────────────────────

def _plan_to_key_events(plan: List[Tuple[str, int]]) -> np.ndarray:
    """
    Convert a Scout plan [(action, ms), ...] into the key_events array expected by
    the outcome model (MAX_KEY_EVENTS × 3 float32: [key_idx, t_down_ms, t_up_ms]).
    """
    from reward.collect_episodes import KEY_TO_IDX, MAX_KEY_EVENTS, pack_key_events
    events = []
    t_cursor = 0.0
    for action, ms in plan:
        parts = [p.strip().lower() for p in action.split("+") if p.strip()]
        for p in parts:
            key_idx = KEY_TO_IDX.get(p)
            if key_idx is not None:
                events.append((key_idx, t_cursor, t_cursor + ms))
        t_cursor += ms
    return pack_key_events(events)


def _score_plan(
    plan: List[Tuple[str, int]],
    frame: np.ndarray,
    outcome_model,
    outcome_device,
    physics: Optional[dict] = None,
) -> float:
    """Score a plan with the outcome model. Returns P(survived) in [0, 1]."""
    from reward.collect_episodes import compute_edge_distances, compute_flow_features, N_BUCKETS
    import numpy as np

    key_events   = _plan_to_key_events(plan)
    from PIL import Image as _Image
    frame_224 = np.array(_Image.fromarray(frame.astype(np.uint8)).resize((224, 224), _Image.Resampling.LANCZOS))

    edge_dists   = compute_edge_distances(frame_224)
    flow_mag     = np.zeros(N_BUCKETS, dtype=np.float32)
    flow_dir     = np.zeros(N_BUCKETS, dtype=np.float32)

    return outcome_model.predict_survival(
        frame_224, edge_dists, flow_mag, flow_dir, key_events,
        device=outcome_device, physics=physics,
    )


def _variant_plans(base_plan: List[Tuple[str, int]]) -> List[List[Tuple[str, int]]]:
    """Generate 2 simple variants of a plan for outcome-model comparison."""
    import random
    variants = []

    # Variant 1: shorten all durations by 30%
    v1 = [(a, max(100, int(ms * 0.7))) for a, ms in base_plan]
    variants.append(v1)

    # Variant 2: insert a W+space jump after the first W action
    v2 = list(base_plan)
    for i, (a, ms) in enumerate(v2):
        if a.lower() == "w":
            v2.insert(i + 1, ("W+space", 350))
            break
    # Trim if too long
    variants.append(v2[:len(base_plan) + 1])

    return variants


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Idle takeover — Scout plans + optional outcome scoring")
    parser.add_argument("--idle",          type=float, default=3.0,   help="Idle seconds before takeover")
    parser.add_argument("--full-screen",   action="store_true",        help="Capture full monitor instead of Roblox window")
    parser.add_argument("--region",        type=str,   default=None,   help="left,top,width,height")
    parser.add_argument("--no-scout",      action="store_true",        help="Skip Scout (use default fallback plans)")
    parser.add_argument("--outcome-model", type=str,   default=None,   help="Path to outcome_model.pt for plan scoring")
    parser.add_argument("--monitor",       type=str,   default=None,   help="Log decisions to this file")
    parser.add_argument("--max-step-ms",   type=int,   default=2000,   help="Cap any single action to this many ms")
    args = parser.parse_args()

    # ── region ──────────────────────────────────────────────────────────────────
    region = None
    if args.region:
        parts = [int(x.strip()) for x in args.region.split(",")]
        if len(parts) != 4:
            raise ValueError("--region must be left,top,width,height")
        region = {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
        print("Using region from --region:", region, flush=True)
    elif args.full_screen:
        with mss.mss() as m:
            mon = m.monitors[0]
        region = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}
        print("Using full primary monitor:", region, flush=True)
    else:
        for attempt in range(30):
            region = get_roblox_region()
            if region is not None:
                break
            if attempt == 0:
                print("Waiting for Roblox window... (Ctrl+C to abort)", flush=True)
            time.sleep(3)
        if region is None:
            print("Error: Roblox window not found after 90s. Use --region or --full-screen.", file=sys.stderr)
            sys.exit(1)
        print("Using Roblox window:", region, flush=True)

    # ── api key ──────────────────────────────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY")
    if not args.no_scout and not api_key:
        print("Warning: GROQ_API_KEY not set; Scout will be skipped.", file=sys.stderr)

    # ── outcome model ────────────────────────────────────────────────────────────
    outcome_model  = None
    outcome_device = None
    physics        = None
    if args.outcome_model and os.path.isfile(args.outcome_model):
        import torch
        from reward.outcome_model import load_outcome_model
        outcome_device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        outcome_model = load_outcome_model(args.outcome_model, device=outcome_device)
        print(f"Loaded outcome model from {args.outcome_model}", flush=True)
        physics_path = os.path.join("episode_data", "physics.json")
        if os.path.isfile(physics_path):
            with open(physics_path) as f:
                physics = json.load(f)
            print(f"Loaded physics from {physics_path}", flush=True)
    elif args.outcome_model:
        print(f"Warning: --outcome-model path not found: {args.outcome_model}", file=sys.stderr)

    # ── misc setup ───────────────────────────────────────────────────────────────
    start_listener()
    sct           = mss.mss()
    idle_sec      = args.idle
    max_step_ms   = max(1, args.max_step_ms)
    monitor_path  = args.monitor
    n_actions     = 0
    last_bot_time = [0.0]
    BOT_COOLDOWN  = 0.8    # ignore "user active" for this long after a bot key press

    last_actions:     deque = deque(maxlen=20)
    last_objective:   Optional[str] = None
    idle_since        = [None]
    current_plan:     List[Tuple[str, int]] = []
    plan_index        = 0
    precomputed_plan: Optional[List[Tuple[str, int]]] = None
    precomputed_obj:  Optional[str] = None
    precomputed_popup = False
    look_streak       = [0]   # consecutive look_left/look_right actions

    def log(msg: str):
        print(msg, flush=True)

    def monitor_log(action: str, ms: int, source: str = "scout"):
        if not monitor_path:
            return
        try:
            with open(monitor_path, "a") as f:
                f.write(f"{time.time():.3f}\t{source}\t{action}\t{ms}\n")
        except Exception:
            pass

    def need_plan() -> bool:
        return plan_index >= len(current_plan)

    def remaining_plan() -> List[Tuple[str, int]]:
        return current_plan[plan_index:] if plan_index < len(current_plan) else []

    use_scout = not args.no_scout and bool(api_key)

    log(f"Watching. Takeover after {idle_sec}s idle. Scout={'on' if use_scout else 'off'} OutcomeModel={'on' if outcome_model else 'off'}.")
    log("Press Ctrl+C to stop.")

    prev_frame: Optional[np.ndarray] = None

    try:
        while True:
            frame = capture_region(region=region, sct=sct)
            now   = time.perf_counter()
            edge_dists, flow_mean = _compute_spatial(frame, prev_frame)
            prev_frame = frame

            # ── user active? ──────────────────────────────────────────────────
            in_cooldown = (now - last_bot_time[0]) < BOT_COOLDOWN
            if not in_cooldown and is_active(idle_sec):
                if idle_since[0] is not None:
                    log("[takeover] User active — stopping bot.")
                idle_since[0]    = None
                current_plan     = []
                plan_index       = 0
                precomputed_plan = None
                look_streak[0]   = 0
                time.sleep(0.4)
                continue

            if idle_since[0] is None:
                idle_since[0] = time.perf_counter()
                log("[takeover] Idle — starting takeover.")

            # ── acquire plan ──────────────────────────────────────────────────
            if need_plan():
                if precomputed_plan is not None and not precomputed_popup:
                    current_plan     = precomputed_plan
                    plan_index       = 0
                    if precomputed_obj:
                        last_objective = precomputed_obj
                    precomputed_plan = None
                    precomputed_obj  = None
                    log("[takeover] Using precomputed plan.")
                else:
                    if precomputed_popup:
                        log("[takeover] Popup in precomputed plan — clicking close.")
                        click_close(region)
                        time.sleep(0.3)
                        precomputed_plan  = None
                        precomputed_popup = False

                    if use_scout:
                        user_pattern = get_recent_activity_summary(seconds=120.0) or None
                        consecutive_looks = look_streak[0]
                        plan, objectives, popup = plan_next_10s(
                            frame,
                            api_key=api_key,
                            current_plan_remaining=remaining_plan(),
                            executed_recent=list(last_actions),
                            user_pattern=user_pattern,
                            last_objective=last_objective,
                            look_streak=consecutive_looks,
                            edge_distances=edge_dists,
                            flow_mean=flow_mean,
                        )
                        if popup:
                            log("[takeover] Popup detected — clicking close.")
                            click_close(region)
                            time.sleep(0.3)
                            continue
                        if objectives:
                            last_objective = objectives
                    else:
                        plan = _default_plan_10s()

                    # ── outcome scoring: pick best among base + variants ──────
                    if outcome_model is not None and plan:
                        candidates = [plan] + _variant_plans(plan)
                        scores = []
                        for c in candidates:
                            try:
                                s = _score_plan(c, frame, outcome_model, outcome_device, physics)
                            except Exception:
                                s = 0.5
                            scores.append(s)
                        best_idx = int(np.argmax(scores))
                        if best_idx != 0:
                            log(f"[takeover] Outcome model prefers variant {best_idx} "
                                f"(P={scores[best_idx]:.2f} vs base P={scores[0]:.2f})")
                        plan = candidates[best_idx]

                    current_plan = plan
                    plan_index   = 0
                    look_streak[0] = 0
                    precomputed_plan = None

            if need_plan():
                time.sleep(0.3)
                continue

            # ── execute next action ───────────────────────────────────────────
            action, action_ms = current_plan[plan_index]
            action_ms = min(max(1, action_ms), max_step_ms)
            n_actions += 1

            # Focus Roblox before every look (camera drag requires window focus + mouse inside)
            if _is_look_action(action):
                _focus_for_look(region)
                look_streak[0] += 1
            elif n_actions % 8 == 1:
                if focus_roblox_and_click():
                    log("[takeover] Refocused Roblox.")
                look_streak[0] = 0
            else:
                look_streak[0] = 0

            log(f"[takeover] step {plan_index + 1}/{len(current_plan)} | {action!r} {action_ms} ms")
            monitor_log(action, action_ms)
            execute_action_ms(action, duration_ms=action_ms)
            last_bot_time[0] = time.perf_counter()
            last_actions.append(action)
            plan_index += 1

            # ── precompute next plan when near the end ────────────────────────
            if (plan_index >= len(current_plan) - 3
                    and len(current_plan) > 0
                    and precomputed_plan is None
                    and use_scout):
                next_frame    = capture_region(region=region, sct=sct)
                user_pattern  = get_recent_activity_summary(seconds=120.0) or None
                next_edge_dists, next_flow_mean = _compute_spatial(next_frame, frame)
                next_plan, next_obj, next_popup = plan_next_10s(
                    next_frame,
                    api_key=api_key,
                    current_plan_remaining=current_plan[plan_index:],
                    executed_recent=list(last_actions),
                    user_pattern=user_pattern,
                    last_objective=last_objective,
                    look_streak=look_streak[0],
                    edge_distances=next_edge_dists,
                    flow_mean=next_flow_mean,
                )
                precomputed_plan  = next_plan
                precomputed_obj   = next_obj
                precomputed_popup = next_popup
                log("[takeover] Precomputed next 10s plan.")

    except KeyboardInterrupt:
        pass
    finally:
        sct.close()
    print("Done.")


if __name__ == "__main__":
    main()
