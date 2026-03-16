#!/usr/bin/env python3
"""
Full pipeline: when idle, take over using 10-second plans. Planner chooses actions (no forcing).
While one 10s plan runs, we compute the next 10s with context so the next plan follows smoothly.
"""

import argparse
import os
import sys
import time
from collections import deque
from typing import List, Optional, Tuple

import mss
from dotenv import load_dotenv

load_dotenv()
from capture.screen import capture_region, get_roblox_region, focus_roblox_and_click
from reward.input_state import start_listener, is_active, get_recent_activity_summary
from llm_agent.cem import execute_action_ms
from llm_agent.scout import plan_next_10s, _default_plan_10s


def click_close(region: Optional[dict]) -> None:
    """Click close (X) then optionally No button when popup was detected. No Escape (opens Roblox menu)."""
    if not region:
        return
    try:
        import pyautogui
        # X button: top-right of game window
        x = region["left"] + region["width"] - 40
        y = region["top"] + 25
        pyautogui.click(x, y)
        time.sleep(0.15)
        # No button: often center-right or center-bottom
        no_x = int(region["left"] + region["width"] * 0.72)
        no_y = int(region["top"] + region["height"] * 0.65)
        pyautogui.click(no_x, no_y)
        time.sleep(0.1)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Idle takeover with CEM + Llama 4 Scout + reward model + avoids")
    parser.add_argument("--idle", type=float, default=3.0, help="Seconds without key press before we take over")
    parser.add_argument("--full-screen", action="store_true", help="Capture full primary monitor (default: Roblox window only)")
    parser.add_argument("--region", type=str, default=None, help="left,top,width,height (overrides Roblox detection)")
    parser.add_argument("--reward-model", type=str, default="reward_model.pt", help="Path to reward model .pt (or empty to skip)")
    parser.add_argument("--no-reward-model", action="store_true", help="Do not load reward model")
    parser.add_argument("--no-scout", action="store_true", help="Do not call Scout (use only reward model + avoids)")
    args = parser.parse_args()

    region = None
    if args.region:
        parts = [int(x.strip()) for x in args.region.split(",")]
        if len(parts) != 4:
            raise ValueError("--region must be left,top,width,height")
        region = {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
        print("Using region from --region", region, flush=True)
    elif args.full_screen:
        with mss.mss() as m:
            mon = m.monitors[0]
        region = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}
        print("Using full primary monitor", region, flush=True)
    else:
        region = None
        for wait in range(30):  # wait up to ~90s for Roblox
            region = get_roblox_region()
            if region is not None:
                break
            if wait == 0:
                print("Waiting for Roblox window... (start Roblox and get in-game, or Ctrl+C and use --region or --full-screen)", flush=True)
            time.sleep(3)
        if region is None:
            print("Error: Roblox window not found after 90s.", file=sys.stderr)
            print("  Start Roblox and get in-game, or use --region left,top,width,height or --full-screen.", file=sys.stderr)
            sys.exit(1)
        print("Using Roblox window only:", region, flush=True)

    start_listener()
    api_key = os.environ.get("GROQ_API_KEY")
    if not args.no_scout and not api_key:
        print("Warning: GROQ_API_KEY not set; Scout will be skipped. Set it for LLM scoring.", file=sys.stderr)

    idle_sec = args.idle
    sct = mss.mss()
    n_actions = 0

    def log(msg):
        print(msg, flush=True)
    log("Watching. When idle " + str(idle_sec) + " s we take over with 10s plans (no forcing).")
    log("Press Ctrl+C to stop.")
    idle_since = [None]
    last_actions: deque = deque(maxlen=16)
    last_objective: Optional[str] = None
    current_plan: List[Tuple[str, int]] = []
    plan_index = 0
    precomputed_plan: Optional[List[Tuple[str, int]]] = None
    precomputed_objective: Optional[str] = None
    precomputed_popup = False

    def need_plan() -> bool:
        return plan_index >= len(current_plan)

    def remaining_plan() -> List[Tuple[str, int]]:
        return current_plan[plan_index:] if plan_index < len(current_plan) else []

    try:
        while True:
            frame = capture_region(region=region, sct=sct)
            if is_active(idle_sec):
                if idle_since[0] is not None:
                    log("[takeover] User active again — stopping bot, watching.")
                idle_since[0] = None
                current_plan = []
                plan_index = 0
                precomputed_plan = None
                time.sleep(0.5)
                continue
            if idle_since[0] is None:
                idle_since[0] = time.perf_counter()
                log("[takeover] Idle detected — starting 10s-plan takeover.")

            if need_plan():
                if precomputed_plan is not None and not precomputed_popup:
                    current_plan = precomputed_plan
                    if precomputed_objective:
                        last_objective = precomputed_objective
                    plan_index = 0
                    precomputed_plan = None
                    precomputed_objective = None
                    log("[takeover] using precomputed next 10s plan.")
                else:
                    if precomputed_popup:
                        log("[takeover] Popup in precomputed — clicking close.")
                        click_close(region)
                        time.sleep(0.3)
                        precomputed_plan = None
                        precomputed_popup = False
                    use_scout = not args.no_scout and api_key is not None
                    if use_scout:
                        user_pattern = get_recent_activity_summary(seconds=120.0) or None
                        plan, objectives, popup = plan_next_10s(
                            frame,
                            api_key=api_key,
                            current_plan_remaining=remaining_plan() if current_plan else [],
                            executed_recent=list(last_actions),
                            user_pattern=user_pattern,
                            last_objective=last_objective,
                        )
                        if popup:
                            log("[takeover] Popup detected — clicking close (X then No).")
                            click_close(region)
                            time.sleep(0.3)
                            continue
                        if objectives:
                            last_objective = objectives
                    else:
                        plan = _default_plan_10s()
                        popup = False
                    current_plan = plan
                    plan_index = 0
                    precomputed_plan = None

            if need_plan():
                time.sleep(0.3)
                continue

            action, action_ms = current_plan[plan_index]
            n_actions += 1
            if n_actions % 8 == 1:
                if focus_roblox_and_click():
                    log("[takeover] Focused Roblox and clicked game window.")
            log(f"[takeover] step {plan_index + 1}/{len(current_plan)} | {action!r} {action_ms} ms")
            execute_action_ms(action, duration_ms=action_ms)
            last_actions.append(action)
            plan_index += 1

            # When 3 or fewer actions left, precompute next 10s (carry context; next plan does not include current remainder)
            use_scout = not args.no_scout and api_key is not None
            if plan_index >= len(current_plan) - 3 and len(current_plan) > 0 and precomputed_plan is None and use_scout:
                next_frame = capture_region(region=region, sct=sct)
                user_pattern = get_recent_activity_summary(seconds=120.0) or None
                next_plan, next_obj, next_popup = plan_next_10s(
                    next_frame,
                    api_key=api_key,
                    current_plan_remaining=current_plan[plan_index:],
                    executed_recent=list(last_actions),
                    user_pattern=user_pattern,
                    last_objective=last_objective,
                )
                precomputed_plan = next_plan
                precomputed_objective = next_obj
                precomputed_popup = next_popup
                log("[takeover] precomputed next 10s plan.")
    except KeyboardInterrupt:
        pass
    finally:
        sct.close()
    print("Done.")


if __name__ == "__main__":
    main()
