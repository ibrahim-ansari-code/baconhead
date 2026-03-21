#!/usr/bin/env python3
"""
Idle takeover — goal-oriented planning loop with persistent memory.

Architecture (informed by V-GEMS/HaltNav research):
  - Persistent spatial memory survives user-takeover events
  - Survey only runs on first spawn or after 3+ failures (not every takeover)
  - Per-step progress checking via frame diff (not optical flow)
  - Action history fed into every Claude prompt for loop avoidance

Usage:
    python run_takeover.py --idle 7 --game nds --monitor bot_log.tsv
"""

import argparse
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
from reward.input_state import start_listener, is_active, set_bot_pressing
from llm_agent.cem import execute_action_ms
from llm_agent.scout import plan_with_goal, verify_goal, survey_pick_best
from llm_agent.physics import degrees_to_ms, MAX_MOVEMENT_MS


# ── Frame comparison ──────────────────────────────────────────────────────────

def _frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Returns similarity [0, 1]. 1.0 = identical. Uses downscaled grayscale MAD.
    Immune to character jiggle unlike optical flow.
    """
    from PIL import Image as _Image
    sz = (112, 112)
    a_sm = np.array(_Image.fromarray(a.astype(np.uint8)).resize(sz, _Image.Resampling.LANCZOS)).astype(np.float32)
    b_sm = np.array(_Image.fromarray(b.astype(np.uint8)).resize(sz, _Image.Resampling.LANCZOS)).astype(np.float32)
    a_gray = a_sm.mean(axis=2)
    b_gray = b_sm.mean(axis=2)
    diff = np.abs(a_gray - b_gray).mean()
    return max(0.0, 1.0 - diff / 60.0)


# ── Survey ────────────────────────────────────────────────────────────────────

def run_survey(region, sct, api_key, game_name, log_fn) -> int:
    """
    Capture 4 compass frames, send all to Claude, pick the best direction.
    Returns index 0-3 of best direction. Camera ends facing that direction.
    """
    log_fn("[survey] Scanning 4 directions...")
    turn_ms = degrees_to_ms(90)
    frames = []

    for i in range(4):
        frame = capture_region(region=region, sct=sct)
        frames.append(frame)
        log_fn(f"[survey] Captured angle {i * 90}°")
        if i < 3:
            set_bot_pressing(True)
            execute_action_ms("look_right", turn_ms)
            set_bot_pressing(False)
            time.sleep(0.15)

    # Complete 360° back to original facing
    set_bot_pressing(True)
    execute_action_ms("look_right", turn_ms)
    set_bot_pressing(False)
    time.sleep(0.15)

    if not api_key:
        return -1

    best = survey_pick_best(frames, game=game_name, api_key=api_key)
    log_fn(f"[survey] Best direction: angle {best * 90}°")

    if best > 0:
        set_bot_pressing(True)
        execute_action_ms("look_right", degrees_to_ms(best * 90))
        set_bot_pressing(False)
        time.sleep(0.15)

    return best


# ── Persistent memory ─────────────────────────────────────────────────────────

class BotMemory:
    """
    Survives user-takeover events. Tracks what the bot has tried so Claude
    can avoid repeating failed approaches.
    """
    def __init__(self):
        self.action_history: deque = deque(maxlen=50)
        self.goal_history: deque = deque(maxlen=10)
        self.failure_reasons: deque = deque(maxlen=10)
        self.consecutive_failures = 0
        self.total_plans = 0
        self.has_surveyed = False
        self.last_survey_direction = -1

    def record_action(self, action: str, ms: int):
        self.action_history.append((action, ms))

    def record_goal(self, goal: str, status: str, reason: str = ""):
        self.goal_history.append((goal, status, reason))
        if status == "failed":
            self.consecutive_failures += 1
            if reason:
                self.failure_reasons.append(reason)
        elif status == "achieved":
            self.consecutive_failures = 0
        self.total_plans += 1

    def needs_survey(self) -> bool:
        if not self.has_surveyed:
            return True
        return self.consecutive_failures >= 3

    def mark_surveyed(self, direction: int):
        self.has_surveyed = True
        self.last_survey_direction = direction
        self.consecutive_failures = 0

    def get_history_summary(self) -> str:
        """Build a compact history string for Claude prompts."""
        parts = []
        if self.action_history:
            recent = list(self.action_history)[-8:]
            action_strs = [f"{a}({ms}ms)" for a, ms in recent]
            parts.append(f"Recent actions: {', '.join(action_strs)}")

        if self.goal_history:
            recent_goals = list(self.goal_history)[-3:]
            for goal, status, reason in recent_goals:
                entry = f"  - '{goal}' → {status}"
                if reason:
                    entry += f" ({reason})"
                parts.append(entry)
            if parts:
                parts.insert(len(parts) - len(recent_goals), "Recent goals:")

        if self.failure_reasons:
            recent_failures = list(self.failure_reasons)[-3:]
            parts.append(f"Failed approaches to AVOID: {'; '.join(recent_failures)}")

        return "\n".join(parts) if parts else ""

    def get_last_failure(self) -> Optional[str]:
        if self.failure_reasons:
            return self.failure_reasons[-1]
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Idle takeover — goal-oriented Claude loop")
    parser.add_argument("--idle",        type=float, default=3.0)
    parser.add_argument("--full-screen", action="store_true")
    parser.add_argument("--region",      type=str,   default=None)
    parser.add_argument("--no-scout",    action="store_true")
    parser.add_argument("--monitor",     type=str,   default=None)
    parser.add_argument("--max-step-ms", type=int,   default=None)
    parser.add_argument("--game",        type=str,   default=None)
    args = parser.parse_args()

    # ── Region ────────────────────────────────────────────────────────────────
    region = None
    if args.region:
        parts = [int(x.strip()) for x in args.region.split(",")]
        if len(parts) != 4:
            raise ValueError("--region must be left,top,width,height")
        region = {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
    elif args.full_screen:
        with mss.mss() as m:
            mon = m.monitors[0]
        region = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}
    else:
        for attempt in range(30):
            region = get_roblox_region()
            if region is not None:
                break
            if attempt == 0:
                print("Waiting for Roblox window...", flush=True)
            time.sleep(3)
        if region is None:
            print("Error: Roblox window not found.", file=sys.stderr)
            sys.exit(1)
    print(f"Region: {region}", flush=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not args.no_scout and not api_key:
        print("Warning: ANTHROPIC_API_KEY not set.", file=sys.stderr)

    start_listener()
    sct          = mss.mss()
    idle_sec     = args.idle
    max_step_ms  = args.max_step_ms or MAX_MOVEMENT_MS
    monitor_path = args.monitor
    game_name    = (args.game or "").strip().lower()
    use_scout    = not args.no_scout and bool(api_key)

    bot_active   = [False]
    focused_this_session = [False]
    # Track when the bot last generated an event (focus click, action, etc.)
    # CGEventSource can't distinguish bot events from user events, so we
    # suppress is_active() checks for idle_sec after any bot-generated event.
    last_bot_event_time = [0.0]

    # Plan state (cleared on user takeover)
    current_goal:  Optional[str]        = None
    action_queue:  deque                = deque()
    plan_frame:    Optional[np.ndarray] = None
    no_progress_streak = [0]

    NO_PROGRESS_THRESHOLD = 0.88
    NO_PROGRESS_ABORT     = 2

    # Persistent memory (survives user takeover)
    memory = BotMemory()

    def log(msg: str):
        print(msg, flush=True)

    def monitor_log(action: str, ms: int):
        if not monitor_path:
            return
        try:
            with open(monitor_path, "a") as f:
                f.write(f"{time.time():.3f}\tscout\t{action}\t{ms}\n")
        except Exception:
            pass

    def _clear_plan(reason: str = "", failed: bool = False, failure_reason: str = ""):
        nonlocal current_goal, plan_frame
        action_queue.clear()
        if failed and current_goal:
            memory.record_goal(current_goal, "failed", failure_reason or reason)
        current_goal = None
        plan_frame   = None
        no_progress_streak[0] = 0
        if reason:
            log(f"[takeover] Plan cleared: {reason}")

    def _is_dead_or_void(frame: np.ndarray) -> bool:
        """Detect death/respawn (bright white), underwater (solid blue), or loading (black)."""
        mean_bright = float(frame.mean())
        screen_std  = float(frame.std())
        # White respawn screen
        if mean_bright > 210 and screen_std < 25:
            return True
        # Underwater / fell off map: blue is dominant, very little ground/structure visible
        # Check the middle 70% of the screen (skip UI elements at top/bottom)
        if frame.ndim == 3:
            h = frame.shape[0]
            body = frame[int(h * 0.1):int(h * 0.9), :]
            blue_mean = float(body[..., 2].mean())
            red_mean  = float(body[..., 0].mean())
            # Blue dominant and red is very low = underwater
            if blue_mean > 180 and red_mean < 80:
                return True
        # Black loading screen
        if mean_bright < 15:
            return True
        return False

    def _is_map_edge(frame: np.ndarray) -> bool:
        """
        Detect if the character is facing the map edge (water/void ahead).
        Checks if the middle-horizontal band of the frame is predominantly
        blue (water) or sky, indicating the edge of the map.
        """
        if frame.ndim != 3:
            return False
        h, w = frame.shape[:2]
        # Check the middle third of the screen (where the horizon would be)
        mid_strip = frame[h // 3 : 2 * h // 3, w // 4 : 3 * w // 4]
        blue = mid_strip[..., 2].astype(float)
        red  = mid_strip[..., 0].astype(float)
        green = mid_strip[..., 1].astype(float)
        # Water/sky: blue > 120, blue > red, blue > green
        water_pct = float(((blue > 120) & (blue > red + 20) & (blue > green + 20)).mean())
        return water_pct > 0.35

    log(f"Watching. Takeover after {idle_sec}s idle. Scout={'on' if use_scout else 'off'}.")
    log("Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(0.15)
            frame = capture_region(region=region, sct=sct)

            # ── User active? ──────────────────────────────────────────────────
            # CGEventSource can't distinguish bot events from user events.
            # If we recently generated an event (focus click, action), skip
            # the active check — the idle timer was reset by US, not the user.
            bot_cooldown = (time.perf_counter() - last_bot_event_time[0]) < (idle_sec + 1.0)
            if not bot_cooldown and is_active(idle_sec):
                if bot_active[0]:
                    log("[takeover] User active — pausing bot.")
                    bot_active[0] = False
                    focused_this_session[0] = False
                    _clear_plan("user took over")
                continue

            # ── Bot activates ─────────────────────────────────────────────────
            if not bot_active[0]:
                bot_active[0] = True
                log("[takeover] Idle — bot active.")

            # Focus Roblox once per session, not every action.
            # The click resets the CGEventSource HID timer, so we must
            # wait past the idle threshold before resuming the loop,
            # otherwise is_active() will see our own click and stop us.
            if not focused_this_session[0]:
                set_bot_pressing(True)
                focus_roblox_and_click()
                set_bot_pressing(False)
                focused_this_session[0] = True
                last_bot_event_time[0] = time.perf_counter()
                time.sleep(0.3)
                continue

            # ── Death / underwater / loading screen ────────────────────────────
            if _is_dead_or_void(frame):
                log("[takeover] Dead/underwater/loading — waiting 3s")
                _clear_plan("dead")
                memory.has_surveyed = False
                set_bot_pressing(True)
                execute_action_ms("none", 3000)
                set_bot_pressing(False)
                last_bot_event_time[0] = time.perf_counter()
                continue

            # ── Map edge detection — turn around before falling ───────────────
            if _is_map_edge(frame):
                log("[takeover] MAP EDGE detected — turning 180°")
                _clear_plan("map edge", failed=True,
                            failure_reason="walked toward map edge / water — must go opposite direction")
                set_bot_pressing(True)
                execute_action_ms("look_right", degrees_to_ms(180))
                set_bot_pressing(False)
                last_bot_event_time[0] = time.perf_counter()
                continue

            # ── Need a new plan? ──────────────────────────────────────────────
            if not action_queue:
                # Verify previous goal
                if current_goal and plan_frame is not None:
                    sim = _frame_similarity(plan_frame, frame)
                    if sim > NO_PROGRESS_THRESHOLD:
                        log(f"[takeover] Frame diff {sim:.2f} — stuck, plan failed")
                        _clear_plan("no visual change", failed=True,
                                    failure_reason="character stuck — did not move at all")
                    elif use_scout:
                        status, fail_reason = verify_goal(plan_frame, frame, current_goal, api_key)
                        log(f"[takeover] Verify '{current_goal}': {status}")
                        if status == "achieved":
                            memory.record_goal(current_goal, "achieved")
                            _clear_plan("goal achieved")
                        elif status == "failed":
                            _clear_plan("goal failed", failed=True, failure_reason=fail_reason)
                        else:
                            # in_progress — record and continue with new plan from here
                            memory.record_goal(current_goal, "in_progress")
                            _clear_plan("extending plan")
                    else:
                        _clear_plan("no scout")

                # ── Survey (only when needed, not every time) ─────────────────
                if memory.needs_survey() and use_scout:
                    run_survey(region, sct, api_key, game_name, log)
                    last_bot_event_time[0] = time.perf_counter()
                    memory.mark_surveyed(0)
                    frame = capture_region(region=region, sct=sct)

                # ── Build context with memory ────────────────────────────────
                context_lines = []
                h = frame.shape[0]
                top_strip = frame[:max(1, int(h * 0.08)), :]
                red_pct = float(
                    ((top_strip[..., 0].astype(int) > 150) &
                     (top_strip[..., 1].astype(int) < 80)  &
                     (top_strip[..., 2].astype(int) < 80)).mean()
                )
                if red_pct > 0.04:
                    context_lines.append("DISASTER WARNING visible — act urgently!")

                history = memory.get_history_summary()
                if history:
                    context_lines.append(history)

                context_text = "\n".join(context_lines) if context_lines else "First action — no prior context."

                # ── Plan ──────────────────────────────────────────────────────
                if use_scout:
                    goal, steps = plan_with_goal(
                        frame,
                        context_text=context_text,
                        api_key=api_key,
                        game=game_name,
                        last_goal=current_goal,
                        last_failure=memory.get_last_failure(),
                    )
                else:
                    import random
                    goal  = "walk forward and explore"
                    steps = random.choice([
                        [("W", 500), ("W", 400)],
                        [("look_right", degrees_to_ms(90)), ("W", 500)],
                        [("look_left",  degrees_to_ms(90)), ("W", 500)],
                    ])

                current_goal = goal
                plan_frame   = frame
                no_progress_streak[0] = 0
                action_queue.clear()
                for step in steps:
                    action_queue.append(step)
                log(f"[plan] '{goal}' → {steps}")

            # ── Execute next step ─────────────────────────────────────────────
            if not action_queue:
                continue

            action, action_ms = action_queue.popleft()
            is_look = "look_" in action.lower()
            action_ms = max(100, min(action_ms, max_step_ms if not is_look else 2000))

            pre_frame = capture_region(region=region, sct=sct)

            log(f"  → {action} {action_ms}ms (steps_left={len(action_queue)})")
            monitor_log(action, action_ms)
            set_bot_pressing(True)
            execute_action_ms(action, duration_ms=action_ms)
            set_bot_pressing(False)
            last_bot_event_time[0] = time.perf_counter()
            memory.record_action(action, action_ms)

            time.sleep(0.05)
            post_frame = capture_region(region=region, sct=sct)

            # Per-step edge check: if we just walked toward the edge, abort NOW
            if not is_look and action.lower() != "none" and _is_map_edge(post_frame):
                log("[takeover] Walked toward map edge — aborting plan, turning 180°")
                _clear_plan("walked to edge", failed=True,
                            failure_reason="walked toward water/map edge — must go opposite direction")
                set_bot_pressing(True)
                execute_action_ms("look_right", degrees_to_ms(180))
                set_bot_pressing(False)
                last_bot_event_time[0] = time.perf_counter()
                continue

            # Per-step stuck check (movement actions only)
            if not is_look and action.lower() != "none":
                sim = _frame_similarity(pre_frame, post_frame)
                if sim > NO_PROGRESS_THRESHOLD:
                    no_progress_streak[0] += 1
                    log(f"  ⚠ No progress (sim={sim:.2f}, streak={no_progress_streak[0]})")
                    if no_progress_streak[0] >= NO_PROGRESS_ABORT:
                        log("[takeover] STUCK: aborting plan, turning away")
                        _clear_plan("stuck", failed=True,
                                    failure_reason="walked into wall — no movement for 2 steps")
                        import random
                        deg = random.choice([90, -90, 135, -135])
                        turn = "look_right" if deg > 0 else "look_left"
                        set_bot_pressing(True)
                        execute_action_ms(turn, degrees_to_ms(abs(deg)))
                        set_bot_pressing(False)
                        last_bot_event_time[0] = time.perf_counter()
                        memory.record_action(turn, degrees_to_ms(abs(deg)))
                        continue
                else:
                    no_progress_streak[0] = 0

    except KeyboardInterrupt:
        pass
    finally:
        sct.close()
    print("Done.")


if __name__ == "__main__":
    main()
